# --- path bootstrap: baseline dir (local modules) + repo root (shared packages) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)
if _BASE not in _sys.path:
    _sys.path.insert(0, _BASE)
_p = _BASE
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

import argparse
import itertools

from torch.autograd import Variable
import os

import torch
import torch.nn as nn
import torchvision.utils as vutils

from tensorboardX import SummaryWriter

from gan_models import *
from gan_utils import ReplayBuffer, LambdaLR, weights_init_normal, to_gan_range, from_gan_range
from utils.helpers import AverageMeter
from utils.tb import log_images
from config import load_config
from data.nyu import get_train_loader, get_test_loader
import numpy as np
import warnings
# Don't promote warnings to errors: benign PyTorch deprecation/legacy-tensor warnings
# would otherwise crash training (matches the CycleGAN Make3D script).
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint prefix to resume from')
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--decay_epoch', type=int, default=10, help='linearly decaying the learning rate to 0')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')

    opt = parser.parse_args()

    cfg = load_config(opt.config)
    n_epochs = cfg.epochs
    lr = opt.lr if getattr(opt, 'lr', None) is not None else cfg.gan_learning_rate
    batchSize = cfg.batch_size_nyu

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    model_name = "cycle_gan_task"

    ckpt_dir = os.path.join(cfg.checkpoint_dir, 'CycleGAN')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------ Definition of variables -------------------------------------
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)  # discriminate the generated A
    netD_B = Discriminator(opt.output_nc)  # discriminate the generated B

    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        nn.DataParallel(netG_A2B.cuda())
        nn.DataParallel(netG_B2A.cuda())
        nn.DataParallel(netD_A.cuda())
        nn.DataParallel(netD_B.cuda())

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # -------------------------- Losses --------------------------------------------------
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # Single "best" scalar tracked on the generator loss (NYU: training loss,
    # matching the NYU Technique_* scripts). All four sub-networks are bundled
    # into ONE best-only checkpoint instead of four separate files.
    best_loss = float('inf')

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                    lr_lambda=LambdaLR(n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                        lr_lambda=LambdaLR(n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                        lr_lambda=LambdaLR(n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if is_use_cuda else torch.Tensor

    target_real = Variable(Tensor(batchSize, 1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batchSize, 1).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Data
    train_loader = get_train_loader(cfg)

    writer_1 = SummaryWriter(os.path.join(cfg.runs_dir, 'CycleGAN', 'nyu'))

    ckpt_path = os.path.join(ckpt_dir, 'CycleGAN_NYU.ckpt')
    start_epoch = opt.epoch
    if opt.resume is not None:
        ckpt = torch.load(opt.resume, map_location=device)
        netG_A2B.load_state_dict(ckpt['state_dict_G_A2B'])
        netG_B2A.load_state_dict(ckpt['state_dict_G_B2A'])
        netD_A.load_state_dict(ckpt['state_dict_D_A'])
        netD_B.load_state_dict(ckpt['state_dict_D_B'])
        start_epoch = ckpt.get('cur_epoch', 0) + 1

    # ------------------------ Training ----------------------------------------------
    for epoch in range(start_epoch, n_epochs):

        losses_G = AverageMeter()
        losses_G_Identity = AverageMeter()
        losses_G_GAN = AverageMeter()
        losses_G_Cycle = AverageMeter()
        losses_G_D = AverageMeter()

        N = len(train_loader)

        # Switch to train model
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()

        loss_G = 0.0
        loss_D_A = 0.0
        loss_D_B = 0.0

        keep_all_batch_losses_G = []
        running_batch_losses_G = 0.0

        keep_all_batch_losses_D_A = []
        running_batch_losses_D_A = 0.0

        keep_all_batch_losses_D_B = []
        running_batch_losses_D_B = 0.0

        num_batches = 0

        for i, sample_batched in enumerate(train_loader):

            num_batches = num_batches + 1

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image_full'].to(device))  # full size
            input_A = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size ; image_half

            input_B = torch.autograd.Variable(
                sample_batched['complex_noise_img'].to(device))  # half size ; complex_image_tensor

            # Set model input
            real_A = input_A
            real_B = input_B

            del input_A, input_B

            real_A = real_A.to(device, dtype=torch.float)
            real_B = real_B.to(device, dtype=torch.float)
            # CycleGAN trains in [-1,1] (Tanh generators); normalise the [0,1] loader data.
            real_A = to_gan_range(real_A)
            real_B = to_gan_range(real_B)
            # --------------------------------- Generators A2B and B2A -----------------------------------
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)  # generate A from B
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)  # generate B from A
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            running_batch_losses_G += loss_G.item() * image.size(dim=0)

            losses_G.update(loss_G.data.item(), image.size(dim=0))
            losses_G_Identity.update((loss_identity_A + loss_identity_B).data.item(), image.size(dim=0))
            losses_G_GAN.update((loss_GAN_A2B + loss_GAN_B2A).data.item(), image.size(dim=0))
            losses_G_Cycle.update((loss_cycle_ABA + loss_cycle_BAB).data.item(), image.size(dim=0))

            keep_all_batch_losses_G.append(loss_G.item())
            loss_G.backward()
            optimizer_G.step()

            # --------------------------------- Discriminator A --------------------------------------------------
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            running_batch_losses_D_A += loss_D_A.item() * image.size(dim=0)
            keep_all_batch_losses_D_A.append(loss_D_A.item())
            loss_D_A.backward()

            optimizer_D_A.step()

            # ------------------------------------ Discriminator B ------------------------------------------------
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            running_batch_losses_D_B += loss_D_B.item() * image.size(dim=0)
            keep_all_batch_losses_D_B.append(loss_D_B.item())
            loss_D_B.backward()

            optimizer_D_B.step()

            losses_G_D.update((loss_D_A + loss_D_B).data.item(), image.size(dim=0))

        epoch_loss_G = running_batch_losses_G / num_batches  # epoch error
        batch_mean_loss_G = (np.mean(keep_all_batch_losses_G))

        epoch_loss_D_A = running_batch_losses_D_A / num_batches  # epoch error
        batch_mean_loss_D_A = (np.mean(keep_all_batch_losses_D_A))

        epoch_loss_D_B = running_batch_losses_D_B / num_batches  # epoch error
        batch_mean_loss_D_B = (np.mean(keep_all_batch_losses_D_B))

        # Log progress; one line per epoch (matches the Technique_* print style).
        print('[CycleGAN NYU] epoch %d/%d  loss_G=%.4f  loss_D_A=%.4f  loss_D_B=%.4f' % (
            epoch, n_epochs - 1, epoch_loss_G, epoch_loss_D_A, epoch_loss_D_B))

        # save the losses in each epoch
        writer_1.add_scalar('loss_G', loss_G, epoch)
        writer_1.add_scalar('loss_G_identity', (loss_identity_A + loss_identity_B), epoch)
        writer_1.add_scalar('loss_G_GAN', (loss_GAN_A2B + loss_GAN_B2A), epoch)
        writer_1.add_scalar('loss_G_cycle', (loss_cycle_ABA + loss_cycle_BAB), epoch)
        writer_1.add_scalar('loss_D', (loss_D_A + loss_D_B), epoch)

        # Log image grids EVERY epoch (was last-epoch-only, and referenced an
        # unimported `vutils`). log_images handles the grid + normalisation.
        log_images(writer_1, epoch, {'real_A': from_gan_range(real_A), 'fake_B': from_gan_range(fake_B),
                                     'real_B': from_gan_range(real_B), 'fake_A': from_gan_range(fake_A)})

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save the LATEST checkpoint every epoch (overwrite). A GAN's loss_G is not a
        # quality signal — it oscillates as the G/D balance shifts — so "best-by-loss"
        # just froze the FIRST epoch. Keep the most recent weights instead.
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({'state_dict_G_A2B': netG_A2B.state_dict(),
                    'state_dict_G_B2A': netG_B2A.state_dict(),
                    'state_dict_D_A': netD_A.state_dict(),
                    'state_dict_D_B': netD_B.state_dict(),
                    'cur_epoch': epoch, 'epoch_loss_G': epoch_loss_G},
                   ckpt_path)

if __name__ == '__main__':
    main()
