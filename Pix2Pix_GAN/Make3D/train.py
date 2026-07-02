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

"""Pix2Pix GAN / Make3D -- training (clean image_half -> degraded complex image)."""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from gan_models import *
from gan_utils import ReplayBuffer, LambdaLR, Logger, weights_init_normal
from utils.helpers import AverageMeter
from utils.metrics import add_results_1
from config import load_config
from data.make3d import get_train_loader, get_val_loader


def main():
    parser = argparse.ArgumentParser(description='Pix2Pix GAN - Make3D')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='resume from the saved Pix2Pix checkpoints')
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=None, help='override config epochs')
    parser.add_argument('--batchSize', type=int, default=None, help='override config batch_size_make3d')
    parser.add_argument('--lr', type=float, default=None, help='override config learning_rate')
    parser.add_argument('--decay_epoch', type=int, default=10, help='epoch to start linearly decaying lr to 0')
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    opt = parser.parse_args()

    cfg = load_config(opt.config)
    n_epochs = opt.n_epochs if opt.n_epochs is not None else cfg.epochs
    lr = opt.lr if opt.lr is not None else cfg.learning_rate
    if opt.batchSize is not None:
        cfg.batch_size_make3d = opt.batchSize

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    lambda_pixel = 100

    generator = GeneratorUNet_Make3D().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=LambdaLR(n_epochs, opt.epoch, opt.decay_epoch).step)

    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg)
    writer = SummaryWriter(os.path.join(cfg.runs_dir, 'Pix2Pix_GAN', 'make3d'))

    ckpt_dir = os.path.join(cfg.checkpoint_dir, 'Pix2Pix_GAN')
    # ONE bundled checkpoint (generator + discriminator), best-only.
    ckpt_path = os.path.join(ckpt_dir, 'Pix2Pix_Make3D.ckpt')

    start_epoch = opt.epoch
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location=device)
        generator.load_state_dict(ckpt['state_dict_G'])
        discriminator.load_state_dict(ckpt['state_dict_D'])
        start_epoch = ckpt.get('cur_epoch', 0) + 1

    # PatchGAN output size for a 173x230 (HxW) half-resolution Make3D input.
    patch = (1, 173 // 2 ** 4, 230 // 2 ** 4)
    Tensor = torch.cuda.FloatTensor if is_use_cuda else torch.FloatTensor

    # Make3D: "best" is tracked on a held-out VALIDATION split with early stopping
    # (matches the Make3D Technique_* scripts). The single scalar is the generator
    # loss; the discriminator is saved only to allow resume.
    best_loss = float('inf')
    epochs_no_improve = 0
    # GANs are noisy and improve slowly (the lr decays across all 50 epochs), so
    # give early stopping much more room than the regression Techniques' default (5).
    patience = max(cfg.early_stopping_patience, 20)

    for epoch in range(start_epoch, n_epochs):
        generator.train(); discriminator.train()
        losses_G, losses_D = AverageMeter(), AverageMeter()

        for sample_batched in train_loader:
            input_A = Variable(sample_batched['image_half'].to(device))          # clean (domain A)
            input_B = Variable(sample_batched['complex_noise_img'].to(device))   # degraded (domain B)

            valid = Variable(Tensor(np.ones((input_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((input_A.size(0), *patch))), requires_grad=False)

            # ---- Generator ----
            optimizer_G.zero_grad()
            fake_B = generator(input_A)
            fake_B = F.interpolate(fake_B, size=(173, 230), mode='bicubic', align_corners=False)
            pred_fake = discriminator(fake_B, input_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, input_B)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()
            losses_G.update(loss_G.item(), input_A.size(0))

            # ---- Discriminator ----
            optimizer_D.zero_grad()
            pred_real = discriminator(input_B.type_as(fake_B), input_A)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake = discriminator(fake_B.detach(), input_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            losses_D.update(loss_D.item(), input_A.size(0))

        lr_scheduler_G.step(); lr_scheduler_D.step()

        # ---- validation on held-out split (drives checkpointing + early stopping) ----
        generator.eval(); discriminator.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for sample_batched in val_loader:
                input_A = sample_batched['image_half'].to(device)
                input_B = sample_batched['complex_noise_img'].to(device)
                fake_B = generator(input_A)
                fake_B = F.interpolate(fake_B, size=(173, 230), mode='bicubic', align_corners=False)
                # Select the checkpoint by the actual eval metric (abs_rel), not the
                # GAN loss, so "best" tracks accuracy rather than adversarial balance.
                abs_rel = add_results_1(input_B, fake_B, border_crop_size=16)[0]
                if torch.isfinite(abs_rel):
                    val_meter.update(abs_rel.item(), input_A.size(0))
        val_avg = val_meter.avg

        writer.add_scalar('loss_G', losses_G.avg, epoch)
        writer.add_scalar('loss_D', losses_D.avg, epoch)
        writer.add_scalar('loss_val_G', val_avg, epoch)
        writer.flush()
        print('[Pix2Pix Make3D] epoch %d/%d  train_G=%.4f  loss_D=%.4f  val_rel=%.4f' % (
            epoch, n_epochs - 1, losses_G.avg, losses_D.avg, val_avg))

        if val_avg < best_loss:
            best_loss = val_avg
            epochs_no_improve = 0
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({'state_dict_G': generator.state_dict(),
                        'state_dict_D': discriminator.state_dict(),
                        'cur_epoch': epoch, 'best_loss': best_loss},
                       ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('[Pix2Pix Make3D] early stopping at epoch %d (no val improvement for %d epochs)' % (epoch, patience))
                break

    writer.close()


if __name__ == '__main__':
    main()
