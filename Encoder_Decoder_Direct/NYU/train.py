# --- path bootstrap: baseline dir + repo root (shared packages) ---
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

"""Encoder_Decoder_Direct / NYU -- training script.

Direct (encoder->decoder) baseline: a 3-channel image model is trained to
regress the complex (ricardo) degraded image straight from the full-size input.
Logic preserved from the original ``train_4.py``; only plumbing/paths/imports
changed to use the shared packages.
"""

import os
import math
import argparse

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from config import load_config
from models.model_builder import ImageModel
from data.nyu import get_train_loader
from utils.helpers import AverageMeter, DepthNorm, colorize, simple_save_images
from utils.loss import ssim, gradient_loss
from utils.tb import log_images


def main():
    parser = argparse.ArgumentParser(description='Encoder_Decoder_Direct NYU training')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to resume from')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = ImageModel(pretrained=cfg.pretrained_encoder).to(device)
    print('Model created.')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model.cuda())
    print('model and cuda mixing done')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

    best_loss = 100.0
    batch_size = cfg.batch_size_nyu

    # Load data
    train_loader = get_train_loader(cfg)

    # Loss
    l1_criterion = nn.L1Loss()
    print("Total number of batches in train loader are :", len(train_loader))

    writer = SummaryWriter(os.path.join(cfg.runs_dir, 'Encoder_Decoder_Direct', 'nyu'))
    ckpt_dir = os.path.join(cfg.checkpoint_dir, 'Encoder_Decoder_Direct')
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict_3'])
        start_epoch = checkpoint.get('cur_epoch', 0) + 1

    # Start training...
    for epoch in range(start_epoch, cfg.epochs):

        losses = AverageMeter()
        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, cfg.epochs - 1))
        print('-' * 10)

        # Switch to train mode
        model.train()

        keep_all_batch_losses = []
        total_batch_loss = 0.0
        running_batch_loss = 0.0

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = sample_batched['image_full'].to(device)  # full size
            complex_image_tensor = sample_batched['complex_noise_img'].to(device)  # half size

            output_ricardo_image_direct = model(image)

            # Compute the loss for complex haze image which is calculated directly from the original image
            loss_ricardo_image_direct = l1_criterion(output_ricardo_image_direct, complex_image_tensor)
            loss_ssim_ricardo_image_direct = torch.clamp((1 - ssim(output_ricardo_image_direct.float(), complex_image_tensor.float(),
                                                            val_range=1)) * 0.5, 0, 1)
            # DenseDepth [51] loss = L1 + SSIM + edge/gradient term (the last was missing).
            loss_grad_ricardo = gradient_loss(output_ricardo_image_direct, complex_image_tensor)
            loss_ricardo_image_total = (cfg.lambda_l1 * loss_ricardo_image_direct) + (cfg.lambda_ssim * loss_ssim_ricardo_image_direct) + (cfg.lambda_grad * loss_grad_ricardo)
            del complex_image_tensor, output_ricardo_image_direct

            # Update step
            total_batch_loss = loss_ricardo_image_total
            running_batch_loss += total_batch_loss.item() * image.size(0)  # dividing by number of images in the batch

            keep_all_batch_losses.append(total_batch_loss.item())
            losses.update(total_batch_loss.data.item(), image.size(0))
            total_batch_loss.backward()

            optimizer.step()

            torch.cuda.empty_cache()

        if math.isnan(losses.avg):
            print('I need to check you')

        # Log progress; print after every epochs into the console
        print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} '.format(epoch, losses.avg))
        writer.add_scalar('Train/Each Epoch Loss', losses.avg, epoch)
        # Log input / prediction / GT for the last batch (re-forward; the loop frees them).
        model.eval()
        with torch.no_grad():
            _pred = model(sample_batched['image_full'].to(device))
            log_images(writer, epoch, {'input': sample_batched['image_half'].to(device),
                                       'pred': _pred,
                                       'gt': sample_batched['complex_noise_img'].to(device)})
        writer.flush()

        if losses.avg < best_loss:
            print("Here the training loss got reduced, hence printing")
            print('Current best epoch loss is {:.4f}'.format(losses.avg), 'previous best was {}'.format(best_loss))
            best_loss = losses.avg
            _save_best_model(model, best_loss, epoch, ckpt_dir)


def _save_best_model(model, best_loss, epoch, ckpt_dir):
    # Save Model
    state = {
        'state_dict_3': model.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }
    torch.save(state, os.path.join(ckpt_dir, 'Models.ckpt'))


if __name__ == '__main__':
    main()
