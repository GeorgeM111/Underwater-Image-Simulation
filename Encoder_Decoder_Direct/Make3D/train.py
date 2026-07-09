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

"""Encoder_Decoder_Direct / Make3D -- training script.

Direct (encoder->decoder) baseline: a 3-channel image model regresses the
complex (ricardo) degraded image straight from the full-size input.

Aligned with the Technique_* convention: held-out validation split, one
best-on-validation checkpoint, early stopping, and a single per-epoch
``train=.. val=..`` print line.
"""

import os
import argparse

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from config import load_config
from models.model_builder import ImageModel
from data.make3d import get_train_loader, get_val_loader
from utils.helpers import AverageMeter
from utils.loss import ssim, gradient_loss
from utils.tb import log_images

TAG = 'EncDec Make3D'


def main():
    parser = argparse.ArgumentParser(description='Encoder_Decoder_Direct Make3D training')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to resume from')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageModel(pretrained=cfg.pretrained_encoder).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)
    l1 = nn.L1Loss()
    lambda_l1, lambda_ssim = cfg.lambda_l1, cfg.lambda_ssim

    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg)

    writer = SummaryWriter(os.path.join(cfg.runs_dir, 'Encoder_Decoder_Direct', 'make3d'))
    ckpt_dir = os.path.join(cfg.checkpoint_dir, 'Encoder_Decoder_Direct')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'Models_Make3D.ckpt')

    def ssim_loss(pred, target):
        return torch.clamp((1 - ssim(pred.float(), target.float(), val_range=1)) * 0.5, 0, 1)

    def batch_loss(sample):
        image = sample['image_full'].to(device)
        complex_gt = sample['complex_noise_img'].to(device)
        out = model(image)
        # DenseDepth [51] loss = L1 + SSIM + edge/gradient term (the last was missing).
        loss = (lambda_l1 * l1(out, complex_gt) + lambda_ssim * ssim_loss(out, complex_gt)
                + cfg.lambda_grad * gradient_loss(out, complex_gt))
        return loss, image.size(0)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict_3'])
        start_epoch = ckpt.get('cur_epoch', 0) + 1

    best_loss = float('inf')
    patience = cfg.early_stopping_patience
    epochs_no_improve = 0
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        meter = AverageMeter()
        for sample in train_loader:
            optimizer.zero_grad()
            loss, bs = batch_loss(sample)
            meter.update(loss.item(), bs)
            loss.backward()
            optimizer.step()

        # ---- validation on held-out split (drives checkpointing + early stopping) ----
        model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for sample in val_loader:
                loss, bs = batch_loss(sample)
                val_meter.update(loss.item(), bs)
        val_avg = val_meter.avg

        writer.add_scalar('loss/total', meter.avg, epoch)
        writer.add_scalar('loss/val_total', val_avg, epoch)
        # Log input / prediction / GT for the last val batch (model already in eval()).
        with torch.no_grad():
            _pred = model(sample['image_full'].to(device))
            log_images(writer, epoch, {'input': sample['image_half'].to(device),
                                       'pred': _pred,
                                       'gt': sample['complex_noise_img'].to(device)})
        writer.flush()
        print('[%s] epoch %d/%d  train=%.4f  val=%.4f' % (TAG, epoch, cfg.epochs - 1, meter.avg, val_avg))

        if val_avg < best_loss:
            best_loss = val_avg
            epochs_no_improve = 0
            torch.save({'state_dict_3': model.state_dict(), 'cur_epoch': epoch, 'best_loss': best_loss}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('[%s] early stopping at epoch %d (no val improvement for %d epochs)' % (TAG, epoch, patience))
                break

    writer.close()


if __name__ == '__main__':
    main()
