# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Technique 1 / KITTI / base -- training script.

Slim entry point: argparse + config + build_models + data loader + train loop.
All models, losses, physics and datasets come from the shared packages.
"""

import os
import argparse

import torch
import torch.nn as nn

from config import load_config
from models.model_builder import build_models
from data.kitti import get_train_loader, get_val_loader
from utils.helpers import AverageMeter, DepthNorm
from utils.physics import compute_haze_image, compute_complex_image
from utils.loss import ssim, gradient_loss
from utils.tb import make_writer, log_scalars, log_images, log_weights

import warnings
# Keep the per-epoch output clean: torch/torchvision emit benign deprecation warnings.
warnings.filterwarnings("ignore")

TECHNIQUE = 1
VARIANT = 'base'
DATASET = 'KITTI'


def main():
    parser = argparse.ArgumentParser(description='Train Technique 1 KITTI base')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to resume from')
    parser.add_argument('--logdir', default=None, help='override TensorBoard runs_dir root')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_1, model_2, model_3 = build_models(TECHNIQUE, VARIANT)
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    if model_3 is not None:
        model_3 = model_3.to(device)

    params = list(model_1.parameters()) + list(model_2.parameters())
    if model_3 is not None:
        params = params + list(model_3.parameters())
    optimizer = torch.optim.Adam(params, cfg.learning_rate)

    l1 = nn.L1Loss()
    lambda_l1, lambda_ssim, lambda_perc = cfg.lambda_l1, cfg.lambda_ssim, cfg.lambda_perc
    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg)
    writer = make_writer(cfg, TECHNIQUE, DATASET, VARIANT, args.logdir)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model_1.load_state_dict(ckpt['state_dict_1'])
        model_2.load_state_dict(ckpt['state_dict_2'])
        if model_3 is not None and 'state_dict_3' in ckpt:
            model_3.load_state_dict(ckpt['state_dict_3'])
        start_epoch = ckpt.get('cur_epoch', 0) + 1

    def ssim_loss(pred, target, vr):
        return torch.clamp((1 - ssim(pred, target, val_range=vr)) * 0.5, 0, 1)

    best_loss = float('inf')
    patience = cfg.early_stopping_patience
    epochs_no_improve = 0
    for epoch in range(start_epoch, cfg.epochs):
        model_1.train()
        model_2.train()
        if model_3 is not None:
            model_3.train()
        meter = AverageMeter()
        m_depth, m_complex, m_haze, m_direct = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for batch in train_loader:
            optimizer.zero_grad()
            image_full = batch['image_full'].to(device)
            image_half = batch['image_half'].to(device)
            depth = batch['depth'].to(device)
            haze = batch['haze_image'].to(device)
            beta = batch['beta'].to(device)
            a_val = batch['a_val'].to(device)
            unit = batch['unit_mat'].to(device)
            complex_gt = batch['complex_noise_img'].to(device)
            depth_n = DepthNorm(depth)

            out_depth = model_1(image_full)
            out_bb = model_2(image_full)
            pred_complex = compute_complex_image(out_depth.detach(), out_bb, beta, a_val, unit, image_half, max_depth_m=cfg.kitti_max_depth_m)
            pred_haze = compute_haze_image(out_depth, beta, a_val, unit, image_half, max_depth_m=cfg.kitti_max_depth_m)
            loss_depth = lambda_ssim * ssim_loss(out_depth, depth_n, float(depth_n.max())) + lambda_l1 * l1(out_depth, depth_n)
            loss_depth = loss_depth + cfg.lambda_grad * gradient_loss(out_depth, depth_n)  # DenseDepth edge/gradient term
            loss_complex = lambda_ssim * ssim_loss(pred_complex, complex_gt, 1) + lambda_l1 * l1(pred_complex, complex_gt)
            loss_haze = lambda_ssim * ssim_loss(pred_haze, haze, 1) + lambda_l1 * l1(pred_haze, haze)
            # Technique 1 (paper Eq. 11): Ltotal = Ld + Lp only (no Lt/haze term; T1 != T2).
            total_loss = loss_depth + loss_complex

            meter.update(total_loss.item(), image_full.size(0))
            _bs = image_full.size(0)
            m_depth.update(loss_depth.item(), _bs)
            m_complex.update(loss_complex.item(), _bs)
            m_haze.update(loss_haze.item(), _bs)
            if model_3 is not None:
                m_direct.update(loss_direct.item(), _bs)
            total_loss.backward()
            optimizer.step()

        _tb_scalars = {'loss/total': meter.avg, 'loss/depth': m_depth.avg,
                       'loss/complex': m_complex.avg, 'loss/haze': m_haze.avg,
                       'lr': optimizer.param_groups[0]['lr']}
        if model_3 is not None:
            _tb_scalars['loss/direct'] = m_direct.avg
        log_scalars(writer, epoch, _tb_scalars)
        _tb_weights = {}
        if VARIANT != 'base':
            _tb_weights['w_depth'] = w_depth
            if TECHNIQUE == 4:
                _tb_weights['w_residue'] = w_residue
                _tb_weights['w_deg'] = w_deg
            else:
                _tb_weights['w_bb'] = w_bb
            if model_3 is not None:
                _tb_weights['w_dir'] = w_dir
            if VARIANT == 'var2':
                _tb_weights['w_global'] = w_global
        log_weights(writer, epoch, _tb_weights)
        _tb_images = {'input/image_full': image_full, 'depth/pred': out_depth,
                      'depth/gt': depth_n, 'haze/pred': pred_haze, 'haze/gt': haze,
                      'complex/pred': pred_complex, 'complex/gt': complex_gt}
        if model_3 is not None:
            _tb_images['direct/pred'] = out_direct
        log_images(writer, epoch, _tb_images)
        writer.flush()
        # ---- validation on held-out split (drives checkpointing + early stopping) ----
        model_1.eval()
        model_2.eval()
        if model_3 is not None:
            model_3.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                image_full = batch['image_full'].to(device)
                image_half = batch['image_half'].to(device)
                depth_n = DepthNorm(batch['depth'].to(device))
                haze = batch['haze_image'].to(device)
                beta = batch['beta'].to(device)
                a_val = batch['a_val'].to(device)
                unit = batch['unit_mat'].to(device)
                complex_gt = batch['complex_noise_img'].to(device)
                out_depth = model_1(image_full)
                out_bb = model_2(image_full)
                pred_complex = compute_complex_image(out_depth.detach(), out_bb, beta, a_val, unit, image_half, max_depth_m=cfg.kitti_max_depth_m)
                pred_haze = compute_haze_image(out_depth, beta, a_val, unit, image_half, max_depth_m=cfg.kitti_max_depth_m)
                loss_depth = lambda_ssim * ssim_loss(out_depth, depth_n, float(depth_n.max())) + lambda_l1 * l1(out_depth, depth_n)
                loss_depth = loss_depth + cfg.lambda_grad * gradient_loss(out_depth, depth_n)  # DenseDepth edge/gradient term
                loss_complex = lambda_ssim * ssim_loss(pred_complex, complex_gt, 1) + lambda_l1 * l1(pred_complex, complex_gt)
                loss_haze = lambda_ssim * ssim_loss(pred_haze, haze, 1) + lambda_l1 * l1(pred_haze, haze)
                v_loss = loss_depth + loss_complex  # T1: Ld + Lp only (match train objective)
                val_meter.update(v_loss.item(), image_full.size(0))
        val_avg = val_meter.avg
        log_scalars(writer, epoch, {'loss/val_total': val_avg})
        writer.flush()
        print('[T%d %s %s] epoch %d/%d  train=%.4f  val=%.4f' % (TECHNIQUE, DATASET, VARIANT, epoch, cfg.epochs - 1, meter.avg, val_avg))
        if val_avg < best_loss:
            best_loss = val_avg
            epochs_no_improve = 0
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            state = {'state_dict_1': model_1.state_dict(), 'state_dict_2': model_2.state_dict(),
                     'cur_epoch': epoch, 'best_loss': best_loss}
            if model_3 is not None:
                state['state_dict_3'] = model_3.state_dict()
            ckpt_name = 'T%d_%s_%s.ckpt' % (TECHNIQUE, DATASET, VARIANT)
            torch.save(state, os.path.join(cfg.checkpoint_dir, ckpt_name))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('[T%d %s %s] early stopping at epoch %d (no val improvement for %d epochs)' % (TECHNIQUE, DATASET, VARIANT, epoch, patience))
                break


if __name__ == '__main__':
    main()
