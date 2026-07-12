# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Technique 3 / NYU / base -- training script.

Slim entry point: argparse + config + build_models + data loader + train loop.
All models, losses, physics and datasets come from the shared packages.

Technique 3 objective (paper Eq. 21):  Ltotal = Ld + Lp + Lt + Lg
  Ld -- depth               (out_depth        vs DepthNorm(depth))
  Lp -- physics/Eq.11 image (pred_complex     vs complex_gt)
  Lt -- initial degraded    (pred_haze        vs haze)
  Lg -- direct branch       (out_direct       vs complex_gt)
"""

import os
import math
import argparse

import torch
import torch.nn as nn

from config import load_config
from models.model_builder import build_models
from data.nyu import get_train_loader, get_val_loader
from utils.helpers import AverageMeter, DepthNorm
from utils.physics import compute_haze_image, compute_complex_image
from utils.loss import ssim, gradient_loss
from utils.tb import make_writer, log_scalars, log_images, log_health

import warnings
# Keep the per-epoch output clean: torch/torchvision emit benign deprecation warnings.
warnings.filterwarnings("ignore")

TECHNIQUE = 3
VARIANT = 'base'
DATASET = 'NYU'

# SSIM's stabilisers are C1 = (0.01*L)^2 and C2 = (0.03*L)^2, so L must be the FIXED dynamic
# range of the domain being compared. Passing `depth_n.max()` made L a per-BATCH random
# variable (the nearest pixel in that batch), swinging C1/C2 by ~45x between neighbouring
# batches for identical prediction quality. The depth head is bounded to [1, 25] by the
# scaled sigmoid in models/decoder_1ch.py, so 25 is the range. Images stay on L = 1.
DEPTH_VAL_RANGE = 25.0

# Weight-head module names (models/model_builder.py): DepthModel -> trunk/w_sigmoid/w_softmax,
# ImageModel -> trunk/heads. 'base' has none, so the head group ends up empty.
_HEAD_KEYS = ('trunk', 'w_sigmoid', 'w_softmax', 'heads')


def main():
    parser = argparse.ArgumentParser(description='Train Technique 3 NYU base')
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

    # Flat list for gradient clipping; the split below only changes the LEARNING RATES.
    params = list(model_1.parameters()) + list(model_2.parameters())
    if model_3 is not None:
        params = params + list(model_3.parameters())

    head_params, body_params = [], []
    for m in (model_1, model_2, model_3):
        if m is None:
            continue
        for n_, p in m.named_parameters():
            (head_params if any(k in n_ for k in _HEAD_KEYS) else body_params).append(p)
    param_groups = [{'params': body_params, 'lr': cfg.learning_rate}]
    if head_params:  # Adam rejects an empty param group
        param_groups.append({'params': head_params,
                             'lr': cfg.learning_rate * cfg.weight_head_lr_mult})
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,
                                                           eta_min=cfg.lr_min)

    l1 = nn.L1Loss()
    lambda_l1, lambda_ssim = cfg.lambda_l1, cfg.lambda_ssim
    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg)
    writer = make_writer(cfg, TECHNIQUE, DATASET, VARIANT, args.logdir)

    def ssim_loss(pred, target, vr):
        return torch.clamp((1 - ssim(pred, target, val_range=vr)) * 0.5, 0, 1)

    # Declared BEFORE the resume block so a resumed run can override them. Previously
    # best_loss was reset to +inf AFTER resuming, so the first post-resume epoch always
    # overwrote the best checkpoint with a worse model.
    best_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0
    patience = cfg.early_stopping_patience

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model_1.load_state_dict(ckpt['state_dict_1'])
        model_2.load_state_dict(ckpt['state_dict_2'])
        if model_3 is not None and 'state_dict_3' in ckpt:
            model_3.load_state_dict(ckpt['state_dict_3'])
        # Adam's moment estimates and the cosine schedule are part of the training state:
        # restarting them from scratch is a different optimiser than the one that was saved.
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        best_loss = ckpt.get('best_loss', best_loss)
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        start_epoch = ckpt.get('cur_epoch', 0) + 1

    for epoch in range(start_epoch, cfg.epochs):
        model_1.train()
        model_2.train()
        if model_3 is not None:
            model_3.train()
        meter = AverageMeter()
        m_depth, m_complex, m_haze, m_direct = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        m_obj = AverageMeter()
        n_bad = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
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
            out_direct = model_3(image_full)
            # NOT out_depth.detach(). Eq.11 is I_Sim = t(I_Depth)*I + (1-t)*A + I_Residue, so
            # L_p MUST backprop through the transmission t into the depth head — that coupling
            # IS the physics-informed part. With the detach, model_1 was a standalone depth
            # regressor and the residual head absorbed every depth error.
            pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
            pred_haze = compute_haze_image(out_depth, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
            loss_depth = lambda_ssim * ssim_loss(out_depth, depth_n, DEPTH_VAL_RANGE) + lambda_l1 * l1(out_depth, depth_n)
            loss_depth = loss_depth + cfg.lambda_grad * gradient_loss(out_depth, depth_n)  # DenseDepth edge/gradient term
            loss_complex = lambda_ssim * ssim_loss(pred_complex, complex_gt, 1) + lambda_l1 * l1(pred_complex, complex_gt)
            loss_haze = lambda_ssim * ssim_loss(pred_haze, haze, 1) + lambda_l1 * l1(pred_haze, haze)
            loss_direct = lambda_ssim * ssim_loss(out_direct, complex_gt, 1) + lambda_l1 * l1(out_direct, complex_gt)
            total_loss = loss_depth + loss_complex + loss_haze + loss_direct

            # The reported TRAIN loss must be the SAME quantity as the reported VAL loss, or the
            # two numbers printed side by side mean nothing. total_fixed is v_loss term for term
            # (fixed cfg lambdas, unweighted sum) evaluated on the training batch. base has no
            # learned weights / EMA / barrier, so here it coincides with total_loss — it is still
            # computed and logged separately so the TB layout is identical across all 12 trainers.
            with torch.no_grad():
                total_fixed = loss_depth + loss_complex + loss_haze + loss_direct

            # Skip BEFORE the meters: a NaN batch that reached meter.update() poisoned the
            # running mean for the whole epoch, so the logged curve went NaN and stayed NaN.
            if not torch.isfinite(total_loss):
                n_bad += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            _bs = image_full.size(0)
            meter.update(total_fixed.item(), _bs)   # comparable loss -> "train=" / TB loss/total
            m_obj.update(total_loss.item(), _bs)    # what is actually minimised -> TB loss/objective
            m_depth.update(loss_depth.item(), _bs)
            m_complex.update(loss_complex.item(), _bs)
            m_haze.update(loss_haze.item(), _bs)
            m_direct.update(loss_direct.item(), _bs)

            total_loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
            optimizer.step()

        if n_bad > cfg.max_nonfinite_batches_per_epoch:
            raise RuntimeError(
                '[T%d %s %s] %d non-finite batches in epoch %d (limit %d) — training has diverged.'
                % (TECHNIQUE, DATASET, VARIANT, n_bad, epoch, cfg.max_nonfinite_batches_per_epoch))

        _tb_scalars = {'loss/total': meter.avg, 'loss/depth': m_depth.avg,
                       'loss/complex': m_complex.avg, 'loss/haze': m_haze.avg,
                       'loss/direct': m_direct.avg, 'loss/objective': m_obj.avg,
                       'lr': optimizer.param_groups[0]['lr'],
                       'train/nonfinite_batches': n_bad}
        log_scalars(writer, epoch, _tb_scalars)
        _tb_images = {'input/image_full': image_full, 'depth/pred': out_depth,
                      'depth/gt': depth_n, 'haze/pred': pred_haze, 'haze/gt': haze,
                      'complex/pred': pred_complex, 'complex/gt': complex_gt,
                      'direct/pred': out_direct}
        log_images(writer, epoch, _tb_images)
        # Early-warning system for the flat-airlight collapse: watch out_depth_frac_at_bound.
        log_health(writer, epoch, out_depth=out_depth, pred_complex=pred_complex,
                   pred_haze=pred_haze, complex_gt=complex_gt)
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
                out_direct = model_3(image_full)
                pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
                pred_haze = compute_haze_image(out_depth, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
                v_depth = lambda_ssim * ssim_loss(out_depth, depth_n, DEPTH_VAL_RANGE) + lambda_l1 * l1(out_depth, depth_n)
                v_depth = v_depth + cfg.lambda_grad * gradient_loss(out_depth, depth_n)
                v_complex = lambda_ssim * ssim_loss(pred_complex, complex_gt, 1) + lambda_l1 * l1(pred_complex, complex_gt)
                v_haze = lambda_ssim * ssim_loss(pred_haze, haze, 1) + lambda_l1 * l1(pred_haze, haze)
                v_direct = lambda_ssim * ssim_loss(out_direct, complex_gt, 1) + lambda_l1 * l1(out_direct, complex_gt)
                v_loss = v_depth + v_complex + v_haze + v_direct
                val_meter.update(v_loss.item(), image_full.size(0))
        val_avg = val_meter.avg
        log_scalars(writer, epoch, {'loss/val_total': val_avg})
        writer.flush()
        # train/val are the fixed-lambda comparable losses (same expression, different split, so
        # they may be read against each other); obj is the actual minimised objective. Do NOT
        # compare obj against val — for base they happen to coincide, for the variants obj is
        # weighted + EMA-normalised + barrier-augmented and lives on a different scale.
        _line = '[T%d %s %s] epoch %d/%d  train=%.4f  val=%.4f  obj=%.4f' % (
            TECHNIQUE, DATASET, VARIANT, epoch, cfg.epochs - 1, meter.avg, val_avg, m_obj.avg)
        if n_bad:
            _line += '  [%d non-finite batches skipped]' % n_bad
        print(_line)
        scheduler.step()

        # NaN never compares < best_loss, but be explicit: a diverged epoch must not be
        # allowed to define "best".
        improved = math.isfinite(val_avg) and val_avg < best_loss
        if improved:
            best_loss = val_avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        state = {'state_dict_1': model_1.state_dict(), 'state_dict_2': model_2.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'cur_epoch': epoch, 'best_loss': best_loss,
                 'epochs_no_improve': epochs_no_improve}
        if model_3 is not None:
            state['state_dict_3'] = model_3.state_dict()
        # '_last' every epoch so a crash mid-run is recoverable; the unsuffixed name stays
        # the BEST checkpoint (what test.py loads by default).
        torch.save(state, os.path.join(cfg.checkpoint_dir, 'T%d_%s_%s_last.ckpt' % (TECHNIQUE, DATASET, VARIANT)))
        if improved:
            torch.save(state, os.path.join(cfg.checkpoint_dir, 'T%d_%s_%s.ckpt' % (TECHNIQUE, DATASET, VARIANT)))
        elif epochs_no_improve >= patience:
            print('[T%d %s %s] early stopping at epoch %d (no val improvement for %d epochs)' % (TECHNIQUE, DATASET, VARIANT, epoch, patience))
            break


if __name__ == '__main__':
    main()
