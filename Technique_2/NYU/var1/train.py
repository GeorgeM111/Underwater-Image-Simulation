# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Technique 2 / NYU / var1 -- training script.

Slim entry point: argparse + config + build_models + data loader + train loop.
All models, losses, physics and datasets come from the shared packages.

Loss (paper Eq. 14-17):  L_total = L_d + L_p + L_t   (no direct branch -> model_3 is None)
var1 = the three terms are summed UNWEIGHTED, but the L1-vs-SSIM balance INSIDE each term
is a learned sigmoid weight (w, 1-w): model_1 -> w_depth[0], model_2 -> w_bb[0] (L_p),
w_bb[1] (L_t).
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
from utils.loss_balance import EMANormalizer, weight_log_barrier
from utils.tb import make_writer, log_scalars, log_images, log_weights, log_health

import warnings
# Keep the per-epoch output clean: torch/torchvision emit benign deprecation warnings.
warnings.filterwarnings("ignore")

TECHNIQUE = 2
VARIANT = 'var1'
DATASET = 'NYU'

# SSIM's stabilisers are C1 = (0.01*L)^2 and C2 = (0.03*L)^2, so L is part of the LOSS
# DEFINITION, not a property of the batch. Passing depth_n.max() made L a per-batch random
# variable (the nearest pixel in that batch), swinging C1/C2 by ~45x between neighbouring
# batches for identical prediction quality. L must be the fixed dynamic range of the domain:
# the depth head is bounded to [1, 25] by the scaled sigmoid in models/decoder_1ch.py.
DEPTH_VAL_RANGE = 25.0

# EMA slots — one per quantity that is multiplied by a LEARNED weight (see EMANormalizer).
# The gradient term carries a FIXED lambda, so it needs no slot.
N_SSIM_D, N_L1_D, N_SSIM_P, N_L1_P, N_SSIM_T, N_L1_T = range(6)


def main():
    parser = argparse.ArgumentParser(description='Train Technique 2 NYU var1')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to resume from')
    parser.add_argument('--logdir', default=None, help='override TensorBoard runs_dir root')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Technique 2 has no direct-prediction branch: build_models returns model_3 = None.
    model_1, model_2, _ = build_models(TECHNIQUE, VARIANT)
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    # Two param groups: the weight heads emit MULTIPLICATIVE scalars, so they are trained at
    # weight_head_lr_mult * lr. Head module names come from models/model_builder.py
    # (DepthModel: trunk/w_sigmoid/w_softmax; ImageModel: trunk/heads.N).
    params, head_params, body_params = [], [], []
    for m in (model_1, model_2):
        for n_, p in m.named_parameters():
            params.append(p)
            (head_params if any(k in n_ for k in ('trunk', 'w_sigmoid', 'w_softmax', 'heads.'))
             else body_params).append(p)
    groups = [{'params': body_params, 'lr': cfg.learning_rate}]
    if head_params:
        groups.append({'params': head_params, 'lr': cfg.learning_rate * cfg.weight_head_lr_mult})
    optimizer = torch.optim.Adam(groups)
    base_lrs = [g['lr'] for g in optimizer.param_groups]  # captured BEFORE any resume overwrites lr

    l1 = nn.L1Loss()
    lambda_l1, lambda_ssim = cfg.lambda_l1, cfg.lambda_ssim
    # Scale-normalise every term BEFORE it meets a learned weight. In var1 the learned
    # weights live INSIDE each loss (w*L1 + (1-w)*SSIM), and those two components are in
    # different units: the depth L1 is O(1-5) in the reciprocal domain [1, 25] while an SSIM
    # loss is bounded by [0, 1]. A weight minimising w*A + (1-w)*B just picks argmin(A, B),
    # so the depth L1 could never win and its weight was driven to the floor. Dividing each
    # component by a running estimate of its own magnitude makes the weights arbitrate on
    # IMPROVEMENT rather than on units.
    norm = EMANormalizer(n=6, momentum=cfg.loss_ema_momentum)
    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg)
    writer = make_writer(cfg, TECHNIQUE, DATASET, VARIANT, args.logdir)

    # Set BEFORE the resume block so a resumed run can override them. Previously best_loss was
    # written to the checkpoint but never read back and was re-initialised to inf after the
    # resume, so the first post-resume epoch always overwrote the best checkpoint.
    best_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model_1.load_state_dict(ckpt['state_dict_1'])
        model_2.load_state_dict(ckpt['state_dict_2'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])   # else Adam restarts with zero moments
        norm.load_state_dict(ckpt.get('loss_norm'))        # else every term is re-scaled from scratch
        best_loss = float(ckpt.get('best_loss', best_loss))
        epochs_no_improve = int(ckpt.get('epochs_no_improve', 0))
        start_epoch = ckpt.get('cur_epoch', 0) + 1

    # CosineAnnealingLR.step() is a RATIO RECURSION on the CURRENT group lr, so a resumed run
    # must be placed back on the schedule explicitly: the optimizer state restores the lr of the
    # epoch BEFORE the checkpoint, and recent torch keeps that value on construction (get_lr()
    # returns the current lrs while _is_initial). Seeding each group with the closed-form lr for
    # start_epoch makes the resumed lr — and every step after it — identical to a fresh run.
    # A no-op when start_epoch == 0 (cos(0) = 1 -> lr = base_lr).
    _e = min(start_epoch, cfg.epochs)
    for g, blr in zip(optimizer.param_groups, base_lrs):
        g['initial_lr'] = blr
        g['lr'] = cfg.lr_min + (blr - cfg.lr_min) * (1 + math.cos(math.pi * _e / cfg.epochs)) / 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min, last_epoch=start_epoch - 1)

    def ssim_loss(pred, target, vr):
        return torch.clamp((1 - ssim(pred, target, val_range=vr)) * 0.5, 0, 1)

    patience = cfg.early_stopping_patience
    for epoch in range(start_epoch, cfg.epochs):
        model_1.train()
        model_2.train()
        meter = AverageMeter()
        m_depth, m_complex, m_haze, m_obj = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
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

            out_depth, w_depth = model_1(image_full)
            out_bb, w_bb = model_2(image_full)
            # NOT out_depth.detach(): Eq.11 is I_Simulated = t(I_Depth)*I + (1-t)*A + I_Residue,
            # so L_p MUST backprop through the transmission into the depth head — that coupling
            # IS the physics-informed part. Detached, model_1 was a standalone depth regressor
            # and the residual head absorbed every depth error.
            pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
            pred_haze = compute_haze_image(out_depth, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
            w_depth = torch.mean(w_depth, dim=0)
            w_bb = torch.mean(w_bb, dim=0)

            s_depth = ssim_loss(out_depth, depth_n, DEPTH_VAL_RANGE)
            l1_depth = l1(out_depth, depth_n)
            g_depth = gradient_loss(out_depth, depth_n)          # DenseDepth edge/gradient term
            s_complex = ssim_loss(pred_complex, complex_gt, 1)
            l1_complex = l1(pred_complex, complex_gt)
            s_haze = ssim_loss(pred_haze, haze, 1)
            l1_haze = l1(pred_haze, haze)

            # DIAGNOSTIC values: fixed lambdas, no learned weights, no EMA. total_fixed is term
            # for term the SAME expression as v_loss below, so train= and val= measure the same
            # thing and the curve means something across epochs. The optimised objective cannot
            # serve this role: the EMA scale is a MOVING TARGET, so it can fall while the model
            # gets worse. Built under no_grad — reporting must not touch the backward graph.
            with torch.no_grad():
                loss_depth = lambda_ssim * s_depth + lambda_l1 * l1_depth + cfg.lambda_grad * g_depth
                loss_complex = lambda_ssim * s_complex + lambda_l1 * l1_complex
                loss_haze = lambda_ssim * s_haze + lambda_l1 * l1_haze
                total_fixed = loss_depth + loss_complex + loss_haze

            # OPTIMISED objective: learned weights over the EMA-normalised components.
            o_depth = ((1.0 - w_depth[0]) * norm(N_SSIM_D, s_depth) + w_depth[0] * norm(N_L1_D, l1_depth)
                       + cfg.lambda_grad * g_depth)
            o_complex = (1.0 - w_bb[0]) * norm(N_SSIM_P, s_complex) + w_bb[0] * norm(N_L1_P, l1_complex)
            o_haze = (1.0 - w_bb[1]) * norm(N_SSIM_T, s_haze) + w_bb[1] * norm(N_L1_T, l1_haze)
            total_loss = o_depth + o_complex + o_haze
            # Each sigmoid head weights a term as the PAIR (w, 1-w), so both coefficients are
            # learned loss weights and both must be kept away from 0 — the barrier only guards
            # the w -> 0 side, hence 1 - w is passed as well. w_depth[1] is emitted by the
            # shared 2-output sigmoid head but is not consumed by L_d, so it is not regularised.
            total_loss = total_loss + cfg.lambda_weight_reg * weight_log_barrier(
                w_depth[:1], 1.0 - w_depth[:1], w_bb, 1.0 - w_bb)

            # Meters are updated only for FINITE batches: a skipped batch must not poison the
            # running mean (and a NaN would make every downstream epoch average NaN).
            if not torch.isfinite(total_loss):
                n_bad += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            _bs = image_full.size(0)
            meter.update(total_fixed.item(), _bs)
            m_depth.update(loss_depth.item(), _bs)
            m_complex.update(loss_complex.item(), _bs)
            m_haze.update(loss_haze.item(), _bs)
            m_obj.update(total_loss.item(), _bs)
            total_loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
            optimizer.step()

        if n_bad > cfg.max_nonfinite_batches_per_epoch:
            raise RuntimeError('[T%d %s %s] epoch %d: %d non-finite batches (limit %d) — training has diverged.'
                               % (TECHNIQUE, DATASET, VARIANT, epoch, n_bad, cfg.max_nonfinite_batches_per_epoch))

        # loss/total is the fixed-lambda unweighted train loss (directly comparable to
        # loss/val_total); loss/objective is the weighted + EMA-normalised + barrier quantity
        # actually minimised, which is NOT comparable to either.
        _tb_scalars = {'loss/total': meter.avg, 'loss/objective': m_obj.avg,
                       'loss/depth': m_depth.avg,
                       'loss/complex': m_complex.avg, 'loss/haze': m_haze.avg,
                       'lr': optimizer.param_groups[0]['lr'], 'train/nonfinite_batches': n_bad}
        log_scalars(writer, epoch, _tb_scalars)
        _tb_images = {'input/image_full': image_full, 'depth/pred': out_depth,
                      'depth/gt': depth_n, 'haze/pred': pred_haze, 'haze/gt': haze,
                      'complex/pred': pred_complex, 'complex/gt': complex_gt}
        log_images(writer, epoch, _tb_images)
        # Early-warning system for the flat-airlight collapse (depth head pinned at a bound,
        # predictions running out of [0, 1]). Watch stats/out_depth_frac_at_bound.
        log_health(writer, epoch, out_depth=out_depth, pred_complex=pred_complex,
                   pred_haze=pred_haze, complex_gt=complex_gt)
        writer.flush()
        # ---- validation on held-out split (drives checkpointing + early stopping) ----
        model_1.eval()
        model_2.eval()
        val_meter = AverageMeter()
        # The learned weights are LOGGED from the val pass (model.eval() -> dropout OFF, and
        # averaged over the whole split). Reading them off the last TRAINING batch logged a
        # single dropout-noised sample against epoch-averaged loss curves.
        w_depth_sum, w_bb_sum, n_val = None, None, 0
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
                out_depth, w_depth_b = model_1(image_full)
                out_bb, w_bb_b = model_2(image_full)
                pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
                pred_haze = compute_haze_image(out_depth, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
                _bs = image_full.size(0)
                _wd = torch.mean(w_depth_b, dim=0) * _bs
                _wb = torch.mean(w_bb_b, dim=0) * _bs
                w_depth_sum = _wd if w_depth_sum is None else w_depth_sum + _wd
                w_bb_sum = _wb if w_bb_sum is None else w_bb_sum + _wb
                n_val += _bs
                # FIXED-WEIGHT validation. The val loss must NOT use the learned weights: with
                # them, the network can lower val_loss without improving any prediction, purely
                # by shifting weight onto the smallest term. val_avg then improves every epoch,
                # early stopping never fires, and the "best" checkpoint is the most collapsed
                # one. Same fixed lambdas as 'base' -> the score is comparable across variants.
                v_depth = lambda_ssim * ssim_loss(out_depth, depth_n, DEPTH_VAL_RANGE) + lambda_l1 * l1(out_depth, depth_n)
                v_depth = v_depth + cfg.lambda_grad * gradient_loss(out_depth, depth_n)
                v_complex = lambda_ssim * ssim_loss(pred_complex, complex_gt, 1) + lambda_l1 * l1(pred_complex, complex_gt)
                v_haze = lambda_ssim * ssim_loss(pred_haze, haze, 1) + lambda_l1 * l1(pred_haze, haze)
                v_loss = v_depth + v_complex + v_haze
                val_meter.update(v_loss.item(), _bs)
        val_avg = val_meter.avg
        if n_val > 0:
            log_weights(writer, epoch, {'w_depth': w_depth_sum / n_val, 'w_bb': w_bb_sum / n_val})
        log_scalars(writer, epoch, {'loss/val_total': val_avg})
        writer.flush()
        _bad_str = '  nonfinite=%d' % n_bad if n_bad else ''
        # train/val are the fixed-lambda comparable losses (same terms, same cfg lambdas, no
        # learned weights / EMA / barrier) — only these two may be read against each other.
        # obj is the actual minimised objective (weighted + normalised + barrier): do NOT
        # compare it against val, and do not read its trend as progress (the EMA scale moves).
        print('[T%d %s %s] epoch %d/%d  train=%.4f  val=%.4f  obj=%.4f%s' % (TECHNIQUE, DATASET, VARIANT, epoch, cfg.epochs - 1, meter.avg, val_avg, m_obj.avg, _bad_str))

        # A NaN val_avg is never an improvement (NaN < x is False), but say so explicitly.
        improved = math.isfinite(val_avg) and val_avg < best_loss
        if improved:
            best_loss = val_avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        state = {'state_dict_1': model_1.state_dict(), 'state_dict_2': model_2.state_dict(),
                 'optimizer': optimizer.state_dict(), 'loss_norm': norm.state_dict(),
                 'cur_epoch': epoch, 'best_loss': best_loss,
                 'epochs_no_improve': epochs_no_improve}
        # Written EVERY epoch so a crash mid-run is recoverable (the best checkpoint may be
        # many epochs old, and resuming from it would silently replay them).
        torch.save(state, os.path.join(cfg.checkpoint_dir, 'T%d_%s_%s_last.ckpt' % (TECHNIQUE, DATASET, VARIANT)))
        if improved:
            torch.save(state, os.path.join(cfg.checkpoint_dir, 'T%d_%s_%s.ckpt' % (TECHNIQUE, DATASET, VARIANT)))
        elif epochs_no_improve >= patience:
            print('[T%d %s %s] early stopping at epoch %d (no val improvement for %d epochs)' % (TECHNIQUE, DATASET, VARIANT, epoch, patience))
            break
        scheduler.step()


if __name__ == '__main__':
    main()
