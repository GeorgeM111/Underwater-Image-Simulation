# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Technique 1 / NYU / var1 -- training script.

Slim entry point: argparse + config + build_models + data loader + train loop.
All models, losses, physics and datasets come from the shared packages.
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

TECHNIQUE = 1
VARIANT = 'var1'
DATASET = 'NYU'

# SSIM's stabilisers are C1 = (0.01*L)^2 and C2 = (0.03*L)^2, so L must be the FIXED
# dynamic range of the domain. Passing depth_n.max() made L a per-BATCH random variable
# (the nearest pixel in that batch), swinging C1/C2 by ~45x between neighbouring batches
# for identical prediction quality. The depth head is bounded to [1, 25] by the scaled
# sigmoid in models/decoder_1ch.py, so 25 IS the range of the reciprocal-depth domain.
DEPTH_VAL_RANGE = 25.0

# Substrings identifying the learned-weight-head parameters. DepthModel names them
# trunk/w_sigmoid/w_softmax; ImageModel names them trunk/heads.<i> (an nn.ModuleList) --
# 'heads' is REQUIRED or model_2's head lands in the body group and trains 10x too fast.
# All are behind AmpForward's 'module.' prefix, hence the substring match.
HEAD_KEYS = ('trunk', 'w_sigmoid', 'w_softmax', 'heads')

# EMA-normalised loss-term slots. Every term that MEETS A LEARNED WEIGHT gets one, because
# a weight minimising sum_i w_i*L_i simply picks argmin_i L_i: it arbitrates on UNITS, not
# on improvement. In var1 the learned weights sit INSIDE each loss (w*L1 + (1-w)*SSIM), and
# the depth L1 lives in the reciprocal domain [1, 25] (O(1-10)) while its SSIM term is
# bounded by 1 -- so w_depth[0] would be driven to its floor and the L1 signal switched off.
N_SSIM_D, N_L1_D, N_SSIM_C, N_L1_C = 0, 1, 2, 3
N_TERMS = 4


def main():
    parser = argparse.ArgumentParser(description='Train Technique 1 NYU var1')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to resume from')
    parser.add_argument('--logdir', default=None, help='override TensorBoard runs_dir root')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_1, model_2, model_3 = build_models(TECHNIQUE, VARIANT)
    # Technique 1 (paper Eq.12) has NO direct-prediction branch: build_models returns None.
    assert model_3 is None, 'Technique 1 must not build a direct-prediction branch'
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    params = list(model_1.parameters()) + list(model_2.parameters())
    head_params, body_params = [], []
    for m in (model_1, model_2):
        for n_, p in m.named_parameters():
            (head_params if any(k in n_ for k in HEAD_KEYS) else body_params).append(p)
    groups = [{'params': body_params, 'lr': cfg.learning_rate}]
    if head_params:  # weight heads emit MULTIPLICATIVE scalars -> train them slower.
        groups.append({'params': head_params, 'lr': cfg.learning_rate * cfg.weight_head_lr_mult})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    l1 = nn.L1Loss()
    lambda_l1, lambda_ssim = cfg.lambda_l1, cfg.lambda_ssim
    norm = EMANormalizer(n=N_TERMS, momentum=cfg.loss_ema_momentum)
    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg)
    writer = make_writer(cfg, TECHNIQUE, DATASET, VARIANT, args.logdir)

    # Declared BEFORE the resume block so a resumed run restores them instead of
    # restarting from inf/0 (which made the first post-resume epoch always overwrite
    # the best checkpoint and reset early stopping).
    start_epoch = 0
    best_loss = float('inf')
    epochs_no_improve = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model_1.load_state_dict(ckpt['state_dict_1'])
        model_2.load_state_dict(ckpt['state_dict_2'])
        if 'optimizer' in ckpt:
            # Adam's moment estimates are state: without them a resume is a warm restart
            # with an effectively random first step.
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        # The EMA scales are part of the objective: losing them rescales every learned
        # weight's arbitration on the first post-resume batch.
        norm.load_state_dict(ckpt.get('loss_norm'))
        best_loss = ckpt.get('best_loss', best_loss)
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        start_epoch = ckpt.get('cur_epoch', 0) + 1

    def ssim_loss(pred, target, vr):
        return torch.clamp((1 - ssim(pred, target, val_range=vr)) * 0.5, 0, 1)

    def make_state(epoch):
        return {'state_dict_1': model_1.state_dict(), 'state_dict_2': model_2.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'loss_norm': norm.state_dict(),
                'cur_epoch': epoch, 'best_loss': best_loss,
                'epochs_no_improve': epochs_no_improve}

    best_path = os.path.join(cfg.checkpoint_dir, 'T%d_%s_%s.ckpt' % (TECHNIQUE, DATASET, VARIANT))
    last_path = os.path.join(cfg.checkpoint_dir, 'T%d_%s_%s_last.ckpt' % (TECHNIQUE, DATASET, VARIANT))

    patience = cfg.early_stopping_patience
    for epoch in range(start_epoch, cfg.epochs):
        model_1.train()
        model_2.train()
        meter = AverageMeter()
        m_obj = AverageMeter()
        m_depth, m_complex, m_haze = AverageMeter(), AverageMeter(), AverageMeter()
        n_bad = 0
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

            out_depth, w_depth = model_1(image_full)
            out_bb, w_bb = model_2(image_full)
            # NOT out_depth.detach(): Eq.11 is I_Simulated = t(I_Depth)*I + (1-t)*A + I_Residue,
            # so L_p must backprop through the transmission t into the depth head -- that
            # coupling IS the physics-informed part. With the detach, model_1 was a standalone
            # depth regressor and the residual head absorbed every depth error.
            pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half,
                                                 max_depth_m=cfg.nyu_max_depth_m)
            w_depth = torch.mean(w_depth, dim=0)
            w_bb = torch.mean(w_bb, dim=0)

            ssim_d = ssim_loss(out_depth, depth_n, DEPTH_VAL_RANGE)
            l1_d = l1(out_depth, depth_n)
            grad_d = gradient_loss(out_depth, depth_n)   # DenseDepth edge term; fixed lambda, no learned weight
            ssim_c = ssim_loss(pred_complex, complex_gt, 1)
            l1_c = l1(pred_complex, complex_gt)

            loss_depth = ((1.0 - w_depth[0]) * norm(N_SSIM_D, ssim_d)
                          + w_depth[0] * norm(N_L1_D, l1_d)
                          + cfg.lambda_grad * grad_d)
            loss_complex = (1.0 - w_bb[0]) * norm(N_SSIM_C, ssim_c) + w_bb[0] * norm(N_L1_C, l1_c)
            # Technique 1 (paper Eq.12): Ltotal = Ld + Lp. There is NO Lt term (that is T2), so
            # the classical haze image is a pure DIAGNOSTIC here (logged + fed to log_health),
            # scored with the FIXED lambdas so its curve stays comparable with 'base'. Built
            # under no_grad so it does not carry a wasted autograd graph.
            #
            # NOTE: w_bb[1] is model_2's L_t weight. T1 has no L_t, so it is UNUSED -- it is
            # deliberately kept out of the barrier below (barriering a weight nothing consumes
            # would inject gradient into model_2's shared trunk for no reason).
            with torch.no_grad():
                pred_haze = compute_haze_image(out_depth, beta, a_val, unit, image_half,
                                               max_depth_m=cfg.nyu_max_depth_m)
                loss_haze = lambda_ssim * ssim_loss(pred_haze, haze, 1) + lambda_l1 * l1(pred_haze, haze)
                # The reported loss must be COMPARABLE -- with the val score printed beside it,
                # with earlier epochs, and with the other variants. Neither the EMA-normalised
                # terms nor the learned weights qualify: both are MOVING TARGETS, so a curve
                # built from them can fall while the model gets worse (the EMA drives each term
                # toward ~1 regardless of quality, and the weights can shift mass onto whichever
                # term is currently smallest). So the meters are scored with the FIXED cfg
                # lambdas and NO learned weights -- term for term the val loop's v_loss, just
                # on the training batch. Reuses the tensors already computed above.
                fixed_depth = (cfg.lambda_l1 * l1_d + cfg.lambda_ssim * ssim_d
                               + cfg.lambda_grad * grad_d)
                fixed_complex = cfg.lambda_l1 * l1_c + cfg.lambda_ssim * ssim_c
                total_fixed = fixed_depth + fixed_complex   # T1: Ld + Lp only, as in v_loss

            total_loss = loss_depth + loss_complex
            # Log-barrier: sum_i w_i*L_i is LINEAR in w, so its minimiser is a VERTEX (one-hot
            # on argmin_i L_i) and a weight driven to 0 kills its term's gradient permanently.
            # -sum(log w) makes w -> 0 infinitely costly, removing that vertex.
            #
            # BOTH ENDPOINTS of each sigmoid weight are barriered, because the sigmoid weights
            # enter as a PAIR -- (1 - w)*SSIM + w*L1 -- so w -> 1 kills the SSIM term exactly as
            # w -> 0 kills the L1 term. Barriering w alone is not neutral: d/dw[-log w] = -1/w is
            # strictly negative, a monotone push toward w = 1. Once the EMA has equalised the two
            # terms the data gradient (L1_n - SSIM_n) is ~0, so that push dominates and pins SSIM
            # at weight_floor -- the barrier would CAUSE the starvation it exists to prevent.
            # w_depth[1] is excluded for the same reason as w_bb[1]: nothing consumes it.
            total_loss = total_loss + cfg.lambda_weight_reg * weight_log_barrier(
                w_depth[:1], 1.0 - w_depth[:1], w_bb[:1], 1.0 - w_bb[:1])

            # Skip non-finite batches BEFORE the meters see them: a single NaN batch would
            # otherwise poison every running mean for the rest of the epoch.
            if not torch.isfinite(total_loss):
                n_bad += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            _bs = image_full.size(0)
            meter.update(total_fixed.item(), _bs)   # comparable loss -> "train=" and TB loss/total
            m_obj.update(total_loss.item(), _bs)    # actually minimised -> TB loss/objective
            m_depth.update(fixed_depth.item(), _bs)
            m_complex.update(fixed_complex.item(), _bs)
            m_haze.update(loss_haze.item(), _bs)

            total_loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
            optimizer.step()

        if n_bad > cfg.max_nonfinite_batches_per_epoch:
            raise RuntimeError('[T%d %s %s] %d non-finite batches in epoch %d (limit %d) -- aborting.'
                               % (TECHNIQUE, DATASET, VARIANT, n_bad, epoch, cfg.max_nonfinite_batches_per_epoch))

        # loss/total is the FIXED-LAMBDA, UNWEIGHTED Ld + Lp -- the same expression as
        # loss/val_total and as 'base', hence comparable across epochs, variants and runs.
        # loss/objective is what backward() actually saw (EMA-normalised, weighted, + barrier).
        _tb_scalars = {'loss/total': meter.avg, 'loss/objective': m_obj.avg, 'loss/depth': m_depth.avg,
                       'loss/complex': m_complex.avg, 'loss/haze': m_haze.avg,
                       'lr': optimizer.param_groups[0]['lr'], 'train/nonfinite_batches': n_bad}
        log_scalars(writer, epoch, _tb_scalars)
        _tb_images = {'input/image_full': image_full, 'depth/pred': out_depth,
                      'depth/gt': depth_n, 'haze/pred': pred_haze, 'haze/gt': haze,
                      'complex/pred': pred_complex, 'complex/gt': complex_gt}
        log_images(writer, epoch, _tb_images)
        # Early-warning system for the flat-airlight collapse (out_depth pinned at a bound
        # => t = exp(-beta*z) -> 0 => the physics degenerates to pure airlight).
        log_health(writer, epoch, out_depth=out_depth, pred_complex=pred_complex,
                   pred_haze=pred_haze, complex_gt=complex_gt)
        writer.flush()
        # ---- validation on held-out split (drives checkpointing + early stopping) ----
        model_1.eval()
        model_2.eval()
        val_meter = AverageMeter()
        w_depth_sum, w_bb_sum, n_val = None, None, 0
        with torch.no_grad():
            for batch in val_loader:
                image_full = batch['image_full'].to(device)
                image_half = batch['image_half'].to(device)
                depth_n = DepthNorm(batch['depth'].to(device))
                beta = batch['beta'].to(device)
                a_val = batch['a_val'].to(device)
                unit = batch['unit_mat'].to(device)
                complex_gt = batch['complex_noise_img'].to(device)
                out_depth, w_depth = model_1(image_full)
                out_bb, w_bb = model_2(image_full)
                pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half,
                                                     max_depth_m=cfg.nyu_max_depth_m)
                # FIXED-WEIGHT VALIDATION. The val score must NOT use the learned weights: with
                # them the network can lower val_loss without improving a single prediction,
                # simply by shifting weight mass onto the smallest term. val_avg < best_loss then
                # fires every epoch, early stopping never triggers, and the "best" checkpoint is
                # the MOST collapsed one. This is scored exactly as 'base' scores it.
                v_depth = (cfg.lambda_l1 * l1(out_depth, depth_n)
                           + cfg.lambda_ssim * ssim_loss(out_depth, depth_n, DEPTH_VAL_RANGE)
                           + cfg.lambda_grad * gradient_loss(out_depth, depth_n))
                v_complex = (cfg.lambda_l1 * l1(pred_complex, complex_gt)
                             + cfg.lambda_ssim * ssim_loss(pred_complex, complex_gt, 1))
                v_loss = v_depth + v_complex  # T1: Ld + Lp only
                _bs = image_full.size(0)
                val_meter.update(v_loss.item(), _bs)
                # Weights are LOGGED from the val pass (eval mode -> dropout OFF, epoch-averaged),
                # not read off the last training batch (one dropout-noised sample).
                w_d = torch.mean(w_depth, dim=0) * _bs
                w_b = torch.mean(w_bb, dim=0) * _bs
                w_depth_sum = w_d if w_depth_sum is None else w_depth_sum + w_d
                w_bb_sum = w_b if w_bb_sum is None else w_bb_sum + w_b
                n_val += _bs
        val_avg = val_meter.avg
        if n_val:
            log_weights(writer, epoch, {'w_depth': w_depth_sum / n_val, 'w_bb': w_bb_sum / n_val})
        log_scalars(writer, epoch, {'loss/val_total': val_avg})
        writer.flush()
        _bad_note = ('  bad_batches=%d' % n_bad) if n_bad else ''
        # train/val are the fixed-lambda comparable losses (same expression, different split).
        # obj is the actual minimised objective -- weighted, EMA-normalised and barriered -- so
        # it lives on its own scale: do NOT compare it against val, only against itself.
        print('[T%d %s %s] epoch %d/%d  train=%.4f  val=%.4f  obj=%.4f%s'
              % (TECHNIQUE, DATASET, VARIANT, epoch, cfg.epochs - 1, meter.avg, val_avg,
                 m_obj.avg, _bad_note))

        # A NaN val_avg must never win: NaN comparisons are False, but be explicit.
        improved = math.isfinite(val_avg) and val_avg < best_loss
        if improved:
            best_loss = val_avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        # Step BEFORE snapshotting: the checkpoint records cur_epoch=epoch and a resume starts at
        # epoch+1, so the scheduler state it carries must already be the one for epoch+1. Saving
        # the pre-step state made every resumed run re-use the previous epoch's LR and lag the
        # cosine schedule by one epoch for the rest of the run.
        scheduler.step()
        state = make_state(epoch)
        torch.save(state, last_path)   # every epoch, so a crash mid-run is recoverable
        if improved:
            torch.save(state, best_path)
        if epochs_no_improve >= patience:
            print('[T%d %s %s] early stopping at epoch %d (no val improvement for %d epochs)'
                  % (TECHNIQUE, DATASET, VARIANT, epoch, patience))
            break


if __name__ == '__main__':
    main()
