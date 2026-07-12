# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Technique 4 / NYU / var1 -- training script.

Slim entry point: argparse + config + build_models + data loader + train loop.
All models, losses, physics and datasets come from the shared packages.

Technique 4 total (repo extension of the paper's Eq.21):
    Ltotal = Ld + Lp + Lt + Lg
var1 LEARNS the per-loss balance: a sigmoid head over (L1, SSIM) for depth and softmax
heads over (L1, SSIM, Perceptual) for each of the three RGB losses.
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
from utils.physics import compute_haze_image, compute_complex_image, depth_for_complex
from utils.loss import ssim, gradient_loss, VGGPerceptualLoss
from utils.loss_balance import EMANormalizer, weight_log_barrier
from utils.tb import make_writer, log_scalars, log_images, log_weights, log_health

import warnings
# Keep the per-epoch output clean: torch/torchvision emit benign deprecation warnings.
warnings.filterwarnings("ignore")

TECHNIQUE = 4
VARIANT = 'var1'
DATASET = 'NYU'

# SSIM's stabilisers are C1 = (0.01*L)^2 and C2 = (0.03*L)^2, so L must be the FIXED
# dynamic range of the domain. Passing the batch's own max made L -- and therefore the
# whole SSIM surface -- a per-batch random variable: identical prediction quality scored
# differently from one batch to the next. The depth head is bounded to [1, 25] by the
# scaled sigmoid in models/decoder_1ch.py, so 25 is that range. Images stay at 1.
DEPTH_VAL_RANGE = 25.0

# Parameter-name fragments identifying the learned-weight heads (models/model_builder.py:
# DepthModel exposes 'trunk' / 'w_sigmoid' / 'w_softmax'; ImageModel exposes 'trunk' and
# an nn.ModuleList called 'heads'). No encoder/decoder parameter contains these strings.
HEAD_KEYS = ('trunk', 'w_sigmoid', 'w_softmax', 'heads')

# EMA-normaliser slots -- ONE PER LOSS COMPONENT THAT IS MULTIPLIED BY A LEARNED WEIGHT.
# In Technique 4 the learned weights sit INSIDE each loss (softmax over L1/SSIM/Perceptual,
# sigmoid over L1/SSIM), so the components -- not the aggregate losses -- are what the
# weights arbitrate between, and they are what must be put on a common scale. Raw, they are
# not remotely comparable (L1 on [0,1] images ~1e-2, the SSIM loss ~1e-1, the VGG term ~1e0,
# and the depth L1 lives in the reciprocal domain [1,25]); since a weight minimising
# sum_i w_i L_i simply picks argmin_i L_i, the largest-magnitude component -- never the
# worst-fitting one -- was driven to zero weight.
(NX_D_L1, NX_D_SSIM,
 NX_C_L1, NX_C_SSIM, NX_C_PERC,
 NX_H_L1, NX_H_SSIM, NX_H_PERC,
 NX_G_L1, NX_G_SSIM, NX_G_PERC) = range(11)
N_WEIGHTED_TERMS = 11


def main():
    parser = argparse.ArgumentParser(description='Train Technique 4 NYU var1')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to resume from')
    parser.add_argument('--logdir', default=None, help='override TensorBoard runs_dir root')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Technique 4 always has the direct-prediction branch, so model_3 is never None here.
    model_1, model_2, model_3 = build_models(TECHNIQUE, VARIANT)
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    model_3 = model_3.to(device)

    params = list(model_1.parameters()) + list(model_2.parameters()) + list(model_3.parameters())
    head_params, body_params = [], []
    for m in (model_1, model_2, model_3):
        for n_, p in m.named_parameters():
            (head_params if any(k in n_ for k in HEAD_KEYS) else body_params).append(p)
    groups = [{'params': body_params, 'lr': cfg.learning_rate}]
    if head_params:
        # The heads emit MULTIPLICATIVE scalars on the losses -- train them slower.
        groups.append({'params': head_params, 'lr': cfg.learning_rate * cfg.weight_head_lr_mult})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    l1 = nn.L1Loss()
    vgg = VGGPerceptualLoss().to(device)
    lambda_l1, lambda_ssim, lambda_perc = cfg.lambda_l1, cfg.lambda_ssim, cfg.lambda_perc
    norm = EMANormalizer(n=N_WEIGHTED_TERMS, momentum=cfg.loss_ema_momentum)
    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg)
    writer = make_writer(cfg, TECHNIQUE, DATASET, VARIANT, args.logdir)

    # Initialised BEFORE the resume block so a resumed run can override them. Previously
    # best_loss was reset to inf after resuming, so the first post-resume epoch always
    # "improved" and overwrote the best checkpoint with a worse model.
    start_epoch = 0
    best_loss = float('inf')
    epochs_no_improve = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model_1.load_state_dict(ckpt['state_dict_1'])
        model_2.load_state_dict(ckpt['state_dict_2'])
        model_3.load_state_dict(ckpt['state_dict_3'])
        if 'optimizer' in ckpt:
            # Adam's first/second moments are part of the training state: restarting them
            # at zero gives the first post-resume steps an effectively huge step size.
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            # The schedule is restored from its own state, never replayed. Re-stepping the
            # scheduler start_epoch times would decay the cosine a SECOND time: step() is
            # RECURSIVE in param_groups['lr'], which optimizer.load_state_dict has already
            # restored to the correct resumed value (measured: 8.20e-5 instead of 9.05e-5
            # after resuming at epoch 10, and the error compounds with the resume epoch).
            scheduler.load_state_dict(ckpt['scheduler'])
        # The EMA scales are training state too: losing them re-runs the warm-up and
        # rescales every learned weight's objective at the resume boundary.
        norm.load_state_dict(ckpt.get('loss_norm'))
        best_loss = ckpt.get('best_loss', best_loss)
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        start_epoch = ckpt.get('cur_epoch', 0) + 1

    def ssim_loss(pred, target, vr):
        return torch.clamp((1 - ssim(pred, target, val_range=vr)) * 0.5, 0, 1)

    patience = cfg.early_stopping_patience
    ckpt_best = os.path.join(cfg.checkpoint_dir, 'T%d_%s_%s.ckpt' % (TECHNIQUE, DATASET, VARIANT))
    ckpt_last = os.path.join(cfg.checkpoint_dir, 'T%d_%s_%s_last.ckpt' % (TECHNIQUE, DATASET, VARIANT))
    for epoch in range(start_epoch, cfg.epochs):
        model_1.train()
        model_2.train()
        model_3.train()
        meter = AverageMeter()
        m_depth, m_complex, m_haze, m_direct = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        m_obj = AverageMeter()   # the quantity actually minimised (weighted + EMA-normalised + barrier)
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
            out_bb, w_residue, w_deg = model_2(image_full)
            out_direct, w_dir = model_3(image_full)
            # out_depth is NOT detached: Eq.11 is I_Simulated = t(I_Depth)*I + (1-t)*A + I_Residue,
            # so L_p must backprop through the transmission t into the depth head -- that coupling
            # IS the physics-informed part. Detaching it made model_1 a standalone depth regressor
            # and left the residual head to absorb every depth error.
            pred_complex = compute_complex_image(depth_for_complex(out_depth, cfg), out_bb, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
            pred_haze = compute_haze_image(out_depth, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
            w_depth = torch.mean(w_depth, dim=0)
            w_residue = torch.mean(w_residue, dim=0)
            w_deg = torch.mean(w_deg, dim=0)
            w_dir = torch.mean(w_dir, dim=0)

            d_l1 = l1(out_depth, depth_n)
            d_ss = ssim_loss(out_depth, depth_n, DEPTH_VAL_RANGE)
            d_gr = gradient_loss(out_depth, depth_n)          # DenseDepth edge/gradient term
            c_l1 = l1(pred_complex, complex_gt)
            c_ss = ssim_loss(pred_complex, complex_gt, 1)
            c_pc = vgg(pred_complex, complex_gt)
            h_l1 = l1(pred_haze, haze)
            h_ss = ssim_loss(pred_haze, haze, 1)
            h_pc = vgg(pred_haze, haze)
            g_l1 = l1(out_direct, complex_gt)
            g_ss = ssim_loss(out_direct, complex_gt, 1)
            g_pc = vgg(out_direct, complex_gt)

            # d_gr carries a FIXED lambda, so it never meets a learned weight and is not
            # normalised. Everything a weight multiplies is.
            loss_depth = (w_depth[0] * norm(NX_D_L1, d_l1)
                          + (1.0 - w_depth[0]) * norm(NX_D_SSIM, d_ss)
                          + cfg.lambda_grad * d_gr)
            loss_complex = (w_residue[0] * norm(NX_C_L1, c_l1)
                            + w_residue[1] * norm(NX_C_SSIM, c_ss)
                            + w_residue[2] * norm(NX_C_PERC, c_pc))
            loss_haze = (w_deg[0] * norm(NX_H_L1, h_l1)
                         + w_deg[1] * norm(NX_H_SSIM, h_ss)
                         + w_deg[2] * norm(NX_H_PERC, h_pc))
            loss_direct = (w_dir[0] * norm(NX_G_L1, g_l1)
                           + w_dir[1] * norm(NX_G_SSIM, g_ss)
                           + w_dir[2] * norm(NX_G_PERC, g_pc))
            total_loss = loss_depth + loss_complex + loss_haze + loss_direct

            # The depth head is a SIGMOID used as w*L1 + (1-w)*SSIM, so BOTH effective
            # weights must be kept off zero -- w_depth[1] is never applied to any loss and
            # is therefore deliberately left un-regularised.
            w_depth_eff = torch.stack((w_depth[0], 1.0 - w_depth[0]))
            total_loss = total_loss + cfg.lambda_weight_reg * weight_log_barrier(w_depth_eff, w_residue, w_deg, w_dir)

            if not torch.isfinite(total_loss):
                # Skip the batch AND the meters: a single NaN would otherwise poison the
                # running mean for the whole epoch.
                n_bad += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            _bs = image_full.size(0)
            # total_loss is NOT a reportable loss: the EMA drives each normalised component to
            # ~1.0 and the learned weights sum to 1, so it sits near 1.0 whatever the model does.
            # Worse, the EMA scale is a MOVING TARGET, so that curve can fall while the model
            # gets worse, and it is on a different scale from val. What is metered instead is
            # total_fixed: the FIXED cfg lambdas, UNWEIGHTED, no normalisation, no barrier --
            # term for term the same expression the val loop builds as v_loss, so "train=" and
            # "val=" are the same quantity on two splits. The per-term meters follow the same
            # rule. Reuses the loss tensors already computed above; the models are not re-run.
            with torch.no_grad():
                fx_depth = lambda_l1 * d_l1 + lambda_ssim * d_ss + cfg.lambda_grad * d_gr
                fx_complex = lambda_l1 * c_l1 + lambda_ssim * c_ss + lambda_perc * c_pc
                fx_haze = lambda_l1 * h_l1 + lambda_ssim * h_ss + lambda_perc * h_pc
                fx_direct = lambda_l1 * g_l1 + lambda_ssim * g_ss + lambda_perc * g_pc
                total_fixed = fx_depth + fx_complex + fx_haze + fx_direct
                meter.update(total_fixed.item(), _bs)
                m_obj.update(total_loss.item(), _bs)
                m_depth.update(fx_depth.item(), _bs)
                m_complex.update(fx_complex.item(), _bs)
                m_haze.update(fx_haze.item(), _bs)
                m_direct.update(fx_direct.item(), _bs)
            total_loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
            optimizer.step()

        if n_bad > cfg.max_nonfinite_batches_per_epoch:
            raise RuntimeError('[T%d %s %s] %d non-finite batches in epoch %d (limit %d) -- training has diverged.'
                               % (TECHNIQUE, DATASET, VARIANT, n_bad, epoch, cfg.max_nonfinite_batches_per_epoch))

        # 'loss/total' is the fixed-lambda comparable loss (directly against 'loss/val_total');
        # 'loss/objective' is the weighted + EMA-normalised + barrier quantity being minimised,
        # which is on its own moving scale and must never be plotted against the val curve.
        _tb_scalars = {'loss/total': meter.avg, 'loss/objective': m_obj.avg,
                       'loss/depth': m_depth.avg,
                       'loss/complex': m_complex.avg, 'loss/haze': m_haze.avg,
                       'loss/direct': m_direct.avg,
                       'lr': optimizer.param_groups[0]['lr']}
        log_scalars(writer, epoch, _tb_scalars)
        _tb_images = {'input/image_full': image_full, 'depth/pred': out_depth,
                      'depth/gt': depth_n, 'haze/pred': pred_haze, 'haze/gt': haze,
                      'complex/pred': pred_complex, 'complex/gt': complex_gt,
                      'direct/pred': out_direct}
        log_images(writer, epoch, _tb_images)
        # Early-warning scalars for the flat-airlight collapse (depth pinned at a bound,
        # residual leaving [0, 1]). Gate any sweep on these staying healthy.
        log_health(writer, epoch, out_depth=out_depth, pred_complex=pred_complex,
                   pred_haze=pred_haze, complex_gt=complex_gt)
        writer.flush()
        # ---- validation on held-out split (drives checkpointing + early stopping) ----
        model_1.eval()
        model_2.eval()
        model_3.eval()
        val_meter = AverageMeter()
        w_sums = {}
        n_val_batches = 0
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
                out_depth, w_depth = model_1(image_full)
                out_bb, w_residue, w_deg = model_2(image_full)
                out_direct, w_dir = model_3(image_full)
                pred_complex = compute_complex_image(depth_for_complex(out_depth, cfg), out_bb, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
                pred_haze = compute_haze_image(out_depth, beta, a_val, unit, image_half, max_depth_m=cfg.nyu_max_depth_m)
                # The val loss uses FIXED lambdas and is UNWEIGHTED -- identical to 'base'.
                # Scoring with the learned weights let the network lower val_loss without
                # improving a single prediction, purely by moving weight mass onto the
                # smallest term: val "improved" every epoch, early stopping never fired, and
                # the checkpoint kept as "best" was the most collapsed one.
                v_depth = (lambda_l1 * l1(out_depth, depth_n) + lambda_ssim * ssim_loss(out_depth, depth_n, DEPTH_VAL_RANGE)
                           + cfg.lambda_grad * gradient_loss(out_depth, depth_n))
                v_complex = (lambda_l1 * l1(pred_complex, complex_gt) + lambda_ssim * ssim_loss(pred_complex, complex_gt, 1) + lambda_perc * vgg(pred_complex, complex_gt))
                v_haze = (lambda_l1 * l1(pred_haze, haze) + lambda_ssim * ssim_loss(pred_haze, haze, 1) + lambda_perc * vgg(pred_haze, haze))
                v_direct = (lambda_l1 * l1(out_direct, complex_gt) + lambda_ssim * ssim_loss(out_direct, complex_gt, 1) + lambda_perc * vgg(out_direct, complex_gt))
                v_loss = v_depth + v_complex + v_haze + v_direct
                val_meter.update(v_loss.item(), image_full.size(0))
                # Weights are LOGGED from the val pass (eval mode -> dropout OFF, epoch-averaged),
                # not read off the last training batch, which was a single dropout-noised sample.
                for _nm, _w in (('w_depth', w_depth), ('w_residue', w_residue),
                                ('w_deg', w_deg), ('w_dir', w_dir)):
                    _wm = torch.mean(_w.detach(), dim=0)
                    w_sums[_nm] = _wm if _nm not in w_sums else w_sums[_nm] + _wm
                n_val_batches += 1
        val_avg = val_meter.avg
        if n_val_batches:
            log_weights(writer, epoch, {k: v / n_val_batches for k, v in w_sums.items()})
        log_scalars(writer, epoch, {'loss/val_total': val_avg})
        writer.flush()
        scheduler.step()
        # train/val are the fixed-lambda comparable losses (same expression, different split);
        # obj is the actual minimised objective (weighted + normalised + barrier). Do NOT
        # compare obj against val -- they are not on the same scale.
        _msg = '[T%d %s %s] epoch %d/%d  train=%.4f  val=%.4f  obj=%.4f' % (TECHNIQUE, DATASET, VARIANT, epoch, cfg.epochs - 1, meter.avg, val_avg, m_obj.avg)
        if n_bad:
            _msg += '  skipped=%d' % n_bad
        print(_msg)

        # A non-finite val loss must never win the comparison.
        improved = math.isfinite(val_avg) and val_avg < best_loss
        if improved:
            best_loss = val_avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        state = {'state_dict_1': model_1.state_dict(), 'state_dict_2': model_2.state_dict(),
                 'state_dict_3': model_3.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'loss_norm': norm.state_dict(),
                 'cur_epoch': epoch, 'best_loss': best_loss, 'epochs_no_improve': epochs_no_improve}
        torch.save(state, ckpt_last)   # every epoch, so a crash is recoverable
        if improved:
            torch.save(state, ckpt_best)
        elif epochs_no_improve >= patience:
            print('[T%d %s %s] early stopping at epoch %d (no val improvement for %d epochs)' % (TECHNIQUE, DATASET, VARIANT, epoch, patience))
            break


if __name__ == '__main__':
    main()
