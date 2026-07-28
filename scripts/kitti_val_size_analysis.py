# --- repo-root path bootstrap ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""How large does the KITTI validation split actually need to be?

WHY THIS IS *NOT* K-FOLD CROSS-VALIDATION
-----------------------------------------
With ``kitti_train_mode: "subset"`` the training set is a FIXED 10k frames no matter what
``kitti_val_ratio`` is (10k fits inside the train region at every fraction). So the val ratio
does NOT trade away training data — the trained model is essentially the same. The only things
a bigger val set buys are (a) a less noisy early-stopping / checkpoint-selection signal, and
the only things it COSTS are (b) more val-tail GT to generate and (c) more forward passes per
epoch. Retraining N models to compare val sizes would therefore be pure waste: the question is
entirely about the STABILITY of the val estimate, which one trained model answers exactly.

WHAT THIS MEASURES
------------------
Given ONE trained checkpoint, it computes the per-frame validation loss over the held-out
frames, then asks: if the val set were size M, how much would the mean val loss WOBBLE
depending on which frames landed in it? That wobble (the standard error of the val estimate)
is what determines whether early stopping can reliably tell "epoch A is better than epoch B".

THE KITTI TWIST (block bootstrap)
---------------------------------
Consecutive KITTI frames come from the same driving sequence and are near-duplicates, so N
frames are NOT N independent samples — the effective sample size is closer to the number of
DRIVES. A naive per-frame bootstrap would badly understate the noise. This resamples whole
DRIVES (block bootstrap), which is the honest estimate. It reports both so you can see how much
the correlation inflates the real uncertainty.

RECOMMENDATION RULE
-------------------
Pick the SMALLEST val fraction whose standard error is comfortably below the val-loss
improvement early stopping must resolve (``--epoch-delta``): if the val estimate wobbles by
more than the per-epoch improvement, early stopping is choosing between epochs by noise.

Run (on the machine with the data + a trained/partway checkpoint):
    python scripts/kitti_val_size_analysis.py --technique 1 --variant base \
        --resume $PROJECT_OUT/checkpoints/T1_KITTI_base_last.ckpt \
        --fracs 0.05 0.10 0.15 0.20 0.30 --epoch-delta 0.002
"""

import argparse
import glob

import numpy as np
import torch

from config import load_config
from models.model_builder import build_models
from utils.helpers import DepthNorm
from utils.loss import ssim
from utils.physics import compute_complex_image
from utils.depth_range import DEPTH_VAL_RANGE


def ssim_loss(pred, target, vr):
    return float(torch.clamp((1 - ssim(pred, target, val_range=vr)) * 0.5, 0, 1))


def per_frame_val_loss(model_1, model_2, batch, device, cfg, lam_l1, lam_ssim):
    """Base (T1) fixed-lambda v_loss = Ld + Lp, computed PER FRAME.

    Uses the fixed DEPTH_VAL_RANGE (not depth_n.max()) so the number is comparable across
    frames — the whole point here is a stable per-frame signal. Higher techniques add L_t / L_g,
    which are correlated with L_p; the val-SIZE conclusion is unchanged, so base is sufficient.
    """
    image_full = batch['image_full'].to(device)
    image_half = batch['image_half'].to(device)
    beta = batch['beta'].to(device)
    a_val = batch['a_val'].to(device)
    unit = batch['unit_mat'].to(device)
    complex_gt = batch['complex_noise_img'].to(device)
    depth_n = DepthNorm(batch['depth'].to(device))

    r1 = model_1(image_full); out_depth = r1[0] if isinstance(r1, tuple) else r1
    r2 = model_2(image_full); out_bb = r2[0] if isinstance(r2, tuple) else r2
    pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half,
                                         max_depth_m=cfg.kitti_max_depth_m)

    out = []
    for i in range(image_full.size(0)):
        sl = slice(i, i + 1)
        l1_d = float((out_depth[sl] - depth_n[sl]).abs().mean())
        ss_d = ssim_loss(out_depth[sl], depth_n[sl], DEPTH_VAL_RANGE)
        l1_c = float((pred_complex[sl] - complex_gt[sl]).abs().mean())
        ss_c = ssim_loss(pred_complex[sl], complex_gt[sl], 1)
        loss_depth = lam_ssim * ss_d + lam_l1 * l1_d
        loss_complex = lam_ssim * ss_c + lam_l1 * l1_c
        out.append(loss_depth + loss_complex)
    return out


def block_bootstrap_se(drive_losses, n_target, n_boot, rng):
    """Standard error of the mean val loss for a val set of ~n_target frames, resampling whole
    DRIVES with replacement (respects within-drive correlation).

    Each draw: pick drives at random WITH replacement, accumulate their frames until we have at
    least n_target, then trim to exactly n_target and take the mean. n_target is capped at the
    pool size (you cannot validate on more frames than exist)."""
    drive_arrs = [np.asarray(v, dtype=np.float64) for v in drive_losses.values()]
    n_drives = len(drive_arrs)
    pool_total = int(sum(a.size for a in drive_arrs))
    n_target = min(int(n_target), pool_total)
    if n_target <= 0 or n_drives == 0:
        return float('nan')
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        acc, total = [], 0
        while total < n_target:
            a = drive_arrs[rng.randint(n_drives)]
            acc.append(a); total += a.size
        means[b] = float(np.concatenate(acc)[:n_target].mean())
    return float(means.std())


def main():
    ap = argparse.ArgumentParser(description='KITTI validation-split size analysis (no retraining).')
    ap.add_argument('--config', default=None)
    ap.add_argument('--technique', type=int, default=1)
    ap.add_argument('--variant', default='base', choices=['base', 'var1', 'var2'])
    ap.add_argument('--resume', default=None, help='checkpoint (default: T{n}_KITTI_{v}_last.ckpt)')
    ap.add_argument('--fracs', type=float, nargs='+', default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument('--bootstrap', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--epoch-delta', type=float, default=None,
                    help='typical late-epoch val-loss improvement early stopping must resolve. '
                         'The recommendation flags fractions whose SE exceeds 0.25x this.')
    ap.add_argument('--limit', type=int, default=0, help='DEBUG: only evaluate the first N held-out frames')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from data.kitti import list_completed_frames, _KittiDataset
    frames = list_completed_frames('train')
    n_total = len(frames)
    if n_total == 0:
        raise SystemExit('No KITTI train frames found — is the data present and the config right?')

    # Held-out POOL = train frames that (a) are NOT in the training subset and (b) have GT on
    # disk. This is exactly the population any val set is drawn from, and it works with whatever
    # val-tail GT you have already generated (missing frames are skipped and reported).
    gt_dir = cfg.kitti_gt_train_dir
    have_gt = set()
    for p in glob.glob(_os.path.join(gt_dir, '*complex_haze_image.npy')):
        base = _os.path.basename(p)[:-len('complex_haze_image.npy')]
        if base.isdigit():
            have_gt.add(int(base))

    subset_idx = set()
    try:
        from data.kitti import _resolve_subset_path
        sp = _resolve_subset_path(cfg)
        subset_idx = set(int(i) for i in np.load(sp).tolist())
    except Exception:
        pass

    pool = sorted(i for i in have_gt if i not in subset_idx and 0 <= i < n_total)
    if args.limit:
        pool = pool[:args.limit]
    if not pool:
        raise SystemExit(
            "No held-out frames with GT found in %s.\n"
            "Generate the val-tail GT first (make_kitti_test_tail_indices.py + "
            "generate_gt_kitti_subset.py --indices ...), then re-run." % gt_dir)

    print("KITTI val-size analysis")
    print("  total train frames        : %d" % n_total)
    print("  training subset (excluded) : %d" % len(subset_idx))
    print("  held-out frames WITH GT    : %d  (the analysis pool)" % len(pool))
    n_drives_pool = len(set(frames[i]['drive'] for i in pool))
    print("  distinct drives in pool    : %d   <-- the REAL effective sample-size ceiling" % n_drives_pool)

    # ---- build model + load checkpoint ----
    m1, m2, m3 = build_models(args.technique, args.variant)
    ckpt_path = args.resume or _os.path.join(
        cfg.checkpoint_dir, 'T%d_KITTI_%s.ckpt' % (args.technique, args.variant))
    if not _os.path.exists(ckpt_path):
        raise SystemExit('No checkpoint at %s (pass --resume).' % ckpt_path)
    ck = torch.load(ckpt_path, map_location=device)
    m1.load_state_dict(ck['state_dict_1']); m2.load_state_dict(ck['state_dict_2'])
    m1 = m1.to(device).eval(); m2 = m2.to(device).eval()
    print("  checkpoint                 : %s (epoch %s)" % (ckpt_path, ck.get('cur_epoch')))

    # ---- evaluate per-frame loss over the pool ----
    ds = _KittiDataset(frames, gt_dir, cfg.beta_mat_kitti_train, cfg.a_mat_kitti_train,
                       cfg.kitti_max_depth_m, cfg.kitti_beta_scale, augment=False)
    from torch.utils.data import Subset, DataLoader
    loader = DataLoader(Subset(ds, pool), batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    lam_l1, lam_ssim = cfg.lambda_l1, cfg.lambda_ssim

    drive_of = [frames[i]['drive'] for i in pool]
    per_frame, cursor = [], 0
    with torch.no_grad():
        for batch in loader:
            per_frame.extend(per_frame_val_loss(m1, m2, batch, device, cfg, lam_l1, lam_ssim))
            cursor += 1
            if cursor % 50 == 0:
                print('    ...scored %d / %d frames' % (len(per_frame), len(pool)))
    per_frame = np.asarray(per_frame, dtype=np.float64)

    drive_losses = {}
    for d, v in zip(drive_of, per_frame):
        drive_losses.setdefault(d, []).append(v)

    grand_mean = float(per_frame.mean())
    frame_std = float(per_frame.std())
    rng = np.random.RandomState(0)

    print("\n  grand mean val loss over the pool : %.5f" % grand_mean)
    print("  per-frame std                     : %.5f\n" % frame_std)

    # ---- per-fraction analysis ----
    hdr = ('%-7s %-9s %-11s %-12s %-12s %-11s' %
           ('frac', 'n_frames', 'SE (iid)', 'SE (drives)', 'rel_SE%', 'val_GT_cost'))
    print(hdr); print('  ' + '-' * (len(hdr) + 2))
    rows = []
    for f in sorted(args.fracs):
        n_val = int(f * n_total)
        se_iid = frame_std / np.sqrt(max(n_val, 1))                 # naive (overconfident)
        se_blk = block_bootstrap_se(drive_losses, n_val, args.bootstrap, rng)  # honest
        rel = 100.0 * se_blk / grand_mean if grand_mean else float('nan')
        rows.append((f, n_val, se_iid, se_blk, rel))
        print('  %-7.2f %-9d %-11.6f %-12.6f %-12.2f %-11d'
              % (f, n_val, se_iid, se_blk, rel, n_val))

    print("\n  SE(iid) assumes independent frames; SE(drives) resamples whole drives and is the")
    print("  HONEST figure. The gap between them is how much within-drive correlation inflates")
    print("  the real noise — for KITTI it is usually large.")

    # ---- recommendation ----
    print("\n  RECOMMENDATION")
    if args.epoch_delta:
        thr = 0.25 * args.epoch_delta
        print("    early stopping must resolve a per-epoch improvement of ~%.5f (--epoch-delta)."
              % args.epoch_delta)
        print("    A val set is 'good enough' when its SE(drives) < %.5f (0.25x that).\n" % thr)
        ok = [f for (f, n, si, sb, r) in rows if sb < thr]
        if ok:
            best = min(ok)
            print("    -> smallest adequate fraction: %.2f  (%d frames). Larger only wastes"
                  % (best, int(best * n_total)))
            print("       generation + per-epoch compute; the selection signal is already stable.")
        else:
            print("    -> NONE of the tested fractions reach the threshold. Either the pool has too")
            print("       few drives (effective N ~ %d), or --epoch-delta is very small. A larger" % n_drives_pool)
            print("       val set will NOT help much (you are correlation-limited, not count-limited);")
            print("       consider a DIVERSE val split (random drives) rather than the tail.")
    else:
        print("    Pass --epoch-delta <x> (your typical late-epoch val improvement, read off the")
        print("    training log) to get a concrete cutoff. As a rule of thumb, pick the smallest")
        print("    fraction where SE(drives) stops falling meaningfully — past that you are adding")
        print("    correlated frames, not information.")
    print("\n    NOTE: with kitti_train_mode='subset' the training set is a fixed 10k either way,")
    print("    so a smaller val set costs you NOTHING in training data — only a slightly noisier")
    print("    early-stopping signal, while saving val GT generation and per-epoch val compute.")


if __name__ == '__main__':
    main()
