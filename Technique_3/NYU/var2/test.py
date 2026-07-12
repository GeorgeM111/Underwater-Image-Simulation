# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Technique 3 / NYU / var2 -- evaluation script.

Slim entry point: argparse + config + build_models + data loader + eval loop.
Reports NYU-style error metrics on the reconstructed complex (degraded) image.

Technique 3 has TWO predictions of the degraded image: pred_complex (Eq.11 physics —
the one the paper reports) and I_Direct from model_3. Both are evaluated, in separate
blocks; they are never merged. The learned loss weights play no part in evaluation.
"""

import os
import argparse

import torch

from config import load_config
from models.model_builder import build_models
from data.nyu import get_test_loader
from utils.helpers import AverageMeter
from utils.metrics import add_results_1, image_quality
from utils.physics import compute_complex_image

TECHNIQUE = 3
VARIANT = 'var2'
DATASET = 'NYU'

KEYS = ['mae', 'psnr', 'ssim', 'abs_rel', 'rmse', 'log10', 'a1', 'a2', 'a3']


def _score(meters, gt, pred, n):
    """Update every meter and return the names of the metrics that came out non-finite.

    The update is UNCONDITIONAL. The old ``if torch.isfinite(v)`` skip, combined with an
    AverageMeter that never received a value, printed abs_rel/rmse/log10 = 0.0000 for a
    fully-diverged model — i.e. a PERFECT score. A single Inf pixel also took out exactly
    those three while mae/psnr/ssim still looked plausible, so the table had no visible
    anomaly. NaN must reach the caller.
    """
    results = tuple(image_quality(gt, pred)) + tuple(add_results_1(gt, pred))
    bad = [k for k, v in zip(KEYS, results) if not torch.isfinite(v)]
    for k, v in zip(KEYS, results):
        meters[k].update(v.item(), n)
    return bad


def _report(title, meters):
    print(title)
    for k in KEYS:
        # n is printed so an empty meter can never masquerade as a score.
        print('  %-8s %.4f  (n=%d)' % (k, meters[k].avg, meters[k].count))


def main():
    parser = argparse.ArgumentParser(description='Evaluate Technique 3 NYU var2')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to evaluate (default: the best one)')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_1, model_2, model_3 = build_models(TECHNIQUE, VARIANT)
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    if model_3 is not None:
        model_3 = model_3.to(device)

    # A missing --resume used to leave the network at its RANDOM INITIALISATION and print a
    # plausible-looking table. Default to the trained checkpoint and refuse to run without one.
    ckpt_path = args.resume or os.path.join(
        cfg.checkpoint_dir, 'T%d_%s_%s.ckpt' % (TECHNIQUE, DATASET, VARIANT))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('No checkpoint at %s — train first or pass --resume.' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model_1.load_state_dict(ckpt['state_dict_1'])
    model_2.load_state_dict(ckpt['state_dict_2'])
    if model_3 is not None:
        model_3.load_state_dict(ckpt['state_dict_3'])
    print('[T%d %s %s] loaded %s (cur_epoch=%s)' % (TECHNIQUE, DATASET, VARIANT, ckpt_path,
                                                    ckpt.get('cur_epoch')))

    model_1.eval()
    model_2.eval()
    if model_3 is not None:
        model_3.eval()

    test_loader = get_test_loader(cfg)
    meters = {k: AverageMeter() for k in KEYS}
    meters_direct = {k: AverageMeter() for k in KEYS}
    n_bad_batches = 0
    n_batches = 0

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            n_batches += 1
            image_full = batch['image_full'].to(device)
            image_half = batch['image_half'].to(device)
            beta = batch['beta'].to(device)
            a_val = batch['a_val'].to(device)
            unit = batch['unit_mat'].to(device)
            complex_gt = batch['complex_noise_img'].to(device)
            n = image_full.size(0)

            r1 = model_1(image_full)
            out_depth = r1[0] if isinstance(r1, tuple) else r1
            r2 = model_2(image_full)
            out_bb = r2[0] if isinstance(r2, tuple) else r2
            # max_depth_m must be passed: the no-arg fallback reads the module-global CONFIG,
            # so --config would silently not apply to the physics.
            pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half,
                                                 max_depth_m=cfg.nyu_max_depth_m)

            # Image-quality metrics (MAE/PSNR/SSIM) reflect visual quality; the
            # depth-ratio metrics (abs_rel/log10/delta) are kept for continuity.
            bad = _score(meters, complex_gt, pred_complex, n)

            if model_3 is not None:
                r3 = model_3(image_full)
                out_direct = r3[0] if isinstance(r3, tuple) else r3
                bad = bad + ['direct:%s' % k for k in _score(meters_direct, complex_gt, out_direct, n)]

            if bad:
                n_bad_batches += 1
                print('[WARN] batch %d produced non-finite %s' % (bi, ','.join(bad)))

    if n_bad_batches:
        raise SystemExit('ABORT: %d/%d batches were non-finite — these results are INVALID.'
                         % (n_bad_batches, n_batches))

    _report('[T%d %s %s] evaluation:' % (TECHNIQUE, DATASET, VARIANT), meters)
    if model_3 is not None:
        # I_Direct is supervised against complex_gt, so it is a legitimate SECOND prediction.
        # Reported separately: the paper's headline number is pred_complex (Eq.11).
        _report('[T%d %s %s] direct-branch (I_Direct) evaluation:' % (TECHNIQUE, DATASET, VARIANT),
                meters_direct)


if __name__ == '__main__':
    main()
