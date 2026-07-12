# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Technique 2 / NYU / var1 -- evaluation script.

Slim entry point: argparse + config + build_models + data loader + eval loop.
Reports NYU-style error metrics on the reconstructed complex (degraded) image.
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

TECHNIQUE = 2
VARIANT = 'var1'
DATASET = 'NYU'


def main():
    parser = argparse.ArgumentParser(description='Evaluate Technique 2 NYU var1')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to evaluate (default: the trained one)')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Technique 2 has no direct-prediction branch: build_models returns model_3 = None.
    model_1, model_2, _ = build_models(TECHNIQUE, VARIANT)
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    # A checkpoint is MANDATORY. With no --resume this script used to evaluate a RANDOMLY
    # INITIALISED network and print a plausible-looking table of numbers.
    ckpt_path = args.resume or os.path.join(cfg.checkpoint_dir,
                                            'T%d_%s_%s.ckpt' % (TECHNIQUE, DATASET, VARIANT))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('No checkpoint at %s — train first or pass --resume.' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model_1.load_state_dict(ckpt['state_dict_1'])
    model_2.load_state_dict(ckpt['state_dict_2'])
    print('[T%d %s %s] loaded %s (epoch %s)' % (TECHNIQUE, DATASET, VARIANT, ckpt_path, ckpt.get('cur_epoch')))

    model_1.eval()
    model_2.eval()

    test_loader = get_test_loader(cfg)
    keys = ['mae', 'psnr', 'ssim', 'abs_rel', 'rmse', 'log10', 'a1', 'a2', 'a3']
    meters = {k: AverageMeter() for k in keys}
    n_bad_batches, n_batches = 0, 0

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            image_full = batch['image_full'].to(device)
            image_half = batch['image_half'].to(device)
            beta = batch['beta'].to(device)
            a_val = batch['a_val'].to(device)
            unit = batch['unit_mat'].to(device)
            complex_gt = batch['complex_noise_img'].to(device)

            r1 = model_1(image_full)
            out_depth = r1[0] if isinstance(r1, tuple) else r1
            r2 = model_2(image_full)
            out_bb = r2[0] if isinstance(r2, tuple) else r2
            # max_depth_m must be passed explicitly: the no-arg fallback reads the module-global
            # CONFIG, so --config would silently not apply to the physics.
            pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half,
                                                 max_depth_m=cfg.nyu_max_depth_m)

            # Image-quality metrics (MAE/PSNR/SSIM) reflect visual quality; the
            # depth-ratio metrics (abs_rel/log10/delta) are kept for continuity.
            results = tuple(image_quality(complex_gt, pred_complex)) + \
                      tuple(add_results_1(complex_gt, pred_complex))
            n_batches += 1
            # Non-finite values are REPORTED, never skipped. The old `if isfinite` skip meant a
            # diverged model printed abs_rel/rmse/log10 = 0.0000 (a PERFECT score, from an empty
            # meter), while mae/psnr/ssim still looked plausible — the table had no visible anomaly.
            bad = [k for k, v in zip(keys, results) if not torch.isfinite(v)]
            if bad:
                n_bad_batches += 1
                print('[WARN] batch %d produced non-finite %s' % (bi, ','.join(bad)))
            for k, v in zip(keys, results):
                meters[k].update(v.item(), image_full.size(0))

    if n_bad_batches:
        raise SystemExit('ABORT: %d/%d batches were non-finite — these results are INVALID.'
                         % (n_bad_batches, n_batches))

    print('[T%d %s %s] evaluation:' % (TECHNIQUE, DATASET, VARIANT))
    for k in keys:
        # n is printed so an empty meter (avg = NaN) can never masquerade as a score.
        print('  %-8s %.4f  (n=%d)' % (k, meters[k].avg, meters[k].count))


if __name__ == '__main__':
    main()
