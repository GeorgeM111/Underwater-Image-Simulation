# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Technique 4 / NYU / var2 -- evaluation script.

Slim entry point: argparse + config + build_models + data loader + eval loop.
Reports NYU-style error metrics on the reconstructed complex (degraded) image.

Technique 4 has TWO legitimate predictions of the degraded image: the physics-based
reconstruction (Eq.11, ``pred_complex``) and the direct branch (model_3, ``out_direct``).
The paper reports the former, so that is the primary block; the direct branch is scored
in a second, clearly separated block and is never merged into the primary numbers.
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

TECHNIQUE = 4
VARIANT = 'var2'
DATASET = 'NYU'


def main():
    parser = argparse.ArgumentParser(description='Evaluate Technique 4 NYU var2')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to evaluate')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Technique 4 always has the direct-prediction branch, so model_3 is never None here.
    model_1, model_2, model_3 = build_models(TECHNIQUE, VARIANT)
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    model_3 = model_3.to(device)

    # A checkpoint is MANDATORY. Without this, `python test.py` evaluated a randomly
    # initialised network and printed a plausible-looking table of numbers.
    ckpt_path = args.resume or os.path.join(cfg.checkpoint_dir, 'T%d_%s_%s.ckpt' % (TECHNIQUE, DATASET, VARIANT))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('No checkpoint at %s - train first or pass --resume.' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model_1.load_state_dict(ckpt['state_dict_1'])
    model_2.load_state_dict(ckpt['state_dict_2'])
    model_3.load_state_dict(ckpt['state_dict_3'])
    print('[T%d %s %s] loaded %s (cur_epoch=%s)' % (TECHNIQUE, DATASET, VARIANT, ckpt_path, ckpt.get('cur_epoch')))

    model_1.eval()
    model_2.eval()
    model_3.eval()

    test_loader = get_test_loader(cfg)
    keys = ['mae', 'psnr', 'ssim', 'abs_rel', 'rmse', 'log10', 'a1', 'a2', 'a3']
    meters = {k: AverageMeter() for k in keys}
    meters_direct = {k: AverageMeter() for k in keys}
    n_bad_batches = 0
    n_batches = 0

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
            r3 = model_3(image_full)
            out_direct = r3[0] if isinstance(r3, tuple) else r3
            # max_depth_m must be passed explicitly: the no-arg fallback reads the module
            # global CONFIG, so --config would silently not apply to the physics.
            pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half,
                                                 max_depth_m=cfg.nyu_max_depth_m)
            n_batches += 1

            # Image-quality metrics (MAE/PSNR/SSIM) reflect visual quality; the
            # depth-ratio metrics (abs_rel/log10/delta) are kept for continuity.
            results = tuple(image_quality(complex_gt, pred_complex)) + \
                      tuple(add_results_1(complex_gt, pred_complex))
            results_direct = tuple(image_quality(complex_gt, out_direct)) + \
                             tuple(add_results_1(complex_gt, out_direct))

            # NaN is REPORTED, never skipped. The old `if isfinite: update` skip left the
            # meter at count 0, and a zeroed lower-is-better metric (abs_rel/rmse/log10 =
            # 0.0000) reads as a PERFECT score - a diverged model printed a flawless table.
            bad = [k for k, v in zip(keys, results) if not torch.isfinite(v)]
            bad += ['direct:' + k for k, v in zip(keys, results_direct) if not torch.isfinite(v)]
            if bad:
                n_bad_batches += 1
                print('[WARN] batch %d produced non-finite %s' % (bi, ','.join(bad)))
            for k, v in zip(keys, results):
                meters[k].update(v.item(), image_full.size(0))
            for k, v in zip(keys, results_direct):
                meters_direct[k].update(v.item(), image_full.size(0))

    print('[T%d %s %s] evaluation:' % (TECHNIQUE, DATASET, VARIANT))
    for k in keys:
        print('  %-8s %.4f  (n=%d)' % (k, meters[k].avg, meters[k].count))
    print('[T%d %s %s] direct-branch (I_Direct) evaluation:' % (TECHNIQUE, DATASET, VARIANT))
    for k in keys:
        print('  %-8s %.4f  (n=%d)' % (k, meters_direct[k].avg, meters_direct[k].count))

    if n_bad_batches:
        raise SystemExit('ABORT: %d/%d batches were non-finite - these results are INVALID.'
                         % (n_bad_batches, n_batches))


if __name__ == '__main__':
    main()
