# --- repo-root path bootstrap (find the dir containing config.py) ---
import os as _os, sys as _sys
_p = _os.path.abspath(__file__)
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Technique 1 / Make3D / var1 -- evaluation script.

Slim entry point: argparse + config + build_models + data loader + eval loop.
Reports NYU-style error metrics on the reconstructed complex (degraded) image.
"""

import argparse

import torch

from config import load_config
from models.model_builder import build_models
from data.make3d import get_test_loader
from utils.helpers import AverageMeter
from utils.metrics import add_results_1, image_quality
from utils.physics import compute_complex_image

TECHNIQUE = 1
VARIANT = 'var1'
DATASET = 'Make3D'


def main():
    parser = argparse.ArgumentParser(description='Evaluate Technique 1 Make3D var1')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to evaluate')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_1, model_2, model_3 = build_models(TECHNIQUE, VARIANT)
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    if model_3 is not None:
        model_3 = model_3.to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model_1.load_state_dict(ckpt['state_dict_1'])
        model_2.load_state_dict(ckpt['state_dict_2'])
        if model_3 is not None and 'state_dict_3' in ckpt:
            model_3.load_state_dict(ckpt['state_dict_3'])

    model_1.eval()
    model_2.eval()
    if model_3 is not None:
        model_3.eval()

    test_loader = get_test_loader(cfg)
    keys = ['mae', 'psnr', 'ssim', 'abs_rel', 'rmse', 'log10', 'a1', 'a2', 'a3']
    meters = {k: AverageMeter() for k in keys}

    with torch.no_grad():
        for batch in test_loader:
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
            pred_complex = compute_complex_image(out_depth, out_bb, beta, a_val, unit, image_half, max_depth_m=cfg.make3d_max_depth_m)

            # Image-quality metrics (MAE/PSNR/SSIM) reflect visual quality; the
            # depth-ratio metrics (abs_rel/log10/delta) are kept for continuity.
            results = tuple(image_quality(complex_gt, pred_complex)) + \
                      tuple(add_results_1(complex_gt, pred_complex))
            for k, v in zip(keys, results):
                if torch.isfinite(v):
                    meters[k].update(v.item(), image_full.size(0))

    print('[T%d %s %s] evaluation:' % (TECHNIQUE, DATASET, VARIANT))
    for k in keys:
        print('  %-8s %.4f' % (k, meters[k].avg))


if __name__ == '__main__':
    main()
