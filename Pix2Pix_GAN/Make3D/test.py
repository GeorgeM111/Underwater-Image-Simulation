# --- path bootstrap: baseline dir (local modules) + repo root (shared packages) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.dirname(_HERE)
if _BASE not in _sys.path:
    _sys.path.insert(0, _BASE)
_p = _BASE
for _ in range(8):
    _p = _os.path.dirname(_p)
    if _os.path.exists(_os.path.join(_p, 'config.py')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        break

"""Pix2Pix GAN / Make3D -- evaluation (metrics of generated vs GT degraded image)."""

import os
import argparse

import torch
import torch.nn.functional as F

from gan_models import *
from utils.helpers import AverageMeter
from utils.metrics import add_results_1
from config import load_config
from data.make3d import get_test_loader


def main():
    parser = argparse.ArgumentParser(description='Pix2Pix GAN eval - Make3D')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='generator checkpoint (default: config checkpoint dir)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = GeneratorUNet_Make3D().to(device)
    ckpt_path = args.resume or os.path.join(cfg.checkpoint_dir, 'Pix2Pix_GAN', 'Pix2Pix_Make3D.ckpt')
    generator.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict_G'])
    generator.eval()

    test_loader = get_test_loader(cfg)
    keys = ['abs_rel', 'rmse', 'log10', 'a1', 'a2', 'a3']
    meters = {k: AverageMeter() for k in keys}

    with torch.no_grad():
        for sample_batched in test_loader:
            input_A = sample_batched['image_half'].to(device)          # clean
            input_B = sample_batched['complex_noise_img'].to(device)   # degraded GT
            fake_B = generator(input_A)
            fake_B = F.interpolate(fake_B, size=input_B.shape[-2:], mode='bicubic', align_corners=False)

            abs_rel, rmse, log_10, a1, a2, a3 = add_results_1(input_B, fake_B, border_crop_size=16)
            for k, v in zip(keys, [abs_rel, rmse, log_10, a1, a2, a3]):
                if torch.isfinite(v):
                    meters[k].update(v.item(), input_A.size(0))

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(
        meters['a1'].avg, meters['a2'].avg, meters['a3'].avg,
        meters['abs_rel'].avg, meters['rmse'].avg, meters['log10'].avg))


if __name__ == '__main__':
    main()
