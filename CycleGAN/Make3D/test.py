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

"""CycleGAN / Make3D -- evaluation (metrics of generated degraded image vs GT).

Direct inference: translate the clean image (domain A) to the degraded domain B
with netG_A2B and score it against the real degraded GT. Mirrors the Technique_*
/ Pix2Pix evaluation (no pre-saved .pt files, no separate LogProgress pass).
"""

import os
import argparse

import torch
import torch.nn.functional as F

from gan_models import *
from gan_utils import to_gan_range, from_gan_range
from utils.helpers import AverageMeter
from utils.metrics import add_results_1, image_quality
from config import load_config
from data.make3d import get_test_loader


def main():
    parser = argparse.ArgumentParser(description='CycleGAN eval - Make3D')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='CycleGAN checkpoint (default: config checkpoint dir)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG_A2B = Generator(3, 3).to(device)
    ckpt_path = args.resume or os.path.join(cfg.checkpoint_dir, 'CycleGAN', 'CycleGAN_Make3D.ckpt')
    netG_A2B.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict_G_A2B'])
    netG_A2B.eval()

    test_loader = get_test_loader(cfg)
    # MAE/PSNR/SSIM are the meaningful, BOUNDED metrics for a generated IMAGE; the
    # depth-ratio metrics (abs_rel/log10/delta) are kept for continuity with the paper.
    keys = ['mae', 'psnr', 'ssim', 'abs_rel', 'rmse', 'log10', 'a1', 'a2', 'a3']
    meters = {k: AverageMeter() for k in keys}

    with torch.no_grad():
        for sample_batched in test_loader:
            input_A = sample_batched['image_half'].to(device)          # clean [0,1] (domain A)
            input_B = sample_batched['complex_noise_img'].to(device)   # degraded GT [0,1] (domain B)
            # Generator trained in [-1,1]: normalise in, denormalise out to [0,1].
            fake_B = netG_A2B(to_gan_range(input_A))
            fake_B = from_gan_range(F.interpolate(fake_B, size=input_B.shape[-2:], mode='bicubic', align_corners=False))

            results = tuple(image_quality(input_B, fake_B)) + tuple(add_results_1(input_B, fake_B, border_crop_size=16))
            for k, v in zip(keys, results):
                if torch.isfinite(v):
                    meters[k].update(v.item(), input_A.size(0))

    print('[CycleGAN Make3D] evaluation:')
    for k in keys:
        print('  %-8s %.4f' % (k, meters[k].avg))


if __name__ == '__main__':
    main()
