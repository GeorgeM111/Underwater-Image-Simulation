# --- path bootstrap: baseline dir + repo root (shared packages) ---
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

"""Encoder_Decoder_Direct / NYU -- evaluation script.

Computes accuracy metrics (a1/a2/a3, abs_rel, rmse, log_10) of the direct
encoder->decoder prediction against the complex (ricardo) GT image. Logic
preserved from the original ``Perform_Test.py``; only plumbing/paths/imports
changed to use the shared packages (the intermediate ``.pt`` GT/Pred dance is
folded into a single pass over the test loader).
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import load_config
from models.model_builder import ImageModel
from data.nyu import get_test_loader
from utils.helpers import AverageMeter, DepthNorm, colorize, simple_save_images
from utils.loss import ssim


def main():
    parser = argparse.ArgumentParser(description='Encoder_Decoder_Direct NYU evaluation')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--resume', default=None, help='checkpoint to evaluate')
    args = parser.parse_args()
    cfg = load_config(args.config)

    test_loader = get_test_loader(cfg)
    ClacAccuracyOnly(cfg, args, test_loader)


def ClacAccuracyOnly(cfg, args, test_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageModel(pretrained=cfg.pretrained_encoder).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model.cuda())
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict_3'])
    model.eval()

    a1_acc = 0.0
    cnt_1 = 0

    a2_acc = 0.0
    cnt_2 = 0

    a3_acc = 0.0
    cnt_3 = 0

    abs_rel_acc = 0.0
    cnt_4 = 0

    rmse_acc = 0.0
    cnt_5 = 0

    log_10_acc = 0.0
    cnt_6 = 0

    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):

            image = sample_batched['image_full'].to(device)  # full size
            complex_image_tensor = sample_batched['complex_noise_img'].to(device)  # half size

            pred_complex_image = model(image)

            abs_rel, rmse, log_10, a1, a2, a3 = add_results_1(complex_image_tensor, pred_complex_image, border_crop_size=16)

            if (torch.isfinite(a1)):
                a1_acc = a1_acc + a1.detach().to("cpu").numpy()
                cnt_1 = cnt_1 + 1

            if (torch.isfinite(a2)):
                a2_acc = a2_acc + a2.detach().to("cpu").numpy()
                cnt_2 = cnt_2 + 1

            if (torch.isfinite(a3)):
                a3_acc = a3_acc + a3.detach().to("cpu").numpy()
                cnt_3 = cnt_3 + 1

            if (torch.isfinite(abs_rel)):
                abs_rel_acc = abs_rel_acc + abs_rel.detach().to("cpu").numpy()
                cnt_4 = cnt_4 + 1

            if (torch.isfinite(rmse)):
                rmse_acc = rmse_acc + rmse.detach().to("cpu").numpy()
                cnt_5 = cnt_5 + 1

            if (torch.isfinite(log_10)):
                log_10_acc = log_10_acc + log_10.detach().to("cpu").numpy()
                cnt_6 = cnt_6 + 1

    a1_acc = a1_acc / cnt_1
    a2_acc = a2_acc / cnt_2
    a3_acc = a3_acc / cnt_3

    abs_rel_acc = abs_rel_acc / cnt_4
    rmse_acc = rmse_acc / cnt_5
    log_10_acc = log_10_acc / cnt_6

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(a1_acc, a2_acc, a3_acc, abs_rel_acc, rmse_acc, log_10_acc))


def compute_errors_nyu(pred, gt):
    y = gt
    x = pred
    thresh = torch.max((y / x), (x / y))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    abs_rel = torch.mean(torch.abs(y - x) / y)
    rmse = (y - x) ** 2
    rmse = torch.sqrt(rmse.mean())
    log_10 = (torch.abs(torch.log10(y) - torch.log10(x))).nanmean()
    return abs_rel, rmse, log_10, a1, a2, a3


def add_results_1(gt_image, pred_image, border_crop_size=16, use_224=False):

    predictions = []
    testSetDepths = []
    half_border_size = border_crop_size // 2

    gt_image_border_cut = gt_image[:, :, half_border_size:-half_border_size, half_border_size:-half_border_size]  # cutting the border to remove the border problem/issue
    pred_image_border_cut = pred_image[:, :, half_border_size:-half_border_size, half_border_size:-half_border_size]  # cutting the border to remove the border problem/issue

    del gt_image, pred_image

    replicate = nn.ReplicationPad2d(half_border_size)
    gt_image_border_cut = replicate(gt_image_border_cut)  # now extrapolate by using the inside content of the image
    pred_image_border_cut = replicate(pred_image_border_cut)  # now extrapolate by using the inside content of the image

    gt_image_border_cut = F.interpolate(gt_image_border_cut, (480, 640), mode='bilinear', align_corners=True)
    pred_image_border_cut = F.interpolate(pred_image_border_cut, (480, 640), mode='bilinear', align_corners=True)

    # Compute errors per image in batch
    for j in range(len(gt_image_border_cut)):
        predictions.append(pred_image_border_cut[j])
        testSetDepths.append(gt_image_border_cut[j])

    predictions = torch.stack(predictions, axis=0)
    testSetDepths = torch.stack(testSetDepths, axis=0)

    del pred_image_border_cut, gt_image_border_cut
    abs_rel, rmse, log_10, a1, a2, a3 = compute_errors_nyu(predictions, testSetDepths)

    del predictions, testSetDepths

    return abs_rel, rmse, log_10, a1, a2, a3


if __name__ == '__main__':
    main()
