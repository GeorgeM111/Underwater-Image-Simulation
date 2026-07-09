"""Evaluation metrics: NYU-style depth/image error metrics."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    """Peak signal-to-noise ratio (dB) for images in [0, max_val]."""
    mse = torch.mean((pred.clamp(0, max_val) - target.clamp(0, max_val)) ** 2)
    if mse.item() == 0:
        return torch.tensor(float('inf'), device=pred.device)
    return 10.0 * torch.log10((max_val ** 2) / mse)


def image_quality(gt_image, pred_image, border_crop_size=16):
    """Perceptually meaningful metrics for the simulated (complex) IMAGE: MAE, PSNR, SSIM.

    Why this exists: the reported abs_rel / log10 / delta metrics are DEPTH-ratio
    metrics (thresh = max(y/x, x/y), |y-x|/y, log10). On RGB images in [0,1] they
    blow up on dark pixels (y -> 0), so a visually excellent simulated image can
    score terribly. MAE/PSNR/SSIM are bounded and track how good the image
    actually looks, so 'amazing predictions' get good numbers.
    """
    from utils.loss import ssim  # local import avoids any import-time coupling
    hb = border_crop_size // 2
    gt = gt_image[:, :, hb:-hb, hb:-hb].clamp(0, 1)
    pr = pred_image[:, :, hb:-hb, hb:-hb].clamp(0, 1)
    mae = torch.mean(torch.abs(pr - gt))
    ps = psnr(pr, gt)
    ss = ssim(pr, gt, val_range=1.0)
    return mae, ps, ss


def compute_errors_nyu(pred, gt, eps=1e-3):
    # These are ratio/log metrics designed for positive DEPTH. On [0,1] IMAGES with
    # dark/zero pixels, y/x and log10(x) blow up to inf/NaN -> the caller's
    # `if torch.isfinite(v)` then SKIPS the batch, so the meter reads a spurious 0
    # (the classic "log10 is always 0"). Flooring to a small positive eps keeps every
    # term finite; for real depth (10..1000) the floor is negligible.
    y = gt.clamp(min=eps)
    x = pred.clamp(min=eps)
    thresh = torch.max((y / x), (x / y))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    abs_rel = torch.mean(torch.abs(y - x) / y)
    rmse = (y - x) ** 2
    rmse = torch.sqrt(rmse.mean())
    log_10 = (torch.abs(torch.log10(y) - torch.log10(x))).nanmean()
    return abs_rel, rmse, log_10, a1, a2, a3


def add_results(gt_image, pred_image, border_crop_size=16):
    predictions = []
    testSetDepths = []
    gt_image_border_cut = gt_image[:, :, border_crop_size:-border_crop_size, border_crop_size:-border_crop_size]
    pred_image_border_cut = pred_image[:, :, border_crop_size:-border_crop_size, border_crop_size:-border_crop_size]

    del gt_image, pred_image

    for j in range(len(gt_image_border_cut)):
        predictions.append(pred_image_border_cut[j])
        testSetDepths.append(gt_image_border_cut[j])

    predictions = torch.stack(predictions, axis=0)
    testSetDepths = torch.stack(testSetDepths, axis=0)

    del pred_image_border_cut, gt_image_border_cut
    abs_rel, rmse, log_10, a1, a2, a3 = compute_errors_nyu(predictions, testSetDepths)

    del predictions, testSetDepths

    return abs_rel, rmse, log_10, a1, a2, a3


def add_results_1(gt_image, pred_image, border_crop_size=16, use_224=False, target_size=None):
    """Border-crop, then compute error metrics.

    ``target_size`` (H, W): if given, both images are resized to it before scoring.
    Default ``None`` => evaluate at the data's NATIVE resolution. The old code
    hard-coded a resize to (480, 640) (NYU's full res) for BOTH datasets, which
    upsampled Make3D's native 173x230 ~2.7x, magnifying its coarse laser-derived
    GT and the smooth-pred/blocky-GT mismatch. Native-resolution evaluation keeps
    prediction and GT on the same, honest grid.
    """
    predictions = []
    testSetDepths = []
    half_border_size = border_crop_size // 2

    gt_image_border_cut = gt_image[:, :, half_border_size:-half_border_size, half_border_size:-half_border_size]
    pred_image_border_cut = pred_image[:, :, half_border_size:-half_border_size, half_border_size:-half_border_size]

    del gt_image, pred_image

    replicate = nn.ReplicationPad2d(half_border_size)
    gt_image_border_cut = replicate(gt_image_border_cut)
    pred_image_border_cut = replicate(pred_image_border_cut)

    if target_size is not None:
        gt_image_border_cut = F.interpolate(gt_image_border_cut, target_size, mode='bilinear', align_corners=True)
        pred_image_border_cut = F.interpolate(pred_image_border_cut, target_size, mode='bilinear', align_corners=True)

    for j in range(len(gt_image_border_cut)):
        predictions.append(pred_image_border_cut[j])
        testSetDepths.append(gt_image_border_cut[j])

    predictions = torch.stack(predictions, axis=0)
    testSetDepths = torch.stack(testSetDepths, axis=0)

    del pred_image_border_cut, gt_image_border_cut
    abs_rel, rmse, log_10, a1, a2, a3 = compute_errors_nyu(predictions, testSetDepths)

    del predictions, testSetDepths

    return abs_rel, rmse, log_10, a1, a2, a3
