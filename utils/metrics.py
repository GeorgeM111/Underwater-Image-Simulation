"""Evaluation metrics for the simulated (complex) IMAGE.

NaN POLICY (single, uniform, loud):
    Nothing in this module hides a non-finite value. No ``nanmean``, no
    ``isfinite`` skip, no ``sys.exit``. If the model emits NaN/Inf, the metric
    comes out NaN and the CALLER is responsible for reporting it. Previously four
    different NaN policies coexisted in one 9-metric row, so a broken model could
    print a table with no visible anomaly (abs_rel/rmse/log10 -> 0.0000 = perfect,
    while mae/psnr/ssim still looked plausible).

Note on the metric set: per the paper (p.12), abs_rel/rmse/log10/delta are computed
between the ground-truth and predicted IMAGE, not depth. They are ratio/log metrics
borrowed from the depth literature; on [0,1] images they misbehave on dark pixels,
so MAE/PSNR/SSIM are reported alongside as the honest image-quality signal.
"""

import torch


def psnr(pred, target, max_val=1.0):
    """Peak signal-to-noise ratio (dB) for images in [0, max_val].

    The mse is floored instead of returning +inf for a perfect match: an infinite
    PSNR is not finite, so a caller filtering on ``isfinite`` would have DROPPED a
    perfect reconstruction (and, with a zero-initialised meter, printed psnr 0.0000
    — the worst possible value — for the best possible model).
    """
    mse = torch.mean((pred.clamp(0, max_val) - target.clamp(0, max_val)) ** 2)
    mse = mse.clamp(min=1e-12)
    return 10.0 * torch.log10((max_val ** 2) / mse)


def image_quality(gt_image, pred_image, border_crop_size=16):
    """Perceptually meaningful metrics for the simulated IMAGE: MAE, PSNR, SSIM.

    Scores the SAME pixel region as :func:`add_results_1` (both crop
    ``border_crop_size // 2`` from each side and score what remains).
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
    """Ratio/log error metrics.

    The eps floor keeps FINITE-but-extreme inputs finite (y/x and log10(x) blow up
    as y -> 0 on dark image pixels). It does NOT sanitise NaN — ``torch.clamp``
    PROPAGATES NaN — and that is deliberate: a NaN prediction must surface as a NaN
    metric, not be silently laundered into a good-looking number.
    """
    y = gt.clamp(min=eps)
    x = pred.clamp(min=eps)
    thresh = torch.max((y / x), (x / y))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    abs_rel = torch.mean(torch.abs(y - x) / y)
    rmse = torch.sqrt(((y - x) ** 2).mean())
    log_10 = (torch.abs(torch.log10(y) - torch.log10(x))).mean()
    return abs_rel, rmse, log_10, a1, a2, a3


def add_results_1(gt_image, pred_image, border_crop_size=16, use_224=False, target_size=None):
    """Border-crop, then compute the ratio/log error metrics.

    The border crop removes the reconstruction ring at the image edge. The previous
    implementation cropped and then ``ReplicationPad2d``-ed the border straight back,
    which (a) defeated the crop and (b) replicated the surviving edge ring over the
    padded band so those pixels were counted ~9x and their error amplified ~9x in
    abs_rel/rmse. It also meant this function scored a DIFFERENT pixel set than
    ``image_quality`` in the same results row. The crop is now final.

    ``target_size`` (H, W): if given, both images are resized to it before scoring.
    Default ``None`` => evaluate at the data's NATIVE resolution. (Old code hard-coded
    a resize to NYU's (480, 640) for BOTH datasets, upsampling Make3D ~2.7x.)
    """
    hb = border_crop_size // 2
    if hb > 0:
        gt = gt_image[:, :, hb:-hb, hb:-hb]
        pred = pred_image[:, :, hb:-hb, hb:-hb]
    else:
        gt, pred = gt_image, pred_image

    # Clamp to the valid image range, EXACTLY as image_quality does.
    #
    # pred_complex = haze + residual, and the residual head is unbounded, so predictions
    # routinely leave [0, 1]. Without this clamp, `image_quality` (which DOES clamp) and
    # `add_results_1` scored DIFFERENT VALUE RANGES in the same results row, and the ratio
    # metrics were penalised on pixels that cannot exist in any image you would actually
    # look at or save — every renderer clips them anyway. Scoring an out-of-gamut value is
    # measuring an artefact of the parameterisation, not of the reconstruction.
    gt = gt.clamp(0, 1)
    pred = pred.clamp(0, 1)

    if target_size is not None:
        import torch.nn.functional as F
        gt = F.interpolate(gt, target_size, mode='bilinear', align_corners=True)
        pred = F.interpolate(pred, target_size, mode='bilinear', align_corners=True)

    return compute_errors_nyu(pred, gt)
