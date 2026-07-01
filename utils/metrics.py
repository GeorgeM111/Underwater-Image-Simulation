"""Evaluation metrics: NYU-style depth/image error metrics."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def add_results_1(gt_image, pred_image, border_crop_size=16, use_224=False):
    predictions = []
    testSetDepths = []
    half_border_size = border_crop_size // 2

    gt_image_border_cut = gt_image[:, :, half_border_size:-half_border_size, half_border_size:-half_border_size]
    pred_image_border_cut = pred_image[:, :, half_border_size:-half_border_size, half_border_size:-half_border_size]

    del gt_image, pred_image

    replicate = nn.ReplicationPad2d(half_border_size)
    gt_image_border_cut = replicate(gt_image_border_cut)
    pred_image_border_cut = replicate(pred_image_border_cut)

    gt_image_border_cut = F.interpolate(gt_image_border_cut, (480, 640), mode='bilinear', align_corners=True)
    pred_image_border_cut = F.interpolate(pred_image_border_cut, (480, 640), mode='bilinear', align_corners=True)

    for j in range(len(gt_image_border_cut)):
        predictions.append(pred_image_border_cut[j])
        testSetDepths.append(gt_image_border_cut[j])

    predictions = torch.stack(predictions, axis=0)
    testSetDepths = torch.stack(testSetDepths, axis=0)

    del pred_image_border_cut, gt_image_border_cut
    abs_rel, rmse, log_10, a1, a2, a3 = compute_errors_nyu(predictions, testSetDepths)

    del predictions, testSetDepths

    return abs_rel, rmse, log_10, a1, a2, a3
