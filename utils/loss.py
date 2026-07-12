"""Canonical loss functions: SSIM and VGG perceptual loss.

Single source of truth (previously duplicated as a per-technique ``loss.py``).
"""

from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vgg_models


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        # PLAIN mean, deliberately NOT nanmean. nanmean() masked out the NaN pixels
        # and averaged the survivors, so a prediction that was NaN over a quarter of
        # the frame still reported SSIM ~1.0 ("perfect"). It also made the old
        # sys.exit guard below unreachable for partial NaN. A NaN prediction must
        # produce a NaN loss/metric so the caller can see it and act.
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    # NOTE: no sys.exit here. A library must never kill the interpreter — it raised
    # SystemExit (a BaseException, so `except Exception` could not catch it) and could
    # take down a multi-hour training run from inside a loss. Non-finite values now
    # propagate; the training loop skips non-finite batches and the eval loop reports
    # them (see the n_bad_batches guard in each train.py / test.py).

    if full:
        return ret, cs

    return ret


def image_gradients(img):
    """First-order finite-difference gradients (dx, dy) of a (B, C, H, W) tensor."""
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    return dx, dy


def gradient_loss(pred, target):
    """Edge/gradient L1 loss — penalises differences in spatial gradients.

    This is the high-frequency term from DenseDepth (Alhashim & Wonka) that the
    original loss was MISSING. Without it, an L1 + SSIM depth objective tends to
    over-smooth, producing depth that looks like a blurred version of the GT.
    """
    px, py = image_gradients(pred)
    tx, ty = image_gradients(target)
    return (px - tx).abs().mean() + (py - ty).abs().mean()


def depth_loss(pred, target, val_range, lambda_l1=0.1, lambda_grad=1.0, lambda_ssim=1.0):
    """Canonical depth loss = lambda_l1 * L1 + lambda_grad * grad + lambda_ssim * SSIM.

    ``pred`` and ``target`` are in the same (reciprocal DepthNorm) domain. The
    gradient term restores sharp depth edges; the SSIM term must NOT be weighted
    far below the image-reconstruction loss or the depth branch is under-trained.
    """
    l1 = (pred - target).abs().mean()
    grad = gradient_loss(pred, target)
    s = torch.clamp((1 - ssim(pred, target, val_range=val_range)) * 0.5, 0, 1)
    return lambda_l1 * l1 + lambda_grad * grad + lambda_ssim * s


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 feature maps up to relu3_3 (layer index 15).

        L_perc(I, I_hat) = (1 / C_j H_j W_j) * || phi_j(I) - phi_j(I_hat) ||_F^2

    Notes:
        - Applied ONLY to RGB image reconstruction losses (L_p, L_t, L_g), not depth.
        - VGG16 weights are frozen throughout training.
        - Inputs are clamped to [0, 1] and then ImageNet-normalised.

    The ImageNet normalisation is REQUIRED and was missing. VGG16 is pretrained on
    ImageNet-normalised inputs (mean 0, range ~4.4); feeding it raw [0,1] (mean ~+0.45,
    range 1.0) puts relu3_3 off-distribution, so the term was not the perceptual
    distance it claims to be. The paper's "no input normalisation" rule is about the
    TRAINABLE DenseNet encoder, which can adapt — a FROZEN VGG cannot.

    The clamp to [0,1] is kept so VGG16 sees a valid image, but note its derivative is
    exactly 0 outside [0,1]: the perceptual term is inert on out-of-range pixels. Since
    pred_complex (= unbounded residual + haze) and out_direct routinely leave the range,
    the trainers add a small differentiable range penalty (see `range_penalty`).
    """

    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg_models.vgg16(pretrained=True)
        # Extract features up to relu3_3 (first 16 layers, indices 0-15)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:16])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        x = torch.clamp(x.float(), 0.0, 1.0)
        return (x - self.imagenet_mean) / self.imagenet_std

    def train(self, mode=True):
        # Keep the frozen VGG in eval mode no matter what the parent module does.
        super(VGGPerceptualLoss, self).train(mode)
        self.feature_extractor.eval()
        return self

    def forward(self, pred, target):
        phi_pred = self.feature_extractor(self._prep(pred))
        phi_target = self.feature_extractor(self._prep(target))

        _, C_j, H_j, W_j = phi_pred.shape

        diff = phi_pred - phi_target
        loss_per_sample = torch.sum(diff ** 2, dim=[1, 2, 3]) / (C_j * H_j * W_j)
        return loss_per_sample.mean()


def range_penalty(x, lo=0.0, hi=1.0):
    """Differentiable penalty for leaving [lo, hi].

    ``torch.clamp`` has derivative exactly 0 outside its range, so any loss that
    clamps its input (e.g. the VGG perceptual term) provides NO gradient on the very
    pixels that are out of range. This gives those pixels a gradient that pushes them
    back into the valid image range. Zero for an in-range tensor.
    """
    return (x - x.clamp(lo, hi)).abs().mean()
