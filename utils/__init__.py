"""Shared utility package (repo root).

Submodules:
    utils.depth_range  – THE depth axis (d, y=DepthNorm, clips). Single source of truth.
    utils.helpers      – DepthNorm, AverageMeter, colorize, simple_save_images
    utils.physics      – compute_haze_image/compute_complex_image + ricardo GT model
    utils.loss         – ssim, gradient_loss, VGGPerceptualLoss, range_penalty
    utils.loss_balance – EMANormalizer, weight_log_barrier (var1/var2 weight regularisation)
    utils.metrics      – compute_errors_nyu, add_results_1, image_quality, psnr
    utils.transforms   – NYU + Make3D augmentation / tensor transforms
    utils.tb           – TensorBoard writer + log_scalars/log_weights/log_images/log_health

Common helpers are re-exported here for convenience::

    from utils import AverageMeter, DepthNorm, colorize, simple_save_images
"""

from utils.helpers import AverageMeter, DepthNorm, colorize, simple_save_images

__all__ = ["AverageMeter", "DepthNorm", "colorize", "simple_save_images"]
