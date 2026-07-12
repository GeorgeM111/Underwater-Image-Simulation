"""General helper utilities: DepthNorm, AverageMeter, colorize, simple_save_images."""

import matplotlib
import matplotlib.cm

import cv2
import numpy as np

from config import CONFIG


def DepthNorm(depth, maxDepth=None):
    if maxDepth is None:
        maxDepth = CONFIG.depth_norm_max
    return maxDepth / depth


class AverageMeter(object):
    """Running mean.

    ``avg`` starts as NaN, NOT 0. A meter that never receives an update means
    "no measurement", and for the lower-is-better metrics (abs_rel/rmse/log10/mae)
    a 0 would read as a PERFECT score. That is exactly how a fully-diverged model
    used to print ``abs_rel 0.0000``. NaN is loud; 0 is a lie.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = float('nan')
        self.avg = float('nan')
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else float('nan')


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def simple_save_images(nn_noisy_image, name):
    nn_noisy_image = nn_noisy_image.cpu()[1, :, :, :]
    nn_noisy_image_numpy = nn_noisy_image.detach().numpy()
    norm_noisy_generated = cv2.normalize(nn_noisy_image_numpy, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F)

    norm_noisy_generated = norm_noisy_generated.astype(np.uint8)
    norm_noisy_generated = np.swapaxes(norm_noisy_generated, 0, 2)
    norm_noisy_generated = np.swapaxes(norm_noisy_generated, 0, 1)
    # Tensors are RGB; cv2.imwrite expects BGR, so convert to avoid a red/blue swap.
    norm_noisy_generated = cv2.cvtColor(norm_noisy_generated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, norm_noisy_generated)
