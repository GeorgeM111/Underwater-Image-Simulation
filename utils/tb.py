"""TensorBoard logging helpers shared by all Technique_* train scripts.

Backend: ``tensorboardX``.  A single :class:`SummaryWriter` per run is created by
:func:`make_writer`; the train loop then calls :func:`log_scalars`,
:func:`log_weights` and :func:`log_images` once per epoch.

Run-directory layout::

    <runs_dir or --logdir>/T<technique>_<dataset>_<variant>/<YYYY-mm-dd_HH-MM-SS>/

so every launch is a distinct TensorBoard run and nothing is overwritten.
"""

import os
import datetime

import torch
from tensorboardX import SummaryWriter

try:
    import torchvision.utils as vutils
except Exception:  # torchvision should always be present, but never break training
    vutils = None


def make_writer(cfg, technique, dataset, variant, logdir=None, timestamp=None):
    """Create a SummaryWriter at ``<root>/T{technique}_{dataset}_{variant}/<timestamp>``.

    Args:
        cfg:       loaded config (uses ``cfg.runs_dir`` as the default root).
        technique: 1, 2, 3 or 4.
        dataset:   'NYU' or 'Make3D'.
        variant:   'base', 'var1' or 'var2'.
        logdir:    optional override for the runs root (e.g. from ``--logdir``).
        timestamp: optional fixed timestamp string (defaults to now()).
    """
    root = logdir if logdir else cfg.runs_dir
    tag = 'T%d_%s_%s' % (technique, dataset, variant)
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(root, tag, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print('[tensorboard] logging to %s' % run_dir)
    return SummaryWriter(run_dir)


def log_scalars(writer, epoch, scalars):
    """Log a ``{name: value}`` dict of scalars at the given epoch (None skipped)."""
    for name, v in scalars.items():
        if v is None:
            continue
        writer.add_scalar(name, float(v), epoch)


def log_weights(writer, epoch, weights):
    """Log learned loss-weight vectors.

    ``weights`` maps a name to a 1-D tensor (already mean-reduced over the batch);
    each component is logged as ``<name>/<i>``.
    """
    for name, w in weights.items():
        if w is None:
            continue
        w = w.detach().float().cpu().reshape(-1)
        for i in range(w.numel()):
            writer.add_scalar('%s/%d' % (name, i), float(w[i]), epoch)


def _norm_img(t):
    """Detach a tensor to a (B, 3, H, W) float grid in [0, 1] for logging."""
    t = t.detach().float().cpu()
    if t.dim() == 3:            # (B, H, W) -> (B, 1, H, W)
        t = t.unsqueeze(1)
    if t.dim() != 4:
        return None
    if t.size(1) == 1:          # grayscale (e.g. depth) -> 3 channels
        t = t.repeat(1, 3, 1, 1)
    mn = t.min()
    mx = t.max()
    if (mx - mn) > 1e-8:        # per-tensor min-max normalize for visibility
        t = (t - mn) / (mx - mn)
    return t.clamp(0, 1)


def log_images(writer, epoch, images, max_imgs=4):
    """Log a ``{name: (B,C,H,W) tensor}`` dict as image grids (None skipped)."""
    if vutils is None:
        return
    for name, t in images.items():
        if t is None:
            continue
        grid_src = _norm_img(t)
        if grid_src is None:
            continue
        grid = vutils.make_grid(grid_src[:max_imgs], nrow=max_imgs)
        writer.add_image(name, grid, epoch)
