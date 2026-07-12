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


def _norm_img(t, normalize=True):
    """Detach a tensor to a (B, 3, H, W) float grid in [0, 1] for logging.

    ``normalize=False`` is MANDATORY for RGB tensors that are already in [0, 1]
    (predictions and GT). Min-max stretching an RGB prediction is actively harmful: a
    collapsed model that outputs a flat airlight frame, or one whose values have run far
    out of [0, 1], gets rescaled into a plausible-looking image. Un-normalised, those
    pixels visibly saturate — which is how you SEE the failure. Depth maps (single
    channel, arbitrary range) still get normalised for visibility.
    """
    t = t.detach().float().cpu()
    if t.dim() == 3:            # (B, H, W) -> (B, 1, H, W)
        t = t.unsqueeze(1)
    if t.dim() != 4:
        return None
    if t.size(1) == 1:          # grayscale (e.g. depth) -> 3 channels
        t = t.repeat(1, 3, 1, 1)
    if normalize:
        mn = t.min()
        mx = t.max()
        if (mx - mn) > 1e-8:
            t = (t - mn) / (mx - mn)
    return t.clamp(0, 1)


# Tensors that are already RGB in [0, 1] and must NOT be min-max stretched.
_RGB_KEYS = ('haze/', 'complex/', 'direct/', 'input/', 'residual/')


def log_images(writer, epoch, images, max_imgs=4):
    """Log a ``{name: (B,C,H,W) tensor}`` dict as image grids (None skipped).

    RGB panels are logged WITHOUT min-max normalisation so out-of-range or degenerate
    predictions saturate visibly instead of being stretched into something plausible.
    """
    if vutils is None:
        return
    for name, t in images.items():
        if t is None:
            continue
        normalize = not name.startswith(_RGB_KEYS)
        grid_src = _norm_img(t, normalize=normalize)
        if grid_src is None:
            continue
        grid = vutils.make_grid(grid_src[:max_imgs], nrow=max_imgs)
        writer.add_image(name, grid, epoch)


def log_health(writer, epoch, out_depth=None, pred_complex=None, pred_haze=None, extra=None):
    """Log the early-warning scalars for the flat-airlight collapse.

    These two are the entire early-warning system and neither existed before:

        stats/out_depth_min            -- the depth head's floor. If it walks toward the
                                          bottom of its range the physics is heading for
                                          t = exp(-beta*z) -> 0, i.e. pure airlight.
        stats/out_depth_frac_at_bound  -- fraction of pixels pinned at either bound. A
                                          saturating sigmoid means the head has given up.

    Plus the prediction ranges, so an exploding residual is visible immediately.

    GATE before launching a sweep: ``out_depth_frac_at_bound`` must stay near 0 and the
    learned weights (``w_global/*``) must stay above the floor.
    """
    stats = {}
    if out_depth is not None:
        d = out_depth.detach().float()
        stats['stats/out_depth_min'] = d.min().item()
        stats['stats/out_depth_max'] = d.max().item()
        stats['stats/out_depth_mean'] = d.mean().item()
        # Depth head is bounded to [1, 25] by a scaled sigmoid; pinning at a bound means
        # the pre-activation has saturated.
        lo, hi = 1.0, 25.0
        eps = 0.02 * (hi - lo)
        at_bound = ((d <= lo + eps) | (d >= hi - eps)).float().mean().item()
        stats['stats/out_depth_frac_at_bound'] = at_bound
    for nm, t in (('pred_complex', pred_complex), ('pred_haze', pred_haze)):
        if t is None:
            continue
        x = t.detach().float()
        stats['stats/%s_min' % nm] = x.min().item()
        stats['stats/%s_max' % nm] = x.max().item()
        # Fraction of pixels outside the valid image range -> an exploding residual head.
        stats['stats/%s_frac_oob' % nm] = ((x < 0.0) | (x > 1.0)).float().mean().item()
    if extra:
        stats.update(extra)
    log_scalars(writer, epoch, stats)
