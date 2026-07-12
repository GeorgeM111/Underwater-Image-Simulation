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


def _grad_mag(x):
    """Mean absolute spatial gradient — a scale-free measure of how SHARP an image is."""
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dx + dy


def log_health(writer, epoch, out_depth=None, pred_complex=None, pred_haze=None,
               complex_gt=None, extra=None):
    """Log the early-warning scalars for the flat-airlight collapse.

    These two are the entire early-warning system and neither existed before:

        stats/out_depth_min            -- the depth head's floor. If it walks toward the
                                          bottom of its range the physics is heading for
                                          t = exp(-beta*z) -> 0, i.e. pure airlight.
        stats/out_depth_frac_at_floor  -- fraction of pixels pinned at the floor (y = 1).
                                          1.0 means the head is dead: z = max_depth everywhere,
                                          t -> 0, and the haze image is pure airlight.
        stats/out_depth_spread         -- max - min. EXACTLY 0 means the depth map is a
                                          CONSTANT, which is what renders as a white panel.
                                          Note it is naturally near 0 at init (the decoder's
                                          final features have little spatial variance) and must
                                          GROW; a small-but-growing spread is benign, a spread
                                          stuck at 0 is not.

    Plus the prediction ranges, so an exploding residual is visible immediately.

    GATE before launching a sweep: ``out_depth_spread`` must be growing, and the learned
    weights (``w_global/*``) must stay above the floor.
    """
    stats = {}
    if out_depth is not None:
        d = out_depth.detach().float()
        stats['stats/out_depth_min'] = d.min().item()
        stats['stats/out_depth_max'] = d.max().item()
        stats['stats/out_depth_mean'] = d.mean().item()
        # The depth head is y = y_min + softplus(raw): floored at y_min = 1, no ceiling. The
        # ONLY degenerate state left is being pinned at the FLOOR (z = max_depth everywhere ->
        # t -> 0 -> flat airlight). Watch this: it should stay small. 1.0 means the head has
        # died and the depth map is a constant (which renders as a WHITE panel).
        stats['stats/out_depth_frac_at_floor'] = (d <= 1.02).float().mean().item()
        # Spread. If this hits 0 the depth map is constant, whatever value it sits at.
        stats['stats/out_depth_spread'] = (d.max() - d.min()).item()
    for nm, t in (('pred_complex', pred_complex), ('pred_haze', pred_haze)):
        if t is None:
            continue
        x = t.detach().float()
        stats['stats/%s_min' % nm] = x.min().item()
        stats['stats/%s_max' % nm] = x.max().item()
        # Fraction of pixels outside the valid image range -> an exploding residual head.
        stats['stats/%s_frac_oob' % nm] = ((x < 0.0) | (x > 1.0)).float().mean().item()

    # --- IS THE RESIDUAL LEARNING THE SCATTERING BLUR? -------------------------------
    # The classical model (Eq.1) is POINTWISE: haze = J*t + A*(1-t). It cannot produce blur
    # at any depth, ever. The GT's blur comes from the forward-scattering PSF (Eq.3-4), so the
    # ONLY component that can reproduce it is the residual head. That is the paper's whole
    # thesis, and this ratio is how you check whether it is actually happening:
    #
    #   sharpness_ratio = |grad(pred_complex)| / |grad(complex_gt)|
    #       >> 1  -> the prediction is SHARPER than the GT: the residual has not learned the
    #                blur yet and the output is still essentially the pointwise haze image.
    #                (Expect this at epoch 1.)
    #       -> 1  -> the residual has learned the scattering. THIS IS THE THING TO WATCH.
    #       << 1  -> over-smoothed.
    #
    # Also logged: how much of the prediction the residual actually contributes. If
    # residual_rel_energy stays ~0, the residual branch is dead and the model has collapsed
    # back onto the plain haze model (i.e. Technique-1 degenerates into Eq.1 alone).
    if pred_complex is not None and complex_gt is not None:
        p = pred_complex.detach().float()
        g = complex_gt.detach().float()
        gg = _grad_mag(g)
        if gg > 1e-8:
            stats['stats/sharpness_ratio'] = (_grad_mag(p) / gg).item()
        if pred_haze is not None:
            h = pred_haze.detach().float()
            resid = p - h                      # == out_bb
            denom = p.abs().mean()
            if denom > 1e-8:
                stats['stats/residual_rel_energy'] = (resid.abs().mean() / denom).item()

    if extra:
        stats.update(extra)
    log_scalars(writer, epoch, stats)
