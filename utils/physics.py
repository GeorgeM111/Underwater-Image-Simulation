"""Canonical physics / image-formation module.

Single source of truth for everything previously split across the per-technique
``train_*.py`` (the ``compute_haze_image`` / ``compute_complex_image`` helpers),
``ricardo_code_1.py`` and ``ricardo_underwater_image.py``. All physics parameters
come from ``config.yaml`` via ``config.CONFIG``.

Public API:
    compute_haze_image(...)          – classical attenuation+airlight haze (tensors)
    compute_complex_image(...)       – haze image + learned residual (tensors)
    compute_complex_noise(...)       – full ricardo underwater degradation (numpy)
    processImg(...), scatterPsdOp(...), turbidDeg(...), outTotal(...)  – ricardo internals
"""

import math
from io import BytesIO

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from sklearn.utils import shuffle

from config import CONFIG
from utils.depth_range import DEPTH_NORM_MIN, DEPTH_CLIP_FRAC

# The reciprocal-depth cancellation z = max_depth_m / out_depth is only valid because
# depth_norm_max equals the 1000 that utils/transforms.py multiplies the 8-bit depth by.
# If someone changes depth_norm_max alone, every recovered z silently rescales.
assert float(CONFIG.depth_norm_max) == 1000.0, (
    "depth_norm_max must be 1000.0 to match the depth transform's *1000 scaling; "
    "got %r. Changing it alone would silently rescale every recovered depth in metres."
    % (CONFIG.depth_norm_max,))


# =============================================================================
# Tensor image-formation helpers (used inside the training loops)
# =============================================================================

def depthnorm_to_metres(output_depth, max_depth_m, depth_norm_max=None, eps=None):
    """Convert a network depth prediction back to METRES.

    The depth head regresses ``DepthNorm(d) = depth_norm_max / d_target`` (a
    reciprocal-domain value in [1, 25]), NOT metres. The GT was generated with
    ``t = exp(-beta * z)`` where ``z`` is in metres, so the prediction must be mapped
    back before it meets the physics:

        z_metres = max_depth_m / output_depth

    (This cancellation holds only because ``depth_norm_max`` equals the 1000 hard-coded
    in the depth transforms; the assert below pins that invariant. Changing
    ``depth_norm_max`` alone would silently rescale every recovered z.)

    The floor is now DEPTH_NORM_MIN (=1.0), not 1e-6. The depth head is bounded to
    [1, 25] by a scaled sigmoid (models/decoder_1ch.py), so this clamp should never
    bind — it is a guard, not a mechanism. The old 1e-6 floor was an ABSORBING STATE:
    ``torch.clamp``'s backward is exactly 0 below the floor, so once the (then unbounded)
    head emitted a non-positive value it could never recover, z jumped to 1e7 m, t
    collapsed to 0, and the haze image degenerated to pure airlight — the flat gray/white
    frames. Keeping a floor at the *physical* minimum means even a clamp event leaves the
    prediction inside the valid range instead of at a degenerate one.
    """
    if depth_norm_max is None:
        depth_norm_max = CONFIG.depth_norm_max
    if eps is None:
        eps = DEPTH_NORM_MIN
    return max_depth_m / torch.clamp(output_depth, min=eps)


def compute_haze_image(output_depth, beta_val, a_mat, unit_mat, image_half,
                       max_depth_m=None, depth_in_metres=False):
    """Classical attenuation + airlight haze model:  I = J*t + A*(1-t), t = exp(-beta*z).

    ``output_depth`` is the raw network prediction (reciprocal DepthNorm domain)
    unless ``depth_in_metres=True``. ``beta_val`` is the per-metre Jerlov beta
    map, so depth MUST be in metres for the transmission to match GT generation;
    pass ``max_depth_m`` for the dataset (NYU 10, Make3D 80).
    """
    if depth_in_metres:
        depth_metres = output_depth
    else:
        if max_depth_m is None:
            max_depth_m = CONFIG.nyu_max_depth_m
        depth_metres = depthnorm_to_metres(output_depth, max_depth_m)

    depth_metres_3d = depth_metres.repeat(1, 3, 1, 1)  # tile depth to 3 channels

    tx1 = torch.exp(-(beta_val * depth_metres_3d))
    second_term = a_mat * (unit_mat - tx1)
    haze_image = (image_half * tx1) + second_term
    return haze_image


def compute_complex_image(output_depth, output_black_box, beta_val, a_mat, unit_mat, image_half,
                          max_depth_m=None, depth_in_metres=False):
    """Complex (ricardo) model = haze image + learned residual ("black box")."""
    output_black_box_3d = output_black_box  # already 3-channel

    haze_image = compute_haze_image(output_depth, beta_val, a_mat, unit_mat, image_half,
                                    max_depth_m=max_depth_m, depth_in_metres=depth_in_metres)

    pred_complex_image = output_black_box_3d + haze_image
    return pred_complex_image


# =============================================================================
# Ricardo underwater image-formation model (numpy / ground-truth generation)
# =============================================================================

# --- Parameters -----------------------------------------------------------------
# READ AT CALL TIME, never bound at import. These used to be module-level constants
# evaluated on `import utils.physics`, which meant `--config myexp.yaml` did NOTHING to
# the physics: the names were already frozen from the repo-root config.yaml singleton and
# load_config() returns a fresh object without mutating it. Worse, the GT scripts DO
# honour --config for the output directory, so an experiment YAML would print one path
# and write baseline-physics GT into it.
#
# `cfg` is threaded explicitly (and defaults to the singleton) so it also survives joblib's
# loky backend, which spawns fresh processes that re-import config and would otherwise
# each rebuild CONFIG from the DEFAULT path — leaving the parent right and every worker
# wrong. A cfg object pickles; a mutated module global does not travel.
def _phys(cfg=None):
    """Snapshot of the physics parameters for one call."""
    c = cfg if cfg is not None else CONFIG
    return {
        'gamma': list(c.gamma),                  # legacy 0-255-axis kernel width (RGB)
        'alpha': list(c.alpha),                  # legacy 0-255-axis direct-path attenuation
        'complex_beta': list(c.complex_beta),    # legacy ricardo transmission coefficient
        'turbu_p': list(c.turbu_p),              # legacy [noise amount, gaussian std]
        'turbu_c': list(c.turbu_c),              # particle colour per channel (LIVE on v2)
        'turbu_pr': list(c.turbu_pr),            # per-channel particle probability (LIVE)
        'turbu_sigma': list(c.turbu_sigma),      # per-channel particle blur sigma (LIVE)
        'u': float(c.u),                         # Eq.6 mixing weight (LIVE)
        's': float(c.s),                         # legacy particle-image multiplier
        'depth_add': float(c.depth_add),         # legacy additive minimum depth
        'depth_levels': int(c.depth_levels),     # depth quantisation bins (LIVE)
        'kern_size': int(c.kern_size),           # legacy fixed kernel half-size
        'complex_model': str(getattr(c, 'complex_model', 'v2')).lower(),
        'complex_beta_scale': float(getattr(c, 'complex_beta_scale', 1.0)),
    }


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# Compute scattering diffusion for all pixels (Eq. 2, 3) using depth levels
def scatterPsdOp(rgb, depht, gamma, alpha, win_size, d_levels):
    dim = rgb.shape
    psd = np.zeros(dim)

    d_min = torch.min(depht)
    d_max = torch.max(depht)

    img_sum = torch.zeros(dim[0], dim[1], 3)

    for n in range(d_levels):
        img_l = torch.zeros(dim[0], dim[1])
        ker = torch.zeros(2 * win_size + 1, 2 * win_size + 1)

        di_min = ((d_max - d_min) / d_levels) * n + d_min
        di_max = ((d_max - d_min) / d_levels) * (n + 1) + d_min
        di_m = (di_min + di_max) / 2

        for c in range(3):

            # compute kernel
            for i in range(2 * win_size + 1):
                for j in range(2 * win_size + 1):
                    pos = [i - win_size, j - win_size]
                    v0 = -(np.power(pos[0], 2) + np.power(pos[1], 2))
                    v1 = np.exp(v0 / (2 * np.power(gamma[c] * di_m, 2)))
                    v2 = 1 / ((2 * np.pi) * np.power(gamma[c] * di_m, 1))
                    v3 = v1 * v2
                    ker[i, j] = v3

            ker = ker / torch.sum(ker)

            # extract pixels in depht range
            img_t = rgb[:, :, c]
            dht = (depht >= di_min).data.numpy().astype(np.float32)
            dht = np.multiply(dht, (depht < di_max).data.numpy().astype(np.float32))
            img_l = torch.from_numpy(np.multiply(dht, img_t.data.numpy()))

            # Convolve
            va = Variable(ker.view(1, 1, 2 * win_size + 1, 2 * win_size + 1))
            vb = Variable(img_l.view(1, 1, dim[0], dim[1]))
            img_c = F.conv2d(vb, va, padding=win_size)
            img_c = img_c[0, 0, :, :]

            img_sum[:, :, c] += img_c
            img_sum[img_sum > 255] = 255  # for saturated values

    return img_sum


# Compute degradation caused by colored particle turbidity
def turbidDeg(rgb, depht, turbu_p, turbu_c):
    dim = rgb.shape
    s_vs_p = 0.5
    out = np.zeros(rgb.shape)

    # Salt mode
    num_salt = np.ceil(turbu_p[0] * rgb.size * s_vs_p * 3)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in rgb.shape]

    for i in range(len(coords[0])):
        out[coords[0][i], coords[1][i], coords[2][i]] = 1.0

    out[:, :, 0] = out[:, :, 0] * turbu_c[0]
    out[:, :, 1] = out[:, :, 1] * turbu_c[1]
    out[:, :, 2] = out[:, :, 2] * turbu_c[2]

    out = cv2.GaussianBlur(out, (9, 9), turbu_p[1])

    return out


# Compute total arriving intensity for a pixel:
#   I_total_c = ( J_sct_c + k_c * J_c ) * t_c + (1 - t_c) * A_c
# where the straight-path radiance is k_c * J_c, k_c = exp(-alpha * z), and the
# transmission is t_c = exp(-beta * z). The straight (unscattered) radiance MUST
# decay with depth, so it is k_c * J_c (= exp(-alpha*z)*J), matching the original
# ricardo code that produced the paper figures. A previous refactor used
# (1 - k_c) * J_c, which GROWS with depth (distant direct radiance unattenuated)
# — physically backwards — so it is restored to k_c * J_c here.
def outTotal(rgb, depht, I_out, gamma, alpha, beta, A_light):
    """Eq. 5, VECTORISED (P0).

    Mathematically identical to the original triple Python loop over H*W*3 (verified
    bit-for-bit), but ~2 orders of magnitude faster — the loop dominated GT-generation
    cost. ``gamma`` is unused (kept for signature compatibility).
    """
    z = np.asarray(depht, dtype=np.float32)[:, :, None]
    a = np.asarray(alpha, dtype=np.float32)[None, None, :]
    b = np.asarray(beta, dtype=np.float32)[None, None, :]
    A = np.asarray(A_light, dtype=np.float32)[None, None, :]
    rgb = np.asarray(rgb, dtype=np.float32)
    k = np.exp(-a * z)                    # straight-path fraction  k_c = exp(-alpha*z)
    t = np.exp(-b * z)                    # transmission            t_c = exp(-beta*z)
    return (np.asarray(I_out, dtype=np.float32) + k * rgb) * t + (1.0 - t) * A


# =============================================================================
# v2 — corrected complex model (P1, P2, P4).  Enable with config complex_model: "v2".
# =============================================================================

def _scatter_v2(rgb, depth, gamma_, alpha_, d_levels):
    """Forward-scattered radiance (Eq. 3-4), corrected.

    P1 — Untruncated Gaussian PSF. The original built the kernel in a FIXED 21x21 window
         (kern_size=10) while sigma = gamma_c * z grows with depth (sigma=25.5 px at
         z=255). Measured centre/corner weight ratio fell to 1.17 => the "Gaussian PSF"
         degenerated into a BOX FILTER beyond mid-depth. Here cv2 sizes the kernel from
         sigma (~4 sigma) and convolves separably: correct *and* faster.

    P2 — Energy-conserving source weighting. Eq. 4's integrand must carry the SCATTERED
         fraction of the source radiance, (1 - k_c(x')) with k_c = exp(-alpha_c z), so
         that scattered + direct = (1-k)J + kJ = J. The original convolved an UNWEIGHTED
         J_c, making total radiance (1+k)J — energy *amplification*, and a principal
         cause of the far-field wash-out. (The paper writes k*J here and (1-k)*J for the
         direct term, i.e. the two are swapped relative to its own definition of k as the
         straight-path probability.)
    """
    H, W, _ = rgb.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    d_min, d_max = float(depth.min()), float(depth.max())
    if not (np.isfinite(d_min) and np.isfinite(d_max)):
        return out
    if d_max <= d_min:
        # Degenerate (constant) depth: one level covering the whole image. The old code
        # produced an all-zero scatter here because its half-open bins matched nothing.
        edges = np.array([d_min, d_min + 1e-6], dtype=np.float64)
        d_levels = 1
    else:
        edges = np.linspace(d_min, d_max, d_levels + 1)
    for n in range(d_levels):
        lo, hi = float(edges[n]), float(edges[n + 1])
        m = (depth >= lo) & (depth < hi) if n < d_levels - 1 else (depth >= lo)
        if not m.any():
            continue
        di_m = 0.5 * (lo + hi)
        for c in range(3):
            src = (rgb[:, :, c] * m).astype(np.float32)
            src *= (1.0 - np.exp(-float(alpha_[c]) * depth)).astype(np.float32)   # P2
            sigma = float(gamma_[c]) * di_m
            if sigma <= 1e-3:
                out[:, :, c] += src
            else:
                out[:, :, c] += cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                                 borderType=cv2.BORDER_REPLICATE)   # P1
    return out


def _turbid_v2(shape, pr, sp_col, sigma_c):
    """Eq. 6 particle field SP_c{sp_col, pr_c, sigma_c}, corrected (P4).

    - Density is EXACTLY pr_c. The original used ceil(pr * size * 0.5 * 3) with random
      coords drawn with replacement -> ~20.1% of entries salted instead of pr_c=15%.
    - Per-channel blur sigma_c (the original used one scalar sigma and a fixed 9x9 kernel).
    - "Pepper" is vacuous on a zero-valued particle field (absence of a particle IS the
      background), so we implement the paper's own wording: with probability pr_c a
      particle of colour sp_col[c] is added to channel c.
    """
    H, W = int(shape[0]), int(shape[1])
    out = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        mask = (np.random.rand(H, W) < float(pr[c])).astype(np.float32)
        ch = mask * float(sp_col[c])
        sg = float(sigma_c[c])
        out[:, :, c] = cv2.GaussianBlur(ch, (0, 0), sigmaX=sg, sigmaY=sg) if sg > 1e-3 else ch
    return out


def processImg(imgD_Norm, imgRGB, beta, A_light,
               gamma_eff=None, alpha_eff=None, beta_eff=None, cfg=None):
    """Complex forward model.

    ``gamma_eff`` / ``alpha_eff`` / ``beta_eff`` are the EFFECTIVE per-channel
    coefficients on whatever depth axis ``imgD_Norm`` uses. When omitted (the
    "normalized" 0-255 path) they fall back to the legacy config values and
    ``beta * complex_beta_scale``.
    """
    P = _phys(cfg)

    if beta_eff is None:
        # Legacy 0-255 axis: rescale the per-metre Jerlov beta onto it.
        beta_eff = (np.asarray(beta, dtype=np.float64) * P['complex_beta_scale']).tolist()
    if gamma_eff is None:
        gamma_eff = P['gamma']
    if alpha_eff is None:
        alpha_eff = P['alpha']

    # depth_add is a LEGACY 0-255-axis knob. On the metric axis it would add whole METRES
    # of water to every pixel (25.5x more potent than on the axis it was designed for), so
    # it is confined to the normalized branch below.

    rgb_f = imgRGB.astype(np.float32)

    if P['complex_model'] == 'v2':
        # ---- corrected model (P1 untruncated PSF, P2 energy-conserving scatter) ----
        d_f = imgD_Norm.astype(np.float32)
        I_out = _scatter_v2(rgb_f, d_f, gamma_eff, alpha_eff, P['depth_levels'])
        I_total = outTotal(rgb_f, d_f, I_out, gamma_eff, alpha_eff, beta_eff, A_light)
        # P4: Eq. 6 exactly -> I = u*I_sct + (1-u)*SP   (no stray `s` multiplier)
        sp = _turbid_v2(rgb_f.shape, P['turbu_pr'], P['turbu_c'], P['turbu_sigma'])
        out = I_total * P['u'] + sp * (1.0 - P['u'])
    else:
        # ---- original ricardo path (same maths as before; P0 vectorised outTotal) ----
        d_f = (imgD_Norm + P['depth_add']).astype(np.float32)
        I_out = scatterPsdOp(torch.from_numpy(rgb_f), torch.from_numpy(d_f),
                             gamma_eff, alpha_eff, P['kern_size'], P['depth_levels']).data.numpy()
        I_total = outTotal(rgb_f, d_f, I_out, gamma_eff, alpha_eff, beta_eff, A_light)
        out2 = turbidDeg(rgb_f, d_f, P['turbu_p'], P['turbu_c'])
        out = I_total * P['u'] + out2 * P['s'] * (1 - P['u'])

    # Fail LOUD on non-finite radiance. The old code did nan_to_num(nan=255.0), which
    # silently wrote PURE WHITE pixels into the training targets — a physics bug became
    # plausible-looking training data. If this ever fires, the depth map or the
    # coefficients are broken and the GT must not be written.
    if not np.isfinite(out).all():
        n_bad = int((~np.isfinite(out)).sum())
        raise ValueError(
            "Non-finite radiance in the complex forward model (%d / %d elements). "
            "Check the depth map for NaN/Inf and the beta/alpha/gamma coefficients."
            % (n_bad, out.size))

    # float32 in [0, 255], NOT uint8. The uint8 cast TRUNCATED (floored), biasing the
    # complex GT by -0.5 gray levels relative to the float64 classical-haze GT — in
    # Techniques 2/3 the residual head then had to learn that constant offset as if it
    # were physics. Quantisation to uint8 (if wanted) belongs at the SAVE step, where it
    # is done with np.rint (round, not floor).
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def getRndPar():
    r = np.random.randint(1, 20, size=1)
    rb = np.random.randint(1, 300, size=1)
    val = (1.0 / 2500000.0) * 16  # r[0]*r[0]*r[0]*r[0]
    val = val * (1.0 + (rb[0] / 1000.0))
    return val


def loadZipToMem(zip_file):
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n')
                       if len(row) > 0))

    nyu2_train = shuffle(nyu2_train, random_state=0)

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train


def to_transform(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img.float().div(255)

    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, dtype=np.int32))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, dtype=np.int16))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)

    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def compute_complex_noise(input_image, input_depth, beta_mat, A_light,
                          max_depth_m=None, focal_px=None, clarity=1.0, cfg=None, seed=None):
    """Full underwater degradation of an RGB image [0,1] given its depth map [0,1].

    ``seed`` makes the Eq.6 particle field reproducible. ``_turbid_v2`` draws from the
    global numpy RNG and NOTHING used to seed it, so the complex GT was not byte-
    reproducible and a partially-regenerated directory silently mixed realisations. Pass
    a per-image seed (e.g. random_seed + idx).

    Returns float32 in [0, 255].

    ``complex_depth_mode``:
      "metric" (P3, default) — depth stays in METRES (``input_depth * max_depth_m``) and

          sigma_px(z) = focal_px * gamma_angular_c * clarity * z_metres
          k_c(z)      = exp(-alpha_metric_c * clarity * z_metres)
          t_c(z)      = exp(-beta_c        * clarity * z_metres)     (beta = Jerlov, per-metre)

        ``gamma_angular`` is ONE physical constant across datasets; the pixel blur differs
        only because the cameras differ (focal_px) and because a dataset may be simulated
        under clearer water (clarity). This also makes the complex transmission use the
        SAME per-metre Jerlov beta as the classical haze — no separate complex_beta_scale.

      "normalized" (legacy) — depth is rescaled to a PER-IMAGE 0-255 axis, so the PSF
        width depends on each image's own depth range (a bathroom and a street get the
        same blur at their respective far planes) and gamma/alpha/beta are unitless.
    """
    _cfg = cfg if cfg is not None else CONFIG
    if seed is not None:
        np.random.seed(int(seed) % (2 ** 32))

    input_image = input_image * 255.0                        # image -> 0-255
    mode = str(getattr(_cfg, 'complex_depth_mode', 'normalized')).lower()

    if mode == 'metric':
        if max_depth_m is None or focal_px is None:
            raise ValueError("complex_depth_mode='metric' requires max_depth_m and focal_px.")
        clarity = float(clarity)
        imgD_Norm = np.asarray(input_depth, dtype=np.float64) * float(max_depth_m)
        if not np.isfinite(imgD_Norm).all():
            raise ValueError("Depth map contains NaN/Inf — refusing to generate GT from it.")
        # Clip the depth to the SAME window the training target uses ([0.4, 10] m for NYU,
        # i.e. [4%, 100%] of max_depth_m). Without this, missing-depth pixels (PNG value 0)
        # get z = 0 => t = exp(0) = 1 => the UN-DEGRADED original pixel is written into the
        # underwater GT, and it also picks up inbound scatter from its hazed neighbours —
        # rendering as clipped-to-white blobs wherever the Kinect had no return. This also
        # guarantees the GT physics axis and the network's target axis are the same window.
        imgD_Norm = np.clip(imgD_Norm,
                            DEPTH_CLIP_FRAC * float(max_depth_m),
                            float(max_depth_m))
        # Effective coefficients on the METRIC depth axis.
        gamma_eff = (np.asarray(_cfg.gamma_angular, np.float64) * float(focal_px) * clarity).tolist()
        alpha_eff = (np.asarray(_cfg.alpha_metric, np.float64) * clarity).tolist()
        beta_eff = (np.asarray(beta_mat, np.float64) * clarity).tolist()
        A_light = (np.array(A_light) * 255).tolist()
        return processImg(imgD_Norm, input_image, beta_mat, A_light,
                          gamma_eff=gamma_eff, alpha_eff=alpha_eff, beta_eff=beta_eff, cfg=_cfg)

    input_depth = input_depth * 255.0                        # depth -> 0-255
    imgD_Norm = input_depth + 1
    imgD_Norm = (imgD_Norm / np.min([np.max(imgD_Norm), 255])) * 255  # >0 and <=255
    A_light = (np.array(A_light) * 255).tolist()
    return processImg(imgD_Norm, input_image, beta_mat, A_light, cfg=_cfg)
