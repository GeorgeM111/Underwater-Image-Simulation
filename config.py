"""Centralised configuration loader.

Every script in the project reads its paths and hyper-parameters from
``config.yaml`` (at the repository root) through this module, instead of
relying on inline literals / hard-coded absolute paths.

Typical usage
-------------
    from config import CONFIG               # ready-to-use, loaded once
    batch_size = CONFIG.batch_size_nyu

or, to load an alternative file (e.g. from an argparse ``--config`` flag)::

    from config import load_config
    cfg = load_config(args.config)

The loader always starts from a complete set of built-in defaults (mirroring
``config.yaml``), then overlays whatever is found in the YAML file, so a missing
key never raises.
"""

import os

try:
    import yaml
except ImportError as exc:  # pragma: no cover - surfaced clearly at runtime
    raise ImportError(
        "PyYAML is required to read config.yaml. Install it with "
        "`pip install pyyaml` (see requirements.txt)."
    ) from exc


# Built-in defaults — kept in sync with config.yaml so the code runs even if the
# YAML file is missing a key (or missing entirely).
_DEFAULTS = {
    # Paths
    # ── NYU ──────────────────────────────────────────────────────────────────────
    "nyu_zip_path": "${PROJECT_DATA}/nyu_data.zip",
    "nyu_save_dir": "${PROJECT_DATA}/ground_truth/nyu/train",
    "nyu_gt_test_dir": "${PROJECT_DATA}/ground_truth/nyu/test",
    "nyu_gt_train_dir": "${PROJECT_DATA}/ground_truth/nyu/train",
    # ── Make3D ───────────────────────────────────────────────────────────────────
    "make3d_train_img_dir": "${PROJECT_DATA}/Make3D/Train400Img",
    "make3d_train_depth_dir": "${PROJECT_DATA}/Make3D/Train400Depth",
    "make3d_test_img_dir": "${PROJECT_DATA}/Make3D/Test134",
    "make3d_test_depth_dir": "${PROJECT_DATA}/Make3D/Test134Depth/Gridlaserdata",
    "make3d_save_dir": "${PROJECT_DATA}/ground_truth/make3d/train",
    "make3d_test_save_dir": "${PROJECT_DATA}/ground_truth/make3d/test",
    # ── KITTI (raw images + completed/annotated depth PNGs, depth_m = png/256) ──────
    "kitti_raw_dir": "${PROJECT_DATA}/kitti",
    # Completed-depth (data_depth_annotated) root, holding train/ and val/ splits:
    #   <dir>/<split>/<drive>/proj_depth/groundtruth/image_02/<frame>.png
    "kitti_completed_depth_dir": "${PROJECT_DATA}/kitti/completed_depth",
    "kitti_gt_train_dir": "${PROJECT_DATA}/ground_truth/kitti/train",
    "kitti_gt_test_dir": "${PROJECT_DATA}/ground_truth/kitti/test",
    "kitti_max_depth_m": 80.0,
    # Option B (as for Make3D): KITTI depths reach ~80 m, so scale the per-metre Jerlov
    # beta for the classical haze transmission ONLY (true metric depth kept), else the
    # haze GT collapses to flat airlight. 10/80 matches NYU's attenuation-vs-depth.
    "kitti_haze_beta_scale": 0.125,
    # Processing resolution [W, H] (generator + loader read these; keep in sync).
    # KITTI frames are ~1242x375 but the Velodyne sees NO sky: the top ~36% of rows
    # (0..133) have ZERO LiDAR returns, so any depth there would be invented. We
    # therefore BOTTOM-CENTER CROP to the LiDAR-covered region instead of resizing the
    # whole frame. /32-divisible. Half = decoder/GT resolution.
    "kitti_full_size": [1216, 224],
    "kitti_half_size": [608, 112],
    # Airlight brightness floor for KITTI GT (see nyu/make3d equivalents).
    "kitti_airlight_brightness_min": 0.5,
    # Informative-subset selection (filter_kitti_subset.py), mirroring NYU.
    "kitti_subset_size": 10000,
    "kitti_subset_w_entropy": 1.0 / 3.0,
    "kitti_subset_w_gradient": 1.0 / 3.0,
    "kitti_subset_w_depth": 1.0 / 3.0,
    # KITTI train/test selection (mirrors nyu_train_mode / nyu_test_mode):
    #   kitti_train_mode: "all" -> full train split | "subset" -> filtered indices
    #   kitti_test_mode:  "tail" -> held-out tail of train | "official" -> val split
    "kitti_train_mode": "subset",
    "kitti_subset_indices": None,
    "kitti_test_mode": "tail",
    # Pre-computed KITTI parameter matrices (beta / atmospheric light).
    "beta_mat_kitti_train": "${PROJECT_DATA}/parameters/Beta_Mat_KITTI_train.npy",
    "a_mat_kitti_train":    "${PROJECT_DATA}/parameters/A_Mat_KITTI_train.npy",
    "beta_mat_kitti_test":  "${PROJECT_DATA}/parameters/Beta_Mat_KITTI_test.npy",
    "a_mat_kitti_test":     "${PROJECT_DATA}/parameters/A_Mat_KITTI_test.npy",
    # ── Parameters ───────────────────────────────────────────────────────────────
    "beta_mat_nyu_train":    "${PROJECT_DATA}/parameters/Beta_Mat_NYU_train.npy",
    "a_mat_nyu_train":       "${PROJECT_DATA}/parameters/A_Mat_NYU_train.npy",
    "beta_mat_nyu_test":     "${PROJECT_DATA}/parameters/Beta_Mat_NYU_test.npy",
    "a_mat_nyu_test":        "${PROJECT_DATA}/parameters/A_Mat_NYU_test.npy",
    "beta_mat_make3d_train": "${PROJECT_DATA}/parameters/Beta_Mat_Make3D_train.npy",
    "a_mat_make3d_train":    "${PROJECT_DATA}/parameters/A_Mat_Make3D_train.npy",
    "beta_mat_make3d_test":  "${PROJECT_DATA}/parameters/Beta_Mat_Make3D_test.npy",
    "a_mat_make3d_test":     "${PROJECT_DATA}/parameters/A_Mat_Make3D_test.npy",
    # ── Outputs ──────────────────────────────────────────────────────────────────
    "checkpoint_dir": "${PROJECT_OUT}/checkpoints",
    "runs_dir": "${PROJECT_OUT}/runs",
    # Training
    "batch_size_nyu": 16,
    "batch_size_make3d": 5,
    "epochs": 50,
    "learning_rate": 1.0e-4,
    # GAN baselines (Pix2Pix / CycleGAN) use the canonical 2e-4 (Isola/Zhu et al.),
    # not the DenseDepth transfer-learning 1e-4 the Technique_* models use.
    "gan_learning_rate": 2.0e-4,
    "early_stopping_patience": 5,
    "train_split_ratio": 0.96,
    "num_workers": 8,
    "n_parallel_jobs": 6,
    # Acceleration (utils/accel.py): bf16 autocast on the model forwards (H200-friendly).
    # bf16 needs no GradScaler. amp=False -> exact fp32 path. amp_dtype: bfloat16|float16.
    "amp": True,
    "amp_dtype": "bfloat16",
    # Informative-subset selection (filter_nyu_subset.py). Composite score =
    # w_ent*entropy + w_grad*gradient + w_depth*depth_range (each min-max
    # normalised across the dataset). Weights need not sum to 1.
    "nyu_subset_size": 10000,
    "nyu_subset_w_entropy": 1.0 / 3.0,
    "nyu_subset_w_gradient": 1.0 / 3.0,
    "nyu_subset_w_depth": 1.0 / 3.0,
    # SSIM diversity pass: after ranking by score, greedily accept images whose
    # SSIM to every already-accepted image is < threshold (drop near-duplicates),
    # backfilling from lower-ranked images until subset_size is reached. SSIM is
    # computed on small ssim_thumb x ssim_thumb grayscale thumbnails.
    "nyu_subset_ssim_threshold": 0.6,
    "nyu_subset_ssim_thumb": 32,
    # NYU training set selection (data.nyu.get_train_loader):
    #   nyu_train_mode: "all"    -> full training split
    #                   "subset" -> only the filtered indices below
    #   nyu_subset_indices: path to the indices .npy; None -> auto
    #                       ({nyu_subset_size}_filtered_nyu.npy in the params dir)
    "nyu_train_mode": "subset",
    "nyu_subset_indices": None,
    # NYU test set selection (data.nyu.get_test_loader):
    #   "tail"     -> held-out tail of nyu2_train (paper's 96/4 protocol; DEFAULT)
    #   "official" -> official NYU-v2 654 test set (nyu2_test.csv + *_NYU_test)
    "nyu_test_mode": "tail",
    # Fraction of the NYU TRAINING POOL held out for validation (checkpoint selection
    # + early stopping), mirroring how data.make3d splits Train400. Independent of
    # train_split_ratio, which already carves off the held-out TEST tail.
    "nyu_val_ratio": 0.05,
    # Seed for the (otherwise non-reproducible) parameter-matrix generation so
    # regenerating betas/atmosphere is idempotent and stays in sync with the GT.
    "random_seed": 42,
    # Loss weights
    "lambda_l1": 0.1,
    "lambda_ssim": 0.1,
    "lambda_perc": 0.1,
    "lambda_grad": 1.0,
    # NOTE: "lambda_depth" was defined here and read by NOTHING (all 39 train scripts).
    # Removed rather than wired up: the paper's Eq.7 has no such term.

    # Learned-loss-weight regularisation (var1/var2). The paper's Eq.12/13/17/21 learn the
    # weights with no regulariser; that objective is LINEAR in w, so its minimiser is a
    # one-hot VERTEX and the losing loss terms lose their gradient permanently. See
    # models/weight_head.py and utils/loss_balance.py. Zeros reproduce the paper exactly.
    "weight_floor": 0.05,
    "lambda_weight_reg": 0.01,
    "loss_ema_momentum": 0.99,
    "weight_head_dropout": 0.1,

    # Optimisation stability
    "grad_clip_norm": 1.0,
    "weight_head_lr_mult": 0.1,
    "lr_min": 1.0e-6,
    "max_nonfinite_batches_per_epoch": 50,
    # Physics — classical model (Jerlov, per-metre).
    # Red is attenuated MOST in water; the original lists had red and blue
    # swapped (red ~0.02, blue ~0.4), inverting the colour physics.
    "beta_val_r": [0.357, 0.416, 0.528, 0.364, 0.423, 0.523],
    "beta_val_g": [0.0398, 0.0598, 0.0460, 0.0661],
    "beta_val_b": [0.0192, 0.0182, 0.0263, 0.0253],
    # Physics — complex model (ricardo). complex_beta is SEPARATE from the
    # classical Jerlov betas: the ricardo model runs on a 0-255 normalised depth
    # axis, so it needs the small ricardo coefficients, not per-metre Jerlov ones.
    "complex_beta": [0.020, 0.005, 0.010],   # fallback water type (superseded by jerlov_water_types)
    # Per-image water-type menu (classical Jerlov scale, [R, G, B] attenuation).
    # ONE type is sampled per image and drives the classical haze, the ricardo
    # transmission, AND the airlight — so images vary blue<->green.
    #   Oceanic types: blue least attenuated (B < G < R) -> blue water.
    #   Coastal types: green least attenuated (G < B < R) -> green water.
    # Red is always attenuated most.
    "jerlov_water_types": [
        [0.30, 0.05, 0.02],   # oceanic  I
        [0.32, 0.06, 0.03],   # oceanic  IA
        [0.34, 0.07, 0.04],   # oceanic  IB
        [0.38, 0.08, 0.05],   # oceanic  II
        [0.42, 0.09, 0.07],   # oceanic  III
        [0.40, 0.06, 0.08],   # coastal  1C
        [0.45, 0.07, 0.11],   # coastal  3C
        [0.50, 0.09, 0.15],   # coastal  5C
        [0.55, 0.11, 0.19],   # coastal  7C
        [0.62, 0.13, 0.24],   # coastal  9C
    ],
    # Maps a classical Jerlov beta (haze axis ~[0,10]) to the ricardo transmission
    # axis (~[0,255]): beta_ricardo = beta * complex_beta_scale. Keeps the ricardo
    # complex GT from collapsing while following the per-image water type.
    "complex_beta_scale": 0.04,
    # Veiling/airlight background distance (on the 0-255 depth axis). Airlight
    # colour = normalise(exp(-beta_ricardo * airlight_bg_depth)) — red-depleted.
    "airlight_bg_depth": 150.0,
    # Airlight (veiling-light) brightness ~ uniform(min, 1) per image. PER-DATASET so
    # regenerating one split can't shift the other's colours. 0.0 = original behaviour;
    # raising the floor avoids near-black/murky GT. REVERSIBLE: the RNG sequence is
    # floor-independent (uniform draws one random() regardless), so 0.0 + regenerate
    # reproduces the original matrices exactly.
    "nyu_airlight_brightness_min": 0.0,
    "make3d_airlight_brightness_min": 0.0,
    # Complex (ricardo) forward-model version:
    #   "ricardo" -> the original code that produced the paper's figures
    #   "v2"      -> corrected: untruncated Gaussian PSF (P1), energy-conserving
    #                (1-k) scatter weighting (P2), Eq.6 hygiene (P4). See utils/physics.
    # MUST default to "v2". load_config() silently tolerates a MISSING config.yaml, so a
    # node/container without it would fall back to these defaults -- and the legacy
    # "ricardo" path convolves an UNWEIGHTED source, making total radiance (1+k)*J: up to
    # 1.84x energy AMPLIFICATION that saturates against the 255 clip. Silent, plausible,
    # and wrong. Fail closed on the corrected model instead.
    "complex_model": "v2",
    # Scattering depth axis: "normalized" (per-image 0-255, the original) or "metric"
    # (true metres, using gamma_angular/alpha_metric + per-dataset focal length).
    "complex_depth_mode": "metric",
    # --- P3: physically-grounded scattering -------------------------------------
    # In pixels the PSF width is  sigma_px = f_px * gamma_angular_c * z_metres * clarity.
    # gamma_angular is ONE physical constant (rad/m) shared by every dataset; the pixel
    # blur differs only because the CAMERAS differ (focal length) and because a dataset
    # may be simulated under clearer water (clarity). Calibrated so NYU reproduces its
    # previous look (sigma=25.5px at z=10m, f=259.4).
    "gamma_angular": [0.0042, 0.0045, 0.0048],            # rad/m — near-flat; blue slightly WIDER (see config.yaml)
    # Straight-path attenuation k_c = exp(-alpha_metric_c * z_m * clarity). Calibrated
    # from the old 0-255-axis alpha: alpha_metric = alpha * 255 / nyu_max_depth_m.
    "alpha_metric": [0.45, 0.50, 0.55],                   # 1/m — scattered fraction; blue-heavy (Rayleigh)
    # Focal length in PIXELS at the GT (half) resolution of each dataset.
    "nyu_focal_px": 259.43,      # NYU-v2 official fx 518.8579 @640 wide -> /2
    "kitti_focal_px": 360.77,    # KITTI P_rect_02 fx ~721.54 @1242 wide; crop keeps f -> /2
    "make3d_focal_px": 246.60,   # Make3D ships NO intrinsics: assumed 50 deg HFOV @230 wide
    # Water clarity per dataset. Multiplies beta, alpha AND gamma together — "clearer
    # water" means fewer absorption *and* scattering events. Replaces the old
    # *_haze_beta_scale (which only scaled beta, an inconsistency).
    "nyu_water_clarity": 1.0,
    "make3d_water_clarity": 0.125,
    "kitti_water_clarity": 0.125,
    # ---- legacy (0-255-axis) coefficients, used only when complex_depth_mode=normalized
    "gamma": [0.1, 0.1, 0.043],
    "alpha": [0.032, 0.032, 0.012],
    "turbu_p": [0.15, 1.5],
    "turbu_c": [200.0, 235.0, 240.0],   # bright near-white marine snow
    # v2 Eq.6: per-channel particle probability pr_c and per-channel blur sigma_c.
    # (ricardo used one scalar each, and an off-by-1.5x density.)
    "turbu_pr": [0.010, 0.010, 0.010],  # 1% seed density (0.15 was a veil, not snow)
    "turbu_sigma": [0.9, 0.9, 0.9],     # ~1 px particle core
    "u": 0.90,                          # Eq.6 mixing (0.99 made particles invisible)
    "s": 2,
    "depth_add": 0,
    "depth_levels": 16,
    "kern_size": 10,
    # Depth
    "nyu_max_depth_m": 10.0,
    "make3d_max_depth_m": 80.0,
    # Option B (Make3D classical haze): treat Make3D as CLEARER water by scaling the
    # per-metre Jerlov beta for the transmission ONLY (t = exp(-(beta*scale)*z), TRUE
    # metric depth kept). Without this, 0-80 m depths drive t~0 everywhere and the
    # haze GT collapses to flat airlight. Default 10/80 puts Make3D's attenuation-vs-
    # depth on the same footing as NYU. Applied in BOTH the GT generator (data_2.py)
    # and the Make3D loader (data/make3d.py); keep them in sync. NYU is unaffected.
    "make3d_haze_beta_scale": 0.125,
    # Make3D processing resolution [W, H] (generator + loader read these). Raise
    # to reduce the pixelated/blocky look; keep generator & loader in sync.
    "make3d_full_size": [460, 345],
    "make3d_half_size": [230, 173],
    "depth_norm_max": 1000.0,
    # Model
    "pretrained_encoder": True,
}

# Default location of the YAML file: alongside this module (repo root).
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


class Config(dict):
    """A plain dict that also supports attribute access (``cfg.epochs``)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


# Portable roots. Paths in the config may use ${PROJECT_DATA} (datasets, ground truth,
# parameter matrices) and ${PROJECT_OUT} (checkpoints, TensorBoard runs). If the env vars
# are unset we fall back to the original absolute locations, so nothing changes locally.
# On another machine (e.g. Grid'5000) just export the two variables:
#     export PROJECT_DATA=$HOME/underwater/data
#     export PROJECT_OUT=$HOME/underwater/out
_DEFAULT_PROJECT_DATA = "/datas/sandbox/gmoussa"


def _expand(value):
    """Recursively expand ${VAR} / $VAR / ~ inside strings (lists and dicts too)."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path=None):
    """Load configuration, overlaying ``path`` (or ``config.yaml``) on defaults.

    After merging, every string value has ``${VAR}``/``~`` expanded, so the same YAML
    works on any machine once PROJECT_DATA / PROJECT_OUT are exported.

    Parameters
    ----------
    path : str, optional
        Path to a YAML config file. Defaults to ``config.yaml`` next to this
        module. A missing file is tolerated (defaults are used).
    """
    os.environ.setdefault("PROJECT_DATA", _DEFAULT_PROJECT_DATA)
    os.environ.setdefault("PROJECT_OUT", os.environ["PROJECT_DATA"])

    cfg = dict(_DEFAULTS)
    cfg_path = path or DEFAULT_CONFIG_PATH
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r") as handle:
            loaded = yaml.safe_load(handle) or {}
        cfg.update(loaded)
    cfg = Config({k: _expand(v) for k, v in cfg.items()})
    # Enable TF32 + cuDNN autotuning once, for every script that loads a config.
    # Safe & numerics-neutral here (kernel selection only); a no-op without CUDA.
    try:
        from utils.accel import setup_perf
        setup_perf(cfg)
    except Exception:
        pass  # never let a perf tweak break config loading
    return cfg


def assert_default_config(path=None):
    """Fail loudly if ``--config`` names anything other than the repo-root config.yaml.

    The GROUND-TRUTH GENERATORS read physics parameters through the ``CONFIG`` singleton,
    which is bound at import from ``DEFAULT_CONFIG_PATH``. ``load_config(other)`` returns a
    FRESH object and does not mutate that singleton, and joblib's loky workers are separate
    processes that re-import ``config`` and rebuild ``CONFIG`` from the default path anyway.

    So for the generation pipeline, ``--config other.yaml`` is honoured only PARTIALLY —
    which is strictly worse than being ignored. It would apply to the printed output
    directory and the job count while the physics silently came from the repo-root YAML,
    so an experiment config could print one path and overwrite the BASELINE GT in another.

    Rather than pretend, refuse. To run different physics, edit config.yaml (it is the
    single source of truth) or point PROJECT_DATA/PROJECT_OUT elsewhere.
    """
    if not path:
        return DEFAULT_CONFIG_PATH
    if os.path.abspath(path) != os.path.abspath(DEFAULT_CONFIG_PATH):
        raise SystemExit(
            "REFUSING to run: --config %s is not the repo-root config.yaml (%s).\n"
            "The GT generators read their physics from the CONFIG singleton (and joblib\n"
            "workers re-import it from the default path), so an alternate YAML would be only\n"
            "PARTIALLY honoured: the output directory would change but the physics would not.\n"
            "Edit config.yaml directly instead." % (path, DEFAULT_CONFIG_PATH))
    return DEFAULT_CONFIG_PATH


# Loaded once at import time for convenient `from config import CONFIG` access.
CONFIG = load_config()
