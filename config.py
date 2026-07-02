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
    "nyu_zip_path": "/datas/sandbox/gmoussa/nyu_data.zip",
    "nyu_save_dir": "/datas/sandbox/gmoussa/ground_truth/nyu/train",
    "nyu_gt_test_dir": "/datas/sandbox/gmoussa/ground_truth/nyu/test",
    "nyu_gt_train_dir": "/datas/sandbox/gmoussa/ground_truth/nyu/train",
    # ── Make3D ───────────────────────────────────────────────────────────────────
    "make3d_train_img_dir": "/datas/sandbox/gmoussa/Make3D/Train400Img",
    "make3d_train_depth_dir": "/datas/sandbox/gmoussa/Make3D/Train400Depth",
    "make3d_test_img_dir": "/datas/sandbox/gmoussa/Make3D/Test134",
    "make3d_test_depth_dir": "/datas/sandbox/gmoussa/Make3D/Test134Depth/Gridlaserdata",
    "make3d_save_dir": "/datas/sandbox/gmoussa/ground_truth/make3d/train",
    "make3d_test_save_dir": "/datas/sandbox/gmoussa/ground_truth/make3d/test",
    # ── KITTI (raw dataset; depth is PROJECTED from Velodyne LiDAR, no depth PNGs)
    "kitti_raw_dir": "/datas/sandbox/gmoussa/kitti",
    "kitti_gt_train_dir": "/datas/sandbox/gmoussa/ground_truth/kitti/train",
    "kitti_gt_test_dir": "/datas/sandbox/gmoussa/ground_truth/kitti/test",
    "kitti_max_depth_m": 80.0,
    "kitti_subset_size": 10000,
    "kitti_subset_w_entropy": 1.0 / 3.0,
    "kitti_subset_w_gradient": 1.0 / 3.0,
    "kitti_subset_w_depth": 1.0 / 3.0,
    # ── Parameters ───────────────────────────────────────────────────────────────
    "beta_mat_nyu_train":    "/datas/sandbox/gmoussa/parameters/Beta_Mat_NYU_train.npy",
    "a_mat_nyu_train":       "/datas/sandbox/gmoussa/parameters/A_Mat_NYU_train.npy",
    "beta_mat_nyu_test":     "/datas/sandbox/gmoussa/parameters/Beta_Mat_NYU_test.npy",
    "a_mat_nyu_test":        "/datas/sandbox/gmoussa/parameters/A_Mat_NYU_test.npy",
    "beta_mat_make3d_train": "/datas/sandbox/gmoussa/parameters/Beta_Mat_Make3D_train.npy",
    "a_mat_make3d_train":    "/datas/sandbox/gmoussa/parameters/A_Mat_Make3D_train.npy",
    "beta_mat_make3d_test":  "/datas/sandbox/gmoussa/parameters/Beta_Mat_Make3D_test.npy",
    "a_mat_make3d_test":     "/datas/sandbox/gmoussa/parameters/A_Mat_Make3D_test.npy",
    # ── Outputs ──────────────────────────────────────────────────────────────────
    "checkpoint_dir": "/datas/sandbox/gmoussa/checkpoints",
    "runs_dir": "/datas/sandbox/gmoussa/runs",
    # Training
    "batch_size_nyu": 10,
    "batch_size_make3d": 5,
    "epochs": 50,
    "learning_rate": 1.0e-4,
    "early_stopping_patience": 5,
    "train_split_ratio": 0.96,
    "num_workers": 4,
    "n_parallel_jobs": 20,
    # Informative-subset selection (filter_nyu_subset.py). Composite score =
    # w_ent*entropy + w_grad*gradient + w_depth*depth_range (each min-max
    # normalised across the dataset). Weights need not sum to 1.
    "nyu_subset_size": 10000,
    "nyu_subset_w_entropy": 1.0 / 3.0,
    "nyu_subset_w_gradient": 1.0 / 3.0,
    "nyu_subset_w_depth": 1.0 / 3.0,
    # NYU training set selection (data.nyu.get_train_loader):
    #   nyu_train_mode: "all"    -> full training split
    #                   "subset" -> only the filtered indices below
    #   nyu_subset_indices: path to the indices .npy; None -> auto
    #                       ({nyu_subset_size}_filtered_nyu.npy in the params dir)
    "nyu_train_mode": "all",
    "nyu_subset_indices": None,
    # NYU test set selection (data.nyu.get_test_loader):
    #   "tail"     -> held-out tail of nyu2_train (paper's 96/4 protocol; DEFAULT)
    #   "official" -> official NYU-v2 654 test set (nyu2_test.csv + *_NYU_test)
    "nyu_test_mode": "tail",
    # Seed for the (otherwise non-reproducible) parameter-matrix generation so
    # regenerating betas/atmosphere is idempotent and stays in sync with the GT.
    "random_seed": 42,
    # Loss weights
    "lambda_l1": 0.1,
    "lambda_ssim": 0.1,
    "lambda_perc": 0.1,
    "lambda_depth": 1.0,
    "lambda_grad": 1.0,
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
    "gamma": [0.1, 0.1, 0.043],
    "alpha": [0.032, 0.032, 0.012],
    "turbu_p": [0.15, 1.5],
    "turbu_c": [34.0, 200.0, 201.0],
    "u": 0.99,
    "s": 2,
    "depth_add": 0,
    "depth_levels": 16,
    "kern_size": 10,
    # Depth
    "nyu_max_depth_m": 10.0,
    "make3d_max_depth_m": 80.0,
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


def load_config(path=None):
    """Load configuration, overlaying ``path`` (or ``config.yaml``) on defaults.

    Parameters
    ----------
    path : str, optional
        Path to a YAML config file. Defaults to ``config.yaml`` next to this
        module. A missing file is tolerated (defaults are used).
    """
    cfg = dict(_DEFAULTS)
    cfg_path = path or DEFAULT_CONFIG_PATH
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r") as handle:
            loaded = yaml.safe_load(handle) or {}
        cfg.update(loaded)
    return Config(cfg)


# Loaded once at import time for convenient `from config import CONFIG` access.
CONFIG = load_config()
