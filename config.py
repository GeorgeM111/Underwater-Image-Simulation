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
    "complex_beta": [0.020, 0.005, 0.010],
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
