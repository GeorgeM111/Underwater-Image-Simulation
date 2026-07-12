# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Sample the per-image water type (Jerlov beta) + airlight for the NYU TRAIN split.

Writes Beta_Mat_NYU_train.npy / A_Mat_NYU_train.npy. The dataloader and the GT generator
index these POSITIONALLY, so they must be generated BEFORE the ground truth, and the GT must
be fully regenerated whenever they change. Seeded by config.random_seed, so a re-run
reproduces the same matrices exactly.
"""

import argparse

from data_2 import generate_and_save_atmosphere_light_beta
from config import assert_default_config


def main():
    parser = argparse.ArgumentParser(description='Generate NYU-train water-type + airlight matrices.')
    parser.add_argument('--config', default=None, help='path to config YAML (must be the repo-root config.yaml)')
    args = parser.parse_args()
    assert_default_config(args.config)
    generate_and_save_atmosphere_light_beta()


if __name__ == "__main__":
    main()
