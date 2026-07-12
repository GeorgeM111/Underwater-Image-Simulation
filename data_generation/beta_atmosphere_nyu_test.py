# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Sample the per-image water type (Jerlov beta) + airlight for the NYU official TEST split.

Writes Beta_Mat_NYU_test.npy / A_Mat_NYU_test.npy. These are only used by
``nyu_test_mode: "official"`` (the 654-image nyu2_test set). The DEFAULT
``nyu_test_mode: "tail"`` reuses the TRAIN matrices, because the tail is a slice of
nyu2_train. See beta_atmosphere_nyu_train.py.
"""

import argparse

from data_2 import generate_and_save_atmosphere_light_beta_test
from config import assert_default_config


def main():
    parser = argparse.ArgumentParser(description='Generate NYU-test water-type + airlight matrices.')
    parser.add_argument('--config', default=None, help='path to config YAML (must be the repo-root config.yaml)')
    args = parser.parse_args()
    assert_default_config(args.config)
    generate_and_save_atmosphere_light_beta_test()


if __name__ == "__main__":
    main()
