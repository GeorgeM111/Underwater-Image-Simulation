# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from data_2 import generate_and_save_atmosphere_light_beta_kitti


def main():
    generate_and_save_atmosphere_light_beta_kitti()


if __name__ == "__main__":
    main()
