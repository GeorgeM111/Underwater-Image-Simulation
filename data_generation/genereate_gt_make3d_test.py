# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import sys
import glob
import time
import joblib
from joblib import Parallel, delayed

from config import CONFIG
from data_2 import generate_and_save_ricardo_image_make_3D_Test

START_IDX = 0
# Derive the end index from the actual number of test images so we never index
# past the dataset (there are 134 test images; the old hard-coded 135 ran off
# the end -> IndexError, leaving the GT only partially generated).
END_IDX = len(glob.glob(_os.path.join(CONFIG.make3d_test_img_dir, '*.jpg')))
CHUNK_SIZE = 40


def main():
    chunks = [
        (st, min(st + CHUNK_SIZE, END_IDX))
        for st in range(START_IDX, END_IDX, CHUNK_SIZE)
    ]

    print("Python Version :", sys.version)
    print("Joblib Version :", joblib.__version__)
    n_cpu = CONFIG.n_parallel_jobs
    print("The number of CPU is :", n_cpu)
    print(f"Generating indices [{START_IDX}, {END_IDX}) in {len(chunks)} chunk(s) of up to {CHUNK_SIZE}")

    t0 = time.time()
    with Parallel(n_jobs=n_cpu) as parallel:
        parallel(delayed(generate_and_save_ricardo_image_make_3D_Test)(st, en) for st, en in chunks)
    print(f"Total computation time : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()