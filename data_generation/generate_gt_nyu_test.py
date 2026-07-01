# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import sys
import time
import joblib
from joblib import Parallel, delayed

from config import CONFIG
from data_2 import generate_and_save_haze_image_test


START_IDX = 0
END_IDX = 654
CHUNK_SIZE = 150


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
        parallel(delayed(generate_and_save_haze_image_test)(st, en) for st, en in chunks)
    print(f"Total computation time : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()