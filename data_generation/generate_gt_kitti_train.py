# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Generate KITTI haze/complex GT for the ENTIRE completed-depth train split.

This is the full-dataset path (~86k frames — slow). To iterate on a smaller set,
use ``generate_gt_kitti_subset.py`` with a filtered-index file instead.
"""

import sys
import time
import joblib
from joblib import Parallel, delayed

from config import CONFIG
from data_2 import generate_and_save_ricardo_image_kitti
from data.kitti import list_completed_frames

START_IDX = 0
END_IDX = len(list_completed_frames('train'))
CHUNK_SIZE = 2000


def main():
    chunks = [(st, min(st + CHUNK_SIZE, END_IDX)) for st in range(START_IDX, END_IDX, CHUNK_SIZE)]
    print("Python :", sys.version.split()[0], "| Joblib :", joblib.__version__)
    print("Generating KITTI train GT indices [%d, %d) in %d chunk(s) of up to %d -> %s"
          % (START_IDX, END_IDX, len(chunks), CHUNK_SIZE, CONFIG.kitti_gt_train_dir))
    _os.makedirs(CONFIG.kitti_gt_train_dir, exist_ok=True)
    t0 = time.time()
    with Parallel(n_jobs=CONFIG.n_parallel_jobs) as parallel:
        parallel(delayed(generate_and_save_ricardo_image_kitti)(st, en) for st, en in chunks)
    print("Total computation time : %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
