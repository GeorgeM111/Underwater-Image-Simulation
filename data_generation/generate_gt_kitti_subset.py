# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Generate KITTI haze/complex GT for ONLY a subset of train frames.

Which indices (positions into data.kitti.list_completed_frames('train')):
    --indices <path.npy>   explicit index file, OR
    (default)              config.kitti_subset_indices, else the auto-derived
                           {kitti_subset_size}_filtered_kitti.npy in the params dir
                           (produced by filter_kitti_subset.py). Also accepts the
                           test-tail file from make_kitti_test_tail_indices.py.

Run:
    python data_generation/generate_gt_kitti_subset.py --config config.yaml
    python data_generation/generate_gt_kitti_subset.py --indices /path/tail.npy
"""

import os
import time
import argparse

import numpy as np
import joblib
from joblib import Parallel, delayed

from config import load_config
from data_2 import generate_and_save_ricardo_image_kitti


def _resolve_indices_path(cfg, override):
    if override:
        return override
    p = getattr(cfg, 'kitti_subset_indices', None)
    if p:
        return p
    return os.path.join(os.path.dirname(cfg.beta_mat_kitti_train),
                        "%d_filtered_kitti.npy" % cfg.kitti_subset_size)


def main():
    parser = argparse.ArgumentParser(description='Generate KITTI GT for a filtered subset only.')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--indices', default=None, help='path to a .npy of indices (overrides config)')
    parser.add_argument('--chunk-size', type=int, default=2000, help='indices per parallel job')
    parser.add_argument('--jobs', type=int, default=None, help='parallel jobs (default config.n_parallel_jobs)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    idx_path = _resolve_indices_path(cfg, args.indices)
    indices = np.asarray(np.load(idx_path), dtype=np.int64).tolist()
    jobs = args.jobs if args.jobs is not None else cfg.n_parallel_jobs
    os.makedirs(cfg.kitti_gt_train_dir, exist_ok=True)

    chunks = [indices[i:i + args.chunk_size] for i in range(0, len(indices), args.chunk_size)]
    print("Python :", __import__('sys').version.split()[0], "| Joblib :", joblib.__version__)
    print("Subset indices file :", idx_path)
    print("Generating GT for %d frames -> %s" % (len(indices), cfg.kitti_gt_train_dir))
    print("Chunks: %d (size <= %d)  |  jobs: %d" % (len(chunks), args.chunk_size, jobs))

    t0 = time.time()
    with Parallel(n_jobs=jobs) as parallel:
        parallel(delayed(generate_and_save_ricardo_image_kitti)(0, 0, indices=chunk) for chunk in chunks)
    print("Total computation time : %.1fs" % (time.time() - t0))


if __name__ == '__main__':
    main()
