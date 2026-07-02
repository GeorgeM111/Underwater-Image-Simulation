# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Generate NYU haze/complex GT for ONLY a subset of images.

Instead of generating all ~50k images (generate_gt_nyu_train.py), this builds GT
for just the indices in a ``.npy`` file — e.g. the informative subset produced by
``filter_nyu_subset.py`` — so you can iterate on a smaller set.

Which indices:
    --indices <path.npy>   explicit index file, OR
    (default)              config.nyu_subset_indices, else the auto-derived
                           {nyu_subset_size}_filtered_nyu.npy in the params dir.

Run:
    python data_generation/generate_gt_nyu_subset.py --config config.yaml
    python data_generation/generate_gt_nyu_subset.py --indices /path/to/5000_filtered_nyu.npy
"""

import os
import time
import argparse

import numpy as np
import joblib
from joblib import Parallel, delayed

from config import load_config
from data_2 import generate_and_save_haze_image


def _resolve_indices_path(cfg, override):
    if override:
        return override
    p = getattr(cfg, 'nyu_subset_indices', None)
    if p:
        return p
    return os.path.join(os.path.dirname(cfg.beta_mat_nyu_train),
                        "%d_filtered_nyu.npy" % cfg.nyu_subset_size)


def main():
    parser = argparse.ArgumentParser(description='Generate NYU GT for a filtered subset only.')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--indices', default=None, help='path to a .npy of indices (overrides config)')
    parser.add_argument('--chunk-size', type=int, default=2000, help='indices per parallel job')
    parser.add_argument('--jobs', type=int, default=None, help='parallel jobs (default config.n_parallel_jobs)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    idx_path = _resolve_indices_path(cfg, args.indices)
    indices = np.asarray(np.load(idx_path), dtype=np.int64).tolist()
    jobs = args.jobs if args.jobs is not None else cfg.n_parallel_jobs

    # chunk the index list into sublists for the parallel workers
    chunks = [indices[i:i + args.chunk_size] for i in range(0, len(indices), args.chunk_size)]

    print("Python :", __import__('sys').version.split()[0], "| Joblib :", joblib.__version__)
    print("Subset indices file :", idx_path)
    print("Generating GT for %d images -> %s" % (len(indices), cfg.nyu_gt_train_dir))
    print("Chunks: %d (size <= %d)  |  jobs: %d" % (len(chunks), args.chunk_size, jobs))

    t0 = time.time()
    with Parallel(n_jobs=jobs) as parallel:
        parallel(delayed(generate_and_save_haze_image)(0, 0, indices=chunk) for chunk in chunks)
    print("Total computation time : %.1fs" % (time.time() - t0))


if __name__ == '__main__':
    main()
