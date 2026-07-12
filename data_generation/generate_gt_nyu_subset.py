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


def _test_tail_indices(cfg):
    """The held-out test tail: [split_idx, N).

    ``nyu_test_mode: "tail"`` builds the test set from exactly these indices and loads GT
    for them — but the SUBSET generator only writes GT for the filtered TRAIN pool
    (filter_nyu_subset restricts to [0, split_idx)). So on a pre-existing GT directory the
    tail files SURVIVE FROM THE PREVIOUS PHYSICS: you train on new-physics GT and are
    scored against old-physics GT, silently. Always regenerate the tail alongside the
    subset after any physics change.
    """
    from data.nyu import loadZipToMem
    _, rows = loadZipToMem(cfg.nyu_zip_path)
    split_idx = int(cfg.train_split_ratio * len(rows))
    return list(range(split_idx, len(rows)))


def main():
    parser = argparse.ArgumentParser(description='Generate NYU GT for a filtered subset only.')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--indices', default=None, help='path to a .npy of indices (overrides config)')
    parser.add_argument('--chunk-size', type=int, default=2000, help='indices per parallel job')
    parser.add_argument('--jobs', type=int, default=None, help='parallel jobs (default config.n_parallel_jobs)')
    parser.add_argument('--with-test-tail', action='store_true',
                        help="ALSO generate GT for the held-out test tail [split_idx, N). "
                             "Required with nyu_test_mode='tail' — otherwise test.py either "
                             "crashes on a missing file or, worse, silently scores you against "
                             "STALE ground truth left over from the previous physics.")
    parser.add_argument('--tail-only', action='store_true',
                        help='Generate ONLY the test tail (skip the training subset).')
    args = parser.parse_args()

    cfg = load_config(args.config)
    idx_path = _resolve_indices_path(cfg, args.indices)

    if args.tail_only:
        indices = []
    else:
        indices = np.asarray(np.load(idx_path), dtype=np.int64).tolist()

    n_subset = len(indices)
    n_tail = 0
    if args.with_test_tail or args.tail_only:
        tail = _test_tail_indices(cfg)
        existing = set(indices)
        tail = [i for i in tail if i not in existing]
        n_tail = len(tail)
        indices = indices + tail
        print("[tail] adding %d held-out test-tail indices [%d, %d)"
              % (n_tail, tail[0] if tail else -1, (tail[-1] + 1) if tail else -1))
    elif str(getattr(cfg, 'nyu_test_mode', 'tail')) == 'tail':
        print("[WARNING] nyu_test_mode='tail' but --with-test-tail was NOT passed. The test "
              "tail will have NO GT from this run. If a GT dir already exists, its tail files "
              "are STALE (previous physics) and test.py will silently score against them.")

    jobs = args.jobs if args.jobs is not None else cfg.n_parallel_jobs

    # chunk the index list into sublists for the parallel workers
    chunks = [indices[i:i + args.chunk_size] for i in range(0, len(indices), args.chunk_size)]

    print("Python :", __import__('sys').version.split()[0], "| Joblib :", joblib.__version__)
    print("Subset indices file :", idx_path)
    print("Generating GT for %d images (%d subset + %d test-tail) -> %s"
          % (len(indices), n_subset, n_tail, cfg.nyu_gt_train_dir))
    print("Chunks: %d (size <= %d)  |  jobs: %d" % (len(chunks), args.chunk_size, jobs))

    t0 = time.time()
    with Parallel(n_jobs=jobs) as parallel:
        parallel(delayed(generate_and_save_haze_image)(0, 0, indices=chunk) for chunk in chunks)
    print("Total computation time : %.1fs" % (time.time() - t0))


if __name__ == '__main__':
    main()
