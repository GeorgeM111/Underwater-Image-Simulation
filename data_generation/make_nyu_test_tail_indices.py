# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Write the NYU held-out *test-tail* index file used by ``nyu_test_mode='tail'``.

The tail-mode test loader (``data.nyu.get_test_loader``) evaluates the indices
``[split_idx, len(nyu2_train))`` where ``split_idx = int(train_split_ratio * N)``
and ``N`` is the number of rows in ``nyu2_train.csv``. Those images need pre-computed
GT in ``nyu_gt_train_dir``. This script computes exactly that index list (matching
the loader's split arithmetic) and saves it so you can feed it to
``generate_gt_nyu_subset.py --indices <file>``.

Run:
    python data_generation/make_nyu_test_tail_indices.py            # -> params dir
    python data_generation/make_nyu_test_tail_indices.py --out /path/tail.npy
"""

import argparse
from zipfile import ZipFile

import numpy as np

from config import load_config


def _csv_row_count(zip_path, csv_name='data/nyu2_train.csv'):
    # Count non-empty rows the SAME way loadZipToMem does (split('\n')), so the
    # split index matches data.nyu exactly.
    with ZipFile(zip_path) as zf:
        return sum(1 for r in zf.read(csv_name).decode('utf-8').splitlines() if len(r) > 0)


def main():
    parser = argparse.ArgumentParser(description='Generate the NYU tail-test index .npy')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--out', default=None, help='output .npy path (default: params dir/nyu_test_tail_indices.npy)')
    args = parser.parse_args()
    cfg = load_config(args.config)

    n = _csv_row_count(cfg.nyu_zip_path)
    split_idx = int(cfg.train_split_ratio * n)
    tail = np.arange(split_idx, n, dtype=np.int64)

    out = args.out or __import__('os').path.join(
        __import__('os').path.dirname(cfg.beta_mat_nyu_train), 'nyu_test_tail_indices.npy')
    np.save(out, tail)
    print("N=%d  split_idx=%d (train_split_ratio=%.3f)" % (n, split_idx, cfg.train_split_ratio))
    print("Wrote %d tail indices [%d, %d) -> %s" % (tail.size, tail[0], tail[-1] + 1, out))
    print("\nNext: python data_generation/generate_gt_nyu_subset.py --indices %s" % out)


if __name__ == '__main__':
    main()
