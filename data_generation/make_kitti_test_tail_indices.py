# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Write the KITTI held-out *test-tail* index file used by kitti_test_mode='tail'.

The tail-mode test loader (data.kitti.get_test_loader) evaluates train-frame indices
[split_idx, N) where split_idx = int(train_split_ratio * N) and N is the number of
completed-depth train frames. Those frames need pre-computed GT in kitti_gt_train_dir.
This computes exactly that index list (matching the loader's split) and saves it so
you can feed it to generate_gt_kitti_subset.py --indices <file>.

Run:
    python data_generation/make_kitti_test_tail_indices.py
    python data_generation/make_kitti_test_tail_indices.py --out /path/tail.npy
"""

import os
import argparse

import numpy as np

from config import load_config
from data.kitti import list_completed_frames


def main():
    parser = argparse.ArgumentParser(description='Generate the KITTI tail-test index .npy')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--out', default=None, help='output .npy (default: params dir/kitti_test_tail_indices.npy)')
    args = parser.parse_args()
    cfg = load_config(args.config)

    n = len(list_completed_frames('train'))
    split_idx = int(cfg.train_split_ratio * n)
    tail = np.arange(split_idx, n, dtype=np.int64)

    out = args.out or os.path.join(os.path.dirname(cfg.beta_mat_kitti_train), 'kitti_test_tail_indices.npy')
    np.save(out, tail)
    print("N=%d  split_idx=%d (train_split_ratio=%.3f)" % (n, split_idx, cfg.train_split_ratio))
    print("Wrote %d tail indices [%d, %d) -> %s" % (tail.size, tail[0], tail[-1] + 1, out))
    print("\nNext: python data_generation/generate_gt_kitti_subset.py --indices %s" % out)


if __name__ == '__main__':
    main()
