# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

"""Write the KITTI held-out TAIL index file — the frames in [split_idx, N).

With kitti_test_mode='official' (recommended) this tail is the VALIDATION set: get_val_loader
reads its GT from kitti_gt_train_dir, and the subset generator does NOT cover it (the subset is
clipped to [0, split_idx)). So its GT must be generated separately — feed this file to
generate_gt_kitti_subset.py --indices <file>. (With the deprecated 'tail' test-mode the same
frames double as the test set.)

The split matches the loaders EXACTLY via data.kitti.kitti_split_idx (which uses
kitti_val_ratio), so the indices cannot drift from what get_val_loader will actually read.

Run:
    python data_generation/make_kitti_test_tail_indices.py
    python data_generation/make_kitti_test_tail_indices.py --out /path/tail.npy
"""

import os
import argparse

import numpy as np

from config import load_config
from data.kitti import list_completed_frames, kitti_split_idx


def main():
    parser = argparse.ArgumentParser(description='Generate the KITTI val/test tail index .npy')
    parser.add_argument('--config', default=None, help='path to config YAML (default ./config.yaml)')
    parser.add_argument('--out', default=None, help='output .npy (default: params dir/kitti_test_tail_indices.npy)')
    args = parser.parse_args()
    cfg = load_config(args.config)

    n = len(list_completed_frames('train'))
    split_idx = kitti_split_idx(cfg, n)          # SAME split the loaders use (kitti_val_ratio)
    tail = np.arange(split_idx, n, dtype=np.int64)

    out = args.out or os.path.join(os.path.dirname(cfg.beta_mat_kitti_train), 'kitti_test_tail_indices.npy')
    np.save(out, tail)
    vr = getattr(cfg, 'kitti_val_ratio', 1.0 - cfg.train_split_ratio)
    print("N=%d  split_idx=%d  (kitti_val_ratio=%.3f -> %.0f%% held out)"
          % (n, split_idx, vr, 100 * vr))
    print("Wrote %d tail (validation) indices [%d, %d) -> %s" % (tail.size, tail[0], tail[-1] + 1, out))
    print("\nNext: python data_generation/generate_gt_kitti_subset.py --indices %s" % out)


if __name__ == '__main__':
    main()
