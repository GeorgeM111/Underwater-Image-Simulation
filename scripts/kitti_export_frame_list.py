"""Export the KITTI frames the pipeline uses, in the monodepth2 filename format.

Runs in the PROJECT environment. Writes one line per frame in exactly the format the official
nianticlabs/depth-hints ``precompute_depth_hints.py`` expects for its ``--filenames`` option:

    <date>/<drive>_sync <frame_index> l

(``l`` = left camera, image_02 — the side the underwater pipeline uses for its ground truth).
Passing this file as ``--filenames`` makes the OFFICIAL depth-hints precompute produce stereo
depth hints for our completed-depth frames instead of the default Eigen split, so the hints line
up with the frames data.kitti actually loads.

Coverage mirrors data.kitti exactly:
  * training frames  -> subset indices (kitti_train_mode='subset') or the full train split ('all')
  * validation / test-tail -> range(split_idx, N)          (get_val_loader / tail test)
  * official test    -> every frame of the 'val' split      (only if kitti_test_mode='official')

Usage:
    python scripts/kitti_export_frame_list.py [--config config.yaml] [--out FILE]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import load_config
import data.kitti as K


def _subset_path(cfg):
    if cfg.kitti_subset_indices:
        return cfg.kitti_subset_indices
    params_dir = os.path.dirname(cfg.beta_mat_kitti_train)
    return os.path.join(params_dir, "%d_filtered_kitti.npy" % int(cfg.kitti_subset_size))


def _md2_line(f):
    """A frame dict -> the monodepth2 filename line '<date>/<drive> <frame_int> l'."""
    return "%s/%s %d l" % (f["date"], f["drive"], int(f["frame"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output list file (default: params dir)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    train_frames = K.list_completed_frames("train")
    if not train_frames:
        raise SystemExit("No KITTI 'train' frames found under %s" % cfg.kitti_completed_depth_dir)
    n = len(train_frames)
    split_idx = int(cfg.train_split_ratio * n)

    # --- training frames actually consumed ---
    mode = str(cfg.kitti_train_mode).lower()
    if mode == "subset":
        idx_path = _subset_path(cfg)
        if not os.path.exists(idx_path):
            raise SystemExit("kitti_train_mode='subset' but %s is missing (run filter_kitti_subset.py)." % idx_path)
        idx = np.load(idx_path).astype(int)
        idx = idx[(idx >= 0) & (idx < split_idx)]
        train_used = [train_frames[i] for i in idx]
    else:
        train_used = train_frames[:split_idx]

    # --- validation / held-out tail (get_val_loader and 'tail' test) ---
    tail_used = train_frames[split_idx:]

    selected = list(train_used) + list(tail_used)

    # --- official test split, if configured ---
    if str(cfg.kitti_test_mode).lower() == "official":
        selected += K.list_completed_frames("val")

    # de-duplicate by (drive, frame), keep first occurrence
    seen, lines = set(), []
    for f in selected:
        key = (f["drive"], f["frame"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(_md2_line(f))

    out = args.out or os.path.join(os.path.dirname(cfg.beta_mat_kitti_train), "kitti_depth_hints_files.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[kitti] wrote %d unique frames (monodepth2 format) -> %s" % (len(lines), out))
    print("        train_used=%d  tail/val=%d  official=%s"
          % (len(train_used), len(tail_used), str(cfg.kitti_test_mode).lower() == "official"))
    print("        feed this to the official precompute:  --filenames %s" % out)


if __name__ == "__main__":
    main()
