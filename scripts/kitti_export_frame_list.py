"""Export the list of KITTI frames the pipeline actually uses, for depth prediction.

Runs in the PROJECT environment. Writes one line per unique frame:

    <raw_image_path>\t<drive>\t<frame>

so that scripts/kitti_predict_depth.py (run in the monodepth2 / depth-hints legacy env)
predicts a dense depth map ONLY for the frames that training and evaluation touch. This
keeps the predicted-depth footprint small (subset + held-out tail, not all ~90k frames),
which matters under a storage budget.

Coverage mirrors data.kitti exactly:
  * training frames  -> subset indices (kitti_train_mode='subset') or range(0, split_idx)
  * validation / test-tail -> range(split_idx, N)          (get_val_loader / tail test)
  * official test    -> every frame of the 'val' split      (only if kitti_test_mode='official')

Usage:
    python scripts/kitti_export_frame_list.py [--config config.yaml] [--out FILE]
"""

import os
import sys
import argparse

# repo-root on sys.path (scripts/ -> repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import load_config
import data.kitti as K


def _subset_path(cfg):
    if cfg.kitti_subset_indices:
        return cfg.kitti_subset_indices
    params_dir = os.path.dirname(cfg.beta_mat_kitti_train)
    return os.path.join(params_dir, "%d_filtered_kitti.npy" % int(cfg.kitti_subset_size))


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
    seen, rows = set(), []
    for f in selected:
        key = (f["drive"], f["frame"])
        if key in seen:
            continue
        seen.add(key)
        rows.append((f["image"], f["drive"], f["frame"]))

    out = args.out or os.path.join(os.path.dirname(cfg.beta_mat_kitti_train), "kitti_frames_for_depth.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        for img, drive, frame in rows:
            fh.write("%s\t%s\t%s\n" % (img, drive, frame))
    print("[kitti] wrote %d unique frames -> %s" % (len(rows), out))
    print("        train_used=%d  tail/val=%d  official=%s"
          % (len(train_used), len(tail_used), str(cfg.kitti_test_mode).lower() == "official"))


if __name__ == "__main__":
    main()
