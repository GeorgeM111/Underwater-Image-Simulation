"""Compute dense SGM stereo depth ("depth hints") for KITTI, to stabilise the depth GT.

Runs in the PROJECT environment (only OpenCV + numpy — no network, no pretrained weights, no
GPU, no downloads). For each frame it semi-global-matches the left (image_02) and right
(image_03) KITTI cameras into a dense disparity, converts it to METRIC depth with the stereo
baseline and focal length from ``calib_cam_to_cam.txt``, and saves it at the raw image
resolution as ``<kitti_depth_hints_dir>/<drive>/<frame>.npy`` (float16, metres).

These maps are a *geometric measurement* from the two real cameras, not a network prediction.
They are consumed by data.kitti.load_frame_image_depth when ``kitti_depth_source: "depth_hints"``,
which keeps the real LiDAR ``completed_depth`` measurements and fills only the holes/sky with this
stereo depth (scale-aligned per frame). The depth-estimation network of the method is untouched.

Usage (project env):
    python scripts/kitti_depth_hints_sgm.py [--config config.yaml] [--frames_file FILE]
    # --frames_file defaults to the output of kitti_export_frame_list.py (only the used frames).
    # Omit it (or pass --all) to process every completed-depth frame.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from config import load_config
import data.kitti as K


def _read_calib(date_dir):
    """Return (fx_pixels, baseline_metres) for the cam2<->cam3 stereo pair from calib_cam_to_cam.txt."""
    path = os.path.join(date_dir, "calib_cam_to_cam.txt")
    P = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("P_rect_02:") or line.startswith("P_rect_03:"):
                key = line.split(":")[0].strip()
                vals = np.array([float(x) for x in line.split(":")[1].split()]).reshape(3, 4)
                P[key] = vals
    fx = float(P["P_rect_02"][0, 0])
    # tx_i = P_rect_0i[0,3] = -fx * (offset of cam i from cam0). Baseline 2<->3 = |tx2 - tx3| / fx.
    baseline = abs(float(P["P_rect_02"][0, 3]) - float(P["P_rect_03"][0, 3])) / fx
    return fx, baseline


def _build_sgbm(num_disp, block):
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,                 # must be divisible by 16
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _right_image_path(left_path):
    """image_02/.../frame.png -> image_03/.../frame.png (the raw right-camera frame)."""
    head, tail = os.path.split(left_path)
    return os.path.join(head.replace("image_02", "image_03"), tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--frames_file", default=None,
                    help="output of kitti_export_frame_list.py (default). Only these frames are processed.")
    ap.add_argument("--all", action="store_true", help="process every completed-depth frame instead")
    ap.add_argument("--num_disparities", type=int, default=192, help="max disparity (multiple of 16)")
    ap.add_argument("--block_size", type=int, default=7)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = args.out_dir or cfg.kitti_depth_hints_dir
    max_depth = float(cfg.kitti_max_depth_m)

    # --- resolve the frame list (left image path, drive, frame) ---
    if args.all:
        rows = [(f["image"], f["drive"], f["frame"])
                for split in ("train", "val") for f in K.list_completed_frames(split)]
    else:
        ff = args.frames_file or os.path.join(os.path.dirname(cfg.beta_mat_kitti_train),
                                              "kitti_frames_for_depth.txt")
        if not os.path.exists(ff):
            raise SystemExit("Frame list %s not found. Run scripts/kitti_export_frame_list.py first, "
                             "or pass --all." % ff)
        with open(ff) as fh:
            rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]

    sgbm = _build_sgbm(args.num_disparities, args.block_size)
    calib_cache = {}
    done = skipped = 0
    for left_path, drive, frame in rows:
        out_path = os.path.join(out_dir, drive, frame + ".npy")
        if os.path.exists(out_path):
            done += 1
            continue
        right_path = _right_image_path(left_path)
        if not (os.path.exists(left_path) and os.path.exists(right_path)):
            skipped += 1
            continue
        date = K._drive_date(drive)
        if date not in calib_cache:
            calib_cache[date] = _read_calib(os.path.join(cfg.kitti_raw_dir, date))
        fx, baseline = calib_cache[date]

        left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        disp = sgbm.compute(left, right).astype(np.float32) / 16.0   # SGBM returns fixed-point x16
        depth = np.zeros_like(disp, dtype=np.float32)
        m = disp > 0.0
        depth[m] = (fx * baseline) / disp[m]                          # metric depth
        depth = np.clip(depth, 0.0, max_depth)                        # invalid stays 0 (a "hole")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, depth.astype(np.float16))
        done += 1
        if done % 500 == 0:
            print("[depth-hints] %d / %d" % (done, len(rows)))

    print("[depth-hints] finished: %d written/present, %d skipped (missing pair) -> %s"
          % (done, skipped, out_dir))


if __name__ == "__main__":
    main()
