"""KITTI raw dataset support.

KITTI raw has NO depth-map images — depth is a Velodyne LiDAR point cloud
(``*.bin``) that must be PROJECTED into the ``image_02`` (left colour) camera
using the per-date calibration. This module provides the reusable pieces:

    list_frames(kitti_raw_dir)                 -> deterministic list of frames
    load_calib(date_dir)                       -> projection matrices for a date
    project_velodyne_to_depth(velo, calib, W,H)-> sparse depth map (metres)
    frame_depth(frame, calib_cache)            -> depth map for one frame

Frames are ordered by (date, drive, frame#), so an index into ``list_frames``
is stable and can be used by the subset filter, GT generation and the loader —
the same "fixed ordering" contract NYU/Make3D use.

get_train_loader / get_test_loader are not implemented yet: they need the
underwater GT (haze/complex .npy) to be generated first (see the KITTI GT
generation step). They raise NotImplementedError until then.
"""

import os
import glob

import numpy as np
import cv2

from config import CONFIG


# ---------------------------------------------------------------------------
# Calibration + Velodyne projection
# ---------------------------------------------------------------------------

def _read_calib_file(path):
    """Parse a KITTI 'Key: v0 v1 ...' calibration file into {key: np.array}."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            try:
                out[key.strip()] = np.array([float(x) for x in val.split()])
            except ValueError:
                pass
    return out


def load_calib(date_dir):
    """Return the velodyne->image_02 projection pieces for a KITTI date folder.

    P_rect_02 (3x4), R_rect_00 (4x4 homogeneous), Tr_velo_to_cam (4x4 homogeneous).
    """
    c2c = _read_calib_file(os.path.join(date_dir, 'calib_cam_to_cam.txt'))
    v2c = _read_calib_file(os.path.join(date_dir, 'calib_velo_to_cam.txt'))
    P_rect_02 = c2c['P_rect_02'].reshape(3, 4)
    R_rect = np.eye(4)
    R_rect[:3, :3] = c2c['R_rect_00'].reshape(3, 3)
    Tr = np.eye(4)
    Tr[:3, :3] = v2c['R'].reshape(3, 3)
    Tr[:3, 3] = v2c['T']
    return {'P': P_rect_02, 'R_rect': R_rect, 'Tr': Tr}


def project_velodyne_to_depth(velo_path, calib, width, height):
    """Project a Velodyne .bin into image_02 -> sparse depth map (metres).

    Returns an (H, W) float32 array; 0 where no LiDAR point landed. When several
    points fall on one pixel the CLOSEST depth is kept.
    """
    velo = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    velo = velo[velo[:, 0] > 0]                                   # points in front of the car
    pts = np.concatenate([velo[:, :3], np.ones((len(velo), 1))], axis=1).T  # 4xN
    cam = calib['P'] @ calib['R_rect'] @ calib['Tr'] @ pts        # 3xN
    z = cam[2]
    u = cam[0] / z
    v = cam[1] / z
    m = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[m].astype(np.int32), v[m].astype(np.int32), z[m]

    depth = np.zeros((height, width), dtype=np.float32)
    order = np.argsort(-z)                                        # far first so near overwrites
    depth[v[order], u[order]] = z[order]
    return depth


# ---------------------------------------------------------------------------
# Frame index (the fixed ordering used everywhere)
# ---------------------------------------------------------------------------

def list_frames(kitti_raw_dir=None):
    """Return a deterministic list of KITTI frames.

    Each entry is a dict: {'image', 'velo', 'date_dir', 'date', 'drive', 'frame'}.
    Only frames that have BOTH an image_02 png and a matching velodyne .bin are
    kept. Ordered by (date, drive, frame#).
    """
    if kitti_raw_dir is None:
        kitti_raw_dir = CONFIG.kitti_raw_dir

    frames = []
    for date in sorted(os.listdir(kitti_raw_dir)):
        date_dir = os.path.join(kitti_raw_dir, date)
        if not os.path.isdir(date_dir):
            continue
        for drive in sorted(os.listdir(date_dir)):
            drive_dir = os.path.join(date_dir, drive)
            img_dir = os.path.join(drive_dir, 'image_02', 'data')
            velo_dir = os.path.join(drive_dir, 'velodyne_points', 'data')
            if not (os.path.isdir(img_dir) and os.path.isdir(velo_dir)):
                continue
            for img in sorted(glob.glob(os.path.join(img_dir, '*.png'))):
                stem = os.path.splitext(os.path.basename(img))[0]
                velo = os.path.join(velo_dir, stem + '.bin')
                if os.path.exists(velo):
                    frames.append({'image': img, 'velo': velo, 'date_dir': date_dir,
                                   'date': date, 'drive': drive, 'frame': stem})
    return frames


def frame_depth(frame, calib_cache=None):
    """Sparse depth map (metres) for one frame from list_frames().

    ``calib_cache`` (dict keyed by date_dir) avoids re-reading calibration files.
    """
    if calib_cache is None:
        calib_cache = {}
    dd = frame['date_dir']
    if dd not in calib_cache:
        calib_cache[dd] = load_calib(dd)
    img = cv2.imread(frame['image'])
    h, w = img.shape[:2]
    return project_velodyne_to_depth(frame['velo'], calib_cache[dd], w, h)


# ---------------------------------------------------------------------------
# Loaders — not implemented until the KITTI underwater GT is generated.
# ---------------------------------------------------------------------------

def get_train_loader(config):
    raise NotImplementedError(
        "KITTI train loader needs the underwater GT generated first "
        "(project velodyne -> depth -> compute_complex_noise). Not built yet.")


def get_test_loader(config):
    raise NotImplementedError(
        "KITTI test loader needs the underwater GT generated first. Not built yet.")
