"""KITTI dataset support (raw images + completed/annotated depth).

KITTI raw has no dense depth. We use the official **completed depth**
(``data_depth_annotated``: ``<root>/<split>/<drive>/proj_depth/groundtruth/
image_02/<frame>.png``, 16-bit, ``depth_m = png/256``), paired with the raw
``image_02`` colour frames. The completed maps are still only semi-dense
(~10-30% valid), so :func:`densify_depth` fills the holes (the physics haze /
scatter model needs a DENSE depth), exactly analogous to how NYU/Make3D provide
dense depth.

Public interface (mirrors data.make3d / data.nyu):
    list_completed_frames(split, ...)     -> deterministic [{image, depth, ...}]
    read_completed_depth(path, max_depth) -> dense depth (metres)  [densified]
    get_train_loader(config) / get_val_loader(config) / get_test_loader(config)

The frame ORDER (sorted by drive, then frame#) is the fixed contract shared by
the subset filter, the GT generator and the loader, so index i always selects
the same frame everywhere (same as NYU's shuffled-list / Make3D's id-paired list).

Legacy Velodyne-projection helpers (``list_frames``, ``project_velodyne_to_depth``)
are kept for backward compatibility but are no longer the depth source.
"""

import os
import glob
import random

import numpy as np
from numpy import load
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
from scipy import ndimage

from config import CONFIG
from utils.depth_range import DEPTH_CLAMP_MIN, DEPTH_CLAMP_MAX

# Spatial size [W, H] — from config so loader/generator stay in sync.
FULL_W, FULL_H = CONFIG.kitti_full_size
HALF_W, HALF_H = CONFIG.kitti_half_size


# ---------------------------------------------------------------------------
# Completed-depth frame index (the fixed ordering used everywhere)
# ---------------------------------------------------------------------------

def _split_root(split, completed_dir=None):
    root = completed_dir or CONFIG.kitti_completed_depth_dir
    return os.path.join(root, split)


def _drive_date(drive):
    """`2011_09_26_drive_0001_sync` -> `2011_09_26` (the raw-image date folder)."""
    return '_'.join(drive.split('_')[:3])


def list_completed_frames(split, raw_dir=None, completed_dir=None, camera='image_02'):
    """Deterministic list of frames that have BOTH a completed-depth PNG and a raw
    colour image, for ``split`` in {'train', 'val'}.

    Each entry: {'image', 'depth', 'drive', 'date', 'frame'}. Ordered by
    (drive, frame#) so an index is stable across the filter / GT gen / loader.
    """
    raw_dir = raw_dir or CONFIG.kitti_raw_dir
    split_root = _split_root(split, completed_dir)
    frames = []
    if not os.path.isdir(split_root):
        return frames
    for drive in sorted(os.listdir(split_root)):
        gt_dir = os.path.join(split_root, drive, 'proj_depth', 'groundtruth', camera)
        if not os.path.isdir(gt_dir):
            continue
        date = _drive_date(drive)
        img_dir = os.path.join(raw_dir, date, drive, camera, 'data')
        for dp in sorted(glob.glob(os.path.join(gt_dir, '*.png'))):
            frame = os.path.basename(dp)
            img = os.path.join(img_dir, frame)
            if os.path.exists(img):
                frames.append({'image': img, 'depth': dp, 'drive': drive,
                               'date': date, 'frame': os.path.splitext(frame)[0]})
    return frames


# ---------------------------------------------------------------------------
# Depth reading + densification
# ---------------------------------------------------------------------------

# IP-Basic morphological kernels (Ku et al., "In Defense of Classical Image
# Processing: Fast Depth Completion", 2018).
_DIAMOND_5 = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)
_FULL_5 = np.ones((5, 5), np.uint8)
_FULL_7 = np.ones((7, 7), np.uint8)
_FULL_31 = np.ones((31, 31), np.uint8)


def _nearest_fill(depth):
    """Guarantee density: fill any remaining <=0 pixel with the nearest valid depth."""
    invalid = depth <= 0
    if not invalid.any() or invalid.all():
        return depth
    idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return depth[tuple(idx)]


def densify_depth(depth, invert_max=100.0):
    """Densify a semi-dense (metres) depth map with IP-Basic + a nearest-fill backstop.

    IP-Basic (classical morphology) fills the LiDAR holes far more cleanly than a raw
    nearest fill — it closes small gaps, extends the topmost measured depth up into the
    (LiDAR-less) sky, and smooths edges — which keeps the downstream haze/scatter GT
    from inheriting blocky depth artefacts. A final nearest fill guarantees 0 holes.
    Returns a dense float32 map in metres. All-invalid input is returned unchanged.
    """
    depth = depth.astype(np.float32)
    if (depth > 0).sum() == 0:
        return depth
    d = depth.copy()
    valid = d > 0.1
    d[valid] = invert_max - d[valid]                                  # invert: near -> large

    d = cv2.dilate(d, _DIAMOND_5)                                     # fill small holes
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, _FULL_5)                 # close gaps
    empty = d < 0.1                                                   # medium fill
    d[empty] = cv2.dilate(d, _FULL_7)[empty]

    # Extend the highest measured value up to the top of the frame (sky has no LiDAR).
    top = np.argmax(d > 0.1, axis=0)
    for c in range(d.shape[1]):
        if d[top[c], c] > 0.1:
            d[0:top[c], c] = d[top[c], c]
    empty = d < 0.1                                                   # large fill for the rest
    d[empty] = cv2.dilate(d, _FULL_31)[empty]

    d = cv2.medianBlur(d, 5)                                          # smooth
    blurred = cv2.GaussianBlur(d, (5, 5), 0)
    valid = d > 0.1
    d[valid] = blurred[valid]

    valid = d > 0.1
    d[valid] = invert_max - d[valid]                                 # invert back
    d = _nearest_fill(d)

    # Restore the REAL LiDAR measurements exactly. IP-Basic's dilate/close/blur perturbs
    # the measured pixels by ~0.2-0.4 m (p95 ~1.6 m); for GT generation we want the
    # measurements untouched and only the HOLES synthesised.
    measured = depth > 0.1
    d[measured] = depth[measured]
    return d


def read_completed_depth(path, max_depth_m=None, densify=True):
    """Read a completed-depth PNG -> float32 depth map in METRES.

    ``depth_m = png/256`` (KITTI convention); 0 = invalid. Densified by default.
    """
    if max_depth_m is None:
        max_depth_m = CONFIG.kitti_max_depth_m
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)        # 16-bit, HxW
    depth = raw.astype(np.float32) / 256.0
    if densify:
        depth = densify_depth(depth)
    return np.clip(depth, 0.0, max_depth_m)


def kb_crop(arr, w=None, h=None):
    """Bottom-center crop to the LiDAR-covered region.

    The Velodyne's vertical FOV (+2 deg .. -24.8 deg) means the TOP ~36% of every KITTI
    frame (rows 0..133 of 375) receives ZERO returns — depth there can only be invented.
    Cropping to the bottom ``h`` rows (centre ``w`` columns) keeps the frame fully
    data-backed and roughly DOUBLES the valid-pixel density (e.g. 19.5% -> 32.7%).
    """
    w = w or FULL_W
    h = h or FULL_H
    H, W = arr.shape[:2]
    top = max(H - h, 0)
    left = max((W - w) // 2, 0)
    return arr[top:top + h, left:left + w]


def load_frame_image_depth(f, max_depth_m=None):
    """Canonical (image_rgb_uint8, dense_depth_metres) for one frame.

    CROP first, then densify — so the hole-filling never extrapolates from the
    LiDAR-less sky. Used by BOTH the GT generator (data_2) and the loader, so the two
    can never drift apart.
    """
    if max_depth_m is None:
        max_depth_m = CONFIG.kitti_max_depth_m
    img = cv2.cvtColor(cv2.imread(f['image']), cv2.COLOR_BGR2RGB)
    depth = read_completed_depth(f['depth'], max_depth_m, densify=False)
    img = kb_crop(img)
    depth = densify_depth(kb_crop(depth))
    return img, np.clip(depth, 0.0, max_depth_m)


# ---------------------------------------------------------------------------
# Dataset + loaders (mirror data.make3d)
# ---------------------------------------------------------------------------

def _create_spatial(vec, m, n):
    v = np.reshape(np.asarray(vec, dtype=np.float32), [3, 1, 1])
    v = np.tile(v, [1, m, n])
    v = np.swapaxes(v, 0, 2)
    v = np.swapaxes(v, 0, 1)
    return v


def _hwc_to_chw(arr):
    return np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32)


def _depth_to_tensor(depth_metres, max_depth_m):
    """Metric depth -> the canonical stored-depth axis d = z/max_depth * 1000.

    Floor is DEPTH_CLAMP_MIN (40), not 10. The reciprocal target y = 1000/d must land in
    [1, 25] — the range the shared depth head (models/decoder_1ch.Decoder1Ch) is bounded to
    by its scaled sigmoid. The old floor of 10 allowed y up to 100, i.e. a target range 4x
    wider than the head can even represent. Because the floor is a FRACTION of max_depth,
    this is z >= 0.4 m for NYU and z >= 3.2 m for Make3D/KITTI (both sane near clips).
    """
    d = torch.from_numpy(depth_metres).float()
    d = d / max_depth_m * 1000.0
    d = torch.clamp(d, DEPTH_CLAMP_MIN, DEPTH_CLAMP_MAX)
    return d.unsqueeze(0)


class _KittiDataset(Dataset):
    def __init__(self, frames, cache_dir, beta_path, a_path, max_depth_m,
                 beta_scale, augment=False):
        self.frames = frames
        self.cache_dir = cache_dir
        self.max_depth_m = max_depth_m
        self.beta_scale = beta_scale
        self.augment = augment
        self.beta_mat_arr = load(beta_path)
        self.a_mat_arr = load(a_path)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        haze_image = load(os.path.join(self.cache_dir, str(idx) + 'haze_image.npy'))
        complex_noisy = load(os.path.join(self.cache_dir, str(idx) + 'complex_haze_image.npy'))
        H, W = haze_image.shape[0], haze_image.shape[1]

        f = self.frames[idx]
        # Bottom-center crop to the LiDAR region (shared helper => identical to the GT gen).
        image_full, depth_m = load_frame_image_depth(f, self.max_depth_m)
        image_half = cv2.resize(image_full, (HALF_W, HALF_H), interpolation=cv2.INTER_LINEAR)
        depth_half = cv2.resize(depth_m, (HALF_W, HALF_H), interpolation=cv2.INTER_LINEAR)

        image_full = torch.from_numpy(_hwc_to_chw(image_full)).float() / 255.0
        image_half = torch.from_numpy(_hwc_to_chw(image_half)).float() / 255.0
        haze_image = torch.from_numpy(_hwc_to_chw(haze_image)).float()
        complex_noisy = torch.from_numpy(_hwc_to_chw(complex_noisy)).float()
        depth_t = _depth_to_tensor(depth_half, self.max_depth_m)

        if self.augment and random.random() < 0.5:
            image_full = torch.flip(image_full, dims=[2])
            image_half = torch.flip(image_half, dims=[2])
            haze_image = torch.flip(haze_image, dims=[2])
            complex_noisy = torch.flip(complex_noisy, dims=[2])
            depth_t = torch.flip(depth_t, dims=[2])

        # Option B: deliver the CLEARER-water (scaled) beta so the classical haze at
        # training matches the GT generator (keep true depth via max_depth_m).
        beta_spatial = _create_spatial(np.asarray(self.beta_mat_arr[idx], dtype=np.float32) * self.beta_scale, H, W)
        a_spatial = _create_spatial(self.a_mat_arr[idx], H, W)
        unit_spatial = _create_spatial([1.0, 1.0, 1.0], H, W)

        return {
            'image_full': image_full,
            'image_half': image_half,
            'depth': depth_t,
            'haze_image': haze_image,
            'beta': torch.from_numpy(_hwc_to_chw(beta_spatial)),
            'a_val': torch.from_numpy(_hwc_to_chw(a_spatial)),
            'unit_mat': torch.from_numpy(_hwc_to_chw(unit_spatial)),
            'complex_noise_img': complex_noisy,
        }


def _resolve_subset_path(config):
    p = getattr(config, 'kitti_subset_indices', None)
    if p:
        return p
    params_dir = os.path.dirname(config.beta_mat_kitti_train)
    return os.path.join(params_dir, "%d_filtered_kitti.npy" % config.kitti_subset_size)


def _train_frames(config):
    return list_completed_frames('train')


def get_train_loader(config):
    """KITTI training loader (mirrors data.nyu.get_train_loader).

    ``kitti_train_mode``: 'all' -> full train split; 'subset' -> filtered indices,
    both clipped to [0, split_idx) so the held-out test tail can't leak in.
    """
    frames = _train_frames(config)
    dataset = _KittiDataset(frames, config.kitti_gt_train_dir, config.beta_mat_kitti_train,
                            config.a_mat_kitti_train, config.kitti_max_depth_m,
                            config.kitti_beta_scale, augment=True)
    split_idx = int(config.train_split_ratio * len(frames))
    mode = getattr(config, 'kitti_train_mode', 'all')
    if mode == 'subset':
        path = _resolve_subset_path(config)
        idx = np.asarray(np.load(path), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < split_idx)]
        if idx.size == 0:
            raise ValueError("kitti_train_mode='subset' but '%s' has no indices in [0, %d)." % (path, split_idx))
        indices = idx.tolist()
        print("[data.kitti] train mode=subset  file=%s  using %d/%d frames" % (path, len(indices), split_idx))
    else:
        indices = list(range(0, split_idx))
        print("[data.kitti] train mode=all  using full train split (%d frames)" % split_idx)
    return DataLoader(Subset(dataset, indices), batch_size=config.batch_size_make3d,
                      shuffle=True, num_workers=config.num_workers, drop_last=True,
                      pin_memory=True, persistent_workers=config.num_workers > 0)


def get_val_loader(config):
    """Held-out tail of the train split (no augmentation) for early stopping."""
    frames = _train_frames(config)
    dataset = _KittiDataset(frames, config.kitti_gt_train_dir, config.beta_mat_kitti_train,
                            config.a_mat_kitti_train, config.kitti_max_depth_m,
                            config.kitti_beta_scale, augment=False)
    split_idx = int(config.train_split_ratio * len(frames))
    val_idx = list(range(split_idx, len(frames)))
    return DataLoader(Subset(dataset, val_idx), batch_size=config.batch_size_make3d,
                      shuffle=False, num_workers=config.num_workers, drop_last=False,
                      pin_memory=True, persistent_workers=config.num_workers > 0)


def get_test_loader(config):
    """KITTI test loader. ``kitti_test_mode``: 'tail' -> held-out tail of train
    (its GT lives in kitti_gt_train_dir); 'official' -> the val split (its own
    params + kitti_gt_test_dir)."""
    mode = getattr(config, 'kitti_test_mode', 'tail')
    if mode == 'official':
        frames = list_completed_frames('val')
        dataset = _KittiDataset(frames, config.kitti_gt_test_dir, config.beta_mat_kitti_test,
                                config.a_mat_kitti_test, config.kitti_max_depth_m,
                                config.kitti_beta_scale, augment=False)
        print("[data.kitti] test mode=official  using %d val frames" % len(frames))
        return DataLoader(dataset, batch_size=config.batch_size_make3d, shuffle=False,
                          num_workers=config.num_workers, drop_last=False)

    frames = _train_frames(config)
    dataset = _KittiDataset(frames, config.kitti_gt_train_dir, config.beta_mat_kitti_train,
                            config.a_mat_kitti_train, config.kitti_max_depth_m,
                            config.kitti_beta_scale, augment=False)
    split_idx = int(config.train_split_ratio * len(frames))
    test_idx = list(range(split_idx, len(frames)))
    print("[data.kitti] test mode=tail  using held-out tail (%d frames)" % len(test_idx))
    return DataLoader(Subset(dataset, test_idx), batch_size=config.batch_size_make3d,
                      shuffle=False, num_workers=config.num_workers, drop_last=False)


# ===========================================================================
# Legacy: Velodyne calibration + projection (kept for backward compatibility)
# ===========================================================================

def _read_calib_file(path):
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
    velo = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    velo = velo[velo[:, 0] > 0]
    pts = np.concatenate([velo[:, :3], np.ones((len(velo), 1))], axis=1).T
    cam = calib['P'] @ calib['R_rect'] @ calib['Tr'] @ pts
    z = cam[2]
    u = cam[0] / z
    v = cam[1] / z
    m = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[m].astype(np.int32), v[m].astype(np.int32), z[m]
    depth = np.zeros((height, width), dtype=np.float32)
    order = np.argsort(-z)
    depth[v[order], u[order]] = z[order]
    return depth


def list_frames(kitti_raw_dir=None):
    """Legacy: raw image_02 frames that have a matching velodyne .bin."""
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
    if calib_cache is None:
        calib_cache = {}
    dd = frame['date_dir']
    if dd not in calib_cache:
        calib_cache[dd] = load_calib(dd)
    img = cv2.imread(frame['image'])
    h, w = img.shape[:2]
    return project_velodyne_to_depth(frame['velo'], calib_cache[dd], w, h)
