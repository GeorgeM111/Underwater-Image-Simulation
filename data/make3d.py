"""Make3D dataset + dataloaders (single canonical copy).

Consolidates the Make3D logic previously in data_4_Make_3D.py / data_make3d.py.

Public interface:
    get_train_loader(config) -> DataLoader
    get_test_loader(config)  -> DataLoader
"""

import os
import glob
import random

import numpy as np
from numpy import load
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
from scipy import io

# Spatial dimensions
FULL_W, FULL_H = 460, 345
HALF_W, HALF_H = 230, 173


def create_reorganize_dimension(data, m, n):
    data = np.reshape(np.array(data, dtype=np.float32), [3, 1, 1])
    data = np.tile(data, [1, m, n])
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 0, 1)
    return data


def hwc_to_chw(arr):
    return np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32)


def _depth_to_tensor(depth_metres, max_depth_m):
    d = torch.from_numpy(depth_metres).float()
    d = d / max_depth_m * 1000.0
    d = torch.clamp(d, 10.0, 1000.0)
    return d.unsqueeze(0)


class _Make3DDataset(Dataset):
    """Make3D dataset (train or test split), driven entirely by config paths."""

    def __init__(self, img_dir, depth_dir, cache_dir, beta_path, a_path,
                 max_depth_m, augment=False):
        self.cache_dir = cache_dir
        self.max_depth_m = max_depth_m
        self.augment = augment

        images = glob.glob(os.path.join(img_dir, '*.jpg'))
        depths = glob.glob(os.path.join(depth_dir, '*.mat'))
        self.images = sorted(images, key=lambda p: os.path.basename(p).split('img-')[-1])
        self.depths = sorted(depths, key=lambda p: os.path.basename(p).split('depth_sph_corr-')[-1])

        self.beta_mat_arr = load(beta_path)
        self.a_mat_arr = load(a_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Cached GT is already RGB float in [0, 1] (see data_generation/data_2.py).
        haze_image = load(os.path.join(self.cache_dir, str(idx) + 'haze_image.npy'))
        complex_noisy = load(os.path.join(self.cache_dir, str(idx) + 'complex_haze_image.npy'))

        H, W = haze_image.shape[0], haze_image.shape[1]

        image_full = cv2.imread(self.images[idx])
        image_full = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)  # cv2 is BGR; pipeline is RGB
        image_full = cv2.resize(image_full, (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)
        image_half = cv2.resize(image_full, (HALF_W, HALF_H), interpolation=cv2.INTER_LINEAR)

        depth_full = io.loadmat(self.depths[idx])['Position3DGrid'][:, :, 3]
        depth_full = cv2.resize(depth_full.astype(np.float32), (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)
        depth_half = cv2.resize(depth_full, (HALF_W, HALF_H), interpolation=cv2.INTER_LINEAR)
        del depth_full

        # To tensors. Images are uint8 [0,255] -> /255; cached GT is already [0,1].
        image_full = torch.from_numpy(hwc_to_chw(image_full)).float() / 255.0
        image_half = torch.from_numpy(hwc_to_chw(image_half)).float() / 255.0
        haze_image = torch.from_numpy(hwc_to_chw(haze_image)).float()
        complex_noisy = torch.from_numpy(hwc_to_chw(complex_noisy)).float()
        depth_t = _depth_to_tensor(depth_half, self.max_depth_m)

        # Augmentation: a SINGLE shared horizontal flip applied to every tensor so
        # image / depth / haze / complex stay pixel-aligned and the input<->GT
        # relationship is preserved. (The previous code ran independent random
        # photometric+geometric transforms on each, scrambling the pairing and
        # corrupting the [0,1] GT with [0,255]-scale brightness/contrast jitter.)
        if self.augment and random.random() < 0.5:
            image_full = torch.flip(image_full, dims=[2])
            image_half = torch.flip(image_half, dims=[2])
            haze_image = torch.flip(haze_image, dims=[2])
            complex_noisy = torch.flip(complex_noisy, dims=[2])
            depth_t = torch.flip(depth_t, dims=[2])

        beta_spatial = create_reorganize_dimension(self.beta_mat_arr[idx], H, W)
        a_spatial = create_reorganize_dimension(self.a_mat_arr[idx], H, W)
        unit_spatial = create_reorganize_dimension([1.0, 1.0, 1.0], H, W)

        return {
            'image_full': image_full,
            'image_half': image_half,
            'depth': depth_t,
            'haze_image': haze_image,
            'beta': torch.from_numpy(hwc_to_chw(beta_spatial)),
            'a_val': torch.from_numpy(hwc_to_chw(a_spatial)),
            'unit_mat': torch.from_numpy(hwc_to_chw(unit_spatial)),
            'complex_noise_img': complex_noisy,
        }


def _train_dataset(config, augment):
    return _Make3DDataset(
        img_dir=config.make3d_train_img_dir,
        depth_dir=config.make3d_train_depth_dir,
        cache_dir=config.make3d_save_dir,
        beta_path=config.beta_mat_make3d_train,
        a_path=config.a_mat_make3d_train,
        max_depth_m=config.make3d_max_depth_m,
        augment=augment,
    )


def _split_indices(n, ratio):
    """Deterministic train/val index split (first ``ratio`` -> train, rest -> val)."""
    k = int(ratio * n)
    return list(range(0, k)), list(range(k, n))


def get_train_loader(config):
    # Train on the first ``train_split_ratio`` of Train400 (augmented); the tail
    # is held out for validation / early stopping (see get_val_loader).
    dataset = _train_dataset(config, augment=True)
    train_idx, _ = _split_indices(len(dataset), config.train_split_ratio)
    return DataLoader(Subset(dataset, train_idx), batch_size=config.batch_size_make3d,
                      shuffle=True, num_workers=config.num_workers, drop_last=True)


def get_val_loader(config):
    # Held-out validation split (no augmentation, deterministic order).
    dataset = _train_dataset(config, augment=False)
    _, val_idx = _split_indices(len(dataset), config.train_split_ratio)
    return DataLoader(Subset(dataset, val_idx), batch_size=config.batch_size_make3d,
                      shuffle=False, num_workers=config.num_workers, drop_last=False)


def get_test_loader(config):
    dataset = _Make3DDataset(
        img_dir=config.make3d_test_img_dir,
        depth_dir=config.make3d_test_depth_dir,
        cache_dir=config.make3d_test_save_dir,
        beta_path=config.beta_mat_make3d_test,
        a_path=config.a_mat_make3d_test,
        max_depth_m=config.make3d_max_depth_m,
        augment=False,
    )
    return DataLoader(dataset, batch_size=config.batch_size_make3d, shuffle=False,
                      num_workers=config.num_workers, drop_last=False)
