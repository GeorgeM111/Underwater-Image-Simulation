import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from scipy import io
from numpy import load

from custom_transforms import Augmentation

# ── Spatial dimensions ────────────────────────────────────────────────────────
FULL_W, FULL_H = 460, 345
HALF_W, HALF_H = 230, 173

# ── Depth scale ───────────────────────────────────────────────────────────────
MAX_DEPTH_M = 80.0

# ── num_workers ───────────────────────────────────────────────────────────────
NUM_WORKERS = 0 if sys.platform == 'win32' else 8

# ── Paths ─────────────────────────────────────────────────────────────────────
MAKE3D_ROOT = r'C:\home\Georges\Make3D'

TRAIN_IMG_DIR   = os.path.join(MAKE3D_ROOT, 'Train400Img')
TRAIN_DEPTH_DIR = os.path.join(MAKE3D_ROOT, 'Train400Depth')
TRAIN_CACHE_DIR = r'C:\home\Georges\data\save_data_water\all_data_make_3D'

TEST_IMG_DIR    = os.path.join(MAKE3D_ROOT, 'Test134')
TEST_DEPTH_DIR  = os.path.join(MAKE3D_ROOT, 'Test134Depth', 'Gridlaserdata')
TEST_CACHE_DIR  = r'C:\home\Georges\data\save_data_water\all_data_make_3D_test'

NPY_ROOT       = r'C:\home\Georges\DenseDepth_1'

BETA_MAT_TRAIN = os.path.join(NPY_ROOT, 'Beta_Mat_Make_3D.npy')
A_MAT_TRAIN    = os.path.join(NPY_ROOT, 'A_Mat_Make_3D.npy')
BETA_MAT_TEST  = os.path.join(NPY_ROOT, 'Beta_Mat_Make_3D_Test.npy')
A_MAT_TEST     = os.path.join(NPY_ROOT, 'A_Mat_Make_3D_Test.npy')


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_reorganize_dimension(data, m, n):
    data = np.reshape(np.array(data, dtype=np.float32), [3, 1, 1])
    data = np.tile(data, [1, m, n])
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 0, 1)
    return data


def hwc_to_chw(arr):
    return np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32)


def depth_to_tensor(depth_metres):
    d = torch.from_numpy(depth_metres).float()
    d = d / MAX_DEPTH_M * 1000.0
    d = torch.clamp(d, 10.0, 1000.0)
    return d.unsqueeze(0)


# ── Train dataset ─────────────────────────────────────────────────────────────

class Make3DDataset(Dataset):
    """
    Make3D training dataset.

    Cache files expected at TRAIN_CACHE_DIR:
        {idx}haze_image.npy               shape (HALF_H, HALF_W, 3)
        {idx}complex_haze_image_name.npy  shape (HALF_H, HALF_W, 3)
    (filename matches original data_4_Make_3D.py convention)
    """

    def __init__(self, transforms=None):
        self.transforms = transforms

        train_images = glob.glob(f'{TRAIN_IMG_DIR}/*.jpg')
        train_depth  = glob.glob(f'{TRAIN_DEPTH_DIR}/*.mat')

        self.train_images = sorted(
            train_images, key=lambda p: p.split('/')[-1].split('img-')[-1])
        self.train_depth  = sorted(
            train_depth,  key=lambda p: p.split('/')[-1].split('depth_sph_corr-')[-1])

        self.beta_mat_arr = load(BETA_MAT_TRAIN)
        self.a_mat_arr    = load(A_MAT_TRAIN)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        # FIX: filename matches original data_4_Make_3D.py — note _name suffix
        haze_image    = load(f'{TRAIN_CACHE_DIR}\\{idx}haze_image.npy')
        complex_noisy = load(f'{TRAIN_CACHE_DIR}\\{idx}complex_haze_image.npy')

        H, W = haze_image.shape[0], haze_image.shape[1]

        image_full = cv2.imread(self.train_images[idx])
        image_full = cv2.resize(image_full, (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)
        image_half = cv2.resize(image_full, (HALF_W, HALF_H), interpolation=cv2.INTER_LINEAR)

        depth_full = io.loadmat(self.train_depth[idx])['Position3DGrid'][:, :, 3]
        depth_full = cv2.resize(depth_full.astype(np.float32), (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)
        depth_half = cv2.resize(depth_full, (HALF_W, HALF_H), interpolation=cv2.INTER_LINEAR)
        del depth_full

        if self.transforms is not None:
            image_full, _             = self.transforms(image_full, depth_half)
            image_half, depth_half    = self.transforms(image_half, depth_half)
            haze_image, complex_noisy = self.transforms(haze_image, complex_noisy)

        beta_spatial = create_reorganize_dimension(self.beta_mat_arr[idx], H, W)
        a_spatial    = create_reorganize_dimension(self.a_mat_arr[idx],    H, W)
        unit_spatial = create_reorganize_dimension([1.0, 1.0, 1.0],        H, W)

        img_full_t  = torch.from_numpy(hwc_to_chw(image_full)).float() / 255.0
        img_half_t  = torch.from_numpy(hwc_to_chw(image_half)).float() / 255.0
        haze_t      = torch.from_numpy(hwc_to_chw(haze_image)).float() / 255.0
        complex_t   = torch.from_numpy(hwc_to_chw(complex_noisy)).float() / 255.0
        beta_t      = torch.from_numpy(hwc_to_chw(beta_spatial))
        a_t         = torch.from_numpy(hwc_to_chw(a_spatial))
        unit_t      = torch.from_numpy(hwc_to_chw(unit_spatial))
        depth_t     = depth_to_tensor(depth_half)

        img_full_t.requires_grad = False
        img_half_t.requires_grad = False
        depth_t.requires_grad    = False

        return {
            'image_full':        img_full_t,
            'image_half':        img_half_t,
            'depth_half':        depth_t,
            'haze_image':        haze_t,
            'beta':              beta_t,
            'a_val':             a_t,
            'unit_mat':          unit_t,
            'complex_noise_img': complex_t,
        }


# ── Test dataset ──────────────────────────────────────────────────────────────

class Make3DDatasetTest(Dataset):

    def __init__(self):
        test_images = glob.glob(f'{TEST_IMG_DIR}/*.jpg')
        test_depth  = glob.glob(f'{TEST_DEPTH_DIR}/*.mat')

        self.test_images = sorted(
            test_images, key=lambda p: p.split('/')[-1].split('img-')[-1])
        self.test_depth  = sorted(
            test_depth,  key=lambda p: p.split('/')[-1].split('depth_sph_corr-')[-1])

        self.beta_mat_arr = load(BETA_MAT_TEST)
        self.a_mat_arr    = load(A_MAT_TEST)

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        # FIX: filename matches original data_4_Make_3D_Test.py — note _name suffix
        haze_image    = load(f'{TEST_CACHE_DIR}\\{idx}haze_image.npy')
        complex_noisy = load(f'{TEST_CACHE_DIR}\\{idx}complex_haze_image.npy')

        H, W = haze_image.shape[0], haze_image.shape[1]

        image_full = cv2.imread(self.test_images[idx])
        image_full = cv2.resize(image_full, (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)
        image_half = cv2.resize(image_full, (HALF_W, HALF_H), interpolation=cv2.INTER_LINEAR)

        depth_full = io.loadmat(self.test_depth[idx])['Position3DGrid'][:, :, 3]
        depth_full = cv2.resize(depth_full.astype(np.float32), (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)
        depth_half = cv2.resize(depth_full, (HALF_W, HALF_H), interpolation=cv2.INTER_LINEAR)
        del depth_full

        beta_spatial = create_reorganize_dimension(self.beta_mat_arr[idx], H, W)
        a_spatial    = create_reorganize_dimension(self.a_mat_arr[idx],    H, W)
        unit_spatial = create_reorganize_dimension([1.0, 1.0, 1.0],        H, W)

        img_full_t  = torch.from_numpy(hwc_to_chw(image_full)).float() / 255.0
        img_half_t  = torch.from_numpy(hwc_to_chw(image_half)).float() / 255.0
        haze_t      = torch.from_numpy(hwc_to_chw(haze_image)).float() / 255.0
        complex_t   = torch.from_numpy(hwc_to_chw(complex_noisy)).float() / 255.0
        beta_t      = torch.from_numpy(hwc_to_chw(beta_spatial))
        a_t         = torch.from_numpy(hwc_to_chw(a_spatial))
        unit_t      = torch.from_numpy(hwc_to_chw(unit_spatial))
        depth_t     = depth_to_tensor(depth_half)

        return {
            'image_full':        img_full_t,
            'image_half':        img_half_t,
            'depth_half':        depth_t,
            'haze_image':        haze_t,
            'beta':              beta_t,
            'a_val':             a_t,
            'unit_mat':          unit_t,
            'complex_noise_img': complex_t,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def getTrainingTestingData(batch_size):
    dataset = Make3DDataset(transforms=Augmentation())
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=NUM_WORKERS, drop_last=True)


def getTestingData(batch_size):
    dataset = Make3DDatasetTest()
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)