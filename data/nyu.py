"""NYU Depth V2 dataset + dataloaders (single canonical copy).

Consolidates the NYU logic previously spread across data_2.py / data_3.py /
data_4.py in every technique folder.

Public interface:
    get_train_loader(config) -> DataLoader
    get_test_loader(config)  -> DataLoader

The paper holds out the last ``(1 - train_split_ratio)`` of the data for
testing; this is a deterministic index split (train = [0, split_idx),
test = [split_idx, end)).
"""

import os
from io import BytesIO

import numpy as np
from numpy import load
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.utils import shuffle

from utils.physics import compute_complex_noise  # noqa: F401  (kept for GT parity / reuse)
from utils.transforms import getDefaultTrainTransform, getNoTransform


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def loadZipToMem(zip_file, csv='data/nyu2_train.csv'):
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    rows = list((row.split(',') for row in (data[csv]).decode("utf-8").split('\n') if len(row) > 0))
    rows = shuffle(rows, random_state=0)
    print('Loaded ({0}).'.format(len(rows)))
    return data, rows


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, beta_mat_arr, a_mat_arr, gt_dir, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.beta_mat_arr = beta_mat_arr
        self.a_mat_arr = a_mat_arr
        self.gt_dir = gt_dir
        self.transform = transform

    def __getitem__(self, idx):
        haze_image_name = os.path.join(self.gt_dir, str(idx) + "haze_image" + ".npy")
        complex_haze_image_name = os.path.join(self.gt_dir, str(idx) + "complex_haze_image" + ".npy")

        haze_image = load(haze_image_name)
        complex_noisy_img = load(complex_haze_image_name)

        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))

        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        image_full, image_half, depth_half_10_1000, depth_half_0_1 = (
            sample['image_norm'], sample['image_half_norm'],
            sample['depth_half_norm_10_1000'], sample['depth_half_norm_0_1'])

        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]
        del depth_half_0_1

        beta_mat = self.beta_mat_arr[idx]
        beta_mat_mod = self.create_reorganize_dimension(beta_mat, m, n)

        a_mat = self.a_mat_arr[idx]
        a_mat_mod = self.create_reorganize_dimension(a_mat, m, n)

        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = self.create_reorganize_dimension(unit_mat, m, n)

        image_half_numpy = np.array(image_half)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)

        del a_mat, beta_mat

        image_half_numpy = self.only_reorganize_dimension(image_half_numpy)
        haze_image = self.only_reorganize_dimension(haze_image)
        a_mat_mod = self.only_reorganize_dimension(a_mat_mod)
        beta_mat_mod = self.only_reorganize_dimension(beta_mat_mod)
        unit_mat = self.only_reorganize_dimension(unit_mat)
        complex_noisy_img = self.only_reorganize_dimension(complex_noisy_img)

        image_half_tensor = torch.from_numpy(image_half_numpy)
        haze_image_tensor = torch.from_numpy(haze_image)
        a_mat_mod = torch.from_numpy(a_mat_mod)
        beta_mat_mod = torch.from_numpy(beta_mat_mod)
        unit_mat = torch.from_numpy(unit_mat)
        complex_image_tensor = torch.from_numpy(complex_noisy_img)

        del complex_noisy_img

        return {'image_full': image_full, 'image_half': image_half_tensor, 'depth': depth_half_10_1000,
                'haze_image': haze_image_tensor, 'beta': beta_mat_mod,
                'a_val': a_mat_mod, 'unit_mat': unit_mat, 'complex_noise_img': complex_image_tensor}

    def __len__(self):
        return len(self.nyu_dataset)

    def create_reorganize_dimension(self, data, m, n):
        data = np.reshape(data, [3, 1, 1])
        data = np.tile(data, [1, m, n])
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 0, 1)
        return data

    def only_reorganize_dimension(self, data):
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 1, 2)
        return data


def _build_full_dataset(config, transform):
    data, nyu2_train = loadZipToMem(config.nyu_zip_path)
    beta_mat_arr = load(config.beta_mat_nyu_train)
    a_mat_arr = load(config.a_mat_nyu_train)
    return depthDatasetMemory(data, nyu2_train, beta_mat_arr, a_mat_arr,
                              gt_dir=config.nyu_gt_train_dir, transform=transform)


def get_train_loader(config):
    full = _build_full_dataset(config, getDefaultTrainTransform())
    split_idx = int(config.train_split_ratio * len(full))
    train_subset = Subset(full, list(range(0, split_idx)))
    return DataLoader(train_subset, config.batch_size_nyu, shuffle=True, drop_last=True,
                      num_workers=config.num_workers)


def get_test_loader(config):
    full = _build_full_dataset(config, getNoTransform(is_test=True))
    split_idx = int(config.train_split_ratio * len(full))
    test_subset = Subset(full, list(range(split_idx, len(full))))
    return DataLoader(test_subset, config.batch_size_nyu, shuffle=False, drop_last=True,
                      num_workers=config.num_workers)
