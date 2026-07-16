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
import random
from io import BytesIO

import numpy as np
from numpy import load
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.utils import shuffle

from config import CONFIG
from utils.physics import compute_complex_noise  # noqa: F401  (kept for GT parity / reuse)
from utils.transforms import getDefaultTrainTransform, getNoTransform
from utils.provenance import check_physics_manifest


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# The NYU archive is ~4.4 GB and loadZipToMem holds it entirely in RAM. A trainer now
# builds a train AND a val (AND possibly a test) dataset, so without this cache the same
# archive would be read into memory two or three times over.
_ZIP_CACHE = {}


def loadZipToMem(zip_file, csv='data/nyu2_train.csv'):
    key = (zip_file, csv)
    if key in _ZIP_CACHE:
        return _ZIP_CACHE[key]
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    # splitlines(), NOT split('\n'). The CSVs inside the zip do not all use LF: nyu2_test.csv
    # is CRLF, so split('\n') leaves a trailing '\r' glued to the last field of every row and
    # every subsequent zip lookup dies with KeyError: 'data/nyu2_test/01216_depth.png\r'.
    # splitlines() handles LF, CRLF and lone CR identically. (Row ORDER is unchanged, so the
    # shuffle(random_state=0) and every positional beta/A/GT index stay exactly as before.)
    rows = list((row.split(',') for row in (data[csv]).decode("utf-8").splitlines() if len(row) > 0))
    rows = shuffle(rows, random_state=0)
    print('Loaded ({0}).'.format(len(rows)))
    _ZIP_CACHE[key] = (data, rows)
    return _ZIP_CACHE[key]


def _load_gt(path):
    """Load a ground-truth array as float32 in [0, 1], whatever it was stored as.

    The GT generator now writes uint8 (3.2x smaller, zero information loss at these
    amplitudes) but older dirs hold float64 already in [0, 1]. Without this dtype guard
    the new loader would divide a pre-existing FLOAT gt by 255 a second time, producing a
    near-black target that trains happily and silently. Dispatch on dtype, never on
    assumption.
    """
    arr = load(path)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, beta_mat_arr, a_mat_arr, gt_dir, transform=None,
                 augment=False):
        self.data, self.nyu_dataset = data, nyu2_train
        self.beta_mat_arr = beta_mat_arr
        self.a_mat_arr = a_mat_arr
        self.gt_dir = gt_dir
        self.transform = transform
        # Shared horizontal flip applied to the input AND every pre-computed GT
        # target together, so input<->GT stay pixel-aligned (unlike a channel swap,
        # which cannot be mirrored onto the cached colour GT). beta/A are spatially
        # uniform, so they need no flip.
        self.augment = augment

    def __getitem__(self, idx):
        haze_image_name = os.path.join(self.gt_dir, str(idx) + "haze_image" + ".npy")
        complex_haze_image_name = os.path.join(self.gt_dir, str(idx) + "complex_haze_image" + ".npy")

        haze_image = _load_gt(haze_image_name)
        complex_noisy_img = _load_gt(complex_haze_image_name)

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

        # beta is scaled by *_beta_scale (attenuation), NOT clarity (scattering). The GT
        # generator uses the same key, so t = exp(-beta*z) matches exactly.
        beta_mat = np.asarray(self.beta_mat_arr[idx], dtype=np.float32) * CONFIG.nyu_beta_scale
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

        # Cast everything to float32. The Beta/A parameter matrices and unit_mat
        # are float64, which would otherwise make the physics output Double and
        # clash with the float32 SSIM window / model weights.
        image_half_tensor = torch.from_numpy(image_half_numpy).float()
        haze_image_tensor = torch.from_numpy(haze_image).float()
        a_mat_mod = torch.from_numpy(a_mat_mod).float()
        beta_mat_mod = torch.from_numpy(beta_mat_mod).float()
        unit_mat = torch.from_numpy(unit_mat).float()
        complex_image_tensor = torch.from_numpy(complex_noisy_img).float()

        del complex_noisy_img

        image_full = image_full.float()
        depth_half_10_1000 = depth_half_10_1000.float()

        # Aligned augmentation: flip input and every GT target on the width axis
        # together (a channel swap cannot be applied to the cached colour GT).
        if self.augment and random.random() < 0.5:
            image_full = torch.flip(image_full, dims=[-1])
            image_half_tensor = torch.flip(image_half_tensor, dims=[-1])
            depth_half_10_1000 = torch.flip(depth_half_10_1000, dims=[-1])
            haze_image_tensor = torch.flip(haze_image_tensor, dims=[-1])
            complex_image_tensor = torch.flip(complex_image_tensor, dims=[-1])

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


def _build_full_dataset(config, transform, csv='data/nyu2_train.csv',
                        beta_path=None, a_path=None, gt_dir=None, augment=False,
                        required_indices=None):
    """Build the full NYU dataset for a given CSV / parameter-matrix / GT dir.

    Defaults reproduce the training split (nyu2_train.csv + *_NYU_train params +
    nyu_gt_train_dir); pass the test paths to build the official-654 test set.
    ``augment`` enables the aligned horizontal-flip augmentation (train only).
    """
    data, rows = loadZipToMem(config.nyu_zip_path, csv=csv)
    beta_mat_arr = load(beta_path or config.beta_mat_nyu_train)
    a_mat_arr = load(a_path or config.a_mat_nyu_train)
    resolved_gt = gt_dir or config.nyu_gt_train_dir
    # Pass the indices we are about to READ. The physics hash alone is NOT enough: a
    # subset-only regeneration stamps a fresh manifest while the test-TAIL files in the same
    # directory are still the old physics, so a hash-only check compares fresh-against-fresh
    # and waves the stale files through.
    check_physics_manifest(resolved_gt, config, required_indices=required_indices)
    return depthDatasetMemory(data, rows, beta_mat_arr, a_mat_arr,
                              gt_dir=resolved_gt, transform=transform,
                              augment=augment)


def _resolve_subset_path(config):
    """Path to the filtered-indices .npy used in 'subset' mode.

    Uses config.nyu_subset_indices if set, else the size-specific file
    ``{nyu_subset_size}_filtered_nyu.npy`` in the parameters directory.
    """
    p = getattr(config, 'nyu_subset_indices', None)
    if p:
        return p
    params_dir = os.path.dirname(config.beta_mat_nyu_train)
    return os.path.join(params_dir, "%d_filtered_nyu.npy" % config.nyu_subset_size)


def _training_pool(config, split_idx):
    """Indices available for training (never the held-out test tail)."""
    mode = getattr(config, 'nyu_train_mode', 'all')
    if mode == 'subset':
        path = _resolve_subset_path(config)
        idx = np.asarray(np.load(path), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < split_idx)]      # keep only training-pool indices
        if idx.size == 0:
            raise ValueError(
                "nyu_train_mode='subset' but '%s' has no indices in the training "
                "split [0, %d)." % (path, split_idx))
        # The subset file is ordered by information score (desc). Sort ascending by
        # dataset index so the train/val cut is NOT "val = the lowest-scoring images".
        # nyu2_train is already shuffled (random_state=0), so index order is scene-random.
        return sorted(idx.tolist()), "subset(%s)" % os.path.basename(path)
    return list(range(0, split_idx)), "all"


def _train_val_indices(config):
    """Deterministic train/val split of the training pool (mirrors data.make3d)."""
    data, rows = loadZipToMem(config.nyu_zip_path)
    split_idx = int(config.train_split_ratio * len(rows))
    pool, tag = _training_pool(config, split_idx)
    val_ratio = float(getattr(config, 'nyu_val_ratio', 0.05))
    k = int((1.0 - val_ratio) * len(pool))
    return pool[:k], pool[k:], tag, split_idx


def get_train_loader(config):
    """NYU training loader (augmented), excluding the held-out validation slice.

    ``config.nyu_train_mode``: 'all' -> full training split; 'subset' -> filtered
    indices. Either way indices are clipped to [0, split_idx) so the test tail can
    never leak in, and the last ``nyu_val_ratio`` of the pool is reserved for
    validation (same contract as data.make3d).
    """
    train_idx, val_idx, tag, split_idx = _train_val_indices(config)
    full = _build_full_dataset(config, getDefaultTrainTransform(), augment=True,
                               required_indices=train_idx)
    print("[data.nyu] train mode=%s  train=%d  val=%d  (pool clipped to [0,%d))"
          % (tag, len(train_idx), len(val_idx), split_idx))
    return DataLoader(Subset(full, train_idx), config.batch_size_nyu, shuffle=True, drop_last=True,
                      num_workers=config.num_workers, pin_memory=True,
                      persistent_workers=config.num_workers > 0)


def get_val_loader(config):
    """Held-out validation slice of the training pool (no augmentation, deterministic).

    Drives checkpoint selection + early stopping, exactly like data.make3d.get_val_loader.
    """
    _, val_idx, _, _ = _train_val_indices(config)
    full = _build_full_dataset(config, getNoTransform(), augment=False,
                               required_indices=val_idx)
    return DataLoader(Subset(full, val_idx), config.batch_size_nyu, shuffle=False, drop_last=False,
                      num_workers=config.num_workers, pin_memory=True,
                      persistent_workers=config.num_workers > 0)


def get_test_loader(config):
    """NYU test loader.

    ``config.nyu_test_mode`` selects the test set:
        'tail'     -> held-out tail of nyu2_train (indices [split_idx, end)); this
                      is the paper's 96%/4% protocol and the DEFAULT. Its GT lives
                      in nyu_gt_train_dir (produced by the train GT generation).
        'official' -> the official NYU-v2 654-image test set (nyu2_test.csv), with
                      its own params (*_NYU_test) and GT (nyu_gt_test_dir).
    """
    mode = getattr(config, 'nyu_test_mode', 'tail')
    if mode == 'official':
        full = _build_full_dataset(
            config, getNoTransform(is_test=True), csv='data/nyu2_test.csv',
            beta_path=config.beta_mat_nyu_test, a_path=config.a_mat_nyu_test,
            gt_dir=config.nyu_gt_test_dir)
        print("[data.nyu] test mode=official  using %d images (nyu2_test.csv)" % len(full))
        # drop_last=False: with True, the final partial batch was DISCARDED, so the
        # reported score was a function of batch_size_nyu — the same checkpoint scored
        # different image sets (and printed different numbers) at bs=10 vs bs=16.
        # AverageMeter already weights by the batch size, so a short last batch is exact.
        return DataLoader(full, config.batch_size_nyu, shuffle=False, drop_last=False,
                          num_workers=config.num_workers)

    # Resolve the tail indices BEFORE building, so the provenance guard can assert their GT
    # was actually written by the current physics (this is the exact set that goes stale when
    # the subset is regenerated without --with-test-tail).
    _data, _rows = loadZipToMem(config.nyu_zip_path)
    split_idx = int(config.train_split_ratio * len(_rows))
    tail_idx = list(range(split_idx, len(_rows)))
    full = _build_full_dataset(config, getNoTransform(is_test=True), required_indices=tail_idx)
    test_subset = Subset(full, tail_idx)
    print("[data.nyu] test mode=tail  using held-out tail (%d images)" % len(test_subset))
    return DataLoader(test_subset, config.batch_size_nyu, shuffle=False, drop_last=False,
                      num_workers=config.num_workers)
