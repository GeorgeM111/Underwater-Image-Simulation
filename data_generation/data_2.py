# --- repo-root path bootstrap (auto-added) ---
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image

import torch.nn.functional as F
from PIL import Image

from io import BytesIO
from sklearn.utils import shuffle
import random

from numpy import asarray
from numpy import save
from numpy import load

from scipy import io

from numpy import savez_compressed

import cv2
import glob

from config import CONFIG
from utils.provenance import (physics_fingerprint as _fingerprint,
                             write_physics_manifest as _write_manifest)
from utils.physics import compute_complex_noise
from utils.depth_range import DEPTH_CLIP_FRAC, DEPTH_CLAMP_MIN, DEPTH_CLAMP_MAX
#GM - define a menu of physically plausible water types that will be used to draw a value at random per image.
#GM - This accounts for the fact that water might appear different based on geography, seasons, depth, weather, etc..
beta_val_r = CONFIG.beta_val_r
beta_val_g = CONFIG.beta_val_g
beta_val_b = CONFIG.beta_val_b

# Directory holding the pre-computed A / Beta parameter matrices. Derived from
# the configured NYU train parameter-matrix path so all parameter .npy files
# live alongside it (replaces the old hard-coded /datas/.../parameters dir).
_PARAMS_DIR = _os.path.dirname(CONFIG.beta_mat_nyu_train)

# Physics provenance: the manifest hash lives in utils/provenance.py so the DATALOADERS can
# import the check without depending on this generation package (a guard that can silently
# fail to import is not a guard).
def physics_fingerprint(cfg=None):
    return _fingerprint(cfg if cfg is not None else CONFIG)


def write_physics_manifest(gt_dir, cfg=None, covered_indices=None):
    return _write_manifest(gt_dir, cfg if cfg is not None else CONFIG, covered_indices)


def _paired_make3d_files(img_dir, depth_dir):
    """Return (images, depths) paired by shared Make3D scene id.

    Make3D files are ``img-<id>.jpg`` / ``depth_sph_corr-<id>.mat``. Globbing and
    sorting the two lists INDEPENDENTLY and zipping them by position silently
    misaligns every pair if a single file is missing/extra; pairing by the shared
    id guarantees image[i] and depth[i] are the same scene. The order (sorted by
    id) is deterministic and identical to the loader, so the positional beta/A
    indexing stays consistent.
    """
    images = glob.glob(_os.path.join(img_dir, '*.jpg'))
    depths = glob.glob(_os.path.join(depth_dir, '*.mat'))

    def _img_id(p):
        return _os.path.basename(p).split('img-')[-1].rsplit('.', 1)[0]

    def _dep_id(p):
        return _os.path.basename(p).split('depth_sph_corr-')[-1].rsplit('.', 1)[0]

    dmap = {_dep_id(p): p for p in depths}
    paired_imgs, paired_deps = [], []
    for ip in sorted(images, key=_img_id):
        sid = _img_id(ip)
        if sid in dmap:
            paired_imgs.append(ip)
            paired_deps.append(dmap[sid])
    return paired_imgs, paired_deps


def _seed_params(offset=0):
    """Seed the RNGs so parameter-matrix (beta / atmospheric-light) generation is
    reproducible. A re-run then yields the SAME matrices, so the pre-computed GT
    can never silently desync from a regenerated matrix. Distinct offsets keep the
    four splits (NYU train/test, Make3D train/test) independent."""
    seed = int(getattr(CONFIG, 'random_seed', 42)) + offset
    random.seed(seed)
    np.random.seed(seed)


def _water_airlight(brightness, beta=None):
    """Water-coloured (red-depleted) veiling light for THIS image's water type.

    A GRAY airlight makes the haze term (1 - t) * A dump equal brightness into
    every channel, so the most-attenuated channel (red) becomes the BRIGHTEST ->
    a pink/gray veil. Real underwater veiling light is the water colour: ambient
    light that has travelled a background distance keeps the low-attenuation
    wavelengths and loses red. We derive that colour from the image's own water
    type ``beta`` (rescaled to the ricardo axis, the same coefficients the
    transmission uses), so the airlight matches the scene's water — blue for
    oceanic types, green for coastal ones. ``brightness`` (0-1) scales intensity.
    ``beta`` is the classical Jerlov triple; if None, falls back to complex_beta.
    """
    if beta is None:
        cb = np.asarray(CONFIG.complex_beta, dtype=np.float64)
    else:
        cb = np.asarray(beta, dtype=np.float64) * CONFIG.complex_beta_scale
    water = np.exp(-cb * CONFIG.airlight_bg_depth)   # [R, G, B]; red << surviving channel
    water = water / water.max()                       # brightest channel -> 1
    return (brightness * water).tolist()

# loader uses the transforms function that comes with torchvision
# here we will pre-compute the images and will save them from before. Later, we will just load the images inside the get_item() function

#GM - this loader isn't used so I commented it
# loader = transforms.Compose([
#     transforms.ToTensor()])
#GM- Used to convert a C*H*W tensor to a PIL image.
un_loader = transforms.ToPILImage()

#GM- checks if the input is a PIL image
def _is_pil_image(img):
    return isinstance(img, Image.Image)

#GM- checks if the input is a numpy image
def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

#GM- converts a C*H*W tensor to a PIL image
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0) #GM- remove batch dimension B if present (tensors here are either [B,C,H,W] or [C,H,W])
    image = un_loader(image)
    return image

#GM- a custom random horizontal flip is used to avoid either flipping the image without the depth or using two seperate horizontal flips which has a chance that one of the inputs won't flip
#GM - this guarantees that both inputs will either be flipped together or not. 
class RandomHorizontalFlipCustom(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

#GM- swap color channels- this is used to tell the model that the depth is independent from colors
class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        #GM -produce random permuations : RGB - RBG - BRG- BGR - GRB - GBR
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            #GM- convert PIL image to numpy H*W*3 uint8
            image = np.asarray(image)
            #GM - pick one of the permutations at random and reorder, then convert back to PIL
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


# Per-PROCESS cache of the decompressed NYU archive.
#
# The archive is ~4.4 GB and loadZipToMem holds it entirely in RAM. The GT generators call
# it ONCE PER CHUNK, so without this cache a chunked run re-reads and re-decompresses the
# whole 4.4 GB for every chunk: at --chunk-size 4 over a 10k subset that is 2500 full reloads
# and the job looks hung for hours while making almost no progress. data/nyu.py already had
# this cache; the generator did not.
#
# joblib's loky backend uses separate PROCESSES, so each worker still pays exactly one load
# (which is why n_parallel_jobs must stay small — each worker costs ~4.4 GB of RAM).
_ZIP_CACHE = {}


def _load_zip(zip_file, csv):
    key = (zip_file, csv)
    if key in _ZIP_CACHE:
        return _ZIP_CACHE[key]
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    # splitlines(): tolerate CRLF. A stray '\r' on the last field breaks every zip lookup.
    rows = list((row.split(',') for row in (data[csv]).decode("utf-8").splitlines()
                 if len(row) > 0))
    rows = shuffle(rows, random_state=0)
    print('Loaded ({0}).'.format(len(rows)))
    _ZIP_CACHE[key] = (data, rows)
    return _ZIP_CACHE[key]


def loadZipToMem(zip_file):
    return _load_zip(zip_file, 'data/nyu2_train.csv')


def loadZipToMemTest(zip_file):
    return _load_zip(zip_file, 'data/nyu2_test.csv')

#GM ------------------------------ BETA AND ATMOSPHERIC LIGHT START -------
#GM - NYU TRAIN
def generate_and_save_atmosphere_light_beta():
    _seed_params(0)
    _amin = CONFIG.nyu_airlight_brightness_min
    data, nyu2_train = loadZipToMem(CONFIG.nyu_zip_path)
    data_len = len(nyu2_train)
    #GM
    print(f"Loaded {data_len} items")
    del data, nyu2_train

    beta_arr = []
    a_mat_arr = []
    #GM - added the next line and the if statement in the loop
    idx = 0
    for x in range(data_len):
        if idx % 1000 == 0:
            print(f"Generating water type for {idx} - {idx+1000}...")
        # Sample ONE Jerlov water type for this image (blue-oceanic .. green-coastal),
        # instead of drawing each channel independently (which fixed the ordering).
        beta_mat = list(random.choice(CONFIG.jerlov_water_types))
        beta_arr.append(beta_mat)

        # a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        #GM - a gray ambient light that models the surface light only.
        #GM - color comes purely from the water wavelength dependent absorption beta.
        rand_val_a = random.uniform(_amin, 1.0)
        a_mat = _water_airlight(rand_val_a, beta_mat)   # airlight tinted by THIS image's water type

        a_mat_arr.append(a_mat)
        idx = idx+ 1
    save(_os.path.join(_PARAMS_DIR, 'A_Mat_NYU_train.npy'), a_mat_arr)
    save(_os.path.join(_PARAMS_DIR, 'Beta_Mat_NYU_train.npy'), beta_arr)
    #GM - added the next line
    print("Sucessfully generated Beta and Atmospheric Light matrices for NYU - Train.")

#GM - NYU TEST
def generate_and_save_atmosphere_light_beta_test():
    _seed_params(1)
    _amin = CONFIG.nyu_airlight_brightness_min
    data, nyu2_test = loadZipToMemTest(CONFIG.nyu_zip_path)
    data_len = len(nyu2_test)
    #GM
    print(f"Loaded {data_len} items")
    del data, nyu2_test

    beta_arr = []
    a_mat_arr = []
    for x in range(data_len):

        # Sample ONE Jerlov water type for this image (blue-oceanic .. green-coastal),
        # instead of drawing each channel independently (which fixed the ordering).
        beta_mat = list(random.choice(CONFIG.jerlov_water_types))
        beta_arr.append(beta_mat)

        # a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        rand_val_a = random.uniform(_amin, 1.0)
        a_mat = _water_airlight(rand_val_a, beta_mat)   # airlight tinted by THIS image's water type

        a_mat_arr.append(a_mat)

    save(_os.path.join(_PARAMS_DIR, 'A_Mat_NYU_test.npy'), a_mat_arr)
    save(_os.path.join(_PARAMS_DIR, 'Beta_Mat_NYU_test.npy'), beta_arr)
    #GM - 1 line
    print("Sucessfully generated Beta and Atmospheric Light matrices for NYU - Test.")

#GM - Make3D Train
def generate_and_save_atmosphere_light_beta_make_3D():
    _seed_params(2)
    _amin = CONFIG.make3d_airlight_brightness_min
    # Use the id-paired file list so the matrix length matches the GT count exactly.
    train_images, _ = _paired_make3d_files(CONFIG.make3d_train_img_dir, CONFIG.make3d_train_depth_dir)
    data_len = len(train_images)
    del train_images

    beta_arr = []
    a_mat_arr = []
    for x in range(data_len):

        # Sample ONE Jerlov water type for this image (blue-oceanic .. green-coastal),
        # instead of drawing each channel independently (which fixed the ordering).
        beta_mat = list(random.choice(CONFIG.jerlov_water_types))
        beta_arr.append(beta_mat)

        # a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        rand_val_a = random.uniform(_amin, 1.0)
        a_mat = _water_airlight(rand_val_a, beta_mat)   # airlight tinted by THIS image's water type

        a_mat_arr.append(a_mat)

    save(_os.path.join(_PARAMS_DIR, 'A_Mat_Make_3D_Train.npy'), a_mat_arr)
    save(_os.path.join(_PARAMS_DIR, 'Beta_Mat_Make_3D_Train.npy'), beta_arr)
    #GM - 1 line
    print("Sucessfully generated Beta and Atmospheric Light matrices for Make3D - Train.")

#GM - Make3D Test
def generate_and_save_atmosphere_light_beta_make_3D_Test():
    _seed_params(3)
    _amin = CONFIG.make3d_airlight_brightness_min
    # Use the id-paired file list so the matrix length matches the GT count exactly.
    train_images, _ = _paired_make3d_files(CONFIG.make3d_test_img_dir, CONFIG.make3d_test_depth_dir)
    data_len = len(train_images)
    del train_images

    beta_arr = []
    a_mat_arr = []
    for x in range(data_len):

        # Sample ONE Jerlov water type for this image (blue-oceanic .. green-coastal),
        # instead of drawing each channel independently (which fixed the ordering).
        beta_mat = list(random.choice(CONFIG.jerlov_water_types))
        beta_arr.append(beta_mat)

        # a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        rand_val_a = random.uniform(_amin, 1.0)
        a_mat = _water_airlight(rand_val_a, beta_mat)   # airlight tinted by THIS image's water type

        a_mat_arr.append(a_mat)

    save(_os.path.join(_PARAMS_DIR, 'A_Mat_Make_3D_Test.npy'), a_mat_arr)
    save(_os.path.join(_PARAMS_DIR, 'Beta_Mat_Make_3D_Test.npy'), beta_arr)
    #GM - 1 line
    print("Sucessfully generated Beta and Atmospheric Light matrices for Make3D - Test.")

#GM ------------------------------ BETA AND ATMOSPHERIC LIGHT END -------

#GM ------------------------------ GROUND TRUTH GENERATION START -------

#GM- NYU TRAIN
def generate_and_save_haze_image(stIndex, endIndex, indices=None):
    """Generate + save NYU haze/complex GT.

    If ``indices`` is given (an iterable of dataset indices), ONLY those images
    are generated (used to build GT for a filtered subset); ``stIndex``/``endIndex``
    are then ignored. Otherwise the contiguous range [stIndex, endIndex) is used.
    """
    _os.makedirs(CONFIG.nyu_gt_train_dir, exist_ok=True)
    data, nyu_dataset = loadZipToMem(CONFIG.nyu_zip_path)
    a_mat_arr = load(_os.path.join(_PARAMS_DIR, 'A_Mat_NYU_train.npy'))
    beta_mat_arr = load(_os.path.join(_PARAMS_DIR, 'Beta_Mat_NYU_train.npy'))
    # NOTE: the manifest is written by the ENTRY-POINT script (the parent), NOT here. This
    # function runs inside joblib worker PROCESSES, and the manifest's read-modify-write of
    # covered_indices would race across them and lose coverage. The parent writes it once, after
    # every chunk has completed, with the full index list.
    targets = list(indices) if indices is not None else list(range(stIndex, endIndex))
    for idx in targets:
        idx = int(idx)
        sample = nyu_dataset[idx]
        image = Image.open(BytesIO(data[sample[0]]))
        depth = Image.open(BytesIO(data[sample[1]]))
        sample_duo = {'image': image, 'depth': depth}

        sample = ToTensorCustom(sample_duo, False)

        # all the image matrices are normalized here within 0-1
        image_full, image_half, depth_half_10_1000, depth_half_0_1 = sample['image_norm'], \
            sample['image_half_norm'], sample['depth_half_norm_10_1000'], sample['depth_half_norm_0_1']

        # -> METRES, then clip to the paper's [0.4, 10] m target window.
        # Without the clip, missing-depth pixels (Kinect returns PNG 0) get z = 0 => t = 1
        # => the UN-DEGRADED original pixel is written straight into the underwater GT, and
        # it additionally picks up inbound scatter from its hazed neighbours -> clipped-to-
        # white blobs. This also puts the GT physics on exactly the same depth axis as the
        # network's training target (utils/transforms.DEPTH_CLAMP_MIN).
        depth_half_0_1 = torch.clamp(depth_half_0_1 * 10.0,
                                     DEPTH_CLIP_FRAC * CONFIG.nyu_max_depth_m,
                                     CONFIG.nyu_max_depth_m)
        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]

        depth_half_0_1_numpy = np.array(depth_half_0_1)
        del depth_half_0_1

        # need to modify the dimension ordering to use in opencv
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 2)
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 1)
        depth_half_0_1_3d = cv2.cvtColor(depth_half_0_1_numpy, cv2.COLOR_GRAY2RGB)  # transformed into 3 channels

        del depth_half_0_1_numpy

        beta_mat = beta_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)

        a_mat = a_mat_arr[idx]
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)

        tx1 = np.exp(-np.multiply(beta_mat_mod * CONFIG.nyu_water_clarity, depth_half_0_1_3d))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = create_reorganize_dimension_custom(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))

        image_half_numpy = np.array(image_half)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)  # making m * n *3
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)  # making m * n *3
        haze_image = np.add((np.multiply(image_half_numpy, tx1)), second_term)

        # seed=random_seed+idx: _turbid_v2 draws the Eq.6 particle field from the global
        # numpy RNG and NOTHING used to seed it, so the complex GT was not byte-reproducible
        # and a partially-regenerated directory silently mixed realisations. Now that the
        # particles are actually VISIBLE (u=0.90), that matters.
        complex_noisy_img = compute_complex_noise(image_half_numpy, depth_half_0_1_3d[:, :, 0]/10, beta_mat,
                                                  a_mat, max_depth_m=CONFIG.nyu_max_depth_m,
                                                  focal_px=CONFIG.nyu_focal_px,
                                                  clarity=CONFIG.nyu_water_clarity,
                                                  seed=int(getattr(CONFIG, 'random_seed', 42)) + idx)
        complex_noisy_img = complex_noisy_img/255

        del a_mat
        del beta_mat

        # asarray will help to convert the input into an array
        # keep_all_haze_image.append(asarray(haze_image))
        # keep_all_complex_haze_image.append(asarray(complex_noisy_img))
        # keep_all_depth_half_3d.append(asarray(depth_half_0_1_3d))

        asarray(haze_image)  # values are between 0-1
        asarray(complex_noisy_img)  # values are between 0-1
        asarray(depth)
        asarray(image)

        haze_image_name = _os.path.join(CONFIG.nyu_gt_train_dir, str(idx) + "haze_image" + ".npy")
        complex_haze_image_name = _os.path.join(CONFIG.nyu_gt_train_dir, str(idx) + "complex_haze_image" + ".npy")
        #orig_depth_image_name =  "/sandbox1/gmoussa/ground_truth/nyu/train/" +  str(idx) + "orig_depth_image" + ".npy"
        #orig_image_name =  "/sandbox1/gmoussa/ground_truth/nyu/train/" +  str(idx) + "orig_image" + ".npy"


        # float64 GT was 3.69 MB/pair (37 GB for the 10k subset) and the loader down-casts
        # to fp32 anyway. haze is a convex combination so it is exactly in [0,1] -> float32.
        # complex quantises to uint8 with np.rint (ROUND, not the old .astype(uint8) floor,
        # which biased the complex GT by -0.5 gray levels relative to the float haze GT and
        # made the residual head learn that constant offset as if it were physics).
        # data.nyu._load_gt dispatches on dtype, so old float dirs still load correctly.
        # BOTH as uint8. haze is a convex combination of J and A so it is exactly in [0,1];
        # 1/255 quantisation is a 0.004 noise floor, far below the ~0.02-0.05 MAE the model
        # achieves. Storage for the FULL 50688-image dataset: float32 haze would be 58 GB,
        # uint8 for both is 23 GB. np.rint = round (the old .astype(uint8) FLOORED, biasing the
        # target by -0.5 gray levels). data/nyu._load_gt dispatches on dtype, so old float dirs
        # still load correctly.
        save(haze_image_name,
             np.rint(np.clip(haze_image, 0.0, 1.0) * 255.0).astype(np.uint8))
        save(complex_haze_image_name,
             np.rint(np.clip(complex_noisy_img, 0.0, 1.0) * 255.0).astype(np.uint8))

        # print("I am done with image no : ", idx)

    # save to npy file
    # savez_compressed('all_haze_image.npz', keep_all_haze_image)
    # savez_compressed('all_complex_haze_image.npz', keep_all_complex_haze_image)
    # savez_compressed('all_complex_depth_half_3d.npz', keep_all_depth_half_3d)

#GM - NYU Test
def generate_and_save_haze_image_test(stIndex, endIndex):
    _os.makedirs(CONFIG.nyu_gt_test_dir, exist_ok=True)
    write_physics_manifest(CONFIG.nyu_gt_test_dir)
    data, nyu_dataset = loadZipToMemTest(CONFIG.nyu_zip_path)
    a_mat_arr = load(_os.path.join(_PARAMS_DIR, 'A_Mat_NYU_test.npy'))
    beta_mat_arr = load(_os.path.join(_PARAMS_DIR, 'Beta_Mat_NYU_test.npy'))

    # keep_all_haze_image = []
    # keep_all_complex_haze_image = []
    # keep_all_depth_half_3d = []
    # for idx in range(len(nyu_dataset)):
    for idx in range(stIndex, endIndex):
        sample = nyu_dataset[idx]
        image = Image.open(BytesIO(data[sample[0]]))
        depth = Image.open(BytesIO(data[sample[1]]))
        sample_duo = {'image': image, 'depth': depth}

        sample = ToTensorCustom(sample_duo, False)

        # all the image matrices are normalized here within 0-1
        image_full, image_half, depth_half_10_1000, depth_half_0_1 = sample['image_norm'], \
            sample['image_half_norm'], sample['depth_half_norm_10_1000'], sample['depth_half_norm_0_1']

        # -> METRES, clipped to the paper's [0.4, 10] m window (see the train generator).
        depth_half_0_1 = torch.clamp(depth_half_0_1 * 10.0,
                                     DEPTH_CLIP_FRAC * CONFIG.nyu_max_depth_m,
                                     CONFIG.nyu_max_depth_m)
        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]

        depth_half_0_1_numpy = np.array(depth_half_0_1)
        del depth_half_0_1

        # need to modify the dimension ordering to use in opencv
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 2)
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 1)
        depth_half_0_1_3d = cv2.cvtColor(depth_half_0_1_numpy, cv2.COLOR_GRAY2RGB)  # transformed into 3 channels

        del depth_half_0_1_numpy

        beta_mat = beta_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)

        a_mat = a_mat_arr[idx]
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)

        tx1 = np.exp(-np.multiply(beta_mat_mod * CONFIG.nyu_water_clarity, depth_half_0_1_3d))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = create_reorganize_dimension_custom(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))

        image_half_numpy = np.array(image_half)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)  # making m * n *3
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)  # making m * n *3
        haze_image = np.add((np.multiply(image_half_numpy, tx1)), second_term)

        # Distinct seed stream from the train split (+2**20) so the two never correlate.
        complex_noisy_img = compute_complex_noise(image_half_numpy, depth_half_0_1_3d[:, :, 0]/10, beta_mat,
                                                  a_mat, max_depth_m=CONFIG.nyu_max_depth_m,
                                                  focal_px=CONFIG.nyu_focal_px,
                                                  clarity=CONFIG.nyu_water_clarity,
                                                  seed=int(getattr(CONFIG, 'random_seed', 42)) + (1 << 20) + idx)
        complex_noisy_img = complex_noisy_img/255

        del a_mat
        del beta_mat

        # asarray will help to convert the input into an array
        # keep_all_haze_image.append(asarray(haze_image))
        # keep_all_complex_haze_image.append(asarray(complex_noisy_img))
        # keep_all_depth_half_3d.append(asarray(depth_half_0_1_3d))

        asarray(haze_image)  # values are between 0-1
        asarray(complex_noisy_img)  # values are between 0-1
        asarray(depth)
        asarray(image)
        
        haze_image_name = _os.path.join(CONFIG.nyu_gt_test_dir, str(idx) + "haze_image" + ".npy")
        complex_haze_image_name = _os.path.join(CONFIG.nyu_gt_test_dir, str(idx) + "complex_haze_image" + ".npy")
        # orig_depth_image_name =  "/sandbox1/gmoussa/ground_truth/nyu/test/" +  str(idx) + "orig_depth_image" + ".npy"
        # orig_image_name =  "/sandbox1/gmoussa/ground_truth/nyu/test/" +  str(idx) + "orig_image" + ".npy"

        # Same compact, round-not-floor encoding as the train split (see there).
        # BOTH as uint8. haze is a convex combination of J and A so it is exactly in [0,1];
        # 1/255 quantisation is a 0.004 noise floor, far below the ~0.02-0.05 MAE the model
        # achieves. Storage for the FULL 50688-image dataset: float32 haze would be 58 GB,
        # uint8 for both is 23 GB. np.rint = round (the old .astype(uint8) FLOORED, biasing the
        # target by -0.5 gray levels). data/nyu._load_gt dispatches on dtype, so old float dirs
        # still load correctly.
        save(haze_image_name,
             np.rint(np.clip(haze_image, 0.0, 1.0) * 255.0).astype(np.uint8))
        save(complex_haze_image_name,
             np.rint(np.clip(complex_noisy_img, 0.0, 1.0) * 255.0).astype(np.uint8))

#GM - Make3D Train
def generate_and_save_ricardo_image_make_3D(stIndex, endIndex):
    
    train_images, train_depth = _paired_make3d_files(
        CONFIG.make3d_train_img_dir, CONFIG.make3d_train_depth_dir)

    a_mat_arr = load(_os.path.join(_PARAMS_DIR, 'A_Mat_Make_3D_Train.npy'))
    beta_mat_arr = load(_os.path.join(_PARAMS_DIR, 'Beta_Mat_Make_3D_Train.npy'))

    for idx in range(stIndex, endIndex):

        image = cv2.imread(train_images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 loads BGR; keep the whole pipeline RGB
        depth = io.loadmat(train_depth[idx])['Position3DGrid'][:, :, 3]

        image = cv2.resize(image, tuple(CONFIG.make3d_full_size), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, tuple(CONFIG.make3d_full_size), interpolation=cv2.INTER_LINEAR)

        image_half = cv2.resize(image, tuple(CONFIG.make3d_half_size), interpolation=cv2.INTER_LINEAR)
        depth_half_m = cv2.resize(depth, tuple(CONFIG.make3d_half_size), interpolation=cv2.INTER_LINEAR)  # metres

        del image, depth

        image_half = torch.from_numpy(image_half).permute(2, 0, 1).float() / 255
        # Keep depth in METRES for the classical (per-metre Jerlov) haze model so
        # the attenuation matches NYU. The complex model self-normalises depth to
        # 0-255 internally, so it gets the [0,1]-scaled version.
        depth_half_m = torch.from_numpy(depth_half_m).float()
        depth_norm01 = depth_half_m / CONFIG.make3d_max_depth_m

        depth_half_m = torch.unsqueeze(depth_half_m, 0).repeat(3, 1, 1)
        depth_norm01 = torch.unsqueeze(depth_norm01, 0).repeat(3, 1, 1)

        depth_half_m.requires_grad = False
        depth_norm01.requires_grad = False
        image_half.requires_grad = False

        m = depth_half_m.shape[1]
        n = depth_half_m.shape[2]

        image_half = image_half.numpy()  # converting into numpy array
        image_half = np.swapaxes(image_half, 0, 2)  # making m * n *3
        image_half = np.swapaxes(image_half, 0, 1)  # making m * n *3

        depth_half_m = depth_half_m.numpy()
        depth_half_m = np.swapaxes(depth_half_m, 0, 2)
        depth_half_m = np.swapaxes(depth_half_m, 0, 1)

        depth_norm01 = depth_norm01.numpy()
        depth_norm01 = np.swapaxes(depth_norm01, 0, 2)
        depth_norm01 = np.swapaxes(depth_norm01, 0, 1)

        beta_mat = beta_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)

        a_mat = a_mat_arr[idx]
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)

        # Option B / water clarity: Make3D as CLEARER water — scale beta for the
        # transmission, keeping TRUE metric depth, so 0-80 m scenes don't go flat.
        tx1 = np.exp(-np.multiply(beta_mat_mod * CONFIG.make3d_water_clarity, depth_half_m))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = create_reorganize_dimension_custom(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))
        haze_image = np.add((np.multiply(image_half, tx1)), second_term)

        complex_noisy_img = compute_complex_noise(image_half, depth_norm01[:, :, 0], beta_mat,
                                                  a_mat, max_depth_m=CONFIG.make3d_max_depth_m,
                                                  focal_px=CONFIG.make3d_focal_px,
                                                  clarity=CONFIG.make3d_water_clarity)

        complex_noisy_img = complex_noisy_img/255

        del a_mat
        del beta_mat

        asarray(haze_image)  # values are between 0-1
        asarray(complex_noisy_img)  # values are between 0-1

        haze_image_name = _os.path.join(CONFIG.make3d_save_dir, str(idx) + "haze_image" + ".npy")
        # Filename matches the Make3D loader (data/make3d.py) and the NYU convention.
        complex_haze_image_name = _os.path.join(CONFIG.make3d_save_dir, str(idx) + "complex_haze_image" + ".npy")

        save(haze_image_name, haze_image)
        save(complex_haze_image_name, complex_noisy_img)

#GM - Make3DTest
def generate_and_save_ricardo_image_make_3D_Test(stIndex, endIndex):
    train_images, train_depth = _paired_make3d_files(
        CONFIG.make3d_test_img_dir, CONFIG.make3d_test_depth_dir)

    a_mat_arr = load(_os.path.join(_PARAMS_DIR, 'A_Mat_Make_3D_Test.npy'))
    beta_mat_arr = load(_os.path.join(_PARAMS_DIR, 'Beta_Mat_Make_3D_Test.npy'))

    for idx in range(stIndex, endIndex):

        image = cv2.imread(train_images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 loads BGR; keep the whole pipeline RGB
        depth = io.loadmat(train_depth[idx])['Position3DGrid'][:, :, 3]

        image = cv2.resize(image, tuple(CONFIG.make3d_full_size), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, tuple(CONFIG.make3d_full_size), interpolation=cv2.INTER_LINEAR)

        image_half = cv2.resize(image, tuple(CONFIG.make3d_half_size), interpolation=cv2.INTER_LINEAR)
        depth_half_m = cv2.resize(depth, tuple(CONFIG.make3d_half_size), interpolation=cv2.INTER_LINEAR)  # metres

        del image, depth

        image_half = torch.from_numpy(image_half).permute(2, 0, 1).float() / 255
        # Keep depth in METRES for the classical (per-metre Jerlov) haze model so
        # the attenuation matches NYU. The complex model self-normalises depth to
        # 0-255 internally, so it gets the [0,1]-scaled version.
        depth_half_m = torch.from_numpy(depth_half_m).float()
        depth_norm01 = depth_half_m / CONFIG.make3d_max_depth_m

        depth_half_m = torch.unsqueeze(depth_half_m, 0).repeat(3, 1, 1)
        depth_norm01 = torch.unsqueeze(depth_norm01, 0).repeat(3, 1, 1)

        depth_half_m.requires_grad = False
        depth_norm01.requires_grad = False
        image_half.requires_grad = False

        m = depth_half_m.shape[1]
        n = depth_half_m.shape[2]

        image_half = image_half.numpy()  # converting into numpy array
        image_half = np.swapaxes(image_half, 0, 2)  # making m * n *3
        image_half = np.swapaxes(image_half, 0, 1)  # making m * n *3

        depth_half_m = depth_half_m.numpy()
        depth_half_m = np.swapaxes(depth_half_m, 0, 2)
        depth_half_m = np.swapaxes(depth_half_m, 0, 1)

        depth_norm01 = depth_norm01.numpy()
        depth_norm01 = np.swapaxes(depth_norm01, 0, 2)
        depth_norm01 = np.swapaxes(depth_norm01, 0, 1)

        beta_mat = beta_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)

        a_mat = a_mat_arr[idx]
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)

        # Option B / water clarity: Make3D as CLEARER water — scale beta for the
        # transmission, keeping TRUE metric depth, so 0-80 m scenes don't go flat.
        tx1 = np.exp(-np.multiply(beta_mat_mod * CONFIG.make3d_water_clarity, depth_half_m))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = create_reorganize_dimension_custom(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))
        haze_image = np.add((np.multiply(image_half, tx1)), second_term)

        complex_noisy_img = compute_complex_noise(image_half, depth_norm01[:, :, 0], beta_mat,
                                                  a_mat, max_depth_m=CONFIG.make3d_max_depth_m,
                                                  focal_px=CONFIG.make3d_focal_px,
                                                  clarity=CONFIG.make3d_water_clarity)

        complex_noisy_img = complex_noisy_img/255

        del a_mat
        del beta_mat

        asarray(haze_image)  # values are between 0-1
        asarray(complex_noisy_img)  # values are between 0-1

        haze_image_name = _os.path.join(CONFIG.make3d_test_save_dir, str(idx) + "haze_image" + ".npy")
        # Filename matches the Make3D loader (data/make3d.py) and the NYU convention.
        complex_haze_image_name = _os.path.join(CONFIG.make3d_test_save_dir, str(idx) + "complex_haze_image" + ".npy")
        
        save(haze_image_name, haze_image)
        save(complex_haze_image_name, complex_noisy_img)



#GM ------------------------------ KITTI (completed-depth) START -------

def _kitti_gen_params(split, seed_offset, beta_name, a_name):
    """Sample one Jerlov water type + water-tinted airlight per KITTI frame, for the
    given completed-depth ``split`` ('train' or 'val'). Mirrors the Make3D generator."""
    from data.kitti import list_completed_frames
    _seed_params(seed_offset)
    _amin = CONFIG.kitti_airlight_brightness_min
    frames = list_completed_frames(split)
    data_len = len(frames)
    del frames
    print("KITTI %s frames: %d" % (split, data_len))
    beta_arr, a_mat_arr = [], []
    for x in range(data_len):
        if x % 5000 == 0:
            print("  params %d/%d" % (x, data_len))
        beta_mat = list(random.choice(CONFIG.jerlov_water_types))
        beta_arr.append(beta_mat)
        rand_val_a = random.uniform(_amin, 1.0)
        a_mat_arr.append(_water_airlight(rand_val_a, beta_mat))
    save(_os.path.join(_PARAMS_DIR, a_name), a_mat_arr)
    save(_os.path.join(_PARAMS_DIR, beta_name), beta_arr)
    print("Sucessfully generated Beta and Atmospheric Light matrices for KITTI - %s." % split)


def generate_and_save_atmosphere_light_beta_kitti():
    _kitti_gen_params('train', 4, 'Beta_Mat_KITTI_train.npy', 'A_Mat_KITTI_train.npy')


def generate_and_save_atmosphere_light_beta_kitti_test():
    _kitti_gen_params('val', 5, 'Beta_Mat_KITTI_test.npy', 'A_Mat_KITTI_test.npy')


def _kitti_generate_gt(split, save_dir, beta_name, a_name, stIndex, endIndex, indices=None):
    """Generate + save KITTI haze/complex GT for a range (or explicit ``indices``).

    Reads the raw image_02 frame and its DENSE completed depth (holes nearest-filled),
    resizes to kitti_half_size, and applies the SAME physics as Make3D: classical haze
    with the Option-B beta scale (true metric depth) + the ricardo complex model.
    """
    from data.kitti import list_completed_frames, load_frame_image_depth
    _os.makedirs(save_dir, exist_ok=True)
    frames = list_completed_frames(split)
    a_mat_arr = load(_os.path.join(_PARAMS_DIR, a_name))
    beta_mat_arr = load(_os.path.join(_PARAMS_DIR, beta_name))
    half = tuple(CONFIG.kitti_half_size)
    targets = list(indices) if indices is not None else range(stIndex, endIndex)
    for idx in targets:
        idx = int(idx)
        f = frames[idx]
        # Bottom-center crop to the LiDAR region + densify (shared helper, so the loader
        # and this generator can never disagree on the image/depth pair).
        image_full, depth_m = load_frame_image_depth(f, CONFIG.kitti_max_depth_m)

        image_half = cv2.resize(image_full, half, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0  # (H,W,3)
        depth_half_m = cv2.resize(depth_m, half, interpolation=cv2.INTER_LINEAR).astype(np.float32)           # (H,W) metres

        depth_norm01 = depth_half_m / CONFIG.kitti_max_depth_m       # (H,W) 0-1
        m, n = depth_half_m.shape
        depth_half_m_3d = np.repeat(depth_half_m[:, :, None], 3, axis=2)

        beta_mat = beta_mat_arr[idx]
        a_mat = a_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)
        unit_mat = create_reorganize_dimension_custom([1.0, 1.0, 1.0], m, n)

        # Water clarity: scale beta for the classical transmission (true metric depth).
        tx1 = np.exp(-np.multiply(beta_mat_mod * CONFIG.kitti_water_clarity, depth_half_m_3d))
        second_term = np.multiply(a_mat_mod, np.subtract(unit_mat, tx1))
        haze_image = np.add(np.multiply(image_half, tx1), second_term)

        complex_noisy_img = compute_complex_noise(
            image_half, depth_norm01, beta_mat, a_mat,
            max_depth_m=CONFIG.kitti_max_depth_m, focal_px=CONFIG.kitti_focal_px,
            clarity=CONFIG.kitti_water_clarity) / 255

        save(_os.path.join(save_dir, str(idx) + "haze_image.npy"), haze_image)
        save(_os.path.join(save_dir, str(idx) + "complex_haze_image.npy"), complex_noisy_img)


def generate_and_save_ricardo_image_kitti(stIndex, endIndex, indices=None):
    _kitti_generate_gt('train', CONFIG.kitti_gt_train_dir,
                       'Beta_Mat_KITTI_train.npy', 'A_Mat_KITTI_train.npy',
                       stIndex, endIndex, indices=indices)


def generate_and_save_ricardo_image_kitti_Test(stIndex, endIndex):
    _kitti_generate_gt('val', CONFIG.kitti_gt_test_dir,
                       'Beta_Mat_KITTI_test.npy', 'A_Mat_KITTI_test.npy',
                       stIndex, endIndex)

#GM ------------------------------ KITTI (completed-depth) END -------


def another_simple_image_save(image, path):
    image = image.astype(np.float32)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = image.astype(np.uint8)
    cv2.imwrite(path, image)



def testing_image_saving_code(self, idx):
    # idx = 49652  # **************************************

    haze_image_name = _os.path.join(CONFIG.nyu_save_dir, str(idx) + "haze_image" + ".npy")
    complex_haze_image_name = _os.path.join(CONFIG.nyu_save_dir, str(idx) + "complex_haze_image_name" + ".npy")
    orig_depth_image_name = _os.path.join(CONFIG.nyu_save_dir, str(idx) + "orig_depth_image" + ".npy")
    orig_image_name = _os.path.join(CONFIG.nyu_save_dir, str(idx) + "orig_image" + ".npy")

    haze_image = load(haze_image_name)
    complex_noisy_img = load(complex_haze_image_name)
    image = load(orig_image_name)
    depth = load(orig_depth_image_name)

    # sample = self.nyu_dataset[idx]
    # image = Image.open(BytesIO(self.data[sample[0]]))
    # depth = Image.open(BytesIO(self.data[sample[1]]))
    sample = {'image': image, 'depth': depth}
    if self.transform:
        sample = self.transform(sample)
    
    # all the image matrices are normalized here within 0-1
    image_full, image_half, depth_half_10_1000, depth_half_0_1 = sample['image_norm'], \
        sample['image_half_norm'], sample['depth_half_norm_10_1000'], sample['depth_half_norm_0_1']

    # self.create_image_to_verify(image_half, depth_half_0_1)
    depth_half_0_1 = depth_half_0_1*10
    m = depth_half_0_1.shape[1]
    n = depth_half_0_1.shape[2]

    depth_half_0_1_numpy = np.array(depth_half_0_1)
    del depth_half_0_1

    # need to modify the dimension ordering to use in opencv
    depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 2)
    depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 1)
    depth_half_0_1_3d = cv2.cvtColor(depth_half_0_1_numpy, cv2.COLOR_GRAY2RGB)  # transformed into 3 channels
    del depth_half_0_1_numpy

    beta_mat = self.beta_mat_arr[idx]
    beta_mat_mod = self.create_reorganize_dimension(beta_mat, m, n)

    a_mat = self.a_mat_arr[idx]
    a_mat_mod = self.create_reorganize_dimension(a_mat, m, n)

    tx1 = np.exp(-np.multiply(beta_mat_mod, depth_half_0_1_3d))

    # generate 3D unit matrix
    unit_mat = [1.0, 1.0, 1.0]
    unit_mat = self.create_reorganize_dimension(unit_mat, m, n)

    second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))

    image_half_numpy = np.array(image_half)
    image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)  # making m * n *3
    image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)  # making m * n *3
    
    del a_mat
    del beta_mat

    image_half_numpy = self.only_reorganize_dimension(image_half_numpy)  # reorganize the dimension as m*n*3
    haze_image = self.only_reorganize_dimension(haze_image)  # reorganize the dimension as m*n*3

    # depth_half_0_1_3d = self.only_reorganize_dimension(depth_half_0_1_3d)  # reorganize the dimension as m*n*3

    a_mat_mod = self.only_reorganize_dimension(a_mat_mod)  # reorganize the dimension as m*n*3
    beta_mat_mod = self.only_reorganize_dimension(beta_mat_mod)  # reorganize the dimension as m*n*3
    unit_mat = self.only_reorganize_dimension(unit_mat)  # reorganize the dimension as m*n*3
    complex_noisy_img = self.only_reorganize_dimension(complex_noisy_img)  # reorganize the dimension as m*n*3

    image_half_tensor = torch.from_numpy(image_half_numpy)  # convert into tensor
    haze_image_tensor = torch.from_numpy(haze_image)  # convert into tensor

    # depth_half_0_1_3d = torch.from_numpy(depth_half_0_1_3d)  # convert into tensor
    
    a_mat_mod = torch.from_numpy(a_mat_mod)  # convert into tensor
    beta_mat_mod = torch.from_numpy(beta_mat_mod)  # convert into tensor
    unit_mat = torch.from_numpy(unit_mat)  # convert into tensor
    complex_image_tensor = torch.from_numpy(complex_noisy_img)  # convert into tensor

    del complex_noisy_img

    return {'image': image_full, 'image_half': image_half_tensor, 'depth': depth_half_10_1000,
            'depth_norm_simple': depth_half_0_1_3d, 'haze_image': haze_image_tensor, 'beta': beta_mat_mod,
            'a_val': a_mat_mod, 'unit_mat': unit_mat, 'complex_noise_img': complex_image_tensor}


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        # beta_arr = []
        # a_mat_arr = []
        # for x in range(len(nyu2_train)):
        #     rand_indx_r = random.randint(0, len(beta_val_r) - 1)
        #     rand_indx_g = random.randint(0, len(beta_val_g) - 1)
        #     rand_indx_b = random.randint(0, len(beta_val_b) - 1)
        #     beta_mat = [beta_val_r[rand_indx_r], beta_val_g[rand_indx_g], beta_val_b[rand_indx_b]]
        #     beta_arr.append(beta_mat)

        #     a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        #     a_mat_arr.append(a_mat)

        self.beta_mat_arr = load('Beta_Mat.npy')
        self.a_mat_arr = load('A_Mat.npy')
        self.transform = transform

    def __getitem__(self, idx):
        # idx = 49652  # **************************************
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        # all the image matrices are normalized here within 0-1
        image_full, image_half, depth_half_10_1000, depth_half_0_1 = sample['image_norm'], \
            sample['image_half_norm'], sample['depth_half_norm_10_1000'], sample['depth_half_norm_0_1']

        # self.create_image_to_verify(image_half, depth_half_0_1)
        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]

        depth_half_0_1 = depth_half_0_1*10
        depth_half_0_1_numpy = np.array(depth_half_0_1)
        del depth_half_0_1
        # need to modify the dimension ordering to use in opencv
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 2)
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 1)
        depth_half_0_1_3d = cv2.cvtColor(depth_half_0_1_numpy, cv2.COLOR_GRAY2RGB)  # transformed into 3 channels

        del depth_half_0_1_numpy

        beta_mat = self.beta_mat_arr[idx]
        beta_mat_mod = self.create_reorganize_dimension(beta_mat, m, n)

        # a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        a_mat = self.a_mat_arr[idx]
        a_mat_mod = self.create_reorganize_dimension(a_mat, m, n)

        tx1 = np.exp(-np.multiply(beta_mat_mod * CONFIG.nyu_water_clarity, depth_half_0_1_3d))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = self.create_reorganize_dimension(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))

        image_half_numpy = np.array(image_half)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)  # making m * n *3
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)  # making m * n *3
        haze_image = np.add((np.multiply(image_half_numpy, tx1)), second_term)

        # self.another_simple_image_save(haze_image, '/homes/t20monda/Water_Correction_Test_1/DenseDepth/haze_image_NN.png')
        # self.another_simple_image_save(image_half_numpy, '/homes/t20monda/Water_Correction_Test_1/DenseDepth/orig_imag_NN.png')
        # self.another_simple_image_save(depth_half_0_1_3d, '/homes/t20monda/Water_Correction_Test_1/DenseDepth/depth_imag_NN.png')

        # seed=random_seed+idx: _turbid_v2 draws the Eq.6 particle field from the global
        # numpy RNG and NOTHING used to seed it, so the complex GT was not byte-reproducible
        # and a partially-regenerated directory silently mixed realisations. Now that the
        # particles are actually VISIBLE (u=0.90), that matters.
        complex_noisy_img = compute_complex_noise(image_half_numpy, depth_half_0_1_3d[:, :, 0]/10, beta_mat,
                                                  a_mat, max_depth_m=CONFIG.nyu_max_depth_m,
                                                  focal_px=CONFIG.nyu_focal_px,
                                                  clarity=CONFIG.nyu_water_clarity,
                                                  seed=int(getattr(CONFIG, 'random_seed', 42)) + idx)
        complex_noisy_img = complex_noisy_img/255
        del a_mat
        del beta_mat

        image_half_numpy = self.only_reorganize_dimension(image_half_numpy)  # reorganize the dimension as m*n*3
        haze_image = self.only_reorganize_dimension(haze_image)  # reorganize the dimension as m*n*3
        depth_half_0_1_3d = self.only_reorganize_dimension(depth_half_0_1_3d)  # reorganize the dimension as m*n*3
        a_mat_mod = self.only_reorganize_dimension(a_mat_mod)  # reorganize the dimension as m*n*3
        beta_mat_mod = self.only_reorganize_dimension(beta_mat_mod)  # reorganize the dimension as m*n*3
        unit_mat = self.only_reorganize_dimension(unit_mat)  # reorganize the dimension as m*n*3
        complex_noisy_img = self.only_reorganize_dimension(complex_noisy_img)  # reorganize the dimension as m*n*3

        image_half_tensor = torch.from_numpy(image_half_numpy)  # convert into tensor
        haze_image_tensor = torch.from_numpy(haze_image)  # convert into tensor
        depth_half_0_1_3d = torch.from_numpy(depth_half_0_1_3d)  # convert into tensor
        a_mat_mod = torch.from_numpy(a_mat_mod)  # convert into tensor
        beta_mat_mod = torch.from_numpy(beta_mat_mod)  # convert into tensor
        unit_mat = torch.from_numpy(unit_mat)  # convert into tensor
        complex_image_tensor = torch.from_numpy(complex_noisy_img)  # convert into tensor

        # haze_image_cpu = haze_image_tensor.cpu()
        # save_image(haze_image_cpu, './haze_image_tensor.png')

        # image_half_cpu = image_half_tensor.cpu()
        # save_image(image_half_cpu, './image_half_tensor.png')

        # depth_half_cpu = depth_half_0_1_3d.cpu()
        # save_image(depth_half_cpu, './depth_half_tensor.png')

        del complex_noisy_img

        return {'image': image_full, 'image_half': image_half_tensor, 'depth': depth_half_10_1000,
                'depth_norm_simple': depth_half_0_1_3d, 'haze_image': haze_image_tensor, 'beta': beta_mat_mod,
                'a_val': a_mat_mod, 'unit_mat': unit_mat, 'complex_noise_img': complex_image_tensor}

    def __len__(self):
        return len(self.nyu_dataset)

    def create_image_to_verify(self, image, depth):
        image = (image - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255
        depth = (depth - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255

        image_numpy = np.array(image)
        image_numpy = np.swapaxes(image_numpy, 0, 2)
        image_numpy = np.swapaxes(image_numpy, 0, 1)

        depth_numpy = np.array(depth)
        depth_numpy = np.swapaxes(depth_numpy, 0, 2)
        depth_numpy = np.swapaxes(depth_numpy, 0, 1)

        depth_numpy = depth_numpy[:, :, 0]

        imgD_Norm = depth_numpy
        imgD_Norm = imgD_Norm + 1
        imgD_Norm = (imgD_Norm / np.min([np.max(imgD_Norm), 255])) * 255  # values are > 0 & <= 255

        imgRGB = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        fname_orig_imag = '/home/tmondal/Documents/Dataset/Dehazing/NYU_Depth_V2/test_orig_imag_NN.png'
        imgRGB = imgRGB.astype(np.uint8)
        cv2.imwrite(fname_orig_imag, imgRGB)

        # imgD_Norm = cv2.cvtColor(imgD_Norm, cv2.COLOR_BGR2RGB)
        imgD_Norm = cv2.normalize(imgD_Norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        imgD_Norm = imgD_Norm.astype(np.uint8)
        fname_orig_depth = '/home/tmondal/Documents/Dataset/Dehazing/NYU_Depth_V2/test_orig_depth_NN.png'
        cv2.imwrite(fname_orig_depth, imgD_Norm)

    def create_reorganize_dimension(self, data, m, n):
        data = np.reshape(data, [3, 1, 1])
        data = np.tile(data, [1, m, n])
        data = np.swapaxes(data, 0, 2)  # making m * n *3
        data = np.swapaxes(data, 0, 1)  # making m * n *3
        return data

    def only_reorganize_dimension(self, data):
        data = np.swapaxes(data, 0, 2)  # making m * n *3
        data = np.swapaxes(data, 1, 2)  # making m * n *3
        return data

    def simple_image_save(self, image, path):
        image = (image - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255
        # image = np.swapaxes(image, 0, 2)
        # image = np.swapaxes(image, 1, 2)
        image = image.astype(np.float32)
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bringing back in the range of 0-255
        imgRGB = imgRGB.astype(np.uint8)
        cv2.imwrite(path, imgRGB)

    def another_simple_image_save(self, image, path):
        image = image.astype(np.float32)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = image.astype(np.uint8)
        cv2.imwrite(path, image)    


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        # MUST match utils/transforms.ToTensor EXACTLY. The GT is rendered from THIS depth map,
        # while the network trains against the LOADER's. PIL's default for mode-L is BICUBIC, which
        # OVERSHOOTS at depth discontinuities and invents depths present in neither the foreground
        # nor the background (an 8-bit step edge 0->255 resamples to 0,52,203,255). A generator/loader
        # mismatch makes the GT's transmission t = exp(-beta*z) UNREPRODUCIBLE at every object
        # boundary: L_d and L_t then pull against each other along every edge and a permanent
        # residual error sits exactly on the boundaries.
        image_half = image.resize((320, 240), resample=Image.BICUBIC)
        depth_half = depth.resize((320, 240), resample=Image.NEAREST)

        image_half_norm_0_1 = self.to_tensor(image_half).float()
        image_norm_0_1 = self.to_tensor(image).float()
        depth_half_0_1 = self.to_tensor(depth_half).float()

        if self.is_test:
            depth_half_norm_10_1000 = self.to_tensor(depth_half).float() / 1000
        else:
            depth_half_norm_10_1000 = self.to_tensor(depth_half).float() * 1000

        # put in expected range
        # Floor 40 (= 0.4 m), matching utils.transforms.DEPTH_CLAMP_MIN and the paper.
        depth_half_norm_10_1000 = torch.clamp(depth_half_norm_10_1000, DEPTH_CLAMP_MIN, DEPTH_CLAMP_MAX)
        return {'image_norm': image_norm_0_1, 'image_half_norm': image_half_norm_0_1,
                'depth_half_norm_10_1000': depth_half_norm_10_1000, 'depth_half_norm_0_1': depth_half_0_1}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, dtype=np.int32))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, dtype=np.int16))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def ToTensorCustom(sample, is_test):
    image, depth = sample['image'], sample['depth']

    # MUST match utils/transforms.ToTensor EXACTLY. The GT is rendered from THIS depth map,
    # while the network trains against the LOADER's. PIL's default for mode-L is BICUBIC, which
    # OVERSHOOTS at depth discontinuities and invents depths present in neither the foreground
    # nor the background (an 8-bit step edge 0->255 resamples to 0,52,203,255). A generator/loader
    # mismatch makes the GT's transmission t = exp(-beta*z) UNREPRODUCIBLE at every object
    # boundary: L_d and L_t then pull against each other along every edge and a permanent
    # residual error sits exactly on the boundaries.
    image_half = image.resize((320, 240), resample=Image.BICUBIC)
    depth_half = depth.resize((320, 240), resample=Image.NEAREST)

    image_half_norm_0_1 = to_tensor_custom(image_half).float()
    image_norm_0_1 = to_tensor_custom(image).float()
    depth_half_0_1 = to_tensor_custom(depth_half).float()

    if is_test:
        depth_half_norm_10_1000 = to_tensor_custom(depth_half).float() / 1000
    else:
        depth_half_norm_10_1000 = to_tensor_custom(depth_half).float() * 1000


    # put in expected range
    # Floor 40 (= 0.4 m), matching utils.transforms.DEPTH_CLAMP_MIN and the paper.
    depth_half_norm_10_1000 = torch.clamp(depth_half_norm_10_1000, DEPTH_CLAMP_MIN, DEPTH_CLAMP_MAX)
    return {'image_norm': image_norm_0_1, 'image_half_norm': image_half_norm_0_1,
            'depth_half_norm_10_1000': depth_half_norm_10_1000, 'depth_half_norm_0_1': depth_half_0_1}


def to_tensor_custom(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError(
            'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))

        return img.float().div(255)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, dtype=np.int32))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, dtype=np.int16))
    else:
        img = torch.ByteTensor(
            torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)

    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def create_reorganize_dimension_custom(data, m, n):
    data = np.reshape(data, [3, 1, 1])
    data = np.tile(data, [1, m, n])
    data = np.swapaxes(data, 0, 2)  # making m * n *3
    data = np.swapaxes(data, 0, 1)  # making m * n *3
    return data


def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlipCustom(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])


def getTrainingTestingData(batch_size):
    data, nyu2_train = loadZipToMem(CONFIG.nyu_zip_path)

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size,
                                                                                  shuffle=False)

def getTestingDataOnly(batch_size):
    data, nyu2_test = loadZipToMemTest(CONFIG.nyu_zip_path)

    transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform(True))
    return DataLoader(transformed_testing, batch_size, shuffle=True, drop_last=True)
