"""Canonical data-augmentation / tensor transforms.

Contains both transform families used in the project:

NYU (operate on a {'image', 'depth'} sample dict of PIL images):
    RandomGamma, RandomHorizontalFlipCustom, RandomChannelSwap, ToTensor,
    getNoTransform(), getDefaultTrainTransform()

Make3D (operate on (image, depth) numpy pairs):
    Augmentation and its building blocks (Compose, PhotometricDistort, ...).
"""

import random

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# =============================================================================
# NYU transforms (sample dict of PIL images)
# =============================================================================

class RandomGamma(object):
    """Apply Random Gamma Correction to the images."""

    def __init__(self, gamma=0):
        self.gamma = gamma

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if self.gamma == 0:
            return {'image': image, 'depth': depth}
        gamma_ratio = random.uniform(1 / self.gamma, self.gamma)
        return {'image': TF.adjust_gamma(image, gamma_ratio, gain=1), 'depth': depth}


class RandomHorizontalFlipCustom(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


class ToTensor(object):
    """NYU ToTensor: resizes to half size and normalises into the sample dict."""

    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image_half = image.resize((320, 240))
        depth_half = depth.resize((320, 240))

        image_half_norm_0_1 = self.to_tensor(image_half).float()
        image_norm_0_1 = self.to_tensor(image).float()
        depth_half_0_1 = self.to_tensor(depth_half).float()

        # Train and test use the SAME depth scaling. The old is_test branch divided
        # by 1000, which (with a [0,1] source depth) collapsed every test depth to
        # the clamp floor of 10 -> a degenerate constant depth map. Depth is the
        # same 8-bit source in both cases, so both use *1000 -> [10, 1000].
        depth_half_norm_10_1000 = self.to_tensor(depth_half).float() * 1000
        depth_half_norm_10_1000 = torch.clamp(depth_half_norm_10_1000, 10, 1000)
        return {'image_norm': image_norm_0_1, 'image_half_norm': image_half_norm_0_1,
                'depth_half_norm_10_1000': depth_half_norm_10_1000, 'depth_half_norm_0_1': depth_half_0_1}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)

        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
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


def getNoTransform(is_test=False):
    return transforms.Compose([ToTensor(is_test=is_test)])


def getDefaultTrainTransform():
    # NOTE: RandomChannelSwap MUST NOT be used here. The haze/complex ground-truth
    # targets are PRE-COMPUTED on disk from the ORIGINAL-channel clean image, but a
    # channel swap only permutes the *input* image at load time -> the network would
    # be asked to map a channel-permuted input to an unpermuted, colour-dependent
    # target (an impossible, non-deterministic mapping on ~42% of samples). This is
    # correct in DenseDepth (whose only target is channel-invariant depth) but WRONG
    # here where two of the three losses are colour-dependent. It also poisons the
    # depth branch through the (non-detached) haze loss. Geometric augmentation that
    # keeps input<->GT aligned (a shared horizontal flip) is done in data.nyu instead.
    return transforms.Compose([
        ToTensor()
    ])


# =============================================================================
# Make3D transforms (numpy (image, depth) pairs)
# =============================================================================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, depth):
        for t in self.transforms:
            image, depth = t(image, depth)
        return image, depth


class ConvertFromInts(object):
    def __call__(self, image, depth):
        return image.astype(np.float32), depth.astype(np.float32)


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, depth):
        if random.random() < 0.5:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, depth


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, depth):
        if random.random() < 0.5:
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, depth


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, depth):
        if random.random() < 0.5:
            swap = self.perms[random.randint(0, len(self.perms) - 1)]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, depth


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, depth):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, depth


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, depth):
        if random.random() < 0.5:
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, depth


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, depth):
        if random.random() < 0.5:
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, depth


class RandomMirror(object):
    def __call__(self, image, depth):
        if random.random() < 0.5:
            image = np.copy(np.fliplr(image))
            depth = np.copy(np.fliplr(depth))
        return image, depth


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, depth):
        im = image.copy()
        im, depth = self.rand_brightness(im, depth)
        if random.random() < 0.5:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, depth = distort(im, depth)
        return self.rand_light_noise(im, depth)


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crops them to keep same size as before."""

    def __call__(self, image, depth):
        in_h, in_w, _ = image.shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        scaled_images = cv2.resize(image, (scaled_w, scaled_h))
        scaled_depth = cv2.resize(depth, (scaled_w, scaled_h))

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)

        cropped_images = scaled_images[offset_y:offset_y + in_h, offset_x:offset_x + in_w]
        cropped_depth = scaled_depth[offset_y:offset_y + in_h, offset_x:offset_x + in_w]

        return cropped_images, cropped_depth


class ClipRGB(object):
    def __call__(self, image, depth):
        image = image.astype(np.float32)
        return np.clip(image, 0, 255), depth


class ToTensorImageDepth(object):
    """Make3D tensor conversion of an (image, depth) numpy pair."""

    def __call__(self, image, depth):
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        depth = torch.from_numpy(depth).float()
        return image, depth


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, depth):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), depth


class Augmentation(object):
    def __init__(self):
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            RandomScaleCrop(),
            ClipRGB()
        ])

    def __call__(self, img, depth):
        return self.augment(img, depth)
