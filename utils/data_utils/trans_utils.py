import numpy as np
import random

import torch
from torchvision import transforms

from utils.constants import *
from utils.data_utils.angle_utils import rot_aa


def fftHighpass(images, cutoff=2):
    out_images = []
    for image in images:
        if isinstance(image, torch.Tensor):
            image = image.to("cpu").numpy()

        fft_image = np.fft.fft2(image)
        fft_image_shift = np.fft.fftshift(fft_image)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        fft_image_shift[
            crow - cutoff : crow + cutoff, ccol - cutoff : ccol + cutoff
        ] = 0
        fft_image = np.fft.ifftshift(fft_image_shift)
        image = np.fft.ifft2(fft_image)
        image = np.abs(image)
        out_images.append(torch.tensor(image, device=DEVICE).float())
    return out_images


# ##################################################################
# Data Augmentation Functions
# ##################################################################
def get_random_value(lower, upper):
    return np.random.uniform() * (lower - upper) + upper


def RandomRotate(pose, images, deg=15, p=0.9):
    """Rotate the pose and images by a small random angle, need to change both images and labels"""
    if np.random.uniform() < (1 - p):
        return pose, images

    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    rot = np.clip(np.random.randn(), -2.0, 2.0) * deg
    pose[:3] = rot_aa(pose[:3], -rot)
    pose = torch.tensor(pose).float()

    for i, image in enumerate(images):
        if image is not None and (image != MISS_MODALITY_FLAG).any():
            fill = 0.0
            # fill = image.max().item()  # 0.0
            image_rot = transforms.functional.rotate(image, rot, fill=fill)
            images[i] = image_rot
    return pose, images


def RandomAffine(images, translate=0.15, p=0.7, separate=False):
    """Translate the images by a small random factor, need to change both images and labels root (xyz)"""
    if np.random.uniform() < (1 - p):
        return images

    def get_params():
        return (
            get_random_value(-translate, translate),
            get_random_value(-translate, translate),
            1,
        )

    trans_factor_x, trans_factor_y, _ = get_params()
    out_images = []
    for i, image in enumerate(images):
        try:
            if separate:
                trans_factor_x, trans_factor_y, _ = get_params()
            fill = image.min().item()
            out_images.append(
                transforms.functional.affine(
                    image,
                    angle=0,
                    translate=[
                        trans_factor_x * image.shape[2],
                        trans_factor_y * image.shape[1],
                    ],
                    scale=1,
                    shear=0,
                    # fill=fill,
                )
            )
        except:
            out_images.append(None)

    return out_images


def RandomBlur(images, p=0.7, kernel_size=3, sigma=(0.5, 5.0), num_cut=5):
    if np.random.uniform() < (1 - p):
        return images

    blur = transforms.GaussianBlur(kernel_size, sigma)
    out_images = []
    padding = kernel_size // 2
    h, w = images[0].shape[1], images[0].shape[2]

    for i, image in enumerate(images):
        if i % 2 == 0:
            # skip the uncover reference image
            out_images.append(image)
            continue

        if image is not None and (image != MISS_MODALITY_FLAG).any():
            image_blur = image.clone()
            for _ in range(num_cut):
                # randomly get a bbox
                try:
                    x1, y1 = random.randint(padding, w - 1 - padding), random.randint(
                        padding, h - 1 - padding
                    )
                    x2, y2 = random.randint(
                        x1 + padding + 1, w - 1 - padding
                    ), random.randint(y1 + padding + 1, h - 1 - padding)

                    # apply blur
                    image_blur[:, y1:y2, x1:x2] = blur(image_blur[:, y1:y2, x1:x2])
                except:
                    pass

            out_images.append(image_blur)
        else:
            out_images.append(None)
    return out_images


def RandomErase(
    images, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0, p=0.6, num_cuts=5
):  # earlier scale was (0.02, 0.33)
    transform = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)
    out_images = []
    for i, image in enumerate(images):
        if i % 2 == 0:
            # skip the uncover reference image
            out_images.append(image)
            continue

        try:
            for _ in range(num_cuts):
                image = transform(image)
        except:
            image = None
        out_images.append(image)
    return out_images


def RandomNoise(images, p=0.5, drop_prob=0.005):
    if np.random.uniform() < (1 - p):
        return images
    out_images = []
    bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([drop_prob]))
    for i, image in enumerate(images):
        if i % 2 == 0:
            # skip the uncover reference image
            out_images.append(image)
            continue

        try:
            noise = bernoulli.sample(image.shape).squeeze(-1)
            image_out = image * (1 - noise)
        except:
            image_out = None
        out_images.append(image_out)
    return out_images


def ToTensor(images):
    out_images = []
    totensor = transforms.ToTensor()
    for image in images:
        out_images.append(totensor(image))
    return out_images


def Resize(images, size=224):
    transform = transforms.Resize((size, size), antialias=None)
    out_images = []
    for i, image in enumerate(images):
        try:
            out_images.append(transform(image))
        except:
            out_images.append(None)
    return out_images


# ##################################################################
# Data Scaling Functions
# ##################################################################
def Normalize(image, lower=0.0, upper=None):
    if upper is None:
        upper = image.max()
    image = image.clamp(min=lower, max=upper)
    return (image - lower) / (upper - lower)


def DeNormalize(image, lower=0.0, upper=1.0):
    return image * (upper - lower) + lower


def SigmoidTransform(image, k=K_DEPTH, x0=MAX_DEPTH_SYNTH):
    assert image.max() <= x0, f"Max value {image.max()} is greater than x0 {x0}"
    return 2 / (1 + torch.exp(-k * (image - x0)))


def DeSigmoidTransform(image, k=K_DEPTH, x0=MAX_DEPTH_SYNTH):
    assert image.max() <= 1, f"Max value {image.max()} is greater than 1"
    return x0 - torch.log(2 / image - 1) / k


def get_scaling_func(scaling, modality):
    try:
        if scaling == "minmax":
            lower = globals()[f"MIN_{modality.upper()}_SYNTH"]
            upper = globals()[f"MAX_{modality.upper()}_SYNTH"]
            return lambda x: Normalize(x, lower, upper)
        elif scaling == "sigmoid":
            k = globals()[f"K_{modality.upper()}"]
            x0 = globals()[f"MAX_{modality.upper()}_SYNTH"]
            return lambda x: SigmoidTransform(x, k, x0)
        elif scaling == "bodymap":
            return lambda x: Normalize(x)
        else:  # scaling == "none"
            return lambda x: x
    except:
        return lambda x: x


def get_descaling_func(scaling, modality):
    try:
        if scaling == "minmax":
            lower = globals()[f"MIN_{modality.upper()}_SYNTH"]
            upper = globals()[f"MAX_{modality.upper()}_SYNTH"]
            return lambda x: DeNormalize(x, lower, upper)
        elif scaling == "sigmoid":
            k = globals()[f"K_{modality.upper()}"]
            x0 = globals()[f"MAX_{modality.upper()}_SYNTH"]
            return lambda x: DeSigmoidTransform(x, k, x0)
        else:  # scaling == "none"
            return lambda x: x
    except:
        return lambda x: x
