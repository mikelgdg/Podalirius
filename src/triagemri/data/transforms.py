"""MONAI 3D augmentation pipelines for training and validation."""

from typing import Optional, Sequence

import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityd,
    SqueezeDimd,
)


def get_train_transforms(
    intensity_jitter: float = 0.1,
    multi_sequence: bool = False,
) -> Compose:
    """Build the training augmentation pipeline.

    Augmentations are calibrated for medical imaging:
    - Flips are applied independently per axis (no simultaneous 3-axis flip).
    - Superior-inferior (axis=2) flip is disabled — not anatomically realistic.
    - Rotation is limited to ±12.5° (0.218 rad) to avoid unrealistic poses.
    - Elastic deformation magnitude is reduced to 20–40 voxels (physiological range).
    - Intensity jitter is followed by re-clamping to [0, 1].

    Args:
        intensity_jitter: Magnitude of intensity shift/scale perturbations.
        multi_sequence: If True, apply transforms compatible with multi-channel
            volumes ``(C, D, H, W)``.

    Returns:
        A MONAI ``Compose`` pipeline.
    """
    keys = ("volume", "mask")

    transforms = [
        RandFlipd(keys=keys, spatial_axis=0, prob=0.5, allow_missing_keys=True),
        RandFlipd(keys=keys, spatial_axis=1, prob=0.5, allow_missing_keys=True),
        RandAffined(
            keys=keys,
            spatial_size=(96, 96, 96),
            rotate_range=(0.218, 0.218, 0.218),
            scale_range=(0.85, 1.15),
            translate_range=(5.0, 5.0, 5.0),
            prob=0.7,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            allow_missing_keys=True,
        ),
        Rand3DElasticd(
            keys=keys,
            sigma_range=(5.0, 8.0),
            magnitude_range=(20.0, 40.0),
            prob=0.4,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            allow_missing_keys=True,
        ),
        RandBiasFieldd(keys=("volume",), coeff_range=(0.0, 0.15), prob=0.3),
        RandScaleIntensityd(keys=("volume",), factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=("volume",), offsets=intensity_jitter / 2.0, prob=0.5),
        ScaleIntensityd(keys=("volume",), minv=0.0, maxv=1.0),
        RandGaussianNoised(keys=("volume",), std=0.01, prob=0.5),
        RandAdjustContrastd(keys=("volume",), gamma=(0.8, 1.2), prob=0.5),
        EnsureTyped(keys=("volume", "mask"), dtype=torch.float32, allow_missing_keys=True),
    ]

    return Compose(transforms)


def get_val_transforms(multi_sequence: bool = False) -> Compose:
    """Build the validation / test transform pipeline (no augmentation).

    Args:
        multi_sequence: If True, apply transforms compatible with multi-channel
            volumes ``(C, D, H, W)``.  Currently a no-op since only
            ``EnsureTyped`` is applied.

    Returns:
        A MONAI ``Compose`` pipeline.
    """
    keys = ("volume", "mask")

    return Compose([
        EnsureTyped(keys=keys, dtype=torch.float32, allow_missing_keys=True),
    ])
