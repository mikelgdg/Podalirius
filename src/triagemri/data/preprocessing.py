"""Preprocessing utilities for 3D MRI volumes.

Functions for loading NIfTI volumes, intensity normalisation,
centre cropping / padding to a target shape, and multi-sequence handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def validate_file(path: str) -> bool:
    """Check whether a NIfTI/Analyze file can be opened and read.

    Returns True if the file is readable, False otherwise.
    """
    try:
        import nibabel as nib

        img = nib.load(path)
        _ = img.dataobj
        if len(img.shape) not in (3, 4):
            return False
        return True
    except Exception:
        return False


def validate_volume(volume: torch.Tensor, min_nonzero_ratio: float = 0.01) -> bool:
    """Check whether a preprocessed volume is usable for training.

    Args:
        volume: Tensor of shape ``(C, D, H, W)`` or ``(D, H, W)``, normalised to [0, 1].
        min_nonzero_ratio: Minimum fraction of voxels that must be > 0.

    Returns:
        True if the volume passes quality checks.
    """
    if not torch.isfinite(volume).all():
        return False
    nonzero_ratio = (volume > 0.001).float().mean().item()
    return nonzero_ratio >= min_nonzero_ratio


def _load_mha(path: str) -> Tuple[torch.Tensor, np.ndarray]:
    """Load a MetaImage (.mha) file using SimpleITK."""
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img)
        data = np.asarray(data, dtype=np.float32)
        while data.ndim > 3 and data.shape[-1] == 1:
            data = data.squeeze(-1)
        if data.ndim == 4:
            data = data[..., 0]
        direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin = np.array(img.GetOrigin(), dtype=np.float64)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = direction * spacing
        affine[:3, 3] = origin
        return torch.from_numpy(data), affine
    except ImportError:
        raise ImportError("SimpleITK is required to load .mha files.")


def load_nifti(path: str) -> Tuple[torch.Tensor, np.ndarray]:
    """Load a NIfTI, Analyze (.img/.hdr), or MetaImage (.mha) file.

    Args:
        path: Path to .nii, .nii.gz, .img, or .mha file.

    Returns:
        data: 3D tensor of shape (D, H, W).
        affine: 4x4 affine matrix.
    """
    if str(path).lower().endswith(".mha"):
        return _load_mha(path)

    try:
        import nibabel as nib

        img = nib.load(path)
        data = img.get_fdata()
        data = np.asarray(data, dtype=np.float32)
        while data.ndim > 3 and data.shape[-1] == 1:
            data = data.squeeze(-1)
        if data.ndim == 4:
            data = data[..., 0]
        affine = img.affine.copy()
        return torch.from_numpy(data), affine
    except ImportError:
        raise ImportError("nibabel is required to load NIfTI/Analyze files.")


def normalize_intensity(
    volume: torch.Tensor | np.ndarray,
    percentiles: Tuple[float, float] = (0.5, 99.5),
) -> torch.Tensor:
    """Normalise volume intensities to [0, 1] using percentile clipping.

    Volumes containing NaN or Inf values raise ``ValueError`` rather than
    silently returning a zero-filled tensor.

    Args:
        volume: Input volume as tensor or ndarray of shape (D, H, W) or (C, D, H, W).
        percentiles: (lower, upper) percentiles for clipping.

    Returns:
        Normalised tensor in range [0, 1].

    Raises:
        ValueError: If the volume contains NaN or Inf values.
    """
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume.astype(np.float32))
    else:
        volume = volume.float()

    if not torch.isfinite(volume).all():
        raise ValueError("Volume contains NaN/Inf values — cannot normalise.")

    vol_np = volume.numpy()
    vmin = np.nanquantile(vol_np, percentiles[0] / 100.0)
    vmax = np.nanquantile(vol_np, percentiles[1] / 100.0)

    if not np.isfinite(vmax) or not np.isfinite(vmin) or vmax - vmin < 1e-8:
        raise ValueError(
            f"Volume intensity range too narrow ({vmin:.4f}–{vmax:.4f}). "
            f"The volume may be empty or constant."
        )

    volume = torch.clamp(volume, vmin, vmax)
    volume = (volume - vmin) / (vmax - vmin)
    return volume


def harmonize_intensity(volume: torch.Tensor) -> torch.Tensor:
    """Apply Z-score normalisation followed by clipping to [-3, 3].

    This reduces domain shift between datasets with different acquisition
    parameters by standardising the intensity distribution.

    Args:
        volume: Input tensor of shape ``(D, H, W)`` or ``(C, D, H, W)``.

    Returns:
        Z-score normalised tensor clipped to [-3, 3], still in float32.
    """
    v = volume.float()
    mean = v.mean()
    std = v.std()
    if std < 1e-8:
        return torch.zeros_like(v)
    v = (v - mean) / std
    return torch.clamp(v, -3.0, 3.0)


def crop_or_pad(
    volume: torch.Tensor,
    target_shape: Tuple[int, int, int] = (96, 96, 96),
) -> torch.Tensor:
    """Centre crop or pad a 3D volume to the target shape.

    If the volume is larger than *target_shape*, it is centre-cropped.
    If smaller, it is padded symmetrically with zeros.

    Args:
        volume: Tensor of shape (D, H, W) or (C, D, H, W).
        target_shape: (D, H, W) desired output shape.

    Returns:
        Tensor of shape (..., target_shape[0], target_shape[1], target_shape[2]).
    """
    is_3d = volume.dim() == 3
    if is_3d:
        volume = volume.unsqueeze(0)

    C, D, H, W = volume.shape
    tgtD, tgtH, tgtW = target_shape

    pad = [0, 0, 0, 0, 0, 0]
    crop_slices = [slice(None), slice(None), slice(None), slice(None)]

    for i, (sz, tgt) in enumerate([(D, tgtD), (H, tgtH), (W, tgtW)]):
        dim_idx = i + 1
        if sz > tgt:
            if i == 0:
                slice_sums = volume[0].sum(dim=(1, 2)).cpu().numpy()
                best_start = 0
                max_sum = -1
                for s in range(sz - tgt + 1):
                    current_sum = slice_sums[s : s + tgt].sum()
                    if current_sum > max_sum:
                        max_sum = current_sum
                        best_start = s
                crop_slices[dim_idx] = slice(best_start, best_start + tgt)
            else:
                start = (sz - tgt) // 2
                crop_slices[dim_idx] = slice(start, start + tgt)
        elif sz < tgt:
            before = (tgt - sz) // 2
            after = tgt - sz - before
            pad_idx = 4 - 2 * i
            pad[pad_idx] = before
            pad[pad_idx + 1] = after

    if any(p != 0 for p in pad):
        volume = torch.nn.functional.pad(volume, pad, mode="constant", value=0.0)

    volume = volume[crop_slices[0], crop_slices[1], crop_slices[2], crop_slices[3]]

    if is_3d:
        volume = volume.squeeze(0)

    return volume


def load_and_preprocess(path: str | Path) -> torch.Tensor:
    """Load a NIfTI/DICOM volume and preprocess it to a normalised tensor.

    Args:
        path: Path to .nii.gz, .mha, or DICOM directory.

    Returns:
        Tensor of shape (1, 96, 96, 96), float32, intensity in [0, 1].
    """
    data, _affine = load_nifti(str(path))
    volume = normalize_intensity(data)
    volume = crop_or_pad(volume, target_shape=(96, 96, 96))
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)
    return volume.to(torch.float32)


def load_and_preprocess_multi(paths: Dict[str, str | Path]) -> torch.Tensor:
    """Load and preprocess multiple MRI sequences into a multi-channel volume.

    Each sequence is loaded, normalised, and cropped/padded to 96³, then
    stacked along the channel dimension.

    Args:
        paths: Dict mapping sequence name (e.g. ``"t1ce"``) to file path.

    Returns:
        Tensor of shape ``(N_channels, 96, 96, 96)``, float32, intensity in [0, 1].
        Returns ``(1, 96, 96, 96)`` if only one sequence is provided.
    """
    channels: list[torch.Tensor] = []
    for _seq, path in paths.items():
        vol = load_and_preprocess(str(path))
        channels.append(vol.squeeze(0))
    if len(channels) == 1:
        return channels[0].unsqueeze(0).to(torch.float32)
    return torch.stack(channels, dim=0).to(torch.float32)


def load_full_volume_for_display(
    path: str | Path,
) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[int, int, int]]:
    """Load and resize a volume to 96³ for display, preserving the full FOV.

    Returns the display volume, the crop offsets and the crop extent
    (both in 96³ display coordinates).  This allows repositioning the
    model's attention heatmap at the correct scale and location on the
    full-brain display.

    Args:
        path: Path to NIfTI / MHA file.

    Returns:
        volume_display: Tensor of shape (96, 96, 96), float32, [0, 1].
        crop_offsets: (d0, h0, w0) in display coordinates.
        crop_shape: (d_sz, h_sz, w_sz) extent in display coordinates.
    """
    import torch.nn.functional as F

    data, _affine = load_nifti(str(path))
    orig_shape = data.shape
    W_orig, H_orig, D_orig = orig_shape[0], orig_shape[1], orig_shape[2]
    volume = normalize_intensity(data)

    d0_orig = max(0, (D_orig - 96) // 2)
    h0_orig = max(0, (H_orig - 96) // 2)
    w0_orig = max(0, (W_orig - 96) // 2)

    d_sz_orig = min(96, D_orig)
    h_sz_orig = min(96, H_orig)
    w_sz_orig = min(96, W_orig)

    if volume.dim() == 3:
        volume = volume.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 3, 2)
    elif volume.dim() == 4:
        volume = volume.unsqueeze(0).permute(0, 1, 4, 3, 2)

    volume = F.interpolate(volume, size=(96, 96, 96), mode="trilinear", align_corners=False)
    volume = volume.permute(0, 1, 4, 3, 2)
    volume = volume.squeeze(0)

    sc_d = 96.0 / D_orig if D_orig > 0 else 1.0
    sc_h = 96.0 / H_orig if H_orig > 0 else 1.0
    sc_w = 96.0 / W_orig if W_orig > 0 else 1.0

    d0 = int(d0_orig * sc_d)
    h0 = int(h0_orig * sc_h)
    w0 = int(w0_orig * sc_w)
    d_sz = int(d_sz_orig * sc_d)
    h_sz = int(h_sz_orig * sc_h)
    w_sz = int(w_sz_orig * sc_w)

    return volume.to(torch.float32), (d0, h0, w0), (d_sz, h_sz, w_sz)
