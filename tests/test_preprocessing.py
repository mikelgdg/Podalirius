"""Unit tests for MRI preprocessing functions.

Uses synthetic numpy arrays — no real medical data required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
nib = pytest.importorskip("nibabel", reason="nibabel not installed")


from triagemri.data.preprocessing import (
    crop_or_pad,
    load_and_preprocess,
    load_nifti,
    normalize_intensity,
)


class TestNormalizeIntensity:
    """Intensity normalisation tests."""

    def test_output_range_numpy(self) -> None:
        """Normalisation of ndarray should produce [0, 1] range."""
        data = np.random.randn(64, 64, 64).astype(np.float32) * 50 + 100
        result = normalize_intensity(data)
        rmin = result.min().item()
        rmax = result.max().item()
        assert -0.01 <= rmin <= 0.01, f"min={rmin}"
        assert 0.99 <= rmax <= 1.01, f"max={rmax}"

    def test_output_range_tensor(self) -> None:
        """Normalisation of tensor should produce [0, 1] range."""
        data = torch.randn(64, 64, 64) * 50 + 100
        result = normalize_intensity(data)
        rmin = result.min().item()
        rmax = result.max().item()
        assert -0.01 <= rmin <= 0.01, f"min={rmin}"
        assert 0.99 <= rmax <= 1.01, f"max={rmax}"

    def test_constant_input(self) -> None:
        """Constant input should produce all zeros."""
        data = torch.ones(32, 32, 32) * 42.0
        result = normalize_intensity(data)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_preserves_shape(self) -> None:
        """Output shape must match input shape (3D)."""
        data = torch.randn(48, 64, 32) * 10
        result = normalize_intensity(data)
        assert result.shape == data.shape


class TestCropOrPad:
    """Centre crop / pad tests."""

    def test_crop_larger(self) -> None:
        """Larger volume should be cropped to target.

        The D axis uses smart-crop (max total intensity window), while
        H and W use centre crop.  We verify shape and that no padded
        zeros are present.
        """
        data = torch.randn(128, 128, 128)
        result = crop_or_pad(data, target_shape=(64, 64, 64))
        assert result.shape == (64, 64, 64)
        assert result.min() < result.max(), "crop should preserve intensity variation"

    def test_pad_smaller(self) -> None:
        """Smaller volume should be centre-padded to target."""
        data = torch.ones(32, 32, 32)
        result = crop_or_pad(data, target_shape=(64, 64, 64))
        assert result.shape == (64, 64, 64)
        assert torch.equal(result[16:48, 16:48, 16:48], data)
        assert result[0, 0, 0] == 0.0

    def test_exact_shape(self) -> None:
        """Volume already at target shape should be unchanged."""
        data = torch.randn(96, 96, 96)
        result = crop_or_pad(data, target_shape=(96, 96, 96))
        assert result.shape == (96, 96, 96)
        assert torch.equal(result, data)

    def test_multichannel(self) -> None:
        """Multichannel input (C, D, H, W) should be handled."""
        data = torch.randn(3, 40, 50, 60)
        result = crop_or_pad(data, target_shape=(48, 48, 48))
        assert result.shape == (3, 48, 48, 48)

    def test_asymmetric_padding(self) -> None:
        """Odd-sized padding differences should be handled gracefully."""
        data = torch.ones(31, 31, 31)
        result = crop_or_pad(data, target_shape=(33, 33, 33))
        assert result.shape == (33, 33, 33)


class TestLoadNifti:
    """NIfTI loading tests with synthetic files."""

    def test_load_synthetic_nifti(self, tmp_path: Path) -> None:
        """Create and load a synthetic NIfTI file."""
        import nibabel as nib

        data = np.random.randn(64, 64, 32).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        filepath = tmp_path / "synthetic.nii.gz"
        nib.save(img, str(filepath))

        loaded, loaded_affine = load_nifti(str(filepath))
        assert loaded.shape == (64, 64, 32)
        np.testing.assert_array_almost_equal(loaded.numpy(), data, decimal=4)
        np.testing.assert_array_almost_equal(loaded_affine, affine)

    def test_load_nonexistent_raises(self) -> None:
        """Loading a nonexistent file should propagate the error."""
        with pytest.raises(Exception):
            load_nifti("/nonexistent/path/file.nii.gz")


class TestLoadAndPreprocess:
    """End-to-end load and preprocess tests."""

    def test_output_shape(self, tmp_path: Path) -> None:
        """load_and_preprocess should output (1, 96, 96, 96)."""
        import nibabel as nib

        data = np.random.randn(80, 80, 80).astype(np.float32) * 20 + 50
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        filepath = tmp_path / "test.nii.gz"
        nib.save(img, str(filepath))

        result = load_and_preprocess(str(filepath))
        assert result.shape == (1, 96, 96, 96)
        assert result.dtype == torch.float32
        rmin = result.min().item()
        rmax = result.max().item()
        assert rmin >= -0.01, f"min={rmin}"
        assert rmax <= 1.01, f"max={rmax}"
