"""Unit tests for the TriadEncoder module.

These are sanity / interface tests that do not require real Triad weights.
"""

from __future__ import annotations

import pytest
import torch

from triagemri.models.encoder import TriadEncoder


class TestTriadEncoder:
    """Interface and sanity tests for TriadEncoder."""

    def test_encoder_creation_no_weights(self) -> None:
        encoder = TriadEncoder(checkpoint_path="nonexistent.pth", freeze=True)
        assert isinstance(encoder, torch.nn.Module)

    def test_encoder_forward_shape(self) -> None:
        encoder = TriadEncoder(checkpoint_path="nonexistent.pth", freeze=True)
        dummy_input = torch.randn(2, 1, 96, 96, 96)
        output = encoder(dummy_input)
        assert output.shape[0] == 2
        assert output.shape[2] == 768
        assert output.shape[1] == 27, (
            f"Expected 27 instances (3x3x3 grid), got {output.shape[1]}"
        )

    def test_encoder_frozen(self) -> None:
        encoder = TriadEncoder(checkpoint_path="nonexistent.pth", freeze=True)
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        assert trainable == 0

    def test_encoder_unfrozen(self) -> None:
        encoder = TriadEncoder(checkpoint_path="nonexistent.pth", freeze=False)
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        assert trainable > 0

    def test_extract_features(self) -> None:
        encoder = TriadEncoder(checkpoint_path="nonexistent.pth", freeze=True)
        dummy_input = torch.randn(2, 1, 96, 96, 96)
        features = encoder.extract_features(dummy_input)
        assert features.shape == (2, 27, 768)

    def test_variable_input_size(self) -> None:
        encoder = TriadEncoder(checkpoint_path="nonexistent.pth", freeze=True)
        dummy_input = torch.randn(1, 1, 80, 100, 90)
        output = encoder(dummy_input)
        assert output.shape == (1, 27, 768)

    def test_pretrained_strict_loads(self) -> None:
        encoder = TriadEncoder(
            checkpoint_path="weights/triad_swinb_simmim.pth",
            freeze=True,
            pretrained_strict=False,
        )
        assert encoder._pretrained_loaded
        dummy_input = torch.randn(1, 1, 96, 96, 96)
        with torch.no_grad():
            output = encoder(dummy_input)
        assert output.shape == (1, 27, 768)


class TestFreezeProjection:
    """Tests for freeze_projection parameter."""

    def test_projection_not_frozen(self) -> None:
        """Projection should be trainable when freeze_projection=False."""
        encoder = TriadEncoder(
            checkpoint_path="nonexistent.pth",
            freeze=True,
            freeze_projection=False,
        )
        swin_grad = any(p.requires_grad for p in encoder.swin.parameters())
        proj_grad = all(p.requires_grad for p in encoder.projection.parameters())
        assert not swin_grad, "Swin backbone should be frozen"
        assert proj_grad, "Projection should be trainable"

    def test_projection_frozen_by_default(self) -> None:
        """Backward compatibility: freeze_projection defaults to True."""
        encoder = TriadEncoder(
            checkpoint_path="nonexistent.pth",
            freeze=True,
        )
        proj_grad = any(p.requires_grad for p in encoder.projection.parameters())
        assert not proj_grad, "Projection should be frozen by default"

    def test_train_mode_respects_freeze_projection(self) -> None:
        """In train mode with frozen swin + unfrozen projection, projection should be in training mode."""
        encoder = TriadEncoder(
            checkpoint_path="nonexistent.pth",
            freeze=True,
            freeze_projection=False,
        )
        encoder.train()
        assert not encoder.swin.training, "Swin should stay in eval"
        assert encoder.projection.training, "Projection should be in training mode"
