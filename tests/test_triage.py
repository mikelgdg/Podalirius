"""Integration tests for the TriageModel."""

from __future__ import annotations

import pytest
import torch

torch = pytest.importorskip("torch", reason="PyTorch not installed")

import torch.nn as nn

from triagemri.models.encoder import TriadEncoder
from triagemri.models.mil import AttentionMIL, GatedAttentionMIL, build_mil_head
from triagemri.models.triage import TriageModel, build_triage_model


def _dummy_encoder(freeze: bool = True) -> TriadEncoder:
    return TriadEncoder(
        checkpoint_path="nonexistent.pth",
        freeze=freeze,
    )


def _dummy_input(batch_size: int = 2) -> torch.Tensor:
    return torch.randn(batch_size, 1, 96, 96, 96)


class TestTriageModel:
    """Integration tests for TriageModel."""

    def test_forward_pass(self) -> None:
        """Full forward pass with random input should return correct dict."""
        encoder = _dummy_encoder()
        mil = AttentionMIL()
        model = TriageModel(encoder, mil, anatomy_mode="shared")
        model.eval()

        x = _dummy_input(3)
        with torch.no_grad():
            output = model(x)

        assert isinstance(output, dict), "Output must be a dict"
        for key in ("score", "logits", "attention", "features"):
            assert key in output, f"Missing key '{key}' in output"

        assert output["score"].shape == (3,), f"Expected score (3,), got {output['score'].shape}"
        assert output["logits"].shape == (3, 1)
        assert output["attention"].shape == (3, 27, 1)
        assert output["features"].shape == (3, 27, 768)

    def test_score_range(self) -> None:
        """Scores should be in [0, 1] after sigmoid."""
        encoder = _dummy_encoder()
        mil = AttentionMIL()
        model = TriageModel(encoder, mil)
        model.eval()

        x = _dummy_input(5)
        with torch.no_grad():
            output = model(x)

        scores = output["score"]
        assert (scores >= 0).all(), "Scores below 0"
        assert (scores <= 1).all(), "Scores above 1"

    def test_encoder_frozen_during_training(self) -> None:
        """Encoder params should not change during backward pass."""
        encoder = _dummy_encoder(freeze=True)
        mil = AttentionMIL()
        model = TriageModel(encoder, mil)
        model.train()

        initial_encoder_params = {
            name: param.clone() for name, param in encoder.named_parameters()
        }

        x = _dummy_input(2)
        output = model(x)
        loss = output["logits"].sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert torch.equal(param, initial_encoder_params[name]), (
                f"Encoder param '{name}' changed during backward"
            )

    def test_predict_with_threshold(self) -> None:
        """predict() should return binary predictions."""
        encoder = _dummy_encoder()
        mil = AttentionMIL()
        model = TriageModel(encoder, mil)

        x = _dummy_input(4)
        preds = model.predict(x, threshold=0.5)
        assert preds.shape == (4,), f"Expected (4,), got {preds.shape}"
        assert preds.dtype == torch.long, f"Expected long dtype, got {preds.dtype}"
        assert torch.all((preds == 0) | (preds == 1)), "Predictions not binary"

    def test_attention_heatmap_shape(self) -> None:
        """get_attention_heatmap should return correct spatial shape."""
        encoder = _dummy_encoder()
        mil = AttentionMIL()
        model = TriageModel(encoder, mil)

        x = _dummy_input(2)
        heatmap = model.get_attention_heatmap(x)
        assert heatmap.shape == (2, 1, 3, 3, 3), (
            f"Expected (2, 1, 3, 3, 3), got {heatmap.shape}"
        )

    def test_shared_mode_no_anatomy(self) -> None:
        """Shared mode should work without anatomy argument."""
        encoder = _dummy_encoder()
        mil = AttentionMIL()
        model = TriageModel(encoder, mil, anatomy_mode="shared")
        model.eval()

        x = _dummy_input(2)
        with torch.no_grad():
            output = model(x)
        assert "score" in output

    def test_multi_head_mode(self) -> None:
        """Multi-head mode should route to the correct anatomy head."""
        encoder = _dummy_encoder()
        heads = nn.ModuleDict(
            {
                "brain": AttentionMIL(),
                "prostate": GatedAttentionMIL(),
            }
        )
        model = TriageModel(encoder, heads, anatomy_mode="multi_head")
        model.eval()

        x = _dummy_input(2)
        with torch.no_grad():
            out_brain = model(x, anatomy="brain")
            out_prostate = model(x, anatomy="prostate")

        assert out_brain["score"].shape == (2,)
        assert out_prostate["score"].shape == (2,)

    def test_multi_head_per_sample_routing(self) -> None:
        """Multi-head mode with per-sample anatomy list."""
        encoder = _dummy_encoder()
        heads = nn.ModuleDict(
            {
                "brain": AttentionMIL(),
                "prostate": GatedAttentionMIL(),
            }
        )
        model = TriageModel(encoder, heads, anatomy_mode="multi_head")
        model.eval()

        x = _dummy_input(3)
        anatomies = ["brain", "prostate", "brain"]
        with torch.no_grad():
            output = model(x, anatomy=anatomies)

        assert output["score"].shape == (3,)
        assert output["logits"].shape == (3, 1)
        assert output["attention"].shape == (3, 27, 1)

    def test_multi_head_requires_anatomy(self) -> None:
        """Multi-head should raise ValueError if anatomy not provided."""
        encoder = _dummy_encoder()
        heads = nn.ModuleDict({"brain": AttentionMIL()})
        model = TriageModel(encoder, heads, anatomy_mode="multi_head")
        model.eval()

        x = _dummy_input(2)
        with pytest.raises(ValueError, match="anatomy is required"):
            with torch.no_grad():
                model(x)

    def test_multi_head_module_dict_check(self) -> None:
        """Multi-head mode should raise TypeError if mil_head is not ModuleDict."""
        encoder = _dummy_encoder()
        mil = AttentionMIL()
        with pytest.raises(TypeError, match="nn.ModuleDict"):
            TriageModel(encoder, mil, anatomy_mode="multi_head")

    def test_invalid_anatomy_mode(self) -> None:
        """Invalid anatomy_mode should raise ValueError."""
        encoder = _dummy_encoder()
        mil = AttentionMIL()
        with pytest.raises(ValueError, match="Unknown anatomy_mode"):
            TriageModel(encoder, mil, anatomy_mode="invalid_mode")

    def test_encoder_projection_changes_during_training(self) -> None:
        """When freeze_projection=False, projection params should change and swin should not."""
        encoder = TriadEncoder(
            checkpoint_path="nonexistent.pth",
            freeze=True,
            freeze_projection=False,
        )
        mil = AttentionMIL()
        model = TriageModel(encoder, mil)
        model.train()

        initial_proj_params = {
            name: param.clone() for name, param in encoder.projection.named_parameters()
        }
        initial_swin_params = {
            name: param.clone() for name, param in encoder.swin.named_parameters()
        }

        x = torch.randn(2, 1, 96, 96, 96)
        output = model(x)
        loss = output["logits"].sum()
        loss.backward()

        for name, param in encoder.swin.named_parameters():
            assert torch.equal(param, initial_swin_params[name]), (
                f"Swin param '{name}' changed but should be frozen"
            )

        proj_has_grad = any(
            p.grad is not None for p in encoder.projection.parameters()
        )
        assert proj_has_grad, "Projection params should have gradients"


class TestBuildTriageModel:
    """Tests for the build_triage_model factory function."""

    def test_build_from_config_shared(self) -> None:
        """Build from config dict with shared mode."""
        config = {
            "encoder": {
                "checkpoint": "nonexistent.pth",
                "freeze_backbone": True,
                "embed_dim": 768,
            },
            "mil": {
                "hidden_dim": 256,
                "attention_dim": 64,
                "dropout": 0.2,
                "pooling": "attention",
            },
            "anatomy_mode": "shared",
        }
        model = build_triage_model(config, device="cpu")
        assert isinstance(model, TriageModel)
        assert model.anatomy_mode == "shared"

        x = torch.randn(2, 1, 96, 96, 96)
        with torch.no_grad():
            output = model(x)
        assert "score" in output
        assert output["score"].shape == (2,)

    def test_build_from_config_multi_head(self) -> None:
        """Build from config dict with multi_head mode."""
        config = {
            "encoder": {
                "checkpoint": "nonexistent.pth",
                "freeze_backbone": True,
                "embed_dim": 768,
            },
            "mil": {
                "hidden_dim": 256,
                "attention_dim": 64,
            },
            "anatomy_mode": "multi_head",
            "num_anatomies": 2,
            "anatomies": ["brain", "prostate"],
        }
        model = build_triage_model(config, device="cpu")
        assert isinstance(model, TriageModel)
        assert model.anatomy_mode == "multi_head"

        x = torch.randn(1, 1, 96, 96, 96)
        with torch.no_grad():
            output = model(x, anatomy="brain")
        assert output["score"].shape == (1,)

    def test_predict_restores_training_state(self) -> None:
        """predict() should not permanently change model.eval/train state."""
        encoder = _dummy_encoder()
        mil = AttentionMIL()
        model = TriageModel(encoder, mil)
        model.train()
        assert model.training is True

        x = _dummy_input(1)
        model.predict(x)
        assert model.training is True, "predict() should restore training state"

        model.eval()
        assert model.training is False
        model.predict(x)
        assert model.training is False, "predict() should preserve eval state"
