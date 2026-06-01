"""Tests for the TriageLightningModule."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
pl = pytest.importorskip("pytorch_lightning", reason="PyTorch Lightning not installed")

import torch.nn as nn

from triagemri.config import Config
from triagemri.models.encoder import TriadEncoder
from triagemri.models.mil import AttentionMIL
from triagemri.models.triage import TriageModel
from triagemri.training.losses import get_loss_fn
from triagemri.training.trainer import TriageLightningModule


def _make_config(overrides: dict | None = None) -> Config:
    """Return a minimal merged config suitable for tests."""
    base: dict = {
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "max_epochs": 5,
            "gradient_accumulation_steps": 1,
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
            "scheduler": {"name": "cosine_annealing", "warmup_epochs": 1, "min_lr": 1e-6},
            "loss": {"name": "focal_loss", "alpha": 0.25, "gamma": 2.0},
            "early_stopping": {"monitor": "val/roc_auc", "patience": 5, "mode": "max"},
            "checkpoint": {"monitor": "val/roc_auc", "mode": "max", "save_top_k": 1},
            "precision": "32-true",
            "accelerator": "cpu",
            "devices": 1,
            "seed": 42,
        },
        "model": {
            "encoder": {"checkpoint": "nonexistent.pth", "freeze_backbone": True},
            "mil": {"hidden_dim": 128, "attention_dim": 64, "dropout": 0.2, "pooling": "attention"},
            "anatomy_mode": "shared",
            "num_anatomies": 3,
        },
        "data": {
            "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        },
    }
    if overrides:
        _deep_update(base, overrides)
    return Config(base)


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def _dummy_model() -> TriageModel:
    encoder = TriadEncoder(
        checkpoint_path="nonexistent.pth",
        freeze=True,
    )
    mil = AttentionMIL(input_dim=768, hidden_dim=128, attention_dim=64, dropout=0.2)
    return TriageModel(encoder, mil, anatomy_mode="shared")


def _dummy_batch(batch_size: int = 2) -> dict:
    return {
        "volume": torch.randn(batch_size, 1, 96, 96, 96),
        "label": torch.randint(0, 2, (batch_size,)).float(),
        "anatomy": ["brain"] * batch_size,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTriageLightningModule:
    """Tests for the Lightning module."""

    def test_initialization(self) -> None:
        """Module should create without errors."""
        model = _dummy_model()
        loss_fn = get_loss_fn("focal_loss", alpha=0.25, gamma=2.0)
        config = _make_config()

        module = TriageLightningModule(model, loss_fn, config)
        assert module.model is model
        assert module.loss_fn is loss_fn
        assert module.lr == config.training.optimizer.lr
        assert module.weight_decay == config.training.optimizer.weight_decay
        assert module.warmup_epochs == config.training.scheduler.warmup_epochs
        assert module.min_lr == config.training.scheduler.min_lr
        assert module.max_epochs == config.training.max_epochs
        assert module.val_preds == []
        assert module.val_labels == []

    def test_training_step_returns_loss(self) -> None:
        """Training step should return a scalar loss tensor."""
        model = _dummy_model()
        loss_fn = get_loss_fn("focal_loss", alpha=0.25, gamma=2.0)
        config = _make_config()
        module = TriageLightningModule(model, loss_fn, config)

        batch = _dummy_batch(4)
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_configure_optimizers(self) -> None:
        """configure_optimizers should return optimizer + scheduler."""
        model = _dummy_model()
        loss_fn = get_loss_fn("focal_loss", alpha=0.25, gamma=2.0)
        config = _make_config()
        module = TriageLightningModule(model, loss_fn, config)

        opt_config = module.configure_optimizers()
        assert isinstance(opt_config, dict)
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
        assert isinstance(opt_config["optimizer"], torch.optim.Optimizer)
        assert opt_config["lr_scheduler"]["interval"] == "epoch"

    def test_validation_step_collects_predictions(self) -> None:
        """Validation step should populate val_preds/val_labels."""
        model = _dummy_model()
        model.eval()
        loss_fn = get_loss_fn("focal_loss")
        config = _make_config()
        module = TriageLightningModule(model, loss_fn, config)

        batch = _dummy_batch(4)
        with torch.no_grad():
            module.validation_step(batch, batch_idx=0)

        assert len(module.val_preds) == 1
        assert len(module.val_labels) == 1
        assert module.val_preds[0].shape == (4,)
        assert module.val_labels[0].shape == (4,)

    def test_validation_epoch_end_clears_buffers(self) -> None:
        """After on_validation_epoch_end, buffers should be empty."""
        model = _dummy_model()
        model.eval()
        loss_fn = get_loss_fn("focal_loss")
        config = _make_config()
        module = TriageLightningModule(model, loss_fn, config)

        batch = _dummy_batch(4)
        with torch.no_grad():
            module.validation_step(batch, batch_idx=0)
        module.on_validation_epoch_end()
        assert module.val_preds == []
        assert module.val_labels == []

    def test_validation_epoch_end_empty(self) -> None:
        """on_validation_epoch_end with empty buffers should not error."""
        model = _dummy_model()
        loss_fn = get_loss_fn("focal_loss")
        config = _make_config()
        module = TriageLightningModule(model, loss_fn, config)

        module.on_validation_epoch_end()
        assert module.val_preds == []
        assert module.val_labels == []

    def test_forward_delegates_to_model(self) -> None:
        """forward() should produce the same output as model()."""
        model = _dummy_model()
        model.eval()
        loss_fn = get_loss_fn("focal_loss")
        config = _make_config()
        module = TriageLightningModule(model, loss_fn, config)
        module.eval()

        x = torch.randn(2, 1, 96, 96, 96)
        with torch.no_grad():
            out_module = module(x)
            out_model = model(x)

        for key in ("score", "logits", "attention", "features"):
            assert torch.allclose(out_module[key], out_model[key], atol=1e-6), (
                f"Mismatch on key '{key}'"
            )

    def test_linear_warmup_scheduler_configured(self) -> None:
        """With warmup_epochs > 0, use SequentialLR with warmup first."""
        model = _dummy_model()
        loss_fn = get_loss_fn("focal_loss")
        config = _make_config()
        module = TriageLightningModule(model, loss_fn, config)

        opt_config = module.configure_optimizers()
        scheduler = opt_config["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    def test_no_warmup_scheduler(self) -> None:
        """With warmup_epochs = 0, skip SequentialLR."""
        model = _dummy_model()
        loss_fn = get_loss_fn("focal_loss")
        config = _make_config({"training": {"scheduler": {"warmup_epochs": 0}}})
        module = TriageLightningModule(model, loss_fn, config)

        opt_config = module.configure_optimizers()
        scheduler = opt_config["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


class TestWeightedBCELoss:
    """Tests for WeightedBCEWithLogitsLoss."""

    def test_weighted_bce_loss_decreases(self) -> None:
        """Loss should be computable and produce reasonable values."""
        from triagemri.training.losses import WeightedBCEWithLogitsLoss

        loss_fn = WeightedBCEWithLogitsLoss()
        logits = torch.tensor([[3.0]])
        targets = torch.tensor([[1.0]])
        loss_easy_pos = loss_fn(logits, targets)

        logits = torch.tensor([[0.0]])
        loss_hard = loss_fn(logits, targets)

        assert loss_hard > loss_easy_pos, "Hard case should have higher loss than easy case"

    def test_weighted_bce_with_pos_weight(self) -> None:
        """With pos_weight > 1, false negatives should be penalized more."""
        from triagemri.training.losses import WeightedBCEWithLogitsLoss

        loss_fn_high = WeightedBCEWithLogitsLoss(pos_weight=5.0)
        loss_fn_std = WeightedBCEWithLogitsLoss(pos_weight=1.0)

        logits = torch.tensor([[-2.0]])
        targets = torch.tensor([[1.0]])
        loss_high = loss_fn_high(logits, targets)
        loss_std = loss_fn_std(logits, targets)

        assert loss_high > loss_std, "Higher pos_weight should increase loss on false negatives"
