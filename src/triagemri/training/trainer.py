"""PyTorch Lightning module for TriageModel training.

Provides :class:`TriageLightningModule` wrapping the model, loss,
optimiser, and LR scheduling in a single Lightning module.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


class TriageLightningModule(pl.LightningModule):
    """Lightning module wrapping a :class:`~triagemri.models.TriageModel`.

    Handles training step, validation step with epoch-end metric
    computation, and learning rate scheduling with linear warmup
    followed by cosine annealing.

    Args:
        model: A :class:`~triagemri.models.TriageModel` instance.
        loss_fn: A callable loss that accepts ``(logits, targets)``.
        config: Merged :class:`~triagemri.config.Config` instance.  Uses
            ``config.training`` for hyper-parameters:
            ``optimizer.lr``, ``optimizer.weight_decay``,
            ``scheduler.warmup_epochs``, ``scheduler.min_lr``,
            ``max_epochs``.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr: float = config.training.optimizer.lr
        self.weight_decay: float = config.training.optimizer.weight_decay
        self.warmup_epochs: int = config.training.scheduler.warmup_epochs
        self.min_lr: float = config.training.scheduler.min_lr
        self.max_epochs: int = config.training.max_epochs

        self.val_preds: List[torch.Tensor] = []
        self.val_labels: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        anatomy: Optional[str | List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Delegates to ``self.model.forward``."""
        return self.model(x, anatomy)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        volume: torch.Tensor = batch["volume"]
        label: torch.Tensor = batch["label"].float()
        anatomy = batch.get("anatomy", None)

        output = self.model(volume, anatomy)
        logits = output["logits"].squeeze(-1)
        loss = self.loss_fn(logits, label)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        volume = batch["volume"]
        label: torch.Tensor = batch["label"].float()
        anatomy = batch.get("anatomy", None)

        output = self.model(volume, anatomy)
        logits = output["logits"].squeeze(-1)
        loss = self.loss_fn(logits, label)

        scores = torch.sigmoid(logits)
        if self.trainer.world_size > 1:
            gathered_scores = self.all_gather(scores.detach())
            gathered_label = self.all_gather(label.detach())
            self.val_preds.append(gathered_scores.cpu())
            self.val_labels.append(gathered_label.cpu())
        else:
            self.val_preds.append(scores.detach().cpu())
            self.val_labels.append(label.detach().cpu())

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss, "scores": scores, "labels": label}

    def on_validation_epoch_end(self) -> None:
        if not self.val_preds:
            return

        all_preds = torch.cat(self.val_preds).numpy().astype(np.float64)
        all_labels = torch.cat(self.val_labels).numpy().astype(np.float64)

        from triagemri.training.metrics import compute_triage_metrics

        try:
            metrics = compute_triage_metrics(all_labels, all_preds)
        except ValueError:
            warnings.warn(
                "compute_triage_metrics failed — validation may have only "
                "one class. Skipping metric logging this epoch."
            )
            return
        finally:
            self.val_preds.clear()
            self.val_labels.clear()

        for name, value in metrics.items():
            if isinstance(value, (int, float)) and name != "roc_auc":
                if not np.isnan(value):
                    self.log(f"val/{name}", value, on_epoch=True)

        if "roc_auc" in metrics and not np.isnan(metrics["roc_auc"]):
            self.log("val/roc_auc", metrics["roc_auc"], on_epoch=True, prog_bar=True)

        if "ppv" in metrics and not np.isnan(metrics["ppv"]):
            self.log("val/precision@99sens", metrics["ppv"], on_epoch=True, prog_bar=True)
        if "specificity" in metrics and not np.isnan(metrics["specificity"]):
            self.log("val/specificity", metrics["specificity"], on_epoch=True)
        if "discard_rate" in metrics and not np.isnan(metrics["discard_rate"]):
            self.log("val/discard_rate", metrics["discard_rate"], on_epoch=True)

        # Also compute accuracy at threshold 0.5 (simple binary decision)
        try:
            acc_05 = float((all_labels == (all_preds >= 0.5).astype(all_labels.dtype)).mean())
            if not np.isnan(acc_05):
                self.log("val/acc@0.5", acc_05, on_epoch=True, prog_bar=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        decay_params = []
        no_decay_params = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.max_epochs - self.warmup_epochs - 1),
            eta_min=self.min_lr,
        )

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.0,
                total_iters=self.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[self.warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
