"""Lightning module for training the SegmentationModel (Model B).

Minimal training loop: encoder frozen, only decoder trained.
Single-task: heatmap vs GT mask with BCE + Dice loss.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import pytorch_lightning as pl

from triagemri.training.losses import SegmentationLoss


class SegmentationLightningModule(pl.LightningModule):
    """Lightning module for :class:`~triagemri.models.segmentation.SegmentationModel`.

    Args:
        model: :class:`SegmentationModel` instance.
        loss_fn: :class:`SegmentationLoss` instance.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_epochs: LR warmup epochs.
        min_lr: Minimum LR for cosine annealing.
        max_epochs: Total epochs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: SegmentationLoss,
        lr: float = 5.0e-4,
        weight_decay: float = 1.0e-3,
        warmup_epochs: int = 5,
        min_lr: float = 1.0e-6,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_epochs = max_epochs

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        volume = batch["volume"]
        mask = batch["mask"]

        output = self.model(volume)
        heatmap = output["heatmap"]
        loss = self.loss_fn(heatmap, mask)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            dice = 1.0 - self.loss_fn._soft_dice_loss(heatmap, mask).mean()
            self.log("train/dice", dice, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        volume = batch["volume"]
        mask = batch["mask"]

        output = self.model(volume)
        heatmap = output["heatmap"]
        loss = self.loss_fn(heatmap, mask)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            dice = 1.0 - self.loss_fn._soft_dice_loss(heatmap, mask).mean()
            self.log("val/dice", dice, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        t_max = max(1, self.max_epochs - self.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=self.min_lr
        )
        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, total_iters=self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, scheduler],
                milestones=[self.warmup_epochs],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
