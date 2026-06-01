from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Args:
        alpha: Weight for the positive class (class 1). Class 0 gets (1-alpha).
               For imbalanced medical data where positives are the critical
               minority, alpha=0.75 or higher may be more appropriate than
               the default 0.25.
        gamma: Focusing parameter. Higher values increase focus on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss.float())
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - pt.to(bce_loss.dtype)) ** self.gamma
        return (focal_weight * bce_loss).mean()


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(logits)

        if self.pos_weight is None:
            return F.binary_cross_entropy_with_logits(logits, targets)

        pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: int = 4, gamma_pos: int = 0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        probs = torch.sigmoid(logits)
        probs = (probs - self.clip).clamp(min=0) / (1.0 - self.clip)

        xs_pos = (1.0 - probs) ** self.gamma_pos
        xs_neg = probs ** self.gamma_neg

        pos_term = targets * xs_pos * torch.log(probs + 1e-8)
        neg_term = (1.0 - targets) * xs_neg * torch.log(1.0 - probs + 1e-8)

        return -torch.mean(pos_term + neg_term)


LOSS_REGISTRY = {
    "focal_loss": FocalLoss,
    "focal": FocalLoss,
    "weighted_bce": WeightedBCEWithLogitsLoss,
    "asymmetric": AsymmetricLoss,
    "asl": AsymmetricLoss,
}


def get_loss_fn(name: str, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        available = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{name}'. Available: {available}")
    return LOSS_REGISTRY[name](**kwargs)
