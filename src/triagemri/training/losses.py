"""Loss functions for triage model training."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Args:
        alpha: Weight for positive class (default 0.75 for medical triage).
        gamma: Focusing parameter.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
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
    """Binary cross-entropy with optional positive-class weighting."""

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
    """Asymmetric loss with decoupled focusing."""

    def __init__(self, gamma_neg: int = 2, gamma_pos: int = 0, clip: float = 0.05):
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


class PseudoLabelLoss(nn.Module):
    """Instance-level pseudo-label supervision from MIL attention.

    Each of the N attention cells in the MIL head provides a weak label
    for the corresponding region of the decoder heatmap:

    - ``attention[i] > pos_factor / N``  →  label 1 (suspicious region)
    - ``attention[i] < neg_factor / N``  →  label 0 (likely normal region)
    - Intermediate attention             →  ignored (uncertain)

    When multi-scale attention is used (e.g. stage3+stage4 giving 243
    instances), only the coarsest scale (last 27 = stage4) is used for
    pseudo-labels, as it provides the most reliable signal.

    Args:
        pos_factor: Multiplier over uniform baseline for positive label.
        neg_factor: Multiplier over uniform baseline for negative label.
    """

    def __init__(self, pos_factor: float = 2.0, neg_factor: float = 1.0) -> None:
        super().__init__()
        self.pos_factor = pos_factor
        self.neg_factor = neg_factor

    def _extract_coarse_attention(self, attention: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Extract the coarsest cubic attention sub-grid.

        If N is a sum of two cubes (e.g. 216+27=243), returns the
        smaller cube (the last entries).  Otherwise returns the whole tensor.
        """
        B, N, _one = attention.shape
        attn = attention.squeeze(-1)

        for small_side in range(2, 8):
            small_n = small_side ** 3
            large_n = N - small_n
            large_side = round(large_n ** (1 / 3))
            if large_side ** 3 == large_n and small_n > 0:
                return attn[:, -small_n:], small_side

        return attn, round(N ** (1 / 3))

    def forward(
        self, heatmap: torch.Tensor, attention: torch.Tensor
    ) -> torch.Tensor:
        B, _c, _d, _h, _w = heatmap.shape

        attn_flat, side = self._extract_coarse_attention(attention)
        N_eff = side ** 3
        target_size = (side, side, side)

        region_means = F.adaptive_avg_pool3d(heatmap, target_size).view(B, -1)

        pos_mask = attn_flat > (self.pos_factor / N_eff)
        neg_mask = attn_flat < (self.neg_factor / N_eff)

        loss = torch.tensor(0.0, device=heatmap.device)
        count = 0

        for b in range(B):
            pos_regions = region_means[b][pos_mask[b]]
            neg_regions = region_means[b][neg_mask[b]]
            if pos_regions.numel() > 0:
                loss = loss + F.binary_cross_entropy_with_logits(
                    pos_regions, torch.ones_like(pos_regions), reduction="mean"
                )
                count += 1
            if neg_regions.numel() > 0:
                loss = loss + F.binary_cross_entropy_with_logits(
                    neg_regions, torch.zeros_like(neg_regions), reduction="mean"
                )
                count += 1

        if count == 0:
            return loss
        return loss / count

    def coverage(self, attention: torch.Tensor) -> float:
        attn_flat, side = self._extract_coarse_attention(attention)
        N_eff = side ** 3
        pos_mask = attn_flat > (self.pos_factor / N_eff)
        neg_mask = attn_flat < (self.neg_factor / N_eff)
        return (pos_mask | neg_mask).float().mean().item()


class MultiTaskLoss(nn.Module):
    """Combined loss for global classification + voxel heatmap + GT mask supervision.

    ``loss = λ_bce * BCE + λ_smooth * TV + λ_attn * KL(attn, pool(hm)) + λ_mask * BCE(hm, mask) + λ_dice * Dice(hm, mask)``

    Args:
        lambda_bce: Weight for BCE on global score.
        lambda_smooth: Weight for TV smoothness on heatmap.
        lambda_attn: Weight for KL between MIL attention and pooled heatmap.
        lambda_mask: Weight for BCEWithLogits on heatmap vs GT mask.
        lambda_dice: Weight for soft Dice loss on heatmap vs GT mask.
    """

    def __init__(
        self,
        lambda_bce: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_attn: float = 0.05,
        lambda_mask: float = 0.5,
        lambda_dice: float = 0.5,
    ):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_smooth = lambda_smooth
        self.lambda_attn = lambda_attn
        self.lambda_mask = lambda_mask
        self.lambda_dice = lambda_dice

    @staticmethod
    def _soft_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss for 3D volumes.

        Args:
            pred: ``(B, 1, D, H, W)`` logits.
            target: ``(B, 1, D, H, W)`` binary mask.

        Returns:
            Scalar loss (1 - Dice).
        """
        probs = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (probs * target).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return (1.0 - dice).mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        heatmap: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
        total = self.lambda_bce * loss_bce

        if heatmap is not None and self.lambda_smooth > 0:
            from triagemri.models.decoder import total_variation_3d
            tv = total_variation_3d(heatmap)
            total = total + self.lambda_smooth * tv

        if heatmap is not None and attention is not None and self.lambda_attn > 0:
            B, N, _attn_dim = attention.shape
            side = round(N ** (1 / 3))

            target_size: tuple[int, ...]
            if side ** 3 == N:
                target_size = (side, side, side)
            elif N > 30:
                for small_side in range(2, 8):
                    small_n = small_side ** 3
                    if small_n < N and (N - small_n) > 0:
                        target_size = (small_side, small_side, small_side)
                        attention = attention[:, -small_n:, :]
                        break
                else:
                    target_size = (max(1, round(N ** (1 / 3))),) * 3
            else:
                target_size = (max(1, round(N ** (1 / 3))),) * 3

            hm_down = F.adaptive_avg_pool3d(heatmap, target_size)
            hm_flat = hm_down.view(B, -1)
            hm_soft = F.softmax(hm_flat, dim=1)

            attn = attention.squeeze(-1)
            attn_norm = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
            attn_kl = attn_norm * torch.log((attn_norm + 1e-8) / (hm_soft + 1e-8))
            loss_attn = attn_kl.mean()
            total = total + self.lambda_attn * loss_attn

        if heatmap is not None and mask is not None:
            if mask.min() >= 0.0:
                if self.lambda_mask > 0:
                    loss_mask = F.binary_cross_entropy_with_logits(heatmap, mask)
                    total = total + self.lambda_mask * loss_mask
                if self.lambda_dice > 0:
                    loss_dice = self._soft_dice_loss(heatmap, mask)
                    total = total + self.lambda_dice * loss_dice

        return total


LOSS_REGISTRY = {
    "focal_loss": FocalLoss,
    "focal": FocalLoss,
    "weighted_bce": WeightedBCEWithLogitsLoss,
    "asymmetric": AsymmetricLoss,
    "asl": AsymmetricLoss,
    "multi_task": MultiTaskLoss,
}


def get_loss_fn(name: str, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        available = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{name}'. Available: {available}")
    return LOSS_REGISTRY[name](**kwargs)
