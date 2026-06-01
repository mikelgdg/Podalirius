"""Attention-based Multiple Instance Learning (MIL) pooling heads.

Provides standard, gated, and multi-scale attention MIL as described in
Ilse et al. (2018) and CLAM (Lu et al., 2021).  The multi-scale variant
attends at multiple spatial resolutions (stage-3: 6³ and stage-4: 3³)
and combines scores via a learned gate.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AttentionMIL (standard)
# ---------------------------------------------------------------------------


class AttentionMIL(nn.Module):
    """Standard attention-based MIL pooling.

    Args:
        input_dim: Dimensionality of each instance feature vector.
        hidden_dim: Hidden dimension for instance projection.
        attention_dim: Dimension of the attention embedding space.
        dropout: Dropout probability applied after the ReLU activation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        attention_dim: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        self.instance_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.input_skip: nn.Module | None = None
        if input_dim != hidden_dim:
            self.input_skip = nn.Linear(input_dim, hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self, instance_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            instance_features: ``(B, N, input_dim)`` instance embeddings.

        Returns:
            logits: ``(B, 1)`` raw logits for binary classification.
            attention_weights: ``(B, N, 1)`` normalised attention weights.
        """
        H = self.instance_proj(instance_features)

        if self.input_skip is not None:
            H = H + self.input_skip(instance_features)

        A_raw = self.attention(H)
        A = F.softmax(A_raw, dim=1)

        Z = torch.sum(A * H, dim=1)

        logits = self.classifier(Z)
        return logits, A


# ---------------------------------------------------------------------------
# GatedAttentionMIL
# ---------------------------------------------------------------------------


class GatedAttentionMIL(nn.Module):
    """Gated attention mechanism (CLAM / Ilse et al., 2018).

    Two parallel attention pathways with a gating mechanism prevent the
    attention from saturating.

    Args:
        input_dim: Dimensionality of each instance feature vector.
        hidden_dim: Hidden dimension for instance projection.
        attention_dim: Dimension ``L`` of the attention embedding space.
        dropout: Dropout probability applied after the ReLU activation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        attention_dim: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        self.instance_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.input_skip: nn.Module | None = None
        if input_dim != hidden_dim:
            self.input_skip = nn.Linear(input_dim, hidden_dim)

        self.attention_V = nn.Linear(hidden_dim, attention_dim)
        self.attention_U = nn.Linear(hidden_dim, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self, instance_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            instance_features: ``(B, N, input_dim)`` instance embeddings.

        Returns:
            logits: ``(B, 1)`` raw logits for binary classification.
            attention_weights: ``(B, N, 1)`` normalised attention weights.
        """
        H = self.instance_proj(instance_features)

        if self.input_skip is not None:
            H = H + self.input_skip(instance_features)

        V = torch.tanh(self.attention_V(H))
        U = torch.sigmoid(self.attention_U(H))

        A_raw = self.attention_weights(V * U)
        A = F.softmax(A_raw, dim=1)

        Z = torch.sum(A * H, dim=1)

        logits = self.classifier(Z)
        return logits, A


# ---------------------------------------------------------------------------
# MultiScaleAttentionMIL
# ---------------------------------------------------------------------------


class MultiScaleAttentionMIL(nn.Module):
    """Multi-scale Gated Attention MIL operating on a feature pyramid.

    Applies :class:`GatedAttentionMIL` independently to each resolution,
    then fuses the per-scale bag-level representations with a learned gate
    before producing a final score.

    Args:
        embed_dim: Dimensionality of features at each scale.
        hidden_dim: Hidden dim for MIL heads.
        attention_dim: Attention embedding dim.
        dropout: Dropout probability.
        scales: Which pyramid keys to attend over (default ``["stage3", "stage4"]``).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        attention_dim: int = 128,
        dropout: float = 0.15,
        scales: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.scales = scales or ["stage3", "stage4"]

        self.heads = nn.ModuleDict({
            scale: GatedAttentionMIL(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                attention_dim=attention_dim,
                dropout=dropout,
            )
            for scale in self.scales
        })

        combined_dim = hidden_dim * len(self.scales)
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Linear(combined_dim // 2, 1),
        )

        self.fusion = nn.Linear(combined_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self, pyramid: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Multi-scale forward pass.

        Args:
            pyramid: Dict mapping scale name to ``(B, C, D, H, W)``
                feature tensors.

        Returns:
            logits: ``(B, 1)`` final logits.
            attention: ``(B, N_total, 1)`` concatenated attention across all scales.
            per_scale: Dict mapping scale name to per-scale ``(logits, attention)``.
        """
        per_scale_logits: Dict[str, torch.Tensor] = {}
        per_scale_attention: Dict[str, torch.Tensor] = {}
        pooled_list: list[torch.Tensor] = []

        for scale in self.scales:
            feat = pyramid.get(scale)
            if feat is None:
                raise KeyError(
                    f"Scale '{scale}' not found in pyramid. "
                    f"Available: {list(pyramid.keys())}"
                )
            B, C, D, H, W = feat.shape
            instances = feat.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

            logits_s, attn_s = self.heads[scale](instances)
            per_scale_logits[scale] = logits_s
            per_scale_attention[scale] = attn_s

            Z_s = torch.sum(attn_s * self.heads[scale].instance_proj(instances), dim=1)
            if self.heads[scale].input_skip is not None:
                Z_s = Z_s + torch.sum(
                    attn_s * self.heads[scale].input_skip(instances), dim=1
                )
            pooled_list.append(Z_s)

        combined = torch.cat(pooled_list, dim=1)
        gate_weights = torch.sigmoid(self.gate(combined))
        fused = gate_weights * self.fusion(combined)
        logits = self.classifier(fused)

        all_attention = torch.cat(
            [per_scale_attention[s] for s in self.scales], dim=1
        )

        return logits, all_attention, {
            scale: (per_scale_logits[scale], per_scale_attention[scale])
            for scale in self.scales
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_mil_head(
    input_dim: int = 768,
    hidden_dim: int = 512,
    attention_dim: int = 128,
    dropout: float = 0.15,
    pooling: str = "gated_attention",
) -> AttentionMIL | GatedAttentionMIL:
    """Factory function to construct the appropriate MIL head.

    Args:
        input_dim: Dimensionality of each instance feature vector.
        hidden_dim: Hidden dimension for instance projection.
        attention_dim: Dimension of the attention embedding space.
        dropout: Dropout probability.
        pooling: ``"attention"`` or ``"gated_attention"``.

    Returns:
        An MIL head instance.

    Raises:
        ValueError: If *pooling* is not a recognised type.
    """
    if pooling == "attention":
        return AttentionMIL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )
    if pooling == "gated_attention":
        return GatedAttentionMIL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown pooling type '{pooling}'. "
        f"Expected 'attention' or 'gated_attention'."
    )


def build_multiscale_mil(
    embed_dim: int = 768,
    hidden_dim: int = 512,
    attention_dim: int = 128,
    dropout: float = 0.15,
    scales: list[str] | None = None,
) -> MultiScaleAttentionMIL:
    """Factory for multi-scale MIL head.

    Args:
        embed_dim: Feature dimensionality per stage.
        hidden_dim: Hidden dim.
        attention_dim: Attention embedding dim.
        dropout: Dropout.
        scales: Which pyramid stages to attend over.

    Returns:
        A configured :class:`MultiScaleAttentionMIL`.
    """
    return MultiScaleAttentionMIL(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        dropout=dropout,
        scales=scales,
    )
