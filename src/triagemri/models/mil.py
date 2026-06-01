"""Attention-based Multiple Instance Learning (MIL) pooling heads.

Implements standard and gated attention MIL as described in
Ilse et al. (2018) "Attention-based Deep Multiple Instance Learning"
and CLAM (Lu et al., 2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """Standard attention-based MIL pooling.

    Architecture::

        instance_features (B,N,D)
          → Linear(D, hidden_dim) → ReLU → Dropout
          → Linear(hidden_dim, attention_dim) → Tanh
          → Linear(attention_dim, 1) → Softmax (over N)
          → Weighted sum → Linear(hidden_dim, 1) → logits

    The weighted sum is ``Z = sum(A_i * H_i)`` over all N instances,
    then a linear classifier produces a single logit per bag.

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
        dropout: float = 0.3,
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

        A_raw = self.attention(H)
        A = F.softmax(A_raw, dim=1)

        Z = torch.sum(A * H, dim=1)

        logits = self.classifier(Z)
        return logits, A


class GatedAttentionMIL(nn.Module):
    """Gated attention mechanism (CLAM / Ilse et al., 2018).

    Uses two parallel attention pathways with a gating mechanism
    to learn more complex instance relationships and prevent the
    attention from saturating.

    Architecture::

        H = ReLU(Dropout(Linear(D, hidden_dim)(x)))
        V = tanh(Linear(hidden_dim, L)(H))
        U = sigmoid(Linear(hidden_dim, L)(H))
        A = softmax(weight(V ⊙ U), dim=1)
        Z = sum(A * H, dim=1)
        logits = Linear(hidden_dim, 1)(Z)

    where ``L = attention_dim`` and ``weight`` is a learned linear
    projection ``Linear(L, 1)``.

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
        dropout: float = 0.3,
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

        V = torch.tanh(self.attention_V(H))
        U = torch.sigmoid(self.attention_U(H))

        A_raw = self.attention_weights(V * U)
        A = F.softmax(A_raw, dim=1)

        Z = torch.sum(A * H, dim=1)

        logits = self.classifier(Z)
        return logits, A


def build_mil_head(
    input_dim: int = 768,
    hidden_dim: int = 512,
    attention_dim: int = 128,
    dropout: float = 0.3,
    pooling: str = "gated_attention",
) -> AttentionMIL | GatedAttentionMIL:
    """Factory function to construct the appropriate MIL head.

    Args:
        input_dim: Dimensionality of each instance feature vector.
        hidden_dim: Hidden dimension for instance projection.
        attention_dim: Dimension of the attention embedding space.
        dropout: Dropout probability applied after the ReLU activation.
        pooling: Type of attention pooling. One of:

            * ``"attention"`` – Standard attention MIL.
            * ``"gated_attention"`` – Gated attention MIL (default).

    Returns:
        An instance of :class:`AttentionMIL` or :class:`GatedAttentionMIL`.

    Raises:
        ValueError: If ``pooling`` is not a recognised type.
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
