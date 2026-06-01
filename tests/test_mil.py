"""Unit tests for the MIL pooling module."""

from __future__ import annotations

import pytest
import torch

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from triagemri.models.mil import AttentionMIL, GatedAttentionMIL, build_mil_head


class TestAttentionMIL:
    """Interface and correctness tests for AttentionMIL."""

    def test_forward_shape(self) -> None:
        """MIL should accept (B, N, D) and return logits (B, 1) + attention (B, N, 1)."""
        mil = AttentionMIL(input_dim=768, hidden_dim=512, attention_dim=128, dropout=0.3)
        features = torch.randn(4, 27, 768)
        logits, attention = mil(features)
        assert logits.shape == (4, 1), f"Expected logits (4, 1), got {logits.shape}"
        assert attention.shape == (4, 27, 1), (
            f"Expected attention (4, 27, 1), got {attention.shape}"
        )

    def test_attention_sums_to_one(self) -> None:
        """Attention weights should sum to 1 per sample (softmax over N)."""
        mil = AttentionMIL()
        features = torch.randn(2, 10, 768)
        _, attention = mil(features)
        sums = attention.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
            f"Attention sums: {sums.squeeze().tolist()}"
        )

    def test_gated_vs_standard(self) -> None:
        """Both variants should work with same input shape."""
        features = torch.randn(3, 15, 768)
        standard = AttentionMIL()
        gated = GatedAttentionMIL()
        logits_s, attn_s = standard(features)
        logits_g, attn_g = gated(features)
        assert logits_s.shape == logits_g.shape == (3, 1)
        assert attn_s.shape == attn_g.shape == (3, 15, 1)

    def test_deterministic(self) -> None:
        """Same input should produce same output (no randomness in forward)."""
        mil = AttentionMIL()
        mil.eval()
        features = torch.randn(2, 27, 768)
        with torch.no_grad():
            logits1, attn1 = mil(features)
            logits2, attn2 = mil(features)
        assert torch.allclose(logits1, logits2, atol=1e-6), "Logits are not deterministic"
        assert torch.allclose(attn1, attn2, atol=1e-6), "Attention is not deterministic"

    def test_build_mil_head_attention(self) -> None:
        """build_mil_head with 'attention' returns AttentionMIL."""
        head = build_mil_head(pooling="attention")
        assert isinstance(head, AttentionMIL)

    def test_build_mil_head_gated(self) -> None:
        """build_mil_head with 'gated_attention' returns GatedAttentionMIL."""
        head = build_mil_head(pooling="gated_attention")
        assert isinstance(head, GatedAttentionMIL)

    def test_build_mil_head_invalid(self) -> None:
        """build_mil_head with unknown pooling raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pooling type"):
            build_mil_head(pooling="unknown_pooling")

    def test_trainable_parameters_exist(self) -> None:
        """MIL head should have trainable parameters."""
        mil = AttentionMIL()
        trainable = sum(p.numel() for p in mil.parameters() if p.requires_grad)
        assert trainable > 0, "MIL head should have trainable parameters"
