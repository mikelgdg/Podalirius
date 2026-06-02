"""3D anomaly heatmap decoder for pseudo-segmentation.

Provides :class:`AnomalyDecoder`, a lightweight 3D decoder that upscales
encoder feature pyramid stages into a full-resolution (96³) anomaly heatmap.
Uses attention gates on skip connections for precise boundary refinement.
Trained with weak supervision via MIL attention guidance.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention Gate (Oktay et al., 2018 — adapted for 3D)
# ---------------------------------------------------------------------------


class AttentionGate3D(nn.Module):
    """Pixel-wise gating of skip connections based on decoder context.

    The gate learns, per spatial position, how much of the fine-grained
    skip feature to inject into the coarser upsampled feature.  This
    preserves high-resolution detail only where the decoder "needs" it
    (e.g. lesion boundaries) and suppresses it elsewhere, preventing
    blurring from naive addition.

    Args:
        F_g: Number of channels in the gating signal (upsampled decoder).
        F_l: Number of channels in the skip connection.
        F_int: Intermediate channel dimension for the attention map.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()

        self.W_g = nn.Conv3d(F_g, F_int, kernel_size=1, bias=False)
        self.W_x = nn.Conv3d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Conv3d(F_int, 1, kernel_size=1, bias=False)

        for m in [self.W_g, self.W_x]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.psi.weight)

    def forward(
        self, g: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention coefficients.

        Args:
            g: Gating signal from decoder ``(B, F_g, D, H, W)``.
            x: Skip features from encoder ``(B, F_l, D, H, W)``.

        Returns:
            Attention coefficients ``(B, 1, D, H, W)`` in [0, 1].
        """
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="trilinear", align_corners=False)

        alignment = F.relu(self.W_g(g) + self.W_x(x), inplace=True)
        return torch.sigmoid(self.psi(alignment))


# ---------------------------------------------------------------------------
# Anomaly Decoder
# ---------------------------------------------------------------------------


class AnomalyDecoder(nn.Module):
    """Lightweight 3D decoder producing a voxel-level anomaly heatmap.

    Upscales encoder features from coarsest (stage-4, 3³) to 96³ using
    transposed convolutions.  Skip connections from finer stages are
    gated by :class:`AttentionGate3D` for crisp boundary delineation.

    Args:
        embed_dim: Dimensionality of encoder features at each stage.
        upconv_channels: Output channels for each decoder block.
        dropout: Dropout probability after each upconv block.
        use_attention_gates: Enable attention gating on skip connections.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        upconv_channels: list[int] | None = None,
        dropout: float = 0.1,
        use_attention_gates: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_attention_gates = use_attention_gates

        if upconv_channels is None:
            upconv_channels = [256, 128, 64, 32, 16]

        first_ch = upconv_channels[0]

        self.input_proj = nn.Sequential(
            nn.Conv3d(embed_dim, first_ch, kernel_size=1),
            nn.BatchNorm3d(first_ch),
            nn.ReLU(inplace=True),
        )

        prev_ch: int = first_ch
        self.blocks = nn.ModuleList()

        for i, out_ch in enumerate(upconv_channels):
            is_last = i == len(upconv_channels) - 1
            block = _DecoderBlock(
                in_channels=prev_ch,
                skip_channels=embed_dim,
                out_channels=out_ch,
                dropout=0.0 if is_last else dropout,
                use_skip=i < 3,
                use_attention_gate=(i < 3) and use_attention_gates,
            )
            self.blocks.append(block)
            prev_ch = out_ch

        self.final = nn.Conv3d(prev_ch, 1, kernel_size=1)

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        sorted_keys = sorted(
            pyramid.keys(),
            key=lambda k: int(k.replace("stage", "")),
            reverse=True,
        )

        x: torch.Tensor = self.input_proj(pyramid[sorted_keys[0]])

        for i, block in enumerate(self.blocks):
            skip: torch.Tensor | None = None
            skip_idx = i + 1
            if skip_idx < len(sorted_keys):
                skip = pyramid.get(sorted_keys[skip_idx])
            x = block(x, skip)

        heatmap = self.final(x)
        return heatmap


# ---------------------------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------------------------


class _DecoderBlock(nn.Module):
    """Single decoder block: upscale → attention gate → skip fusion → conv.

    If *use_attention_gate* is True, the skip connection is gated pixel-wise
    by an :class:`AttentionGate3D` before projection and addition.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int = 768,
        out_channels: int = 256,
        dropout: float = 0.1,
        use_skip: bool = True,
        use_attention_gate: bool = False,
    ) -> None:
        super().__init__()

        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.use_skip = use_skip
        self.skip_proj: nn.Module | None = None
        self.attn_gate: AttentionGate3D | None = None

        if use_skip:
            self.skip_proj = nn.Conv3d(skip_channels, out_channels, kernel_size=1)
            if use_attention_gate:
                self.attn_gate = AttentionGate3D(
                    F_g=out_channels,
                    F_l=skip_channels,
                    F_int=out_channels // 2,
                )

        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.upconv(x)

        if skip is not None and self.use_skip and self.skip_proj is not None:
            # Attention gate: where should the decoder look at high-res detail?
            if self.attn_gate is not None:
                gate_coeff = self.attn_gate(x, skip)
                skip_gated = skip * gate_coeff
            else:
                skip_gated = skip

            skip_proj = self.skip_proj(skip_gated)
            if x.shape[2:] != skip_proj.shape[2:]:
                skip_proj = F.interpolate(
                    skip_proj, size=x.shape[2:],
                    mode="trilinear", align_corners=False,
                )
            x = x + skip_proj

        x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------


def total_variation_3d(heatmap: torch.Tensor) -> torch.Tensor:
    """Compute total variation loss for a 3D heatmap.

    Encourages spatial smoothness.

    Args:
        heatmap: ``(B, 1, D, H, W)`` tensor.

    Returns:
        Scalar TV loss (mean over batch).
    """
    diff_d = torch.abs(heatmap[:, :, 1:, :, :] - heatmap[:, :, :-1, :, :])
    diff_h = torch.abs(heatmap[:, :, :, 1:, :] - heatmap[:, :, :, :-1, :])
    diff_w = torch.abs(heatmap[:, :, :, :, 1:] - heatmap[:, :, :, :, :-1])
    return (diff_d.mean() + diff_h.mean() + diff_w.mean()) / 3.0
