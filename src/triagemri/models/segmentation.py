"""Segmentation model: frozen Triad encoder + 3D anomaly decoder.

Produces voxel-level heatmaps (96³) for explainability.
No MIL head — decoder-only architecture for supervised mask training.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from triagemri.models.encoder import TriadEncoder
from triagemri.models.decoder import AnomalyDecoder


class SegmentationModel(nn.Module):
    """Frozen Triad encoder → AnomalyDecoder → heatmap 96³.

    Single-task model: only learns to produce anomaly heatmaps from
    ground-truth segmentation masks.  No classification head.

    Args:
        encoder: :class:`TriadEncoder` instance (frozen).
        decoder: :class:`AnomalyDecoder` instance.
        freeze_encoder: Keep encoder frozen during training.
    """

    def __init__(
        self,
        encoder: TriadEncoder,
        decoder: AnomalyDecoder,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pyramid = self.encoder(x, return_pyramid=True)
        heatmap = self.decoder(pyramid)
        return {"heatmap": heatmap}

    def train(self, mode: bool = True) -> "SegmentationModel":
        super().train(mode)
        if self._freeze_encoder:
            self.encoder.eval()
        return self


def build_segmentation_model(
    checkpoint_path: str = "weights/triad_swinb_simmim.pth",
    embed_dim: int = 768,
    upconv_channels: Optional[list[int]] = None,
    use_attention_gates: bool = True,
    freeze_encoder: bool = True,
    device: str = "cuda",
) -> SegmentationModel:
    """Build a :class:`SegmentationModel`.

    Args:
        checkpoint_path: Path to Triad weights.
        embed_dim: Embedding dimension per stage.
        upconv_channels: Decoder channel progression.
        use_attention_gates: Enable attention gates in decoder.
        freeze_encoder: Freeze encoder parameters.
        device: Target device.

    Returns:
        Configured :class:`SegmentationModel`.
    """
    if upconv_channels is None:
        upconv_channels = [256, 128, 64, 32, 16]

    encoder = TriadEncoder(
        checkpoint_path=checkpoint_path,
        freeze=freeze_encoder,
        embed_dim=embed_dim,
        freeze_projection=freeze_encoder,
        freeze_patch_embed=True,
        return_pyramid=True,
    )

    decoder = AnomalyDecoder(
        embed_dim=embed_dim,
        upconv_channels=list(upconv_channels),
        use_attention_gates=use_attention_gates,
    )

    model = SegmentationModel(
        encoder=encoder,
        decoder=decoder,
        freeze_encoder=freeze_encoder,
    )
    model.to(device)
    return model
