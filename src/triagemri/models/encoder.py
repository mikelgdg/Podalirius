"""Triad 3D Swin-B SimMIM encoder for volumetric MRI feature extraction.

Loads the Triad checkpoint (SwinTransformer from MONAI) and extracts
multi-scale features from a 96x96x96 volume. Features at each stage are
spatially aligned, providing instance-level embeddings for MIL pooling.

Architecture (matching Triad QuickStart.py):
    SwinTransformer (use_v2=True, embed_dim=48, depths=(2,2,2,2))
      → multi-stage feature maps
      → interpolate to common spatial size
      → concatenate → (B, N_instances, total_dim)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriadEncoder(nn.Module):
    """3D Swin-B encoder with Triad SimMIM pre-trained weights.

    Wraps MONAI's SwinTransformer and loads the official Triad checkpoint.
    Extracts multi-scale features from a 96-cubed volume and spatially aligns
    them to produce instance-level embeddings for attention MIL.

    Input:  (B, 1, D, H, W) — volume is resized to 96-cubed internally.
    Output: (B, N_instances, embed_dim) — one embedding per spatial cell.

    With stage 4's 32x downsampling on 96³ input, N_instances = 27 (3×3×3 grid).

    Args:
        checkpoint_path: Path to Triad Swin-B SimMIM checkpoint (.pth).
        freeze: Freeze all encoder parameters.
        embed_dim: Output embedding dimension (after projection from
                   concatenated multi-stage features).
        pretrained_strict: Raise error if checkpoint keys mismatch.
    """

    def __init__(
        self,
        checkpoint_path: str = "weights/triad_swinb_simmim.pth",
        freeze: bool = True,
        embed_dim: int = 768,
        pretrained_strict: bool = False,
        freeze_projection: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self._frozen = freeze
        self._freeze_projection = freeze_projection

        self.swin, stage_dims = self._build_swin()

        loaded = self._load_triad_weights(checkpoint_path, strict=pretrained_strict)
        self._pretrained_loaded = loaded

        total_in_dim = sum(stage_dims)
        self.projection = nn.Linear(total_in_dim, embed_dim)

        if freeze:
            self._freeze()

    @staticmethod
    def _build_swin() -> Tuple[nn.Module, List[int]]:
        """Build the SwinTransformer matching Triad's config.

        Returns:
            (swin_module, list_of_stage_output_dims)
        """
        from monai.networks.nets.swin_unetr import SwinTransformer

        swin = SwinTransformer(
            in_chans=1,
            embed_dim=48,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=True,
            spatial_dims=3,
            downsample="merging",
            use_v2=True,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 96, 96, 96)
            stage_features = swin(dummy)
            stage_dims = [f.shape[1] for f in stage_features]
        return swin, stage_dims

    def _load_triad_weights(self, checkpoint_path: str, strict: bool = False) -> bool:
        """Load Triad SimMIM checkpoint into the Swin backbone.

        The checkpoint was saved from TriadHead(backbone=Swin(swinViT=SwinTransformer)).
        Keys are prefixed with ``backbone.swinViT.`` — we strip that prefix.
        Extra keys (layers4c conv layers) are safely ignored with strict=False.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            warnings.warn(
                f"Triad checkpoint not found at {checkpoint_path}. "
                f"Encoder will use randomly initialised weights."
            )
            return False

        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except (FileNotFoundError, RuntimeError) as exc:
            warnings.warn(
                f"Triad checkpoint not found at {checkpoint_path}. Encoder will use randomly initialised weights."
            )
            return False

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]

        if not isinstance(state, dict):
            warnings.warn("Checkpoint is not a state dict. Using random initialisation.")
            return False

        PREFIX = "backbone.swinViT."
        cleaned = {}
        skipped = 0
        for key, value in state.items():
            if key.startswith(PREFIX):
                cleaned[key[len(PREFIX):]] = value
            else:
                skipped += 1

        if not cleaned:
            warnings.warn(
                f"Triad checkpoint at {checkpoint_path} had no keys matching "
                f"'{PREFIX}'. Using random initialisation."
            )
            return False

        missing, unexpected = self.swin.load_state_dict(cleaned, strict=strict)
        if missing or unexpected:
            warnings.warn(
                f"Checkpoint mismatch — missing: {len(missing)}, "
                f"unexpected: {len(unexpected)} keys. "
                f"This is expected for strict=False."
            )

        print(
            f"[TriadEncoder] Loaded {len(cleaned)}/{len(state)} keys "
            f"from {ckpt_path.name} (skipped {skipped} non-backbone keys)"
        )
        return True

    def _freeze(self) -> None:
        for param in self.swin.parameters():
            param.requires_grad_(False)
        self.swin.eval()
        if self._freeze_projection:
            for param in self.projection.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True) -> "TriadEncoder":
        super().train(mode)
        if self._frozen:
            self.swin.eval()
            if self._freeze_projection:
                self.projection.eval()
        return self

    def _extract_multiscale_features(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Run volume through Swin, collect multi-stage features,
        spatially align to the coarsest (stage 4) resolution, and concatenate.

        Args:
            x: (B, 1, D, H, W) — will be interpolated to 96³ if needed.

        Returns:
            features: (B, total_channels, D_s4, H_s4, W_s4)
        """
        B, C, D, H, W = x.shape
        target = (96, 96, 96)
        if (D, H, W) != target:
            x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)

        stage_features: List[torch.Tensor] = self.swin(x)

        ref = stage_features[-1]
        _, _, ref_D, ref_H, ref_W = ref.shape

        aligned: List[torch.Tensor] = []
        for feat in stage_features:
            _, _, sd, sh, sw = feat.shape
            if (sd, sh, sw) == (ref_D, ref_H, ref_W):
                aligned.append(feat)
            else:
                aligned.append(
                    F.interpolate(
                        feat, size=(ref_D, ref_H, ref_W),
                        mode="trilinear", align_corners=False,
                    )
                )

        return torch.cat(aligned, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract instance-level features for MIL.

        Args:
            x: (B, 1, D, H, W) input volume.

        Returns:
            features: (B, N, embed_dim) where         N = D_s × H_s × W_s (27 for 96³ input with 5 stages).
        """
        multiscale = self._extract_multiscale_features(x)
        B, C, D_s, H_s, W_s = multiscale.shape

        multiscale = multiscale.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        projected = self.projection(multiscale)

        return projected

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without gradient tracking.

        Convenience wrapper for inference/evaluation.
        """
        was_training = self.training
        self.eval()
        features = self.forward(x)
        if was_training:
            self.train()
        return features
