"""Triad 3D Swin-B SimMIM encoder for volumetric MRI feature extraction.

Loads the Triad checkpoint (SwinTransformer from MONAI) and extracts
multi-scale features from a 96x96x96 volume. Supports:

- Single-channel and multi-sequence inputs.
- Feature pyramid at native stage resolutions.
- LoRA (Low-Rank Adaptation) adapters injected into attention projections
  for task-specific fine-tuning without modifying the frozen backbone.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation)
# ---------------------------------------------------------------------------


class LoRALayer(nn.Module):
    """Low-Rank Adaptation wrapper around a frozen linear layer.

    ``output = W·x + (B·A)·x * scale`` where A and B are low-rank matrices.
    The original weight W is kept frozen; only A and B are trainable.

    Args:
        base: The original frozen ``nn.Linear`` layer.
        r: Rank of the adaptation matrices.
        alpha: Scaling factor (effective scale = alpha / r).
        dropout: Dropout applied to the input of the LoRA branch.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 0.0

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for param in base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale
        return base_out + lora_out


def _inject_lora_into_swin(
    swin: nn.Module,
    r: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_substrings: tuple[str, ...] = ("qkv", "attn", "proj"),
) -> int:
    """Replace attention-projection Linear layers with LoRALayer wrappers.

    Walks *swin* named modules.  For every ``nn.Linear`` whose name contains
    any of *target_substrings*, replaces it in-place with a :class:`LoRALayer`
    that keeps the original frozen.

    Returns the number of layers replaced.
    """
    replaced = 0
    for parent_name, parent_module in list(swin.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue
            full_lower = f"{parent_name}.{child_name}".lower() if parent_name else child_name.lower()
            if not any(t in full_lower for t in target_substrings):
                continue

            lora_wrapper = LoRALayer(child_module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent_module, child_name, lora_wrapper)
            replaced += 1

    return replaced


# ---------------------------------------------------------------------------
# TriadEncoder
# ---------------------------------------------------------------------------


class TriadEncoder(nn.Module):
    """3D Swin-B encoder with Triad SimMIM pre-trained weights.

    Injecting LoRA adapters into the Swin attention layers enables
    task-specific adaptation while keeping the backbone frozen.

    Args:
        checkpoint_path: Path to Triad Swin-B SimMIM checkpoint (.pth).
        freeze: Freeze all encoder parameters.
        embed_dim: Output embedding dimension per stage.
        pretrained_strict: Raise error if checkpoint keys mismatch.
        in_channels: Number of input channels.
        freeze_patch_embed: Keep ``patch_embed.proj`` trainable for
            multi-channel fusion.
        freeze_projection: Freeze the per-stage projection layers.
        return_pyramid: Default ``return_pyramid`` value.
        lora_r: LoRA rank (0 = disabled).
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout.
    """

    def __init__(
        self,
        checkpoint_path: str = "weights/triad_swinb_simmim.pth",
        freeze: bool = True,
        embed_dim: int = 768,
        pretrained_strict: bool = False,
        in_channels: int = 1,
        freeze_patch_embed: bool = True,
        freeze_projection: bool = True,
        return_pyramid: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self._frozen = freeze
        self._freeze_patch_embed = freeze_patch_embed
        self._freeze_projection = freeze_projection
        self._return_pyramid_default = return_pyramid
        self.in_channels = in_channels
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha

        self.swin, stage_dims = self._build_swin(in_channels=in_channels)

        loaded = self._load_triad_weights(
            checkpoint_path, strict=pretrained_strict, in_channels=in_channels
        )
        self._pretrained_loaded = loaded

        self._lora_count = 0
        if lora_r > 0:
            self._lora_count = _inject_lora_into_swin(
                self.swin, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
            )
            print(f"[TriadEncoder] Injected {self._lora_count} LoRA adapters (r={lora_r})")

        self.stage_projections = nn.ModuleDict({
            f"stage{i}": nn.Linear(dim, embed_dim)
            for i, dim in enumerate(stage_dims)
        })

        if freeze:
            self._freeze()

    @staticmethod
    def _build_swin(in_channels: int = 1) -> Tuple[nn.Module, List[int]]:
        from monai.networks.nets.swin_unetr import SwinTransformer

        swin = SwinTransformer(
            in_chans=in_channels,
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
            dummy = torch.zeros(1, in_channels, 96, 96, 96)
            stage_features = swin(dummy)
            stage_dims = [int(f.shape[1]) for f in stage_features]
        return swin, stage_dims

    def _load_triad_weights(
        self, checkpoint_path: str, strict: bool = False, in_channels: int = 1
    ) -> bool:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            warnings.warn(
                f"Triad checkpoint not found at {checkpoint_path}. "
                f"Encoder will use randomly initialised weights."
            )
            return False

        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location="cpu")
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            warnings.warn(
                f"Failed to load Triad checkpoint: {exc}. "
                f"Encoder will use randomly initialised weights."
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
        cleaned: Dict[str, torch.Tensor] = {}
        skipped = 0
        for key, value in state.items():
            if key.startswith(PREFIX):
                cleaned_key = key[len(PREFIX):]
                cleaned[cleaned_key] = value
            else:
                skipped += 1

        if not cleaned:
            warnings.warn(
                f"Triad checkpoint had no keys matching '{PREFIX}'. "
                f"Using random initialisation."
            )
            return False

        if in_channels > 1 and "patch_embed.proj.weight" in cleaned:
            pretrained_weight = cleaned["patch_embed.proj.weight"]
            if pretrained_weight.shape[1] == 1:
                new_weight = torch.zeros(
                    pretrained_weight.shape[0],
                    in_channels,
                    *pretrained_weight.shape[2:],
                    dtype=pretrained_weight.dtype,
                )
                nn.init.kaiming_normal_(new_weight, mode="fan_out", nonlinearity="relu")
                new_weight[:, 0:1, ...] = pretrained_weight
                cleaned["patch_embed.proj.weight"] = new_weight
                if "patch_embed.proj.bias" not in cleaned:
                    cleaned["patch_embed.proj.bias"] = torch.zeros(pretrained_weight.shape[0])

        # Load with strict=False — LoRA wrappers may change module types
        missing, unexpected = self.swin.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"Checkpoint mismatch — missing: {len(missing)}, "
                f"unexpected: {len(unexpected)} keys. "
                f"This is expected when LoRA wrappers are active."
            )

        print(
            f"[TriadEncoder] Loaded {len(cleaned)}/{len(state)} keys "
            f"from {ckpt_path.name} (skipped {skipped} non-backbone keys)"
        )
        return True

    def _freeze(self) -> None:
        for name, param in self.swin.named_parameters():
            is_lora = "lora_A" in name or "lora_B" in name
            is_patch = name.startswith("patch_embed.proj")
            if is_lora:
                param.requires_grad_(True)
            elif is_patch and not self._freeze_patch_embed:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        self.swin.eval()
        if self._freeze_projection:
            for param in self.stage_projections.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True) -> "TriadEncoder":
        super().train(mode)
        if self._frozen:
            self.swin.eval()
            if self._freeze_patch_embed or self.in_channels == 1:
                for name, param in self.swin.named_parameters():
                    if name.startswith("patch_embed.proj") and "lora" not in name:
                        param.requires_grad_(not self._freeze_patch_embed)
            if self._freeze_projection:
                self.stage_projections.eval()
        return self

    def _extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, D, H, W = x.shape
        target = (96, 96, 96)
        if (D, H, W) != target:
            x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
        return self.swin(x)

    def forward(
        self, x: torch.Tensor, return_pyramid: Optional[bool] = None
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if return_pyramid is None:
            return_pyramid = self._return_pyramid_default

        stage_features = self._extract_multiscale_features(x)

        projected = []
        for i, feat in enumerate(stage_features):
            B_st, C_st, D_st, H_st, W_st = feat.shape
            feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C_st)
            proj = self.stage_projections[f"stage{i}"](feat_flat)
            proj = proj.view(B_st, D_st, H_st, W_st, self.embed_dim)
            proj = proj.permute(0, 4, 1, 2, 3).contiguous()
            projected.append(proj)

        if return_pyramid:
            return {f"stage{i}": p for i, p in enumerate(projected)}

        ref = projected[-1]
        _, _, ref_D, ref_H, ref_W = ref.shape

        aligned: List[torch.Tensor] = []
        for p in projected:
            _, _, sd, sh, sw = p.shape
            if (sd, sh, sw) == (ref_D, ref_H, ref_W):
                aligned.append(p)
            else:
                aligned.append(
                    F.interpolate(
                        p, size=(ref_D, ref_H, ref_W),
                        mode="trilinear", align_corners=False,
                    )
                )

        concatenated = torch.cat(aligned, dim=1)
        B, C_total, D_s, H_s, W_s = concatenated.shape
        features = concatenated.permute(0, 2, 3, 4, 1).reshape(B, -1, C_total)

        return features

    @torch.no_grad()
    def extract_features(
        self, x: torch.Tensor, return_pyramid: Optional[bool] = None
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        features = self.forward(x, return_pyramid=return_pyramid)
        if was_training:
            self.train()
        return features
