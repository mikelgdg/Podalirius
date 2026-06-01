"""Full triage model combining TriadEncoder + AttentionMIL → anomaly score.

Assembles the encoder and MIL pooling head into a single end-to-end
module that maps raw MRI volumes to triage predictions.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from triagemri.models.encoder import TriadEncoder
from triagemri.models.mil import AttentionMIL, GatedAttentionMIL, build_mil_head

MILHead = Union[AttentionMIL, GatedAttentionMIL]


class TriageModel(nn.Module):
    """Full triage model: TriadEncoder → AttentionMIL → Anomaly Score.

    Input:  ``(B, 1, D, H, W)`` raw MRI volume.
    Output: dict with ``score``, ``logits``, ``attention``, ``features``.

    Args:
        encoder: :class:`TriadEncoder` instance for subvolume feature extraction.
        mil_head: :class:`AttentionMIL` or :class:`GatedAttentionMIL` instance,
            or a ``nn.ModuleDict`` mapping anatomy strings to heads for
            ``anatomy_mode="multi_head"``.
        anatomy_mode:

            * ``"shared"`` — single MIL head for all anatomies (default).
            * ``"multi_head"`` — per-anatomy MIL heads; ``mil_head`` must be
              a ``nn.ModuleDict`` keyed by anatomy name.
    """

    def __init__(
        self,
        encoder: TriadEncoder,
        mil_head: Union[MILHead, nn.ModuleDict],
        anatomy_mode: str = "shared",
    ) -> None:
        super().__init__()

        if anatomy_mode not in ("shared", "multi_head"):
            raise ValueError(
                f"Unknown anatomy_mode '{anatomy_mode}'. "
                f"Expected 'shared' or 'multi_head'."
            )

        self.encoder = encoder
        self.anatomy_mode = anatomy_mode

        if anatomy_mode == "multi_head":
            if not isinstance(mil_head, nn.ModuleDict):
                raise TypeError(
                    "mil_head must be an nn.ModuleDict for anatomy_mode='multi_head'"
                )
            self.mil_heads: nn.ModuleDict = mil_head
        else:
            self.mil_head: MILHead = mil_head  # type: ignore[assignment]

    def forward(
        self,
        x: torch.Tensor,
        anatomy: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: encode → pool → score.

        Args:
            x: ``(B, 1, D, H, W)`` normalised volume.
            anatomy: Optional anatomy label(s). Required when
                ``anatomy_mode="multi_head"``.

        Returns:
            Dict with keys:

            * ``"score"``: ``(B,)`` anomaly probability in [0, 1].
            * ``"logits"``: ``(B, 1)`` raw logits.
            * ``"attention"``: ``(B, N, 1)`` attention weights.
            * ``"features"``: ``(B, N, 768)`` intermediate features.
        """
        features = self.encoder(x)

        if self.anatomy_mode == "multi_head":
            if anatomy is None:
                raise ValueError("anatomy is required for multi_head mode")

            if isinstance(anatomy, str):
                try:
                    head = self.mil_heads[anatomy]
                except KeyError:
                    raise KeyError(
                        f"Unknown anatomy '{anatomy}'. Available: {list(self.mil_heads.keys())}"
                    )
                logits, attention = head(features)
            else:
                batch_size = features.shape[0]
                if len(anatomy) != batch_size:
                    raise ValueError(
                        f"anatomy list length ({len(anatomy)}) != batch size ({batch_size})"
                    )
                logits_parts: list[torch.Tensor] = []
                attention_parts: list[torch.Tensor] = []
                for i, anat in enumerate(anatomy):
                    try:
                        head = self.mil_heads[anat]
                    except KeyError:
                        raise KeyError(
                            f"Unknown anatomy '{anat}'. Available: {list(self.mil_heads.keys())}"
                        )
                    log_i, attn_i = head(features[i : i + 1])
                    logits_parts.append(log_i)
                    attention_parts.append(attn_i)
                logits = torch.cat(logits_parts, dim=0)
                attention = torch.cat(attention_parts, dim=0)
        else:
            logits, attention = self.mil_head(features)

        score = torch.sigmoid(logits).squeeze(-1)

        return {
            "score": score,
            "logits": logits,
            "attention": attention,
            "features": features,
        }

    def predict(
        self, x: torch.Tensor, threshold: float = 0.5, anatomy: Optional[str] = None
    ) -> torch.Tensor:
        """Convenience method returning binary predictions.

        Args:
            x: ``(B, 1, D, H, W)`` volume.
            threshold: Decision threshold (default 0.5).
            anatomy: Optional anatomy name for multi_head mode.

        Returns:
            ``(B,)`` long tensor of 0/1 predictions.
        """
        is_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(x, anatomy=anatomy)
        if is_training:
            self.train()
        return (output["score"] > threshold).long()

    def get_attention_heatmap(
        self, x: torch.Tensor, anatomy: Optional[str] = None
    ) -> torch.Tensor:
        """Return attention weights reshaped to a 3D spatial grid.

        The encoder produces a 3×3×3 spatial grid of features.
        Attention weights are reshaped to match.

        Args:
            x: ``(B, 1, D, H, W)`` volume.
            anatomy: Optional anatomy name for multi_head mode.

        Returns:
            ``(B, 1, 3, 3, 3)`` spatial attention grid, or
            ``(B, 1, N, 1)`` if the grid is non-cubic.
        """
        is_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(x, anatomy=anatomy)
        if is_training:
            self.train()

        attention = output["attention"]
        B = attention.shape[0]
        N = attention.shape[1]
        side = round(N ** (1 / 3))
        if side ** 3 != N:
            warnings.warn(
                f"Attention grid is non-cubic (N={N}, side={side}). "
                f"Returning flat attention instead."
            )
            return attention.unsqueeze(1)
        attention_grid = attention.reshape(B, side, side, side)
        return attention_grid.unsqueeze(1)


def build_triage_model(
    config_or_checkpoint: Union[str, dict],
    device: str = "cuda",
) -> TriageModel:
    """Build :class:`TriageModel` from a config dict or load from a checkpoint.

    Args:
        config_or_checkpoint:
            * ``str`` — path to a saved ``.pt`` / ``.pth`` checkpoint.
            * ``dict`` — configuration dictionary with keys:

              - ``encoder``: kwargs for :class:`TriadEncoder`.
              - ``mil``: kwargs for :func:`build_mil_head`.
              - ``anatomy_mode``: ``"shared"`` or ``"multi_head"``.
              - ``num_anatomies``: int (used in multi_head mode).
              - ``anatomies``: optional list of anatomy names.
        device: Target device for the model (default ``"cuda"``).

    Returns:
        Configured :class:`TriageModel`.
    """
    if isinstance(config_or_checkpoint, str):
        checkpoint = torch.load(
            config_or_checkpoint, map_location=device, weights_only=True
        )
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            config = checkpoint["config"]
        else:
            config = {}
            if (
                isinstance(checkpoint, dict)
                and "state_dict" not in checkpoint
            ):
                embed_dim = None
                for key in checkpoint:
                    if "projection.weight" in key:
                        embed_dim = checkpoint[key].shape[0]
                        break
                if embed_dim is None:
                    raise ValueError(
                        "Checkpoint has no 'config' key and embed_dim could not be "
                        "inferred. Pass a config dict or ensure the checkpoint "
                        "contains 'config' or 'encoder.projection.weight'."
                    )
                config = {"encoder": {"embed_dim": embed_dim}}

        anatomy_mode = config.get("anatomy_mode", "shared")

        encoder = TriadEncoder(
            checkpoint_path=config.get("encoder", {}).get(
                "checkpoint", "weights/triad_swinb_simmim.pth"
            ),
            freeze=config.get("encoder", {}).get("freeze_backbone", True),
            freeze_projection=config.get("encoder", {}).get("freeze_projection", True),
            embed_dim=config.get("encoder", {}).get("embed_dim", 768),
        )

        mil_config = config.get("mil", {})

        if anatomy_mode == "multi_head":
            ckpt_anatomies = set()
            for source in [checkpoint, checkpoint.get("state_dict", {})]:
                if not isinstance(source, dict):
                    continue
                for key in source:
                    k = key.removeprefix("module.").removeprefix("model.")
                    if k.startswith("mil_heads."):
                        parts = k.split(".")
                        if len(parts) >= 2:
                            ckpt_anatomies.add(parts[1])
            if ckpt_anatomies:
                anatomies = sorted(ckpt_anatomies)
            else:
                num_anatomies = config.get("num_anatomies", 3)
                anatomies = config.get("anatomies", ["brain", "prostate", "breast"])
                anatomies = anatomies[:num_anatomies]
            mil_head = nn.ModuleDict(
                {
                    anat: build_mil_head(
                        input_dim=encoder.embed_dim,
                        hidden_dim=mil_config.get("hidden_dim", 512),
                        attention_dim=mil_config.get("attention_dim", 128),
                        dropout=mil_config.get("dropout", 0.3),
                        pooling=mil_config.get("pooling", "gated_attention"),
                    )
                    for anat in anatomies
                }
            )
        else:
            mil_head = build_mil_head(
                input_dim=encoder.embed_dim,
                hidden_dim=mil_config.get("hidden_dim", 512),
                attention_dim=mil_config.get("attention_dim", 128),
                dropout=mil_config.get("dropout", 0.3),
                pooling=mil_config.get("pooling", "gated_attention"),
            )

        model = TriageModel(
            encoder=encoder,
            mil_head=mil_head,
            anatomy_mode=anatomy_mode,
        )

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            state_dict = {
                k.removeprefix("module.").removeprefix("model."): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict, strict=False)
        elif isinstance(checkpoint, dict) and any(
            k.startswith("encoder.") or k.startswith("mil_head")
            or k.startswith("model.encoder.") or k.startswith("model.mil_head")
            or k.startswith("module.encoder.") or k.startswith("module.mil_head")
            for k in checkpoint
        ):
            state_dict = {
                k.removeprefix("module.").removeprefix("model."): v
                for k, v in checkpoint.items()
            }
            model.load_state_dict(state_dict, strict=False)

        model.to(device)
        return model

    # Build from config dict
    config = config_or_checkpoint
    anatomy_mode = config.get("anatomy_mode", "shared")

    encoder = TriadEncoder(
        checkpoint_path=config.get("encoder", {}).get(
            "checkpoint", "weights/triad_swinb_simmim.pth"
        ),
        freeze=config.get("encoder", {}).get("freeze_backbone", True),
        freeze_projection=config.get("encoder", {}).get("freeze_projection", True),
        embed_dim=config.get("encoder", {}).get("embed_dim", 768),
    )

    mil_config = config.get("mil", {})

    if anatomy_mode == "multi_head":
        num_anatomies = config.get("num_anatomies", 3)
        anatomies = config.get("anatomies", ["brain", "prostate", "breast"])
        mil_head = nn.ModuleDict(
            {
                anat: build_mil_head(
                    input_dim=encoder.embed_dim,
                    hidden_dim=mil_config.get("hidden_dim", 512),
                    attention_dim=mil_config.get("attention_dim", 128),
                    dropout=mil_config.get("dropout", 0.3),
                    pooling=mil_config.get("pooling", "gated_attention"),
                )
                for anat in anatomies[:num_anatomies]
            }
        )
    else:
        mil_head = build_mil_head(
            input_dim=encoder.embed_dim,
            hidden_dim=mil_config.get("hidden_dim", 512),
            attention_dim=mil_config.get("attention_dim", 128),
            dropout=mil_config.get("dropout", 0.3),
            pooling=mil_config.get("pooling", "gated_attention"),
        )

    model = TriageModel(
        encoder=encoder,
        mil_head=mil_head,
        anatomy_mode=anatomy_mode,
    )
    model.to(device)
    return model
