"""Full triage model combining TriadEncoder + AttentionMIL + (optional) Decoder.

Assembles the encoder, MIL pooling head(s), and an optional 3D anomaly decoder
into a single end-to-end module that maps raw MRI volumes to triage predictions.

Supports:
- Single-sequence and multi-sequence inputs.
- Shared or per-anatomy MIL heads.
- Legacy single-scale or multi-scale MIL.
- Optional decoder producing voxel-level anomaly heatmaps.
- Platt scaling calibration.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from triagemri.models.encoder import TriadEncoder
from triagemri.models.mil import (
    AttentionMIL,
    GatedAttentionMIL,
    MultiScaleAttentionMIL,
    build_mil_head,
    build_multiscale_mil,
)

MILHead = Union[AttentionMIL, GatedAttentionMIL]


class TriageModel(nn.Module):
    """Full triage model: Encoder → MIL → Anomaly Score.

    Input:  ``(B, C, D, H, W)`` raw MRI volume (C=1 for single-sequence).
    Output: dict with ``score``, ``logits``, ``attention``, ``features``,
    and optionally ``heatmap``.

    Args:
        encoder: :class:`TriadEncoder` instance.
        mil_head: MIL head, ``ModuleDict`` of per-anatomy heads, or
            :class:`MultiScaleAttentionMIL`.
        anatomy_mode: ``"shared"`` or ``"multi_head"``.
        decoder: Optional :class:`AnomalyDecoder` for voxel-level heatmaps.
        multi_scale_mil: Whether the MIL head expects a feature pyramid.
    """

    def __init__(
        self,
        encoder: TriadEncoder,
        mil_head: Union[MILHead, nn.ModuleDict, MultiScaleAttentionMIL],
        anatomy_mode: str = "shared",
        decoder: Optional[nn.Module] = None,
        multi_scale_mil: bool = False,
    ) -> None:
        super().__init__()

        if anatomy_mode not in ("shared", "multi_head"):
            raise ValueError(
                f"Unknown anatomy_mode '{anatomy_mode}'. "
                f"Expected 'shared' or 'multi_head'."
            )

        self.encoder = encoder
        self.anatomy_mode = anatomy_mode
        self.decoder = decoder
        self.multi_scale_mil = multi_scale_mil

        if anatomy_mode == "multi_head":
            if not isinstance(mil_head, nn.ModuleDict):
                raise TypeError(
                    "mil_head must be an nn.ModuleDict for anatomy_mode='multi_head'"
                )
            self.mil_heads: nn.ModuleDict = mil_head
        elif multi_scale_mil:
            self.mil_head_ms: MultiScaleAttentionMIL = mil_head
        else:
            self.mil_head: MILHead = mil_head

        self._platt_model: Optional[LogisticRegression] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        anatomy: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, C, D, H, W)`` normalised volume.
            anatomy: Optional anatomy label(s).

        Returns:
            Dict with keys:

            * ``"score"``: ``(B,)`` anomaly probability in [0, 1].
            * ``"logits"``: ``(B, 1)`` raw logits.
            * ``"attention"``: ``(B, N, 1)`` attention weights.
            * ``"attention_fine"``: only in multi_scale MIL mode.
            * ``"per_scale"``: per-scale logits and attention (multi_scale).
            * ``"features"``: ``(B, N, embed_dim)`` or ``(B, C_total, ...)``.
            * ``"heatmap"``: ``(B, 1, 96, 96, 96)`` if decoder is enabled.
        """
        use_pyramid = self.multi_scale_mil or (self.decoder is not None)

        if use_pyramid:
            pyramid_or_features = self.encoder(
                x, return_pyramid=True
            )
            pyramid = pyramid_or_features
        else:
            features = self.encoder(x, return_pyramid=False)

        heatmap: Optional[torch.Tensor] = None
        if self.decoder is not None and use_pyramid:
            heatmap = self.decoder(pyramid)

        if self.anatomy_mode == "multi_head":
            if anatomy is None:
                raise ValueError("anatomy is required for multi_head mode")
            logits, attention, per_scale = self._forward_multi_head(
                features if not use_pyramid else pyramid,
                anatomy,
                use_pyramid=use_pyramid,
            )
        elif self.multi_scale_mil:
            logits, attention, per_scale = self.mil_head_ms(pyramid)
        else:
            logits, attention = self.mil_head(features)
            per_scale = {}

        score = torch.sigmoid(logits).squeeze(-1)

        result: Dict[str, torch.Tensor] = {
            "score": score,
            "logits": logits,
            "attention": attention,
            "features": (
                features if not use_pyramid
                else torch.cat(
                    [
                        p.permute(0, 2, 3, 4, 1).reshape(p.shape[0], -1, p.shape[1])
                        for p in pyramid.values()
                    ],
                    dim=1,
                )
            ),
        }

        if per_scale:
            result["per_scale"] = per_scale
        if use_pyramid and "stage2" in pyramid:
            result["attention_fine"] = pyramid["stage2"]
        if heatmap is not None:
            result["heatmap"] = heatmap

        return result

    def _forward_multi_head(
        self,
        data: torch.Tensor | Dict[str, torch.Tensor],
        anatomy: Union[str, List[str]],
        use_pyramid: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict]:
        if isinstance(anatomy, str):
            head = self.mil_heads[anatomy]
            if use_pyramid:
                logits, attention = head(data)
            else:
                logits, attention = head(data)
            return logits, attention, {}
        else:
            batch_size = (
                data.shape[0] if isinstance(data, torch.Tensor)
                else next(iter(data.values())).shape[0]
            )
            if len(anatomy) != batch_size:
                raise ValueError(
                    f"anatomy list length ({len(anatomy)}) != batch size ({batch_size})"
                )
            logits_parts: list[torch.Tensor] = []
            attention_parts: list[torch.Tensor] = []
            for i, anat in enumerate(anatomy):
                head = self.mil_heads[anat]
                if use_pyramid:
                    sub_data = {
                        k: v[i : i + 1] for k, v in data.items() if isinstance(data, dict)
                    }
                else:
                    sub_data = data[i : i + 1]
                log_i, attn_i = head(sub_data)
                logits_parts.append(log_i)
                attention_parts.append(attn_i)
            logits = torch.cat(logits_parts, dim=0)
            attention = torch.cat(attention_parts, dim=0)
            return logits, attention, {}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        anatomy: Optional[str] = None,
        calibrated: bool = False,
    ) -> torch.Tensor:
        """Binary predictions.

        Args:
            x: ``(B, C, D, H, W)`` volume.
            threshold: Decision threshold (default 0.5).
            anatomy: Optional anatomy name for multi_head mode.
            calibrated: If True, apply Platt scaling before thresholding.

        Returns:
            ``(B,)`` long tensor of 0/1 predictions.
        """
        is_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(x, anatomy=anatomy)
        if is_training:
            self.train()

        scores = output["score"].cpu().numpy()
        if calibrated and self._platt_model is not None:
            scores = self._platt_model.predict_proba(
                scores.reshape(-1, 1)
            )[:, 1]

        return torch.tensor((scores >= threshold).astype(np.int64))

    def get_attention_heatmap(
        self, x: torch.Tensor, anatomy: Optional[str] = None
    ) -> torch.Tensor:
        """Return attention weights reshaped to a 3D spatial grid.

        Args:
            x: ``(B, C, D, H, W)`` volume.
            anatomy: Optional anatomy name for multi_head mode.

        Returns:
            ``(B, 1, S, S, S)`` spatial attention grid, or
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

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self, val_scores: np.ndarray, val_labels: np.ndarray
    ) -> None:
        """Fit Platt scaling on validation scores/labels.

        Args:
            val_scores: 1-D array of uncalibrated scores in [0, 1].
            val_labels: 1-D array of binary ground-truth labels.
        """
        val_scores = np.asarray(val_scores, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.float64)
        mask = np.isfinite(val_scores) & np.isfinite(val_labels)
        val_scores = val_scores[mask]
        val_labels = val_labels[mask]

        if len(np.unique(val_labels)) < 2:
            warnings.warn(
                "Calibration requires both classes in validation set. "
                "Skipping calibration."
            )
            return

        self._platt_model = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=200
        )
        self._platt_model.fit(val_scores.reshape(-1, 1), val_labels)

    def predict_calibrated(
        self, x: torch.Tensor, anatomy: Optional[str] = None
    ) -> np.ndarray:
        """Return calibrated scores after Platt scaling.

        Args:
            x: ``(B, C, D, H, W)`` volume.
            anatomy: Optional anatomy name.

        Returns:
            1-D numpy array of calibrated probabilities.
        """
        is_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(x, anatomy=anatomy)
        if is_training:
            self.train()

        scores = output["score"].cpu().numpy()
        if self._platt_model is not None:
            return self._platt_model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return scores


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def build_triage_model(
    config_or_checkpoint: Union[str, dict],
    device: str = "cuda",
) -> TriageModel:
    """Build :class:`TriageModel` from a config dict or load from a checkpoint.

    Args:
        config_or_checkpoint:
            * ``str`` — path to a saved checkpoint.
            * ``dict`` — configuration dictionary with keys:

              - ``encoder``: kwargs for :class:`TriadEncoder`.
              - ``mil``: kwargs for MIL head construction.
              - ``anatomy_mode``: ``"shared"`` or ``"multi_head"``.
              - ``num_anatomies``: int.
              - ``anatomies``: optional list of anatomy names.
              - ``decoder``: optional dict with ``"enabled"`` and kwargs.
        device: Target device.

    Returns:
        Configured :class:`TriageModel`.
    """
    if isinstance(config_or_checkpoint, str):
        return _build_from_checkpoint(config_or_checkpoint, device)

    return _build_from_config(config_or_checkpoint, device)


def _build_from_checkpoint(checkpoint_path: str, device: str) -> TriageModel:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = _infer_config_from_state(checkpoint)

    anatomy_mode = config.get("anatomy_mode", "shared")
    encoder_cfg = config.get("encoder", {})
    mil_cfg = config.get("mil", {})
    decoder_cfg = config.get("decoder", {})
    multi_scale = mil_cfg.get("multi_scale", False)
    in_channels = encoder_cfg.get("in_channels", 1)
    lora_cfg = encoder_cfg.get("lora", {})

    encoder = TriadEncoder(
        checkpoint_path=encoder_cfg.get("checkpoint", "weights/triad_swinb_simmim.pth"),
        freeze=encoder_cfg.get("freeze_backbone", True),
        freeze_projection=encoder_cfg.get("freeze_projection", True),
        embed_dim=encoder_cfg.get("embed_dim", 768),
        in_channels=in_channels,
        freeze_patch_embed=encoder_cfg.get("freeze_patch_embed", True),
        return_pyramid=(multi_scale or decoder_cfg.get("enabled", False)),
        lora_r=lora_cfg.get("r", 0) if lora_cfg.get("enabled", False) else 0,
        lora_alpha=lora_cfg.get("alpha", 1.0),
        lora_dropout=lora_cfg.get("dropout", 0.0),
    )

    mil_head = _build_mil_component(
        mil_cfg, encoder.embed_dim, anatomy_mode, config, checkpoint, multi_scale,
    )

    decoder = None
    if decoder_cfg.get("enabled", False):
        from triagemri.models.decoder import AnomalyDecoder

        decoder = AnomalyDecoder(
            embed_dim=encoder.embed_dim,
            upconv_channels=list(decoder_cfg.get(
                "upconv_channels", [256, 128, 64, 32, 16],
            )),
            use_attention_gates=decoder_cfg.get("attention_gates", True),
        )

    model = TriageModel(
        encoder=encoder,
        mil_head=mil_head,
        anatomy_mode=anatomy_mode,
        decoder=decoder,
        multi_scale_mil=multi_scale,
    )

    _load_state_into_model(model, checkpoint)

    model.to(device)
    return model


def _build_from_config(config: dict, device: str) -> TriageModel:
    anatomy_mode = config.get("anatomy_mode", "shared")
    encoder_cfg = config.get("encoder", {})
    mil_cfg = config.get("mil", {})
    decoder_cfg = config.get("decoder", {})
    multi_scale = mil_cfg.get("multi_scale", False)
    in_channels = encoder_cfg.get("in_channels", 1)
    lora_cfg = encoder_cfg.get("lora", {})

    encoder = TriadEncoder(
        checkpoint_path=encoder_cfg.get("checkpoint", "weights/triad_swinb_simmim.pth"),
        freeze=encoder_cfg.get("freeze_backbone", True),
        freeze_projection=encoder_cfg.get("freeze_projection", True),
        embed_dim=encoder_cfg.get("embed_dim", 768),
        in_channels=in_channels,
        freeze_patch_embed=encoder_cfg.get("freeze_patch_embed", True),
        return_pyramid=(multi_scale or decoder_cfg.get("enabled", False)),
        lora_r=lora_cfg.get("r", 0) if lora_cfg.get("enabled", False) else 0,
        lora_alpha=lora_cfg.get("alpha", 1.0),
        lora_dropout=lora_cfg.get("dropout", 0.0),
    )

    mil_head = _build_mil_component(
        mil_cfg, encoder.embed_dim, anatomy_mode, config, None, multi_scale,
    )

    decoder = None
    if decoder_cfg.get("enabled", False):
        from triagemri.models.decoder import AnomalyDecoder

        decoder = AnomalyDecoder(
            embed_dim=encoder.embed_dim,
            upconv_channels=list(decoder_cfg.get(
                "upconv_channels", [256, 128, 64, 32, 16],
            )),
            use_attention_gates=decoder_cfg.get("attention_gates", True),
        )

    model = TriageModel(
        encoder=encoder,
        mil_head=mil_head,
        anatomy_mode=anatomy_mode,
        decoder=decoder,
        multi_scale_mil=multi_scale,
    )
    model.to(device)
    return model


def _build_mil_component(
    mil_cfg: dict,
    embed_dim: int,
    anatomy_mode: str,
    config: dict,
    checkpoint: Optional[dict],
    multi_scale: bool,
) -> Union[MILHead, nn.ModuleDict, MultiScaleAttentionMIL]:
    if multi_scale:
        return build_multiscale_mil(
            embed_dim=embed_dim,
            hidden_dim=mil_cfg.get("hidden_dim", 512),
            attention_dim=mil_cfg.get("attention_dim", 128),
            dropout=mil_cfg.get("dropout", 0.15),
        )

    if anatomy_mode == "multi_head":
        ckpt_anatomies = set()
        if checkpoint is not None:
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
        return nn.ModuleDict({
            anat: build_mil_head(
                input_dim=embed_dim * len(config.get("encoder", {}).get(
                    "stages_for_mil",
                    ["stage4"] if not multi_scale else ["stage3", "stage4"],
                )),
                hidden_dim=mil_cfg.get("hidden_dim", 512),
                attention_dim=mil_cfg.get("attention_dim", 128),
                dropout=mil_cfg.get("dropout", 0.15),
                pooling=mil_cfg.get("pooling", "gated_attention"),
            )
            for anat in anatomies
        })

    return build_mil_head(
        input_dim=embed_dim * 5,
        hidden_dim=mil_cfg.get("hidden_dim", 512),
        attention_dim=mil_cfg.get("attention_dim", 128),
        dropout=mil_cfg.get("dropout", 0.15),
        pooling=mil_cfg.get("pooling", "gated_attention"),
    )


def _load_state_into_model(model: TriageModel, checkpoint: dict) -> None:
    for candidate in ["state_dict", "model_state_dict"]:
        if candidate in checkpoint:
            state_dict = {
                k.removeprefix("module.").removeprefix("model."): v
                for k, v in checkpoint[candidate].items()
            }
            model.load_state_dict(state_dict, strict=False)
            return

    if isinstance(checkpoint, dict) and any(
        k.startswith(("encoder.", "mil_head", "mil_heads.", "decoder."))
        or k.startswith("module.")
        or k.startswith("model.")
        for k in checkpoint
    ):
        state_dict = {
            k.removeprefix("module.").removeprefix("model."): v
            for k, v in checkpoint.items()
        }
        model.load_state_dict(state_dict, strict=False)


def _infer_config_from_state(checkpoint: dict) -> dict:
    embed_dim = None
    for source in [checkpoint, checkpoint.get("state_dict", {})]:
        if not isinstance(source, dict):
            continue
        for key in source:
            if "projection.weight" in key or "stage_projections.stage0.weight" in key:
                embed_dim = source[key].shape[0]
                break
        if embed_dim is not None:
            break

    if embed_dim is None:
        warnings.warn(
            "Could not infer embed_dim from checkpoint. Using default 768."
        )
        embed_dim = 768

    return {"encoder": {"embed_dim": embed_dim}}
