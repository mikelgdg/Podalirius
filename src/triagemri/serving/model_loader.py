"""Singleton model loader for the API server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from triagemri.models.triage import build_triage_model, TriageModel

logger = logging.getLogger(__name__)

_MODEL: Optional[TriageModel] = None
_DEVICE: str = "cuda"
_ANATOMY_MODE: str = "shared"


def load_model(checkpoint_path: str, config_dir: str = "configs", device: str = "cuda") -> TriageModel:
    """Load the model once (singleton pattern).

    Args:
        checkpoint_path: Path to .pt/.pth checkpoint.
        config_dir: Directory with config YAML files.
        device: Torch device.

    Returns:
        The loaded TriageModel.
    """
    global _MODEL, _DEVICE, _ANATOMY_MODE
    if _MODEL is not None:
        return _MODEL

    logger.info("Loading model from %s", checkpoint_path)
    _MODEL = build_triage_model(checkpoint_path, device=device)
    _MODEL.eval()
    _DEVICE = device
    _ANATOMY_MODE = _MODEL.anatomy_mode
    logger.info("Model loaded. Anatomy mode: %s", _ANATOMY_MODE)
    return _MODEL


def get_model() -> TriageModel:
    """Get the loaded model. Raises RuntimeError if not loaded."""
    if _MODEL is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _MODEL


def get_device() -> str:
    return _DEVICE


def get_anatomy_mode() -> str:
    return _ANATOMY_MODE
