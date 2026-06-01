"""Triage-MRI: Generalist triage and normality screening for 3D MRI."""

from __future__ import annotations

__version__ = "0.1.0"

from triagemri.config import Config, load_config

__all__ = [
    "__version__",
    "Config",
    "load_config",
]
