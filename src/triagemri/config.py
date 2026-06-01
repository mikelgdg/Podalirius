"""Configuration loading and merging utilities.

Merges data.yaml, model.yaml, and train.yaml into a single typed Config object
with attribute access for autocomplete support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml

_MISSING = object()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*.

    Nested dictionaries are merged recursively.  Top-level keys present in
    *override* but not in *base* are added.  If a key is a dictionary in both
    inputs, their contents are merged; otherwise *override* replaces *base*.

    Returns:
        A new merged dictionary (inputs are not mutated).
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Schema: required top-level sections and per-section required keys.
# ---------------------------------------------------------------------------
_CONFIG_SCHEMA: Dict[str, Set[str]] = {
    "data": {"raw_dir", "volume_size", "split"},
    "model": {"encoder", "mil", "anatomy_mode"},
    "training": {"batch_size", "max_epochs", "optimizer", "scheduler", "loss"},
}

_CONFIG_VALUE_CHECKS: Dict[str, list] = {
    "training.batch_size": [(lambda v: int(v) > 0, "must be > 0")],
    "training.max_epochs": [(lambda v: int(v) > 0, "must be > 0")],
    "training.optimizer.lr": [(lambda v: float(v) > 0, "must be > 0")],
    "training.scheduler.warmup_epochs": [(lambda v: int(v) >= 0, "must be >= 0")],
    "data.split.train_ratio": [(lambda v: 0 < float(v) <= 1, "must be in (0, 1]")],
    "data.split.val_ratio": [(lambda v: 0 <= float(v) < 1, "must be in [0, 1)")],
    "data.split.test_ratio": [(lambda v: 0 <= float(v) < 1, "must be in [0, 1)")],
}


def _config_get_nested(d: Dict[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """Retrieve a value from a nested dict using a dotted key path."""
    parts = dotted_key.split(".")
    for part in parts:
        if not isinstance(d, dict) or part not in d:
            if default is _MISSING:
                raise KeyError(dotted_key)
            return default
        d = d[part]
    return d


def _validate_config(merged: Dict[str, Any]) -> None:
    """Check required sections and value ranges in the merged config dict."""
    for section, required_keys in _CONFIG_SCHEMA.items():
        if section not in merged:
            raise ValueError(
                f"Merged config is missing required section '{section}'. "
                f"Ensure {section}.yaml exists in the config directory."
            )
        for key in required_keys:
            if key not in merged[section]:
                raise ValueError(
                    f"config.{section} is missing required key '{key}'."
                )

    for dotted_key, checks in _CONFIG_VALUE_CHECKS.items():
        try:
            value = _config_get_nested(merged, dotted_key)
        except KeyError:
            continue
        for check_fn, msg in checks:
            if not check_fn(value):
                raise ValueError(f"config.{dotted_key} = {value!r} {msg}")


def load_config(config_dir: str = "configs") -> Config:
    """Load and merge data.yaml, model.yaml, train.yaml into a single Config.

    Args:
        config_dir: Path to the directory containing the YAML config files.

    Returns:
        A Config instance with all merged settings accessible as attributes.
    """
    config_path = Path(config_dir)
    merged: Dict[str, Any] = {}

    for filename in ("data.yaml", "model.yaml", "train.yaml"):
        filepath = config_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, "r") as f:
            content = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, content)

    _validate_config(merged)
    return Config(merged)


class Config:
    """Typed config wrapper with attribute access for autocomplete.

    Nested dicts become nested Config objects recursively, allowing
    dot-access like ``config.model.encoder.embed_dim``.

    Args:
        config_dict: Dictionary of configuration key-value pairs.
    """

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        items = [f"{k}={v!r}" for k, v in self.__dict__.items()]
        return f"Config({', '.join(items)})"

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get that distinguishes missing keys from ``None`` values.

        If *key* exists and its value is ``None``, returns ``None``.
        If *key* is missing, returns *default*.
        """
        if key in self.__dict__:
            return self.__dict__[key]
        return default

    def keys(self) -> Set[str]:
        return set(self.__dict__.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert back to a plain dictionary."""
        result: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result
