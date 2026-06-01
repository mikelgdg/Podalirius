"""Training run card — records hyperparams, git commit, and data hashes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _hash_yaml(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def save_run_card(
    output_dir: str | Path,
    config: Any,
    data_root: str | Path = "data/raw",
) -> Dict[str, Any]:
    """Generate and save a YAML run card with run metadata.

    Args:
        output_dir: Directory to save ``run_card.yaml``.
        config: Merged :class:`~triagemri.config.Config` instance.
        data_root: Root of the data directory.

    Returns:
        The run card dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)

    run_card: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_hash(),
        "config_hash": hashlib.sha256(
            json.dumps(config_dict, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "hyperparams": {
            "batch_size": int(config.training.get("batch_size", 4)),
            "max_epochs": int(config.training.get("max_epochs", 100)),
            "lr": float(config.training.optimizer.get("lr", 5e-4)),
            "weight_decay": float(config.training.optimizer.get("weight_decay", 1e-3)),
            "warmup_epochs": int(config.training.scheduler.get("warmup_epochs", 5)),
        },
        "data": {
            "config": config_dict.get("data", {}),
        },
    }

    data_path = Path(data_root)
    if data_path.exists():
        run_card["data_hash"] = _hash_yaml(
            Path(config_dict.get("config_dir", "configs")) / "data.yaml"
        )

    card_path = output_path / "run_card.yaml"
    with open(card_path, "w") as f:
        yaml.dump(run_card, f, default_flow_style=False, sort_keys=False)

    return run_card
