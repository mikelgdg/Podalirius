"""Segmentation dataset for supervised anomaly heatmap training.

Provides volumes with ground-truth binary masks for decoder supervision.
Includes normal controls (mask=zeros) to prevent hallucination.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from triagemri.data.datasets import BrainMRIDataset, ProstateMRIDataset
from triagemri.data.preprocessing import load_mask

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """Dataset yielding (volume, mask) pairs for decoder training.

    Positive cases (label=1) with GT masks receive real lesion masks.
    Abnormal cases without GT masks are excluded (no harmful zero supervision).
    Normal controls (label=0) receive all-zeros masks.

    Args:
        brain_data_root: Path to brain raw data.
        prostate_data_root: Path to prostate raw data.
        prostate_labels_dir: Path to PI-CAI labels directory.
        split: ``"train"``, ``"val"``, or ``"test"``.
        anatomies: Which anatomies to include.
        normal_ratio: Fraction of normal cases to include (mitigates decoder
            hallucination by training on healthy volumes with zero masks).
        seed: Random seed for patient-level split.
        train_ratio, val_ratio, test_ratio: Split proportions.
    """

    def __init__(
        self,
        brain_data_root: str | Path = "data/raw/brain",
        prostate_data_root: str | Path = "data/raw/prostate/pi_cai",
        prostate_labels_dir: str | Path = "data/raw/prostate/picai_labels",
        split: str = "train",
        anatomies: Optional[List[str]] = None,
        normal_ratio: float = 0.30,
        seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        if anatomies is None:
            anatomies = ["brain", "prostate"]

        self._cases: List[Dict[str, Any]] = []
        split_kwargs = dict(
            seed=seed, train_ratio=train_ratio,
            val_ratio=val_ratio, test_ratio=test_ratio,
        )

        if "brain" in anatomies:
            try:
                ds = BrainMRIDataset(
                    data_root=Path(brain_data_root),
                    split=split,
                    sources=("brats", "oasis", "ixi", "hcp"),
                    **split_kwargs,
                )
                for case in ds._cases:
                    if case["label"] == 1 and case.get("mask_path") is None:
                        continue  # Skip abnormals without GT masks
                    case["_anatomy"] = "brain"
                    self._cases.append(case)
            except Exception as e:
                logger.warning("Failed to load brain cases: %s", e)

        if "prostate" in anatomies:
            try:
                ds = ProstateMRIDataset(
                    data_root=Path(prostate_data_root),
                    labels_dir=Path(prostate_labels_dir),
                    split=split,
                    **split_kwargs,
                )
                for case in ds._cases:
                    if case["label"] == 1 and case.get("mask_path") is None:
                        continue  # Skip abnormals without GT masks
                    case["_anatomy"] = "prostate"
                    case["preferred_sequence"] = "t2w"
                    self._cases.append(case)
            except Exception as e:
                logger.warning("Failed to load prostate cases: %s", e)

        # Subsample normals if normal_ratio < 1.0
        self._subsample_normals(normal_ratio, seed)

        logger.info(
            "SegmentationDataset(split=%s): %d cases "
            "(normal=%d, abnormal=%d)",
            split,
            len(self._cases),
            sum(1 for c in self._cases if c["label"] == 0),
            sum(1 for c in self._cases if c["label"] == 1),
        )

    def _subsample_normals(self, ratio: float, seed: int) -> None:
        if ratio >= 1.0:
            return
        import random
        rng = random.Random(seed)
        normals = [c for c in self._cases if c["label"] == 0]
        abnormals = [c for c in self._cases if c["label"] == 1]
        n_keep = max(1, int(len(normals) * ratio))
        kept_normals = rng.sample(normals, n_keep) if n_keep < len(normals) else normals
        self._cases = abnormals + kept_normals
        rng.shuffle(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from triagemri.data.preprocessing import load_and_preprocess

        case = self._cases[idx]

        # Load volume using the preferred sequence path
        vpaths = case.get("volume_paths", {})
        preferred = case.get("preferred_sequence", "t1ce")
        vol_path = vpaths.get(preferred) or next(iter(vpaths.values()), None)
        if vol_path is None:
            raise RuntimeError(f"No volume path for case {case.get('case_id')}")
        volume = load_and_preprocess(str(vol_path))

        # Load mask: real GT for positives, zeros for normals
        mask_path = case.get("mask_path")
        if mask_path is not None:
            mask = load_mask(mask_path)
        else:
            mask = load_mask(None)  # zeros

        return {
            "volume": volume,
            "mask": mask,
        }
