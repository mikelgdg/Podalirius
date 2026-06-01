"""PyTorch Dataset classes for 3D MRI volumes.

Provides :class:`BrainMRIDataset` for brain MRI triage (BraTS + OASIS-3),
:class:`ProstateMRIDataset` for prostate, :class:`MultiAnatomyDataset` for
combining multiple anatomies, and :func:`create_dataloaders` for building
DataLoader objects from config.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from triagemri.data.labels import (
    extract_label_from_brats,
    extract_label_from_oasis,
    extract_label_from_picai,
    get_label_distribution,
)
from triagemri.data.preprocessing import (
    load_and_preprocess,
    load_and_preprocess_multi,
    validate_file,
    validate_volume,
)

logger = logging.getLogger(__name__)

_SEQUENCE_PATTERNS = {
    "t1ce": ["t1ce", "T1CE", "T1ce"],
    "t1": ["t1", "T1", "T1w"],
    "t2": ["t2", "T2", "T2w"],
    "flair": ["flair", "FLAIR", "Flair"],
}


def _discover_sequences(
    nifti_files: List[Path],
) -> Dict[str, Path]:
    """Map sequence names to file paths from a list of NIfTI files.

    Each file is checked against known sequence name patterns.
    The first match wins per sequence.

    Args:
        nifti_files: List of ``.nii.gz`` or ``.mha`` paths to inspect.

    Returns:
        Dict mapping sequence key to a file path.
    """
    volume_paths: Dict[str, Path] = {}
    for f in nifti_files:
        name = f.name.lower()
        if "seg" in name or "mask" in name or "label" in name:
            continue
        for seq, keywords in _SEQUENCE_PATTERNS.items():
            if seq in volume_paths:
                continue
            if any(kw.lower() in name for kw in keywords):
                volume_paths[seq] = f
                break
    return volume_paths


# ---------------------------------------------------------------------------
# BrainMRIDataset
# ---------------------------------------------------------------------------


class BrainMRIDataset(Dataset):
    """Dataset for brain MRI triage (normal vs abnormal).

    Loads volumes from BraTS, OASIS, IXI, and HCP, applying preprocessing
    and transforms.  Directory scanning is cached at class level so repeated
    instantiations with the same inputs avoid re-scanning.

    Returns (per ``__getitem__``):
        ``{"volume": (C,96,96,96), "label": 0/1, "anatomy": "brain",
        "patient_id": str, "source": str}``
        where ``C`` is 1 for single-sequence or N for multi-sequence mode.

    Args:
        data_root: Root directory containing source subdirectories.
        split: ``"train"``, ``"val"``, or ``"test"``.
        transform: Optional MONAI-style dictionary transform.
        sources: Which data sources to include.
        clinical_csv: Path to OASIS clinical data CSV.
        train_ratio: Fraction of patient-IDs reserved for training.
        val_ratio: Fraction of patient-IDs reserved for validation.
        test_ratio: Fraction of patient-IDs reserved for testing.
        seed: Random seed for reproducible patient-level splits.
        multi_sequence: If True, load all available sequences as channels.
        multi_sequence_list: Which sequences to include when *multi_sequence* is True.
    """

    _case_cache: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        transform: Any = None,
        sources: Sequence[str] = ("brats", "oasis", "ixi", "hcp"),
        clinical_csv: str | Path | None = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        multi_sequence: bool = False,
        multi_sequence_list: Sequence[str] = ("t1ce", "t1", "t2", "flair"),
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.sources = tuple(sources)
        self.clinical_csv = Path(clinical_csv) if clinical_csv else None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.multi_sequence = multi_sequence
        self.multi_sequence_list = tuple(multi_sequence_list)

        cache_key = (
            f"{self.data_root.resolve()}_"
            f"{'_'.join(sorted(self.sources))}_"
            f"{'multi' if multi_sequence else 'single'}"
        )

        if cache_key not in self._case_cache:
            cache_file = self.data_root / ".triage_cache.json"
            if cache_file.exists():
                try:
                    cached = json.loads(cache_file.read_text())
                    for case in cached:
                        case["volume_paths"] = {
                            k: Path(v) for k, v in case.get("volume_paths", {}).items()
                        }
                    self._case_cache[cache_key] = cached
                    logger.info("Loaded %d cases from disk cache", len(cached))
                except Exception:
                    logger.warning("Disk cache corrupt, rescanning")

        if cache_key not in self._case_cache:
            cases = self._scan_all()
            self._case_cache[cache_key] = cases
            try:
                serializable = []
                for case in cases:
                    c = dict(case)
                    c["volume_paths"] = {
                        k: str(v) for k, v in case.get("volume_paths", {}).items()
                    }
                    serializable.append(c)
                cache_file = self.data_root / ".triage_cache.json"
                cache_file.write_text(json.dumps(serializable))
                logger.info("Saved %d cases to disk cache", len(cases))
            except Exception:
                pass

        all_cases = self._case_cache[cache_key]
        self._cases = self._patient_split(all_cases, split)

        dist = get_label_distribution(self._cases)
        total = len(self._cases)
        n_abnormal = sum(d.get("abnormal", 0) for d in dist.values())
        n_normal = sum(d.get("normal", 0) for d in dist.values())
        n_unknown = sum(d.get("unknown", 0) for d in dist.values())
        logger.info(
            "BrainMRIDataset(split=%s): %d cases (%d normal, %d abnormal, %d unknown) "
            "distribution=%s",
            split,
            total,
            n_normal,
            n_abnormal,
            n_unknown,
            {k: v for k, v in dist.items()},
        )

        if multi_sequence:
            valid_cases = sum(
                1
                for c in self._cases
                if len(c.get("volume_paths", {})) >= len(multi_sequence_list)
            )
            logger.info(
                "Multi-sequence mode: %d/%d cases have all %d requested sequences",
                valid_cases,
                total,
                len(multi_sequence_list),
            )

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------

    def _scan_all(self) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        if "brats" in self.sources:
            cases.extend(self._scan_brats(self.data_root / "brats"))
        if "oasis" in self.sources:
            cases.extend(self._scan_oasis(self.data_root / "oasis"))
        if "ixi" in self.sources:
            cases.extend(self._scan_ixi(self.data_root / "ixi"))
        if "hcp" in self.sources:
            cases.extend(self._scan_hcp(self.data_root / "hcp"))
        return cases

    def _scan_brats(self, brats_dir: Path) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        if not brats_dir.is_dir():
            logger.warning("BraTS directory not found: %s", brats_dir)
            return cases

        case_dirs = list(brats_dir.rglob("BraTS-*"))
        case_dirs = [
            d for d in case_dirs if d.is_dir() and list(d.glob("*seg*.nii.gz"))
        ]
        case_dirs = sorted(set(case_dirs), key=lambda d: d.name)
        skipped_no_seg = 0
        skipped_corrupt = 0

        for case_dir in case_dirs:
            try:
                nifti_files = list(case_dir.rglob("*.nii.gz"))
                volume_paths = _discover_sequences(nifti_files)

                if not volume_paths:
                    logger.debug("No NIfTI volumes in %s, skipping", case_dir)
                    continue

                try:
                    label = extract_label_from_brats(case_dir)
                except (FileNotFoundError, ValueError):
                    logger.debug("No valid segmentation in %s, skipping", case_dir)
                    skipped_no_seg += 1
                    continue

                if not any(validate_file(str(p)) for p in volume_paths.values()):
                    logger.warning("Corrupted volume in %s, skipping", case_dir.name)
                    skipped_corrupt += 1
                    continue

                cases.append(
                    {
                        "case_id": case_dir.name,
                        "volume_paths": volume_paths,
                        "label": label,
                        "anatomy": "brain",
                        "patient_id": case_dir.name,
                        "source": "brats",
                        "preferred_sequence": "t1ce",
                    }
                )
            except Exception:
                logger.warning(
                    "Failed to scan BraTS case %s, skipping",
                    case_dir,
                    exc_info=True,
                )

        if skipped_no_seg > 0:
            logger.info("BraTS: skipped %d cases without valid segmentation", skipped_no_seg)
        if skipped_corrupt > 0:
            logger.info("BraTS: skipped %d cases with corrupted volumes", skipped_corrupt)
        return cases

    def _scan_oasis(self, oasis_dir: Path) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        if not oasis_dir.is_dir():
            logger.warning("OASIS directory not found: %s", oasis_dir)
            return cases

        skipped_unknown = 0

        for session_dir in sorted(oasis_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            if session_dir.name.startswith("disc"):
                continue
            if not session_dir.name.startswith("OAS"):
                continue

            try:
                volume_paths = self._find_oasis_volume(session_dir)

                if not volume_paths:
                    logger.debug("No volumes in %s, skipping", session_dir)
                    continue

                try:
                    csv_path = (
                        oasis_dir / "oasis_longitudinal.csv"
                        if session_dir.name.startswith("OAS2")
                        else oasis_dir / "oasis_cross-sectional.csv"
                    )
                    if not csv_path.exists():
                        csv_path = self.clinical_csv
                    label = extract_label_from_oasis(session_dir, csv_path)
                except FileNotFoundError:
                    logger.warning("Clinical CSV not found, OASIS labels unavailable")
                    label = -1

                if label == -1:
                    skipped_unknown += 1
                    continue

                parts = session_dir.name.split("_")
                if session_dir.name.startswith(("OAS1", "OAS2")):
                    patient_id = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
                else:
                    patient_id = parts[0] if parts else session_dir.name

                if not any(validate_file(str(p)) for p in volume_paths.values()):
                    logger.debug("Corrupted volume in %s, skipping", session_dir.name)
                    continue

                cases.append(
                    {
                        "case_id": session_dir.name,
                        "volume_paths": volume_paths,
                        "label": label,
                        "anatomy": "brain",
                        "patient_id": patient_id,
                        "source": "oasis",
                        "preferred_sequence": "t1",
                    }
                )
            except Exception:
                logger.warning(
                    "Failed to scan OASIS session %s, skipping",
                    session_dir,
                    exc_info=True,
                )

        if skipped_unknown > 0:
            logger.info("OASIS: skipped %d sessions with unknown label", skipped_unknown)
        return cases

    def _scan_ixi(self, ixi_dir: Path) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        if not ixi_dir.is_dir():
            logger.warning("IXI directory not found: %s", ixi_dir)
            return cases

        nii_files = sorted(ixi_dir.glob("*.nii.gz"))
        if not nii_files:
            logger.warning("No NIfTI files in %s", ixi_dir)
            return cases

        patients: Dict[str, Dict[str, Path]] = {}
        for f in nii_files:
            name = f.name
            pid = name.split("-")[0] if "-" in name else name.replace(".nii.gz", "")
            if pid not in patients:
                patients[pid] = {}
            name_lower = name.lower()
            if "t1" in name_lower:
                patients[pid]["t1"] = f
            elif "t2" in name_lower:
                patients[pid]["t2"] = f
            elif "pd" in name_lower:
                patients[pid]["pd"] = f

        for pid, seqs in sorted(patients.items()):
            vol = seqs.get("t1") or next(iter(seqs.values()), None)
            if vol is None:
                continue
            if not validate_file(str(vol)):
                logger.warning("Corrupted IXI volume %s, skipping", vol)
                continue
            cases.append(
                {
                    "case_id": pid,
                    "volume_paths": {"t1": vol},
                    "label": 0,
                    "anatomy": "brain",
                    "patient_id": pid,
                    "source": "ixi",
                    "preferred_sequence": "t1",
                }
            )

        logger.info("IXI: %d subjects from %d files", len(cases), len(nii_files))
        return cases

    def _scan_hcp(self, hcp_dir: Path) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        if not hcp_dir.is_dir():
            logger.warning("HCP directory not found: %s", hcp_dir)
            return cases

        for f in sorted(hcp_dir.glob("*_T1w_*.nii.gz")):
            pid = f.name.split("_")[0]
            cases.append(
                {
                    "case_id": pid,
                    "volume_paths": {"t1": f},
                    "label": 0,
                    "anatomy": "brain",
                    "patient_id": pid,
                    "source": "hcp",
                    "preferred_sequence": "t1",
                }
            )

        logger.info("HCP: %d subjects", len(cases))
        return cases

    @staticmethod
    def _find_oasis_volume(session_dir: Path) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}

        if session_dir.name.startswith(("OAS1", "OAS2")):
            if session_dir.name.startswith("OAS2"):
                raw_dir = session_dir / "RAW"
                if raw_dir.exists():
                    for f in sorted(raw_dir.iterdir()):
                        if f.suffix == ".img" and "mpr" in f.name.lower():
                            paths["t1"] = f
                            break
                    if paths:
                        return paths
                    for f in sorted(raw_dir.glob("*.img")):
                        paths["t1"] = f
                        break
                    return paths
            mpr_dir = session_dir / "PROCESSED" / "MPRAGE" / "T88_111"
            if mpr_dir.exists():
                for f in sorted(mpr_dir.iterdir()):
                    name = f.name.lower()
                    if (
                        f.suffix == ".img"
                        and "gfc" in name
                        and "masked" not in name
                        and "fseg" not in name
                    ):
                        paths["t1"] = f
                        break
            if not paths:
                for f in sorted(mpr_dir.rglob("*.img")):
                    if "fseg" not in f.name.lower():
                        paths["t1"] = f
                        break
            return paths

        nifti_files = list(session_dir.rglob("*.nii.gz"))
        return _discover_sequences(nifti_files)

    # ------------------------------------------------------------------
    # Patient-level split
    # ------------------------------------------------------------------

    def _patient_split(
        self,
        cases: List[Dict[str, Any]],
        split: str,
    ) -> List[Dict[str, Any]]:
        if not cases:
            return []

        patient_ids = sorted(set(c["patient_id"] for c in cases))
        patient_labels: Dict[str, int] = {}
        skipped_ties = 0

        for pid in patient_ids:
            pid_labels = [c["label"] for c in cases if c["patient_id"] == pid]
            counts: Dict[int, int] = {}
            for lbl in pid_labels:
                counts[lbl] = counts.get(lbl, 0) + 1
            max_count = max(counts.values())
            winners = [lbl for lbl, cnt in counts.items() if cnt == max_count]
            if len(winners) > 1:
                skipped_ties += 1
                patient_labels[pid] = -1
            else:
                patient_labels[pid] = winners[0]

        if skipped_ties > 0:
            logger.warning(
                "%d patients have tied labels across sessions — excluded from split",
                skipped_ties,
            )

        ids_array = [pid for pid in patient_ids if patient_labels[pid] != -1]
        labels_array = [patient_labels[pid] for pid in ids_array]

        try:
            from sklearn.model_selection import train_test_split

            _has_sklearn = True
        except ImportError:
            _has_sklearn = False
            warnings.warn(
                "scikit-learn not available; using simple sequential split "
                "(prefer installing scikit-learn for stratified sampling)."
            )

        if _has_sklearn:
            val_test_ratio = self.val_ratio + self.test_ratio
            if val_test_ratio >= 1.0 or self.train_ratio <= 0:
                selected = set(ids_array)
            else:
                try:
                    train_ids, temp_ids = train_test_split(
                        ids_array,
                        test_size=val_test_ratio,
                        stratify=labels_array,
                        random_state=self.seed,
                    )
                except ValueError:
                    train_ids, temp_ids = train_test_split(
                        ids_array,
                        test_size=val_test_ratio,
                        random_state=self.seed,
                    )

                if self.test_ratio <= 0 or self.val_ratio <= 0:
                    if split == "train":
                        selected = set(train_ids)
                    else:
                        selected = set(temp_ids)
                else:
                    temp_labels = [patient_labels[pid] for pid in temp_ids]
                    val_fraction = self.val_ratio / (
                        self.val_ratio + self.test_ratio
                    )
                    try:
                        val_ids, test_ids = train_test_split(
                            temp_ids,
                            test_size=1.0 - val_fraction,
                            stratify=temp_labels,
                            random_state=self.seed,
                        )
                    except ValueError:
                        val_ids, test_ids = train_test_split(
                            temp_ids,
                            test_size=1.0 - val_fraction,
                            random_state=self.seed,
                        )

                    if split == "train":
                        selected = set(train_ids)
                    elif split == "val":
                        selected = set(val_ids)
                    else:
                        selected = set(test_ids)
        else:
            n = len(ids_array)
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))
            rng = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(n, generator=rng).tolist()
            if split == "train":
                selected = set(ids_array[i] for i in perm[:train_end])
            elif split == "val":
                selected = set(ids_array[i] for i in perm[train_end:val_end])
            else:
                selected = set(ids_array[i] for i in perm[val_end:])

        return [c for c in cases if c["patient_id"] in selected]

    # ------------------------------------------------------------------
    # Volume loading
    # ------------------------------------------------------------------

    def _load_volume(self, case: Dict[str, Any]) -> torch.Tensor:
        volume_paths = case["volume_paths"]

        if self.multi_sequence:
            filtered: Dict[str, Path] = {}
            for seq in self.multi_sequence_list:
                if seq in volume_paths:
                    filtered[seq] = volume_paths[seq]
            if not filtered:
                preferred = case.get("preferred_sequence")
                if preferred and preferred in volume_paths:
                    filtered = {preferred: volume_paths[preferred]}
                else:
                    first_key = next(iter(volume_paths.keys()))
                    filtered = {first_key: volume_paths[first_key]}
            return load_and_preprocess_multi(
                {k: str(v) for k, v in filtered.items()}
            )
        else:
            preferred = case.get("preferred_sequence")
            if preferred and preferred in volume_paths:
                path = volume_paths[preferred]
            else:
                path = next(iter(volume_paths.values()))
            return load_and_preprocess(str(path))

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self._cases[idx]
        try:
            volume = self._load_volume(case)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to load %s (case %s): %s. Raising RuntimeError.",
                case.get("volume_paths", {}),
                case.get("case_id", "?"),
                exc,
            )
            raise RuntimeError(
                f"Failed to load case {case.get('case_id', '?')}: {exc}"
            ) from exc

        item: Dict[str, Any] = {
            "volume": volume,
            "label": torch.tensor(float(case["label"]), dtype=torch.float32),
            "anatomy": case["anatomy"],
            "patient_id": case["patient_id"],
            "source": case["source"],
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __repr__(self) -> str:
        normal = sum(1 for c in self._cases if c["label"] == 0)
        abnormal = sum(1 for c in self._cases if c["label"] == 1)
        return (
            f"BrainMRIDataset(anatomy=brain, split={self.split}, "
            f"cases={len(self._cases)}, normal={normal}, abnormal={abnormal})"
        )

    def compute_pos_weight(self) -> float:
        """Compute positive class weight for BCEWithLogitsLoss.

        Returns:
            ``num_negatives / num_positives`` clamped to [1, 100].
        """
        n_pos = sum(1 for c in self._cases if c["label"] == 1)
        n_neg = sum(1 for c in self._cases if c["label"] == 0)
        if n_pos == 0:
            return 1.0
        weight = n_neg / n_pos
        return float(np.clip(weight, 1.0, 100.0))


# ---------------------------------------------------------------------------
# ProstateMRIDataset
# ---------------------------------------------------------------------------


class ProstateMRIDataset(Dataset):
    """Dataset for prostate MRI triage using PI-CAI data.

    Expected layout::

        data/raw/prostate/pi_cai/
        ├── images/
        │   ├── 1000000_t2w.nii.gz
        │   ├── 1000000_dwi.nii.gz
        │   ├── 1000000_adc.nii.gz
        │   └── ...
        └── picai_labels/
            └── clinical_information/
                └── marksheet.csv

    Returns:
        ``{"volume": (1,96,96,96), "label": 0/1, "anatomy": "prostate",
          "patient_id": str, "source": "picai"}``
    """

    _case_cache: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(
        self,
        data_root: str | Path,
        labels_dir: str | Path,
        split: str = "train",
        transform: Any = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        preferred_sequence: str = "t2w",
        multi_sequence: bool = False,
        multi_sequence_list: Sequence[str] = ("t2w", "dwi", "adc"),
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.labels_dir = Path(labels_dir)
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.preferred_sequence = preferred_sequence
        self.multi_sequence = multi_sequence
        self.multi_sequence_list = tuple(multi_sequence_list)

        marksheet = self.labels_dir / "clinical_information" / "marksheet.csv"

        cache_key = f"{self.data_root.resolve()}_{'multi' if multi_sequence else 'single'}"
        if cache_key not in self._case_cache:
            cases = self._scan_picai(marksheet)
            self._case_cache[cache_key] = cases

        all_cases = self._case_cache[cache_key]
        self._cases = self._patient_split(all_cases, split)

        dist = get_label_distribution(self._cases)
        logger.info(
            "ProstateMRIDataset(split=%s): %d cases — %s",
            split,
            len(self._cases),
            {k: v for k, v in dist.items()},
        )

    def _scan_picai(self, marksheet_path: Path) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        mha_files = sorted(self.data_root.rglob("*.mha"))
        if not mha_files:
            mha_files = sorted(self.data_root.rglob("*.nii.gz"))

        by_study: Dict[str, Dict[str, Path]] = {}
        for f in mha_files:
            name = f.stem
            parts = name.split("_")
            if parts and parts[-1].endswith(".nii"):
                parts[-1] = parts[-1][:-4]
            study_id = (
                parts[-2]
                if len(parts) >= 2 and parts[-2].isdigit()
                else None
            )
            if study_id is None:
                continue
            if study_id not in by_study:
                by_study[study_id] = {}
            for seq in ["t2w", "dwi", "adc", "hbv"]:
                if seq in name.lower():
                    by_study[study_id]["t2w" if seq == "hbv" else seq] = f
                    break

        for study_id, seqs in sorted(by_study.items()):
            preferred = seqs.get("t2w") or next(iter(seqs.values()), None)
            if preferred is None:
                continue

            label = extract_label_from_picai(study_id, marksheet_path)
            if label == -1:
                continue

            cases.append(
                {
                    "case_id": study_id,
                    "volume_paths": seqs,
                    "label": label,
                    "anatomy": "prostate",
                    "patient_id": study_id,
                    "source": "picai",
                    "preferred_sequence": self.preferred_sequence,
                }
            )

        logger.info("PI-CAI: %d cases found", len(cases))
        return cases

    def _load_volume(self, case: Dict[str, Any]) -> torch.Tensor:
        volume_paths = case["volume_paths"]

        if self.multi_sequence:
            filtered: Dict[str, Path] = {}
            for seq in self.multi_sequence_list:
                if seq in volume_paths:
                    filtered[seq] = volume_paths[seq]
            if not filtered:
                preferred = case.get("preferred_sequence", "t2w")
                if preferred in volume_paths:
                    filtered = {preferred: volume_paths[preferred]}
                else:
                    first_key = next(iter(volume_paths.keys()))
                    filtered = {first_key: volume_paths[first_key]}
            return load_and_preprocess_multi(
                {k: str(v) for k, v in filtered.items()}
            )
        else:
            preferred = case.get("preferred_sequence", "t2w")
            if preferred in volume_paths:
                path = volume_paths[preferred]
            else:
                path = next(iter(volume_paths.values()))
            return load_and_preprocess(str(path))

    def __len__(self) -> int:
        return len(self._cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self._cases[idx]
        try:
            volume = self._load_volume(case)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to load prostate case %s: %s", case.get("case_id"), exc
            )
            raise RuntimeError(
                f"Failed to load prostate case {case.get('case_id')}: {exc}"
            ) from exc

        item: Dict[str, Any] = {
            "volume": volume,
            "label": torch.tensor(float(case["label"]), dtype=torch.float32),
            "anatomy": case["anatomy"],
            "patient_id": case["patient_id"],
            "source": case["source"],
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

    def _patient_split(
        self, cases: List[Dict[str, Any]], split: str
    ) -> List[Dict[str, Any]]:
        if not cases:
            return []

        patient_ids = sorted(set(c["patient_id"] for c in cases))
        patient_labels: Dict[str, int] = {}
        skipped_ties = 0

        for pid in patient_ids:
            pid_labels = [c["label"] for c in cases if c["patient_id"] == pid]
            counts: Dict[int, int] = {}
            for lbl in pid_labels:
                counts[lbl] = counts.get(lbl, 0) + 1
            max_count = max(counts.values())
            winners = [lbl for lbl, cnt in counts.items() if cnt == max_count]
            if len(winners) > 1:
                skipped_ties += 1
                patient_labels[pid] = -1
            else:
                patient_labels[pid] = winners[0]

        if skipped_ties > 0:
            logger.warning(
                "%d prostate patients have tied labels — excluded from split",
                skipped_ties,
            )

        ids_array = [pid for pid in patient_ids if patient_labels[pid] != -1]
        labels_array = [patient_labels[pid] for pid in ids_array]

        try:
            from sklearn.model_selection import train_test_split

            _has_sklearn = True
        except ImportError:
            _has_sklearn = False

        if _has_sklearn:
            val_test_ratio = self.val_ratio + self.test_ratio
            if val_test_ratio >= 1.0 or self.train_ratio <= 0:
                selected = set(ids_array)
            else:
                try:
                    train_ids, temp_ids = train_test_split(
                        ids_array,
                        test_size=val_test_ratio,
                        stratify=labels_array,
                        random_state=self.seed,
                    )
                except ValueError:
                    train_ids, temp_ids = train_test_split(
                        ids_array,
                        test_size=val_test_ratio,
                        random_state=self.seed,
                    )
                if self.test_ratio <= 0 or self.val_ratio <= 0:
                    if split == "train":
                        selected = set(train_ids)
                    else:
                        selected = set(temp_ids)
                else:
                    temp_labels = [patient_labels[pid] for pid in temp_ids]
                    val_fraction = self.val_ratio / (
                        self.val_ratio + self.test_ratio
                    )
                    try:
                        val_ids, test_ids = train_test_split(
                            temp_ids,
                            test_size=1.0 - val_fraction,
                            stratify=temp_labels,
                            random_state=self.seed,
                        )
                    except ValueError:
                        val_ids, test_ids = train_test_split(
                            temp_ids,
                            test_size=1.0 - val_fraction,
                            random_state=self.seed,
                        )
                    if split == "train":
                        selected = set(train_ids)
                    elif split == "val":
                        selected = set(val_ids)
                    else:
                        selected = set(test_ids)
        else:
            n = len(ids_array)
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))
            rng = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(n, generator=rng).tolist()
            if split == "train":
                selected = set(ids_array[i] for i in perm[:train_end])
            elif split == "val":
                selected = set(ids_array[i] for i in perm[train_end:val_end])
            else:
                selected = set(ids_array[i] for i in perm[val_end:])

        return [c for c in cases if c["patient_id"] in selected]

    def __repr__(self) -> str:
        normal = sum(1 for c in self._cases if c["label"] == 0)
        abnormal = sum(1 for c in self._cases if c["label"] == 1)
        return (
            f"ProstateMRIDataset(anatomy=prostate, split={self.split}, "
            f"cases={len(self._cases)}, normal={normal}, abnormal={abnormal})"
        )

    def compute_pos_weight(self) -> float:
        n_pos = sum(1 for c in self._cases if c["label"] == 1)
        n_neg = sum(1 for c in self._cases if c["label"] == 0)
        if n_pos == 0:
            return 1.0
        weight = n_neg / n_pos
        return float(np.clip(weight, 1.0, 100.0))


# ---------------------------------------------------------------------------
# MultiAnatomyDataset
# ---------------------------------------------------------------------------


class MultiAnatomyDataset(Dataset):
    """Wraps multiple anatomy-specific datasets with flat global indexing.

    Args:
        datasets: Dict mapping anatomy name strings to
            :class:`Dataset` instances.
    """

    def __init__(self, datasets: Dict[str, Dataset]) -> None:
        super().__init__()
        self.datasets = datasets
        self._index_map: List[Tuple[str, int]] = []
        for anatomy, ds in datasets.items():
            for local_idx in range(len(ds)):
                self._index_map.append((anatomy, local_idx))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anatomy, local_idx = self._index_map[idx]
        return self.datasets[anatomy][local_idx]

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={len(v)}" for k, v in self.datasets.items())
        return f"MultiAnatomyDataset({parts})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_dataloaders(
    config: Any,
    enabled_anatomies: Optional[Sequence[str]] = None,
) -> Dict[str, DataLoader]:
    """Create train, val, and test DataLoaders from a merged configuration.

    Args:
        config: Merged :class:`~triagemri.config.Config` instance.
        enabled_anatomies: Which anatomies to include.  ``None`` or
            ``["brain"]`` for brain-only.

    Returns:
        ``{"train": DataLoader, "val": DataLoader, "test": DataLoader}``.
    """
    from triagemri.data.transforms import get_train_transforms, get_val_transforms

    if enabled_anatomies is None:
        enabled_anatomies = ["brain"]

    data_cfg = config.data
    train_cfg = config.training

    batch_size: int = int(train_cfg.get("batch_size", 4))
    num_workers: int = int(train_cfg.get("num_workers", 4))
    seed: int = int(train_cfg.get("seed", 42))

    split_cfg = data_cfg.get("split", {})
    train_ratio: float = float(split_cfg.get("train_ratio", 0.70))
    val_ratio: float = float(split_cfg.get("val_ratio", 0.15))
    test_ratio: float = float(split_cfg.get("test_ratio", 0.15))

    multi_seq_cfg = data_cfg.get("multi_sequence", {})
    multi_seq_enabled: bool = bool(multi_seq_cfg.get("enabled", False))
    multi_seq_list: Sequence[str] = multi_seq_cfg.get("input_sequences", [])

    train_transform = get_train_transforms(multi_sequence=multi_seq_enabled)
    val_transform = get_val_transforms(multi_sequence=multi_seq_enabled)

    train_parts: Dict[str, Dataset] = {}
    val_parts: Dict[str, Dataset] = {}
    test_parts: Dict[str, Dataset] = {}

    for anatomy in enabled_anatomies:
        if anatomy == "brain":
            t, v, te = _build_brain_datasets(
                data_cfg,
                train_transform,
                val_transform,
                train_ratio,
                val_ratio,
                test_ratio,
                seed,
                multi_seq_enabled,
                multi_seq_list,
            )
        elif anatomy == "prostate":
            t, v, te = _build_prostate_datasets(
                data_cfg,
                train_transform,
                val_transform,
                train_ratio,
                val_ratio,
                test_ratio,
                seed,
                multi_seq_enabled,
                multi_seq_list,
            )
        elif anatomy == "breast":
            t, v, te = _build_breast_datasets(
                data_cfg,
                train_transform,
                val_transform,
                train_ratio,
                val_ratio,
                test_ratio,
                seed,
            )
        else:
            continue
        train_parts[anatomy] = t
        val_parts[anatomy] = v
        test_parts[anatomy] = te

    if not train_parts:
        raise RuntimeError(
            "No datasets created. Check --anatomies and data availability."
        )

    train_ds = MultiAnatomyDataset(train_parts)
    val_ds = MultiAnatomyDataset(val_parts)
    test_ds = MultiAnatomyDataset(test_parts)

    use_sampler = (train_cfg.get("imbalance") or {}).get("use_weighted_sampler", False)

    def _make_loader(
        dataset: Dataset,
        shuffle: bool,
        drop_last: bool = False,
        use_weighted_sampler: bool = False,
    ) -> DataLoader:
        actual_workers = num_workers
        if len(dataset) == 0:
            shuffle = False
            drop_last = False
            use_weighted_sampler = False
            if num_workers > 0:
                logger.warning(
                    "Empty dataset — using num_workers=0 to avoid idle worker warnings."
                )
                actual_workers = 0

        sampler = None
        if use_weighted_sampler:
            labels: List[int] = []
            anat_list: List[str] = []
            for anatomy, ds in dataset.datasets.items():
                if hasattr(ds, "_cases"):
                    labels.extend(int(c["label"]) for c in ds._cases)
                    anat_list.extend([anatomy] * len(ds._cases))
                else:
                    for i in range(len(ds)):
                        item = ds[i]
                        labels.append(int(item["label"]))
                        anat_list.append(str(item.get("anatomy", "unknown")))
            labels_t = torch.tensor(labels, dtype=torch.long)
            anat_map = {a: i for i, a in enumerate(sorted(set(anat_list)))}
            anat_ids = torch.tensor(
                [anat_map[a] for a in anat_list], dtype=torch.long
            )

            group_key = anat_ids * 2 + labels_t
            _unique_groups, group_counts = torch.unique(
                group_key, return_counts=True
            )
            group_weights_dict = {
                int(g): 1.0 / int(c) for g, c in zip(_unique_groups, group_counts)
            }
            sample_weights = torch.tensor(
                [group_weights_dict[int(g)] for g in group_key]
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights.tolist(),
                num_samples=len(sample_weights),
                replacement=True,
            )
            logger.info(
                "WeightedRandomSampler: class counts=%s, groups=%s",
                torch.bincount(labels_t).tolist(),
                {
                    f"{a}-{int(lbl)}": group_weights_dict.get(
                        anat_map[a] * 2 + int(lbl), 0
                    )
                    for a in sorted(anat_map)
                    for lbl in [0, 1]
                },
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=actual_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

    return {
        "train": _make_loader(
            train_ds, shuffle=True, drop_last=True, use_weighted_sampler=use_sampler
        ),
        "val": _make_loader(val_ds, shuffle=False),
        "test": _make_loader(test_ds, shuffle=False),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_clinical_csv(oasis_dir: Path) -> Optional[Path]:
    candidates = [
        "oasis3_clinical.csv",
        "OASIS3_clinical_data.csv",
        "clinical_data.csv",
        "oasis_cross-sectional.csv",
    ]
    for name in candidates:
        path = oasis_dir / name
        if path.exists():
            return path
    parent_path = oasis_dir.parent / "oasis3_clinical.csv"
    if parent_path.exists():
        return parent_path
    return None


# ---------------------------------------------------------------------------
# Per-anatomy dataset builders
# ---------------------------------------------------------------------------


def _build_brain_datasets(
    data_cfg: Any,
    train_transform: Any,
    val_transform: Any,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    multi_sequence: bool = False,
    multi_sequence_list: Sequence[str] = (),
) -> Tuple[BrainMRIDataset, BrainMRIDataset, BrainMRIDataset]:
    datasets_cfg = data_cfg.get("datasets", {})
    brain_cfg = datasets_cfg.get("brain", {})
    brats_cfg = brain_cfg.get("brats", {})
    oasis_cfg = brain_cfg.get("oasis", {})

    brats_path = Path(brats_cfg.get("path", "data/raw/brain/brats"))
    oasis_path = Path(oasis_cfg.get("path", "data/raw/brain/oasis"))
    brain_root = Path(data_cfg.get("raw_dir", "data/raw")) / "brain"
    clinical_csv = _find_clinical_csv(oasis_path)

    ds_kwargs = dict(
        data_root=brain_root,
        sources=("brats", "oasis", "ixi", "hcp"),
        clinical_csv=clinical_csv,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        multi_sequence=multi_sequence,
        multi_sequence_list=multi_sequence_list,
    )

    train_ds = BrainMRIDataset(split="train", transform=train_transform, **ds_kwargs)
    val_ds = BrainMRIDataset(split="val", transform=val_transform, **ds_kwargs)
    test_ds = BrainMRIDataset(split="test", transform=val_transform, **ds_kwargs)

    max_brats = (data_cfg.get("balance") or {}).get("max_brats", None)
    if max_brats is not None and max_brats > 0:
        train_ds, val_ds, test_ds = _limit_brats(
            train_ds, val_ds, test_ds, max_brats, seed,
        )

    logger.info(
        "Brain: %d train / %d val / %d test cases",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    return train_ds, val_ds, test_ds


def _limit_brats(
    train_ds: BrainMRIDataset,
    val_ds: BrainMRIDataset,
    test_ds: BrainMRIDataset,
    max_total: int,
    seed: int,
) -> Tuple[BrainMRIDataset, BrainMRIDataset, BrainMRIDataset]:
    """Randomly subsample BraTS cases to *max_total*, stratified by label.

    Non-BraTS cases are never touched.
    """
    all_brats = (
        [c for c in train_ds._cases if c["source"] == "brats"]
        + [c for c in val_ds._cases if c["source"] == "brats"]
        + [c for c in test_ds._cases if c["source"] == "brats"]
    )
    if len(all_brats) <= max_total:
        return train_ds, val_ds, test_ds

    fraction = max_total / len(all_brats)
    for ds in (train_ds, val_ds, test_ds):
        brats_indices = [
            i for i, c in enumerate(ds._cases) if c["source"] == "brats"
        ]
        n_keep = max(1, int(len(brats_indices) * fraction))
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(brats_indices, size=n_keep, replace=False))
        ds._cases = [
            c
            for i, c in enumerate(ds._cases)
            if c["source"] != "brats" or i in keep
        ]

    logger.info(
        "BraTS limited: %d -> %d (%.0f%%)",
        len(all_brats),
        max_total,
        fraction * 100,
    )
    return train_ds, val_ds, test_ds


def _build_prostate_datasets(
    data_cfg: Any,
    train_transform: Any,
    val_transform: Any,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    multi_sequence: bool = False,
    multi_sequence_list: Sequence[str] = (),
) -> Tuple[Dataset, Dataset, Dataset]:
    datasets_cfg = data_cfg.get("datasets", {})
    prostate_cfg = datasets_cfg.get("prostate", {})
    pi_cai_cfg = prostate_cfg.get("pi_cai", {})
    pi_cai_dir = Path(pi_cai_cfg.get("path", "data/raw/prostate/pi_cai"))
    labels_dir = pi_cai_dir.parent / "picai_labels"

    nii_count = len(list(pi_cai_dir.rglob("*.nii.gz"))) if pi_cai_dir.is_dir() else 0
    mha_count = len(list(pi_cai_dir.rglob("*.mha"))) if pi_cai_dir.is_dir() else 0

    if nii_count == 0 and mha_count == 0:
        logger.warning(
            "Prostate PI-CAI: no .nii.gz or .mha files found in %s. "
            "Skipping prostate — download the dataset first.",
            pi_cai_dir,
        )
        return _empty_datasets()

    train_ds = ProstateMRIDataset(
        data_root=pi_cai_dir,
        labels_dir=labels_dir,
        split="train",
        transform=train_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        preferred_sequence="t2w",
        multi_sequence=multi_sequence,
        multi_sequence_list=multi_sequence_list,
    )
    val_ds = ProstateMRIDataset(
        data_root=pi_cai_dir,
        labels_dir=labels_dir,
        split="val",
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        preferred_sequence="t2w",
        multi_sequence=multi_sequence,
        multi_sequence_list=multi_sequence_list,
    )
    test_ds = ProstateMRIDataset(
        data_root=pi_cai_dir,
        labels_dir=labels_dir,
        split="test",
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        preferred_sequence="t2w",
        multi_sequence=multi_sequence,
        multi_sequence_list=multi_sequence_list,
    )
    logger.info(
        "Prostate: %d train / %d val / %d test cases",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    return train_ds, val_ds, test_ds


def _build_breast_datasets(
    data_cfg: Any,
    train_transform: Any,
    val_transform: Any,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    datasets_cfg = data_cfg.get("datasets", {})
    breast_cfg = datasets_cfg.get("breast", {})
    mri_cfg = breast_cfg.get("mri_breast", {})
    breast_dir = Path(mri_cfg.get("path", "data/raw/breast/advanced_mri"))

    nii_count = len(list(breast_dir.rglob("*.nii.gz"))) if breast_dir.is_dir() else 0
    if nii_count == 0:
        logger.warning("Breast: no .nii.gz files found in %s. Skipping.", breast_dir)
        return _empty_datasets()

    logger.warning("Breast MRI dataset not yet implemented. Skipping.")
    return _empty_datasets()


def _empty_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    class _EmptyDataset(Dataset):
        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> dict:
            raise IndexError

    empty = _EmptyDataset()
    return empty, empty, empty
