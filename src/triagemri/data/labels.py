import csv
import re
from pathlib import Path
from typing import Dict, Optional, Union

import nibabel as nib
import numpy as np

MIN_VOXELS_POSITIVE: int = 10
"""Minimum number of non-zero voxels in a segmentation mask to consider a case abnormal."""


def extract_label_from_brats(case_dir: Union[str, Path]) -> int:
    """Determine abnormality from BraTS segmentation mask.

    A case is considered abnormal (1) if any segmentation mask
    contains at least ``MIN_VOXELS_POSITIVE`` non-zero voxels;
    otherwise normal (0).

    Args:
        case_dir: Directory containing a BraTS case with *seg*.nii.gz.

    Returns:
        1 if abnormal (tumor present), 0 if normal, -1 if label cannot be determined.
    """
    case_dir = Path(case_dir)
    seg_files = sorted(case_dir.glob("*seg*.nii.gz"))

    if not seg_files:
        raise FileNotFoundError(f"No segmentation file found in {case_dir}")

    for seg_path in seg_files:
        try:
            seg = nib.load(str(seg_path))
            data = seg.get_fdata()
            if not np.isfinite(data).all():
                raise ValueError(f"Segmentation contains NaN/Inf: {seg_path}")
            if (data > 0).sum() >= MIN_VOXELS_POSITIVE:
                return 1
        except Exception as exc:
            raise ValueError(f"Failed to read segmentation {seg_path}: {exc}") from exc

    return 0


def extract_label_from_oasis(
    session_dir: Union[str, Path],
    clinical_csv: Optional[Union[str, Path]] = None,
) -> int:
    """Extract label from OASIS clinical data based on CDR score.

    Supports OASIS-1 (cross-sectional), OASIS-2 (longitudinal), and OASIS-3.

    CDR = 0.0 → normal (0), CDR > 0.0 → abnormal (1).

    OASIS-1 dirs look like ``OAS1_0001_MR1`` (participant = ``OAS1_0001``).
    OASIS-2 dirs look like ``OAS2_0001_MR1`` (participant = ``OAS2_0001``).
    OASIS-3 dirs look like ``OAS30001_MR_d0129`` (participant = ``OAS30001``).

    Args:
        session_dir: Session directory.
        clinical_csv: Path to OASIS clinical data CSV.
    Returns:
        0 (normal), 1 (abnormal), or -1 if unknown.
    """
    session_dir = Path(session_dir)

    if clinical_csv is None:
        return -1

    clinical_csv = Path(clinical_csv)
    if not clinical_csv.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {clinical_csv}")

    dir_name = session_dir.name

    if dir_name.startswith(("OAS1", "OAS2")):
        match_id = dir_name
    elif dir_name.startswith("OAS3"):
        match_id = dir_name.split("_")[0]
    else:
        match_id = dir_name

    try:
        with open(clinical_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get(
                    "MRI ID",
                    row.get("ID", row.get("Subject", row.get("participant_id", ""))),
                )
                if pid.strip() != match_id:
                    continue
                cdr_val = row.get("CDR", row.get("cdr", None))
                if cdr_val is not None and str(cdr_val).strip() != "":
                    try:
                        cdr = float(cdr_val)
                        if np.isnan(cdr):
                            return -1
                        return 0 if cdr == 0.0 else 1
                    except (ValueError, TypeError):
                        return -1
                return -1
    except Exception as exc:
        raise ValueError(f"Failed to parse clinical CSV {clinical_csv}: {exc}") from exc

    return -1


ABNORMAL_PATTERNS = re.compile(
    r"\b(mass|lesion|tumor|tumour|neoplasm|carcinoma|malignan"
    r"|enhancing|enhancement|edema|oedema|hemorrhage|haemorrhage"
    r"|haemorrhagic|hemorrhagic"
    r"|infarct|infarction|metastas|metastasis|metastatic|abnormal"
    r"|positive|suspicious"
    r"|patholog|atrophy|dementia|Alzheimer"
    r"|stroke|isch(a|e)emia|inflammation|demyelinating"
    r"|hydrocephalus|aneurysm|cyst)\b",
    re.IGNORECASE,
)

NORMAL_PATTERNS = re.compile(
    r"\b(no (evidence|sign|abnormal|mass|lesion|tumor|tumour|enhanc)"
    r"|within normal limits|unremarkable|negative|normal study|"
    r"normal exam|no acute|no significant|no suspicious|"
    r"no pathologic|intact|clear)\b",
    re.IGNORECASE,
)


def extract_label_generic(report_text: str) -> int:
    """Extract binary label from a radiology report using regex.

    Abnormal patterns take precedence over normal patterns for safety:
    if both abnormal and normal indicators are present, the case is
    classified as abnormal (1).

    Args:
        report_text: Free-text radiology report.

    Returns:
        1 if abnormal indicators found, 0 if normal indicators found,
        -1 if ambiguous or no indicators found.
    """
    if not report_text or not isinstance(report_text, str):
        return -1

    abnormal_matches = len(ABNORMAL_PATTERNS.findall(report_text))
    normal_matches = len(NORMAL_PATTERNS.findall(report_text))

    if abnormal_matches > 0:
        return 1
    if normal_matches > 0:
        return 0
    return -1


class LabelExtractor:
    """Auto-detect dataset format and extract binary labels.

    Supported formats:
      - BraTS: uses segmentation mask presence
      - OASIS: uses CDR from clinical data CSV
      - generic: regex on radiology report text
    """

    def __init__(self, clinical_csv: Optional[Union[str, Path]] = None):
        self._clinical_csv = clinical_csv

    def extract(self, path: Union[str, Path], report_text: Optional[str] = None) -> int:
        """Auto-detect the dataset format from the path and extract the label.

        Args:
            path: Path to a case directory, session directory, or file.
            report_text: Optional radiology report text for generic extraction.

        Returns:
            1 for abnormal, 0 for normal, -1 if label could not be determined.
        """
        path = Path(path)

        if path.is_dir():
            if path.name.startswith("BraTS") or list(path.glob("*seg*.nii.gz")):
                return extract_label_from_brats(path)
            if path.name.startswith("OAS") or "oasis" in path.name.lower():
                return extract_label_from_oasis(path, self._clinical_csv)

        if report_text:
            return extract_label_generic(report_text)

        return -1

    def extract_brats(self, case_dir: Union[str, Path]) -> int:
        return extract_label_from_brats(case_dir)

    def extract_oasis(self, session_dir: Union[str, Path]) -> int:
        return extract_label_from_oasis(session_dir, self._clinical_csv)

    @staticmethod
    def extract_generic(report_text: str) -> int:
        return extract_label_generic(report_text)


def extract_label_from_picai(
    study_id: str,
    marksheet_path: Union[str, Path],
) -> int:
    """Extract binary label from PI-CAI marksheet CSV.

    Uses ``case_csPCa`` column: YES → 1, NO → 0.  Any other value
    (empty, "UNKNOWN", missing) → -1 (unknown).  Unknown cases are
    skipped by the dataset; they are never treated as normal.

    Returns:
        1 for clinically significant prostate cancer, 0 otherwise, -1 if unknown.
    """
    marksheet_path = Path(marksheet_path)
    if not marksheet_path.exists():
        return -1

    try:
        with open(marksheet_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("study_id", "")).strip() == str(study_id).strip():
                    cspca = row.get("case_csPCa", "").strip().upper()
                    if cspca == "YES":
                        return 1
                    if cspca == "NO":
                        return 0
                    return -1
    except Exception:
        pass
    return -1


def get_label_distribution(cases: list[Dict], source_key: str = "source") -> Dict[str, Dict[str, int]]:
    """Compute label distribution grouped by source.

    Args:
        cases: List of case dicts with ``"label"`` and *source_key* keys.
        source_key: Key to group by (default ``"source"``).

    Returns:
        Nested dict: ``{source_value: {"normal": N, "abnormal": N, "unknown": N}}``.
    """
    distribution: Dict[str, Dict[str, int]] = {}
    for case in cases:
        source = case.get(source_key, "unknown")
        if source not in distribution:
            distribution[source] = {"normal": 0, "abnormal": 0, "unknown": 0}
        label = case.get("label", -1)
        if label == 0:
            distribution[source]["normal"] += 1
        elif label == 1:
            distribution[source]["abnormal"] += 1
        else:
            distribution[source]["unknown"] += 1
    return distribution
