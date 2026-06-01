#!/usr/bin/env python3
"""Verify downloaded datasets against expected file counts and structure."""

from pathlib import Path
import sys

EXPECTED = {
    "data/raw/brain/brats": {"min_nii": 100, "description": "BraTS 2023 brain tumors"},
    "data/raw/brain/oasis": {"min_nii": 500, "description": "OASIS-3 aging/Alzheimer"},
    "data/raw/brain/ixi": {"min_nii": 100, "description": "IXI healthy controls"},
    "data/raw/brain/hcp": {"min_nii": 100, "description": "HCP healthy subjects"},
    "data/raw/prostate/pi_cai": {"min_nii": 100, "description": "PI-CAI prostate cancer"},
    "data/raw/breast/advanced_mri": {"min_nii": 0, "description": "Breast MRI (planned)"},
}


def main():
    project_root = Path(__file__).resolve().parent.parent
    all_ok = True

    for rel_path, spec in EXPECTED.items():
        full_path = project_root / rel_path
        min_nii = spec["min_nii"]
        desc = spec["description"]

        if not full_path.exists():
            if min_nii > 0:
                print(f"[WARN] {desc}: directory not found — {full_path}")
                all_ok = False
            else:
                print(f"[INFO] {desc}: directory not found (expected — planned dataset)")
            continue

        nii_files = list(full_path.rglob("*.nii.gz")) + list(full_path.rglob("*.mha"))
        count = len(nii_files)
        status = "OK" if count >= min_nii else "WARN"
        print(f"[{status}] {desc}: {count} NIfTI/MHA files (min expected: {min_nii})")
        if count < min_nii:
            all_ok = False

    weights = project_root / "weights" / "triad_swinb_simmim.pth"
    if weights.exists():
        size_mb = weights.stat().st_size / (1024 * 1024)
        print(f"[OK] Triad weights: {size_mb:.1f} MB")
    else:
        print(f"[WARN] Triad weights not found — run scripts/download_triad.sh")
        all_ok = False

    if all_ok:
        print("\nAll data verified successfully.")
    else:
        print("\nSome datasets are missing or incomplete. Run download scripts.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
