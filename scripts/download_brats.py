#!/usr/bin/env python3
"""
Download BraTS 2023 subsets from Synapse.

PREREQUISITE: Accept the Data Use Agreement for each subset on the web:
  1. https://www.synapse.org/Synapse:syn51514105  (BraTS-GLI — glioma)
  2. https://www.synapse.org/Synapse:syn51514106  (BraTS-MEN — meningioma)
  3. https://www.synapse.org/Synapse:syn51514107  (BraTS-MET — metastasis)
  4. https://www.synapse.org/Synapse:syn51514108  (BraTS-PED — pediatric)

For each, log in and click the lock icon → Request Access → Accept DUA.

Usage:
    export SYNAPSE_AUTH_TOKEN="token" && python scripts/download_brats.py
    python scripts/download_brats.py --auth-token "token"
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import zipfile
from pathlib import Path

try:
    import synapseclient
    import synapseutils
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "synapseclient"])
    import synapseclient
    import synapseutils

DATA_FOLDERS = {
    "syn51514105": "BraTS-GLI",
    "syn51514106": "BraTS-MEN",
    "syn51514107": "BraTS-MET",
    "syn51514108": "BraTS-PED",
}

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "brain" / "brats"


def _login(token: str):
    syn = synapseclient.Synapse()
    if token:
        syn.login(authToken=token, silent=True)
    else:
        try:
            syn.login(silent=True)
        except Exception:
            print("[brats] Create token: Synapse → Settings → Personal Access Tokens")
            token = getpass.getpass("Token: ").strip()
            syn.login(authToken=token, silent=True)
    print(f"[brats] {syn.username}")
    return syn


def _download(syn, folder_id: str, folder_name: str, base_dir: Path) -> int:
    target = base_dir / folder_name
    target.mkdir(parents=True, exist_ok=True)

    print(f"\n[brats] {folder_name} ({folder_id}) → {target}")
    try:
        files = synapseutils.syncFromSynapse(syn, folder_id, path=str(target))
    except Exception as exc:
        if "access" in str(exc).lower() or "credential" in str(exc).lower():
            print(f"[brats]   Needs DUA: https://www.synapse.org/Synapse:{folder_id}")
            return 0
        raise

    n_zips = 0
    for f in files:
        if f.name.endswith(".zip") and f.path:
            zip_path = target / f.name
            if zip_path.exists():
                size_mb = zip_path.stat().st_size / 1024 / 1024
                print(f"[brats]   OK  {f.name} ({size_mb:.0f} MB)")
                n_zips += 1
            else:
                print(f"[brats]   MISS {f.name} — DUA may be required")
    return n_zips


def _extract(base_dir: Path) -> int:
    count = 0
    for zip_path in sorted(base_dir.rglob("*.zip")):
        print(f"[brats] Extract {zip_path.name}...", end=" ", flush=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(zip_path.parent)
            zip_path.unlink()
            print("OK")
            count += 1
        except zipfile.BadZipFile:
            print("CORRUPT")
            zip_path.unlink()
        except Exception as exc:
            print(f"ERROR: {exc}")
    return count


def _flatten(base_dir: Path) -> None:
    """Move NIfTI files from nested dirs up to base_dir for simpler scanning."""
    import shutil
    for nii in base_dir.rglob("*.nii.gz"):
        parent = nii.parent
        if parent.name.startswith("BraTS-"):
            # Already in a case folder, check if it's nested too deeply
            grandparent = parent.parent
            if grandparent != base_dir and grandparent.name.startswith("BraTS"):
                dest_dir = base_dir / parent.name
                if not dest_dir.exists():
                    shutil.move(str(parent), str(dest_dir))
                    print(f"[brats]   Move {parent.name} → {base_dir.name}/")


def _verify(base_dir: Path) -> bool:
    segs = list(base_dir.rglob("*-seg.nii.gz"))
    case_dirs = {s.parent for s in segs}
    print(f"\n[brats] {len(case_dirs)} cases with segmentation masks")
    total = len(list(base_dir.rglob("*.nii.gz")))
    print(f"[brats] {total} total .nii.gz files")
    return len(case_dirs) > 0


def main():
    parser = argparse.ArgumentParser(description="Download BraTS 2023 subsets.")
    parser.add_argument("--auth-token", "-t", default=os.environ.get("SYNAPSE_AUTH_TOKEN"))
    parser.add_argument("--data-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--subset", help="Single folder ID to download (e.g. syn51514105)")
    args = parser.parse_args()

    base = Path(args.data_dir)
    base.mkdir(parents=True, exist_ok=True)

    syn = _login(args.auth_token)

    folders = {args.subset: DATA_FOLDERS.get(args.subset, args.subset)} if args.subset else DATA_FOLDERS

    total = 0
    for fid, fname in folders.items():
        total += _download(syn, fid, fname, base)

    if total == 0:
        print("\n[brats] No zip files downloaded.")
        print("[brats] Make sure you accepted all DUAs on the Synapse website.")
        return

    extracted = _extract(base)
    print(f"\n[brats] Extracted {extracted} archives")

    _flatten(base)
    _verify(base)


if __name__ == "__main__":
    main()
