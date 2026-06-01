#!/usr/bin/env bash
# Extract T1w_restore.nii.gz from HCP Structural zips in TEMPORAL/
# Saves to data/raw/brain/hcp/ and removes the zip to save space.
set -euo pipefail

TEMP_DIR="TEMPORAL"
HCP_DIR="data/raw/brain/hcp"

mkdir -p "$HCP_DIR"

for zip in "$TEMP_DIR"/??????_StructuralRecommended.zip; do
    [ -f "$zip" ] || continue
    subj_id=$(basename "$zip" | cut -d'_' -f1)
    dest="$HCP_DIR/${subj_id}_T1w_restore.nii.gz"

    if [ -f "$dest" ]; then
        echo "[SKIP] $subj_id already extracted"
        rm -f "$zip"
        continue
    fi

    echo "[EXTRACT] $subj_id ..."
    unzip -jo "$zip" "*/T1w_restore.nii.gz" -d "$HCP_DIR/" 2>/dev/null
    mv "$HCP_DIR/T1w_restore.nii.gz" "$dest" 2>/dev/null || true

    if [ -f "$dest" ]; then
        echo "  -> $(du -h "$dest" | cut -f1)"
        rm -f "$zip"  # delete zip to save space
    else
        echo "  -> FAILED (T1w_restore.nii.gz not found in zip)"
    fi
done

echo ""
echo "Done. HCP subjects: $(ls "$HCP_DIR"/*_T1w_restore.nii.gz 2>/dev/null | wc -l)"
du -sh "$HCP_DIR/"
