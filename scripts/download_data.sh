#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

BRATS_DIR="${DATA_DIR}/raw/brain/brats"
OASIS_DIR="${DATA_DIR}/raw/brain/oasis"

PROSTATE_DIR="${DATA_DIR}/raw/prostate/pi_cai"
BREAST_DIR="${DATA_DIR}/raw/breast/advanced_mri"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Download helper for Triage-MRI datasets.
Creates directory structure and prints download instructions.

Options:
  --check       Verify whether expected dataset files exist on disk
  --help        Show this help message

Datasets:
  BraTS 2023    Brain tumor segmentation (Synapse)
  OASIS-3       Alzheimer's / aging brain MRI (NITRC)
  PI-CAI        Prostate MRI (planned)
  Advanced-MRI  Breast MRI (planned)
EOF
}

print_header() {
    echo ""
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================${NC}"
}

print_step() {
    echo -e "${YELLOW}  -> $1${NC}"
}

create_dirs() {
    print_header "Creating directory structure"

    mkdir -p "${BRATS_DIR}"
    echo -e "  ${GREEN}OK${NC}  ${BRATS_DIR}"

    mkdir -p "${OASIS_DIR}"
    echo -e "  ${GREEN}OK${NC}  ${OASIS_DIR}"

    mkdir -p "${PROSTATE_DIR}"
    echo -e "  ${GREEN}OK${NC}  ${PROSTATE_DIR}"

    mkdir -p "${BREAST_DIR}"
    echo -e "  ${GREEN}OK${NC}  ${BREAST_DIR}"
}

print_brats_instructions() {
    print_header "BraTS 2023 — Brain Tumor Segmentation"

    cat << EOF

  The BraTS 2023 dataset is distributed via Synapse.
  A free Synapse account is required.

  Steps:
    1. Create an account at https://www.synapse.org/Register
    2. Visit: https://www.synapse.org/Synapse:syn51156910
    3. Accept the Data Use Agreement (DUA)
    4. Download the training data archive(s)

  Expected directory layout after placement:

    ${BRATS_DIR}/
    ├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
    │   ├── BraTS-GLI-00000-000/
    │   │   ├── BraTS-GLI-00000-000-t1c.nii.gz
    │   │   ├── BraTS-GLI-00000-000-t1n.nii.gz
    │   │   ├── BraTS-GLI-00000-000-t2f.nii.gz
    │   │   ├── BraTS-GLI-00000-000-t2w.nii.gz
    │   │   └── BraTS-GLI-00000-000-seg.nii.gz
    │   └── ...
    └── ...

  One-liner to download (requires authenticated synapseclient):

    pip install synapseclient
    python -c "
    import synapseclient
    syn = synapseclient.Synapse()
    syn.login()
    syn.get('syn51156910', downloadLocation='${BRATS_DIR}')
    "

  After extracting, the check flag should pass.

EOF
}

print_oasis_instructions() {
    print_header "OASIS-3 — Alzheimer's Disease / Aging MRI"

    cat << EOF

  OASIS-3 is distributed via NITRC and XNAT Central.
  Free registration and a data use agreement are required.

  Steps:
    1. Request access at: https://www.oasis-brains.org/
    2. After approval, you will receive XNAT credentials
    3. Log in to https://central.xnat.org/
    4. Navigate to OASIS-3 → Download Images
    5. Download the NIfTI data (recommended: T1, T2, FLAIR)

  Expected directory layout after placement:

    ${OASIS_DIR}/
    ├── OAS30001_MR_d0129/
    │   ├── anat1/
    │   │   ├── sub-OAS30001_ses-d0129_T1w.nii.gz
    │   │   └── ...
    │   └── ...
    ├── OAS30002_MR_d0215/
    │   └── ...
    └── ...

  The clinical data CSV (needed for CDR labels) can be obtained from:
    https://www.oasis-brains.org/ → Data → Clinical Data

EOF
}

print_prostate_instructions() {
    print_header "PI-CAI — Prostate MRI (planned)"

    echo ""
    echo -e "  ${YELLOW}Not yet implemented.${NC}"
    echo "  Target: https://pi-cai.grand-challenge.org/"
}

print_breast_instructions() {
    print_header "Advanced MRI for Breast Cancer (planned)"

    echo ""
    echo -e "  ${YELLOW}Not yet implemented.${NC}"
    echo "  Target: https://wiki.cancerimagingarchive.net/"
}

do_check() {
    print_header "Checking dataset files"

    local all_ok=true

    # --- BraTS ---
    echo ""
    echo "  BraTS 2023 (${BRATS_DIR}):"
    if [ -d "${BRATS_DIR}" ]; then
        local brats_cases
        brats_cases=$(find "${BRATS_DIR}" -maxdepth 3 -name "*seg.nii.gz" -type f 2>/dev/null | wc -l)
        if [ "${brats_cases}" -gt 0 ]; then
            echo -e "    ${GREEN}FOUND${NC}  ${brats_cases} cases with segmentation masks"
        else
            echo -e "    ${RED}MISSING${NC} No .nii.gz files with segmentation masks found"
            all_ok=false
        fi
    else
        echo -e "    ${RED}MISSING${NC} Directory does not exist"
        all_ok=false
    fi

    # --- OASIS ---
    echo ""
    echo "  OASIS-3 (${OASIS_DIR}):"
    if [ -d "${OASIS_DIR}" ]; then
        local oasis_scans
        oasis_scans=$(find "${OASIS_DIR}" -maxdepth 4 -name "*T1w*.nii.gz" -type f 2>/dev/null | wc -l)
        if [ "${oasis_scans}" -gt 0 ]; then
            echo -e "    ${GREEN}FOUND${NC}  ${oasis_scans} T1w NIfTI files"
        else
            echo -e "    ${RED}MISSING${NC} No T1w .nii.gz files found"
            all_ok=false
        fi
    else
        echo -e "    ${RED}MISSING${NC} Directory does not exist"
        all_ok=false
    fi

    # --- PI-CAI ---
    echo ""
    echo "  PI-CAI (${PROSTATE_DIR}):"
    if [ -d "${PROSTATE_DIR}" ] && [ "$(find "${PROSTATE_DIR}" -type f 2>/dev/null | head -1)" ]; then
        echo -e "    ${GREEN}FOUND${NC} Files present"
    else
        echo -e "    ${YELLOW}EMPTY${NC} No files — planned dataset"
    fi

    # --- Breast ---
    echo ""
    echo "  Breast Advanced MRI (${BREAST_DIR}):"
    if [ -d "${BREAST_DIR}" ] && [ "$(find "${BREAST_DIR}" -type f 2>/dev/null | head -1)" ]; then
        echo -e "    ${GREEN}FOUND${NC} Files present"
    else
        echo -e "    ${YELLOW}EMPTY${NC} No files — planned dataset"
    fi

    echo ""
    if [ "${all_ok}" = true ]; then
        echo -e "  ${GREEN}All expected brain datasets are present.${NC}"
    else
        echo -e "  ${RED}Some datasets are missing. Run without --check for instructions.${NC}"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "${1:-}" in
    --check)
        do_check
        exit $?
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    "")
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        usage
        exit 1
        ;;
esac

create_dirs
print_brats_instructions
print_oasis_instructions
print_prostate_instructions
print_breast_instructions
