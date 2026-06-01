# Dataset Card — Triage-MRI

## Brain Datasets

| Dataset | Version | Date Downloaded | N Cases | Labels | License | Commercial Use | URL |
|---------|---------|----------------|---------|--------|---------|----------------|-----|
| BraTS 2023 | Training set | TBD | ~1,250 with seg | Tumor from mask | TCIA DUA (research only) | No | synapse.org/brats2023 |
| OASIS-3 | TBD | TBD | ~1,378 | CDR from clinical CSV | OASIS DUA (research only) | No | nitrc.org/oasis |
| IXI | v1.0 | TBD | 582 | All normal (0) | CC BY-SA 3.0 | No (SA) | brain-development.org |
| HCP | S1200 | TBD | ~1,200 | All normal (0) | HCP DUA (research only) | No | humanconnectome.org |

## Prostate Datasets

| Dataset | Version | Date Downloaded | N Cases | Labels | License | Commercial Use | URL |
|---------|---------|----------------|---------|--------|---------|----------------|-----|
| PI-CAI | v1.0 | TBD | ~1,500 | csPCa from marksheet | CC BY-NC-SA 4.0 | No (NC+SA) | zenodo.org/pi-cai |

## Breast Datasets

| Dataset | Version | Date Downloaded | N Cases | Labels | License | Commercial Use | URL |
|---------|---------|----------------|---------|--------|---------|----------------|-----|
| Advanced-MRI-Breast | N/A | Planned | ~200 | Pathology | TCIA DUA (research only) | No | TCIA |

## Pre-trained Encoder

| Model | Weights | License | Commercial Use |
|-------|---------|---------|----------------|
| Triad Swin-B SimMIM | `weights/triad_swinb_simmim.pth` | Unknown (verify with authors) | To be confirmed |

## Data Splits

- Train: 70%
- Validation: 15%  
- Test: 15%
- Patient-level stratified split
- Seed: 42

## Notes

- **Important:** All datasets have non-commercial or share-alike licenses. Models trained on this data cannot be sold without relicensing agreements from each data provider.
- BraTS cases are ALL positive (tumors). There are no normal BraTS cases.
- Only one sequence is used per case (preferred_sequence): T1ce for BraTS, T1 for OASIS/IXI/HCP, T2w for PI-CAI.
