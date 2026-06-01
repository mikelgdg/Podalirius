# Data Provenance — Triage-MRI

## Summary

| Dataset | Version | Download Date | N Cases | License | Commercial Use | Contact for Commercial License |
|---------|---------|---------------|---------|---------|----------------|-------------------------------|
| BraTS 2023 | Training set | TBD | ~1,250 | TCIA DUA | No | help@cancerimagingarchive.net |
| OASIS-3 | TBD | TBD | ~1,378 | OASIS DUA | No | oasis-brains.org |
| IXI | v1.0 | TBD | 582 | CC BY-SA 3.0 | No (SA clause) | brain-development.org |
| HCP S1200 | S1200 | TBD | ~1,200 | HCP DUA | No | humanconnectome.org |
| PI-CAI | v1.0 | TBD | ~1,500 | CC BY-NC-SA 4.0 | No (NC+SA) | pi-cai.grand-challenge.org |
| Triad Encoder | v1 | TBD | 131K pre-train | Unknown | To be confirmed | Paper authors (arxiv) |

## License Compatibility Matrix

| License Type | Can sell model? | Can distribute weights? | Can use in SaaS? |
|-------------|----------------|------------------------|-------------------|
| CC0 | Yes | Yes | Yes |
| CC BY 4.0 | Yes | Yes | Yes |
| CC BY-SA 4.0 | Yes* | Yes* | Yes* |
| CC BY-NC 4.0 | No | No | No |
| CC BY-NC-SA 4.0 | No | No | No |
| TCIA DUA | No | Research only | No |
| HCP DUA | No | Research only | No |
| OASIS DUA | No | Research only | No |

*SA (ShareAlike) requires derivative works to use the same license.

## Path to Commercial Viability

To produce a sellable model, all training data must be:
1. CC0, CC BY 4.0, or commercially licensed
2. OR: the model must be re-trained exclusively on data with a commercial license

### Potential replacements

| Anatomy | Current | Commercial Alternative | License |
|---------|---------|----------------------|---------|
| Brain tumors | BraTS | UPenn-GBM (TCIA) | CC BY 4.0 |
| Brain normal | OASIS/IXI/HCP | OASIS-1, UK Biobank | CC BY 4.0 / Commercial |
| Prostate | PI-CAI | ProstateX, hospital data | Verify / Negotiate |
| Breast | MRI-Breast | DDSM, CMMD | CC0 / CC BY |

## Encoder License Status

Triad Swin-B SimMIM was pre-trained on 131K volumes from 3 Chinese hospitals.
License unknown — must be confirmed with authors before any commercial use.

If Triad is non-commercial:
- BrainMVP (OpenMEDLab, CVPR 2025) — check license
- MRI-CORE (Duke) — check license
- Train custom encoder on public CC0/CC BY data
