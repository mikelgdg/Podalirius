# Commercial Viability Assessment — Triage-MRI

## Current License Status: NOT COMMERCIALLY VIABLE

All datasets used for training have non-commercial or share-alike restrictions.
Model weights are derivative works and inherit these restrictions.

## Dataset License Breakdown

| Dataset | License | Can Distribute Weights? | Can Sell? | Can Use in SaaS? |
|---------|---------|------------------------|-----------|-------------------|
| BraTS 2023 | TCIA DUA | No | No | No |
| OASIS-3 | OASIS DUA | No | No | No |
| IXI | CC BY-SA 3.0 | Yes (SA) | No (SA) | Complex |
| HCP S1200 | HCP DUA | No | No | No |
| PI-CAI | CC BY-NC-SA 4.0 | No (NC) | No (NC) | No (NC) |
| **Combined model** | **Most restrictive (NC-SA)** | **No** | **No** | **No** |

The CC BY-NC-SA 4.0 from PI-CAI is the most restrictive and "viral" — any
model trained on it cannot be used commercially.

## Triad Encoder Status — **CONFIRMED MIT ✅**

The Triad Swin-B SimMIM encoder (arxiv 2502.14064, Medical Image Analysis 2026)
is released under the **MIT license** — confirmed on both the [GitHub repo](https://github.com/wangshansong1/Triad)
(license badge) and the [arxiv paper](https://arxiv.org/abs/2502.14064) (CC BY 4.0).

| Item | Value |
|------|-------|
| Paper license | CC BY 4.0 |
| Code license | **MIT** |
| Weights license | **MIT** (inherits repo license) |
| Commercial use | **YES ✅** |
| Can distribute weights | **YES ✅** |
| Can sell | **YES ✅** |

The encoder is **not the blocker** for commercialization. The only remaining
blockers are the MIL head training datasets.
| Contact authors | Email corresponding author for license clarification |
| If non-commercial | Replace encoder with BrainMVP (OpenMEDLab) or train custom |

## Path to Commercial Viability

### Option A: Full retraining on permissive data (recommended, ~3-4 months)

1. Replace all datasets with CC0 or CC BY 4.0 alternatives
2. Re-train MIL head from scratch
3. Verify encoder license or replace encoder
4. Clinical validation on target population

| Component | Current | Replacement | License |
|-----------|---------|------------|---------|
| Brain tumors | BraTS (~600) | UPenn-GBM (TCIA, CC BY 4.0) + LGG-1p19qDeletion (TCIA) + IvyGAP (TCIA) | CC BY 4.0 |
| Brain normal | OASIS+IXI+HCP (~2,600) | OASIS-1 (~400, CC BY 4.0) + IXI (SA constraint) + ADNI (negotiate) | Mixed |
| Prostate cancer | PI-CAI (NC-SA) | ProstateX (TCIA, verify) + PROMISE12 + hospital data | Verify |
| Breast lesions | MRI-Breast (TCIA DUA) | DDSM (~2,500, CC0) or CMMD (~1,800, CC BY) | CC0 / CC BY |

Total cases after replacement: ~4,000-5,000 (similar to current).

### Option B: License negotiation (riskier, ~6+ months)

Contact each data provider for a commercial license:
- TCIA (BraTS): Possible via institutional agreement
- NITRC (OASIS): Unlikely
- HCP: Possible via ConnectomeDB commercial addendum
- PI-CAI authors: Possible via direct contact

Cost estimate: $5,000-50,000+ depending on provider.

### Option C: Hospital-only deployment (easiest, ~1 month)

If the model is deployed ONLY within a hospital's internal network and never
sold or distributed:
- Most research-use restrictions do not apply to internal clinical use
- Check local laws and institution policy
- Cannot sell or distribute externally

## Recommended Action Plan

### Short-term (now)
- [ ] Confirm Triad encoder license
- [ ] Start collecting hospital data for prostate and breast
- [ ] Test zero-shot performance on UPenn-GBM (CC BY)

### Medium-term (2-3 months)
- [ ] Download CC0/CC BY datasets (UPenn-GBM, DDSM, OASIS-1)
- [ ] Re-train MIL head exclusively on permissive data
- [ ] Evaluate performance vs current non-commercial model

### Long-term (6+ months)
- [ ] Clinical validation study (IRB approved)
- [ ] FDA/CE certification pathway assessment
- [ ] Commercial licensing agreements for any remaining restricted data

## Cost Estimate

| Item | Cost |
|------|------|
| GPU compute for retraining | $500-2,000 (cloud) |
| Data acquisition (commercial licenses) | $0-50,000 |
| Clinical validation study | $10,000-100,000 |
| Regulatory certification (FDA 510k) | $50,000-250,000 |
| Legal (license review) | $5,000-15,000 |

## Conclusion

The current model is a research prototype. It cannot be sold, distributed
commercially, or used in a SaaS product. A commercial version requires:

1. Replacing all training data with permissively licensed alternatives
2. (or) Obtaining commercial licenses from each data provider
3. Verifying/replacing the encoder license
4. Clinical validation and possible regulatory clearance

Timeline to commercial prototype: 3-4 months.
Timeline to sellable product: 12-18 months.
