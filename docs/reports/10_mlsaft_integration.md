# Results Report: ML-SAFT Dataset Integration (Step 10)

## Summary

Step 10 encountered a pre-existing issue: the Figshare article ID hardcoded in `model/data/download_mlsaft.py` (article 24689738) resolves to an unrelated soil-science dataset, not a PC-SAFT parameters dataset. The download script now raises a clear error and documents this issue. The RF was retrained on Esper-only data as a controlled baseline.

## Dataset Issue

The `model/data/download_mlsaft.py` script targeted `https://api.figshare.com/v2/articles/24689738`. Downloaded file had columns `['<b>Type</b>', '<b>Apparent density...</b>', ...]` — clearly a soil engineering aggregates dataset. The required columns (`smiles`, `m`, `sigma`, `epsilon_k`) were absent.

**Resolution**: Script updated to raise a `RuntimeError` with guidance rather than silently saving an invalid file. The module docstring now documents:
- The incorrect article ID
- Candidate correct sources (Esper supplementary, Rehner & Gross FeOs, other published PC-SAFT tables with SMILES identifiers)
- How to update the script when the correct URL is found

**Finding the correct dataset** requires manual bibliographic search. Candidate papers:
- Esper et al. (2023) "Combining Machine Learning and Physical Models for PC-SAFT Parameter Predictions" — supplementary tables may contain additional molecules beyond the main CSV
- Rehner & Gross (2023), "FeOs: An Open-Source Framework for Equations of State and Classical Density Functional Theory" — includes PC-SAFT parameters for ~300 molecules
- Any published ML-for-thermodynamics paper with SMILES + PC-SAFT columns

## Esper-Only Retrain (Baseline)

The RF was retrained on the Esper dataset with identical hyperparameters to confirm reproducibility and establish a clean Phase 2 baseline.

| Parameter | R² (Phase 1) | R² (Phase 2 retrain) | Delta |
|-----------|-------------|----------------------|-------|
| m (segments) | 0.62 | 0.6194 | −0.0006 |
| σ (Å) | 0.35 | 0.3527 | +0.0027 |
| ε/k (K) | 0.33 | 0.3298 | −0.0002 |

Results are numerically identical to Phase 1 (differences within rounding). The training pipeline is deterministic and reproducible.

## Comparison to Baseline

| Scenario | N train | R² (m) | R² (σ) | R² (ε/k) |
|----------|---------|--------|--------|----------|
| Phase 1 (Esper, RF) | 1,440 | 0.62 | 0.35 | 0.33 |
| Phase 2 retrain (Esper, RF) | 1,440 | 0.619 | 0.353 | 0.330 |
| ML-SAFT combined | pending | — | — | — |

## Key Findings

1. **The ML-SAFT download script has an incorrect dataset reference.** The Figshare article ID needs to be updated. This is a documentation/infrastructure issue, not a modeling failure.

2. **RF training is fully reproducible.** Phase 1 and Phase 2 results differ by <0.003 R² across all targets. The pipeline is deterministic.

3. **ML-SAFT integration remains high-priority.** The data loader and merge logic (`load_data("combined")`) are implemented and tested. Once the correct dataset is located, integration is a one-command operation.

## Figures

See `figures/10_mlsaft/r2_comparison_esper_baseline.png`: bar chart comparing Phase 1 and Phase 2 Esper-only R² (confirms reproducibility; ML-SAFT comparison bar to be added when dataset is available).

## Deviations

- ML-SAFT dataset could not be downloaded due to incorrect Figshare article ID in the download script. This is a pre-existing issue surfaced by this step.
- ChemBERTa retrain on combined data deferred pending ML-SAFT dataset.

## Readiness Check

- [x] `model/data/download_mlsaft.py` updated: now raises `RuntimeError` with guidance instead of silently saving invalid data; `.xls` extension support added
- [x] Bad `mlsaft_pcsaft.csv` (soil science data) deleted from repository
- [x] RF retrained on Esper-only: metrics confirmed reproducible
- [x] Figure saved to `figures/10_mlsaft/r2_comparison_esper_baseline.png`
- [x] All existing tests pass
- [ ] ML-SAFT combined retrain: pending correct dataset source — deferred to manual follow-up
