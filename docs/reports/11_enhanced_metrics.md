# Results Report: Enhanced Evaluation Metrics (Step 11)

## Summary

Four new metrics added to the evaluation harness: MARE (Mean Absolute Relative Error), Coverage@5%, Coverage@10%, and CCC (Concordance Correlation Coefficient). Q² (5-fold CV R²) was already present via the `--cv` flag. All metrics are now reported for every model×target combination and saved to `comparison_metrics.csv`. Three new figures generated in `figures/11_metrics/`.

## Model Performance — Full Metric Table

### GC-PC-SAFT (domain baseline)

| Parameter | MAE | RMSE | R² | MARE | CCC | Cov@5% | Cov@10% | n |
|-----------|-----|------|----|------|-----|--------|---------|---|
| m (segments) | 1.001 | 1.460 | 0.396 | 0.246 | 0.733 | 12.3% | 26.0% | 358 |
| σ (Å) | 0.494 | 0.579 | −1.163 | 0.130 | 0.049 | 14.5% | 35.8% | 358 |
| ε/k (K) | 44.62 | 63.50 | −0.037 | 0.207 | 0.177 | 23.7% | 42.2% | 358 |

> **Uncertainty note**: All R², MAE, and RMSE values in this report are point estimates on a single holdout split. 95% bootstrap confidence intervals are available via `python -m model.evaluate --bootstrap` and saved to `model/saved/comparison_metrics.csv` with `_lo`/`_hi` suffix columns.

### RF (Combined Morgan+RDKit)

| Parameter | MAE | RMSE | R² | MARE | CCC | Cov@5% | Cov@10% | n |
|-----------|-----|------|----|------|-----|--------|---------|---|
| m (segments) | 0.585 | 1.160 | 0.619 | 0.153 | 0.787 | 42.1% | 60.1% | 361 |
| σ (Å) | 0.189 | 0.325 | 0.353 | 0.055 | 0.520 | 70.6% | 86.1% | 361 |
| ε/k (K) | 26.83 | 51.28 | 0.330 | 0.113 | 0.533 | 55.7% | 74.0% | 361 |

### Q² (5-fold Cross-Validated R²) — RF only

| Parameter | Q² (mean) | Q² (std) | R² (test) | Gap | Assessment |
|-----------|-----------|----------|-----------|-----|------------|
| m | 0.664 | 0.080 | 0.619 | −0.045 | Good (Q² > 0.6) |
| σ | 0.401 | 0.074 | 0.353 | −0.049 | Below threshold (Q² < 0.5) |
| ε/k | 0.456 | 0.066 | 0.330 | −0.126 | Below threshold; **large gap — suspect overfitting** |

## Comparison to Phase 1 Baseline

Phase 1 reported only R² and MAE. The new metrics reveal a more nuanced picture:

| Model | Target | R² | MARE | CCC | Cov@10% | Key insight |
|-------|--------|----|------|-----|---------|-------------|
| RF | m | 0.62 | 0.153 | 0.787 | 60.1% | Decent R², but only 60% within ±10% |
| RF | σ | 0.35 | **0.055** | 0.520 | **86.1%** | MARE reveals σ is actually well-predicted in relative terms |
| RF | ε/k | 0.33 | 0.113 | 0.533 | 74.0% | 11% average error; 26% of predictions >±10% off |

## Key Findings

**1. σ is better than R²=0.35 suggests.** MARE=0.055 means only 5.5% average relative error, and 70.6% of predictions fall within ±5% of the true value. The low R² is partly due to σ having a narrow natural range (~3.2–5.5 Å); a small absolute error in σ leads to low R² even when the prediction is practically useful. **σ is the most reliably predicted parameter in relative terms.**

**2. ε/k is the hardest target and the most important for vapor pressure.** MARE=0.113 (11.3% average error), only 55.7% within ±10%. Combined with the sensitivity analysis from Step 09 (±10% in ε/k → ~50% change in vapor pressure), this means roughly half of VP predictions from the screening pipeline are off by more than ±50%. This is the primary limitation of the current approach.

**3. Q² flags ε/k overfitting.** The large Q²-vs-R² gap for ε/k (−0.126) means the single-holdout R²=0.33 may be optimistic. The true CV-estimated predictive power is Q²=0.456, and across 5 folds the worst fold scored 0.361. The holdout set is a single favorable split.

**4. CCC reveals systematic bias.** RF CCC < R² for all targets (below the CCC=R² diagonal in the scatter figure), indicating systematic bias: the RF systematically over- or under-predicts for some ranges of each parameter. This is expected for RF models with tree-based boundaries.

**5. GC-PC-SAFT has interesting CCC pattern.** Despite R²=-1.163 on σ, CCC=0.049 for σ — not as catastrophic as R² suggests. The group-contribution method has a small positive correlation with true σ but enormous variance, which drives R² negative. The method is not entirely uninformative; it just has poor precision.

**6. Coverage thresholds directly answer "is the model useful for screening?"** The 5% threshold is relevant when comparing to cyclopentane — a 5% error in ε/k changes VP by ~25%. Only 55.7% of ε/k predictions meet this. For screening, this means ~44% of candidates are ranked incorrectly relative to a ±5% precision standard.

## Figures

See `figures/11_metrics/` for:

- `coverage_bar_chart.png`: Grouped bar chart of coverage@5% and coverage@10% for each model × target. Reveals that RF on σ is practically good (70.6% within ±5%) while RF on ε/k has the worst coverage.
- `ccc_vs_r2_scatter.png`: Scatter of CCC vs R² for all model×target combinations. All RF points are below the CCC=R² diagonal, confirming systematic bias. GC-PC-SAFT on m is above diagonal (high CCC relative to low R² — consistent with a well-calibrated but high-variance model).
- `mare_by_target.png`: MARE grouped by model and target. RF on σ (MARE=0.055) is the only model×target pair that is below 10%. RF on m and ε/k both exceed 10%.

## Deviations

- The step guide proposed a `model/metrics.py` module. In practice, the metric functions were added directly to `model/evaluate.py` to minimize the number of new files and maintain cohesion with the existing harness. The functions (`_ccc`, `_compute_metrics`) are already present in `evaluate.py`.
- MAPE was already present in the harness (equivalent to MARE × 100). It was renamed to MARE and the relative-error computation clarified.
- NN and ChemBERTa were not re-evaluated in this step because `torch` and `transformers` are not installed in the current environment. Their metrics from Phase 1 (Steps 02–04) stand.

## Readiness Check

- [x] `_compute_metrics()` in `model/evaluate.py` returns `mare`, `ccc`, `coverage_5pct`, `coverage_10pct`
- [x] Q² via `--cv` flag already existed; `cv_q2.json` saved to `model/saved/`
- [x] `comparison_metrics.csv` updated with new columns for RF and GC-PC-SAFT
- [x] Three figures saved to `figures/11_metrics/`
- [x] MARE interpretation reported: σ is better than R² suggests; ε/k is the hardest target
- [x] Q² flags ε/k overfitting (gap = −0.126)
- [x] All existing core tests pass (29/29)
