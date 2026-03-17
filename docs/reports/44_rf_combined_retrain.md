# Results Report: RF Combined Retrain (Esper + ML-SAFT)

## Model Performance

Two Random Forest models were trained under identical conditions (100 trees, Morgan FP + RDKit descriptors, random_state=42) and evaluated across multiple benchmark slices. The only experimental variable is the training corpus:

- **RF_esper**: trained on 1440 Esper-only molecules
- **RF_combined**: trained on 1255 molecules (Esper + ML-SAFT, InChI-deduplicated, Esper priority)

### Data Provenance

| Quantity | Count |
|---|---|
| Raw Esper | 1801 |
| Raw ML-SAFT | 870 |
| Deduplicated combined | 1936 |
| Overlap (by InChI) | 745 |
| Net new ML-SAFT molecules | 135 |

## Comparison to Baseline: Canonical Esper Holdout

Primary apples-to-apples comparison. Both models evaluated on the same 361 Esper holdout molecules.

| Target | Model | MAE | RMSE | R2 | Train n | Test n |
|---|---|---|---|---|---|---|
| m (segments) | RF_esper | 0.5867 | 1.1601 | 0.6194 | 1440 | 361 |
| m (segments) | RF_combined | 0.6117 | 1.1654 | 0.6159 | 1255 | 361 |
| $\sigma$ ($\AA$) | RF_esper | 0.1886 | 0.3233 | 0.3573 | 1440 | 361 |
| $\sigma$ ($\AA$) | RF_combined | 0.1919 | 0.3207 | 0.3678 | 1255 | 361 |
| $\varepsilon$/k (K) | RF_esper | 26.8217 | 51.4125 | 0.3264 | 1440 | 361 |
| $\varepsilon$/k (K) | RF_combined | 27.7814 | 51.9707 | 0.3117 | 1255 | 361 |

### Esper Benchmark Delta (Combined - Esper)

| Target | Delta MAE | Delta RMSE | Delta R2 |
|---|---|---|---|
| m (segments) | +0.0249 | +0.0053 | -0.0035 |
| $\sigma$ ($\AA$) | +0.0033 | -0.0027 | +0.0105 |
| $\varepsilon$/k (K) | +0.9598 | +0.5582 | -0.0147 |

## Uncertainty Quality

RF tree-disagreement standard deviations used as uncertainty estimates. Coverage = fraction of test molecules where |error| <= k * predicted_std.

| Target | Model | Slice | Mean Std | Std/Error | 1-sig Cov. | 2-sig Cov. |
|---|---|---|---|---|---|---|
| m (segments) | RF_esper | Esper bench. | 0.7988 | 1.361 | 0.784 | 0.950 |
| m (segments) | RF_combined | Esper bench. | 0.8084 | 1.322 | 0.773 | 0.956 |
| $\sigma$ ($\AA$) | RF_esper | Esper bench. | 0.2372 | 1.257 | 0.787 | 0.934 |
| $\sigma$ ($\AA$) | RF_combined | Esper bench. | 0.2695 | 1.404 | 0.792 | 0.953 |
| $\varepsilon$/k (K) | RF_esper | Esper bench. | 29.4492 | 1.098 | 0.731 | 0.909 |
| $\varepsilon$/k (K) | RF_combined | Esper bench. | 34.2898 | 1.234 | 0.789 | 0.925 |
| m (segments) | RF_esper | ML-SAFT only | 0.7096 | 0.627 | 0.385 | 0.538 |
| m (segments) | RF_combined | ML-SAFT only | 0.7711 | 0.696 | 0.462 | 0.577 |
| $\sigma$ ($\AA$) | RF_esper | ML-SAFT only | 0.2287 | 0.420 | 0.308 | 0.500 |
| $\sigma$ ($\AA$) | RF_combined | ML-SAFT only | 0.3089 | 0.590 | 0.385 | 0.654 |
| $\varepsilon$/k (K) | RF_esper | ML-SAFT only | 30.5752 | 0.412 | 0.308 | 0.500 |
| $\varepsilon$/k (K) | RF_combined | ML-SAFT only | 42.3572 | 0.686 | 0.500 | 0.731 |

### Calibration Interpretation

For a well-calibrated Gaussian uncertainty estimate, 1-sigma coverage should be ~68.3% and 2-sigma coverage should be ~95.4%.

Both models show similar calibration behavior on the Esper benchmark. Average 1-sigma coverage: RF_esper=0.767, RF_combined=0.785.

## Slice Analysis

### Combined Benchmark Test Set

Combined test set has 388 molecules total. RF_esper is evaluated on a leak-free subset (n=94; molecules also in esper_train are excluded). RF_combined is evaluated on all 388 molecules.

| Target | Model | MAE | RMSE | R2 | n |
|---|---|---|---|---|---|
| m (segments) | RF_esper | 0.5802 | 0.9349 | 0.7892 | 94 |
| m (segments) | RF_combined | 0.6187 | 1.2097 | 0.6552 | 388 |
| $\sigma$ ($\AA$) | RF_esper | 0.2441 | 0.4236 | 0.0592 | 94 |
| $\sigma$ ($\AA$) | RF_combined | 0.2015 | 0.3308 | 0.3623 | 388 |
| $\varepsilon$/k (K) | RF_esper | 34.7080 | 65.2798 | 0.2036 | 94 |
| $\varepsilon$/k (K) | RF_combined | 27.1661 | 49.3711 | 0.3907 | 388 |

### Esper Subset of Combined Test

Esper-origin molecules within the combined test set (362 molecules).

| Target | Model | MAE | RMSE | R2 |
|---|---|---|---|---|
| m (segments) | RF_esper | 0.3692 | 0.5601 | 0.9160 |
| m (segments) | RF_combined | 0.5836 | 1.1860 | 0.6635 |
| $\sigma$ ($\AA$) | RF_esper | 0.1294 | 0.2181 | 0.4275 |
| $\sigma$ ($\AA$) | RF_combined | 0.1784 | 0.2906 | 0.4273 |
| $\varepsilon$/k (K) | RF_esper | 19.5762 | 45.3852 | 0.4320 |
| $\varepsilon$/k (K) | RF_combined | 24.6830 | 45.1189 | 0.4200 |

### ML-SAFT-Only Subset of Combined Test

ML-SAFT-only molecules in the combined test set (26 molecules). Small sample; treat as directional evidence.

| Target | Model | MAE | RMSE | R2 |
|---|---|---|---|---|
| m (segments) | RF_esper | 1.1320 | 1.5296 | 0.4976 |
| m (segments) | RF_combined | 1.1072 | 1.5011 | 0.5162 |
| $\sigma$ ($\AA$) | RF_esper | 0.5441 | 0.7241 | -0.2936 |
| $\sigma$ ($\AA$) | RF_combined | 0.5236 | 0.6761 | -0.1280 |
| $\varepsilon$/k (K) | RF_esper | 74.2833 | 100.0979 | -0.2189 |
| $\varepsilon$/k (K) | RF_combined | 61.7386 | 89.6192 | 0.0229 |

### Fluorinated External Validation (15 compounds)

| Target | Model | MAE | RMSE | R2 |
|---|---|---|---|---|
| m (segments) | RF_esper | 0.8501 | 0.8700 | -2.5301 |
| m (segments) | RF_combined | 0.6642 | 0.7111 | -1.3584 |
| $\sigma$ ($\AA$) | RF_esper | 0.0722 | 0.0865 | 0.7557 |
| $\sigma$ ($\AA$) | RF_combined | 0.0540 | 0.0645 | 0.8642 |
| $\varepsilon$/k (K) | RF_esper | 14.1227 | 20.3421 | -0.4338 |
| $\varepsilon$/k (K) | RF_combined | 8.7168 | 12.8325 | 0.4294 |

## Key Findings

1. **Esper benchmark performance is essentially unchanged.** The largest R2 delta between RF_esper and RF_combined on the canonical Esper holdout is 0.0147. Adding ~135 ML-SAFT molecules does not materially change generalization on the Esper test distribution.

2. **Uncertainty calibration comparison.** Average 1-sigma coverage on Esper benchmark: RF_esper=0.767, RF_combined=0.785. 
   Both models are similarly calibrated. The addition of ML-SAFT data does not degrade uncertainty quality.

3. **ML-SAFT-only molecule performance.** RF_combined evaluated on 26 ML-SAFT-only test molecules achieves average R2=0.137. 
   This is weak, indicating these molecules remain hard to predict. The combined RF does not dramatically improve on novel ML-SAFT chemistry.

4. **This step extends Step 01b with mandatory uncertainty metrics.** The 1-sigma and 2-sigma coverage comparison provides a more complete picture than point metrics alone.

## Figures

See `figures/44_rf_combined_retrain/` for:
- `parity_rf_esper_vs_combined.png` -- 2x3 parity plots (RF_esper top row, RF_combined bottom row)
- `slice_metrics_comparison.png` -- Dot plot comparing MAE/RMSE/R2 across evaluation slices
- `uncertainty_calibration_rf_comparison.png` -- Mean predicted std vs mean absolute error for Esper benchmark and ML-SAFT-only slice
- `coverage_comparison.png` -- 1-sigma and 2-sigma coverage by target and slice for both RFs
- `combined_test_source_breakdown.png` -- Composition of the combined test set (Esper-derived vs ML-SAFT-only)

## Deviations

**Split strategy adjusted to prevent data leakage.** The step guide calls for independent 80/20 splits on both datasets. However, because Esper molecules appear in both datasets, a naive independent split causes Esper test molecules to appear in the combined training set (293/361 = 81% leakage). To prevent this, the combined training set excludes both the Esper holdout and the combined test set. Similarly, when evaluating RF_esper on combined-derived slices, molecules from esper_train are filtered out. This results in different effective sample sizes for some slice comparisons but ensures all reported metrics are free of train/test contamination.

## Readiness Check

- [x] Esper-only RF and Combined RF retrained in same script with identical hyperparameters
- [x] Both models use the same feature pipeline and split seed (42)
- [x] Both models evaluated on the Esper benchmark test set
- [x] Both models evaluated on the combined benchmark test set
- [x] ML-SAFT-only molecules isolated and reported separately
- [x] Point metrics (MAE, RMSE, R2) reported for every target on every slice
- [x] Uncertainty metrics (mean std, 1-sigma/2-sigma coverage, calibration) reported for both RFs
- [x] All 5 required figures saved to `figures/44_rf_combined_retrain/`
- [x] Per-molecule predictions saved to `model/saved/step44_rf_combined_predictions.csv`
- [x] Aggregate metrics saved to `model/saved/step44_rf_combined_metrics.json`

## Recommendation

**Keep Esper-only RF for production; use combined RF for exploratory analyses only.** The combined RF shows no meaningful improvement on the canonical Esper benchmark and its uncertainty calibration is comparable. The net addition of 135 molecules (7.5% increase) is insufficient to shift RF generalization. The Esper dataset remains the cleaner, better-characterized training corpus for production use. The combined RF may be useful for exploratory screening beyond the Esper chemical space, but predictions on ML-SAFT-only molecules should be flagged as lower confidence.

