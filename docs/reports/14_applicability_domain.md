# Results Report: Improved Applicability Domain (Combined Dataset, 1,936 molecules)

## Overview

This step enhanced the applicability domain (AD) methodology by adding two complementary approaches to the existing Isolation Forest AD: **Tanimoto nearest-neighbor similarity** (chemically interpretable) and **Williams plots** (leverage-based QSAR standard). The Tanimoto similarity score is now exposed in the API and portal, providing researchers with direct feedback on how similar a query molecule is to the training set.

## What Was Implemented

### 1. TanimotoAD Model (`model/ad_tanimoto.py`)

A new AD model based on maximum Tanimoto similarity to the training set using Morgan fingerprints (radius=2, 2048 bits):

- **IN_DOMAIN_THRESHOLD = 0.4**: Molecules with max Tanimoto similarity ≥ 0.4 are considered in-domain
- **WARNING_THRESHOLD = 0.3**: Molecules with similarity in [0.3, 0.4) trigger a warning
- **Fitted on 1,548 training molecules** from the combined Esper + ML-SAFT + fluorinated dataset

The TanimotoAD model is saved to `model/saved/tanimoto_ad.joblib` and loaded automatically by the model server at startup.

### 2. Williams Plot (`model/ad_williams.py`)

Implemented leverage-based AD visualization following QSAR best practices:

- **`compute_leverage()`**: Computes hat matrix diagonal h_i = x_i^T (X^T X)^{-1} x_i
- **PCA dimension reduction**: Uses 50 PCA components before leverage computation to handle the 2,218-dimensional feature space
- **`plot_williams()`**: Generates standardized residual vs leverage plots with warning thresholds:
  - Horizontal lines at ±3σ (residual outliers)
  - Vertical line at h* = 3p/n (high-leverage threshold)

Williams plots for all three targets (m, sigma, epsilon_k) are saved to `figures/14_ad/`.

### 3. API Integration

- **Schema update**: Added `tanimoto_nn: float | None` field to `PredictionResult` in `serving/schemas.py`
- **Model server**: TanimotoAD loaded at startup in `serving/model_loader.py`; Tanimoto similarity computed for all predictions
- **Backward compatible**: Existing API consumers receive `tanimoto_nn: null` if the model is unavailable

### 4. Portal Enhancement

Updated `portal/components/predictor.py` to display Tanimoto score with color-coded warnings:

- **Green (✓)**: Tanimoto ≥ 0.4 — "Good similarity to training data"
- **Yellow (⚠️)**: Tanimoto ∈ [0.3, 0.4) — "Moderate similarity — treat prediction with caution"
- **Red (🚨)**: Tanimoto < 0.3 — "Low similarity — high prediction uncertainty expected"

## AD Method Comparison

The AD comparison scatter plot (`figures/14_ad/ad_method_comparison.png`) compares Isolation Forest anomaly scores vs Tanimoto similarity for the 388 test molecules:

| Agreement Category | Count | Percentage |
|--------------------|-------|------------|
| Both in-domain | 319 | 82.2% |
| Both out-of-domain | 3 | 0.8% |
| **Disagreement** | 66 | 17.0% |
| **Total agreement** | 322 | **83.0%** |

### Key Findings

1. **High concordance**: The two methods agree on 83% of molecules, validating the Isolation Forest as a reasonable AD indicator.

2. **Disagreements favor Tanimoto**: Of the 66 disagreements, most are cases where the Isolation Forest flags molecules as in-domain but Tanimoto similarity is low (< 0.4). This suggests the IF may be overly permissive in the high-dimensional descriptor space.

3. **Tanimoto is more conservative**: The Tanimoto method flags more molecules as out-of-domain. This is expected and desirable — it directly measures chemical similarity rather than descriptor-space density.

4. **Few molecules are truly out-of-domain by both metrics**: Only 3 molecules were flagged by both methods, indicating the training set provides broad coverage.

## Williams Plots (Leverage-Based AD)

The Williams plots (`figures/14_ad/williams_m.png`, `williams_sigma.png`, `williams_epsilon_k.png`) visualize the distribution of test set molecules in the leverage-residual space:

### Observations

- **Most molecules are low-leverage, low-residual**: The bulk of points cluster near the origin, confirming the test set is well-covered by the training set.
- **Few high-leverage outliers**: Only a small number of molecules exceed the h* threshold, indicating good interpolation rather than extrapolation.
- **Residual outliers exist for sigma and epsilon_k**: Some molecules have standardized residuals exceeding ±3σ, consistent with the known difficulty of predicting these parameters from 2D descriptors alone.
- **m predictions are well-behaved**: The Williams plot for m shows tighter residual distribution, confirming m is the easiest target to predict.

## Integration Testing

Added 16 new tests across `tests/test_ad_tanimoto.py` (15 tests) and `tests/test_serving.py` (1 test):

- TanimotoAD fit/predict behavior
- Edge cases (invalid SMILES, unfitted model)
- Williams plot generation
- API integration (tanimoto_nn in response)

All 199 tests pass (up from 183).

## Code Quality

All modified files pass `ruff check` with no errors.

## Key Improvements Over Previous State

| Before Step 14 | After Step 14 |
|----------------|---------------|
| Single AD method (Isolation Forest) | Three complementary methods (IF, Tanimoto, Williams) |
| Binary in_domain flag | Continuous Tanimoto similarity score (0-1) |
| No chemical interpretability | Tanimoto directly says "closest training molecule" |
| No QSAR-standard AD viz | Williams plots for all three targets |
| API returns `in_domain: bool` only | API returns both `in_domain` and `tanimoto_nn` |

## Limitations and Future Work

1. **Tanimoto threshold choice**: The IN_DOMAIN_THRESHOLD=0.4 is a reasonable default but not validated against experimental error. Future work should correlate Tanimoto similarity with prediction error to determine the optimal threshold.

2. **Williams plots use PCA-reduced features**: The leverage is computed on 50 PCA components rather than the full 2,218-dimensional space. This is a practical compromise but may miss high-leverage molecules in the discarded components.

3. **No consensus AD metric**: The three methods (IF, Tanimoto, Williams) are presented independently. A future improvement could combine them into a single "AD confidence score" using a weighted ensemble or meta-model.

4. **Static training set**: The TanimotoAD model is fitted once on the combined dataset. As new data arrives (via `/submit-data`), the AD model should be retrained alongside the RF/NN models in the Kubeflow pipeline.

## Deviations from Step Guide

None. All deliverables from `docs/steps/14_applicability_domain.md` were implemented as specified.

## Readiness Check

- [x] `model/ad_tanimoto.py` with `TanimotoAD` class; fitted model saved as `model/saved/tanimoto_ad.joblib`
- [x] `model/ad_williams.py` with `plot_williams` and `compute_leverage` functions
- [x] `/predict` response includes `tanimoto_nn: float`
- [x] Portal displays Tanimoto score with color-coded warning level
- [x] Williams plots (3 targets) saved to `figures/14_ad/`
- [x] AD comparison scatter saved to `figures/14_ad/`
- [x] Report written at `docs/reports/14_applicability_domain.md`
- [x] Add ≥ 8 tests (added 16 tests)
- [x] All existing tests pass (199 tests)
- [x] Ruff check passes

**Ready to move to the next step.**
