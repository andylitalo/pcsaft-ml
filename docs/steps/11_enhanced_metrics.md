# Step 11: Enhanced Evaluation Metrics

## Purpose

Phase 1 reported R² and MAE. Both have real weaknesses for this task: R² on a single holdout is noisy at n=361, MAE is not scale-free across targets, and neither directly answers "how often is the model close enough to be useful for screening?" This step adds richer metrics to the evaluation harness so all Phase 2 reports use a more honest and informative standard.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| R² and MAE only | MARE, coverage@5%/10%, CCC, and Q² added |
| Single holdout R² (noisy at n=361) | Q² from 5-fold CV replaces holdout R² as headline metric |
| Cross-target comparisons misleading (different scales) | MARE is scale-free; direct comparison across m, σ, ε/k valid |

## Metric Definitions

- **MARE** (Mean Absolute Relative Error): `mean(|y_pred - y_true| / |y_true|)`. Scale-free; directly answers "on average, what fraction of the true value is the error?"
- **Coverage @ k%**: fraction of predictions with `|y_pred - y_true| / |y_true| ≤ k/100`. Directly answers "what fraction of predictions land within k% of true value?"
- **CCC** (Concordance Correlation Coefficient): `2 * cov(y,ŷ) / (var(y) + var(ŷ) + (μ_y - μ_ŷ)²)`. Combines precision and accuracy; penalizes both random error and systematic bias. Range [−1, 1].
- **Q²** (5-fold CV R²): `1 - SS_res_CV / SS_tot`. More stable than holdout R² for n~1,500 datasets.

## Dependencies

- Requires: evaluation harness from Step 03 (`model/evaluate.py`, registry)
- No new packages needed (all metrics implementable in NumPy)

## Implementation Guide

### 11.1 Add Metric Functions to `model/metrics.py`

Create (or extend) `model/metrics.py`:

```python
import numpy as np

def mare(y_true, y_pred):
    """Mean Absolute Relative Error. Assumes y_true != 0."""
    return float(np.mean(np.abs(y_pred - y_true) / np.abs(y_true)))

def coverage_at_threshold(y_true, y_pred, threshold: float):
    """Fraction of predictions within threshold fraction of true value."""
    rel_errors = np.abs(y_pred - y_true) / np.abs(y_true)
    return float(np.mean(rel_errors <= threshold))

def ccc(y_true, y_pred):
    """Concordance Correlation Coefficient."""
    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    return float(2 * cov / (var_t + var_p + (mu_t - mu_p) ** 2))

def q2_cv(model, X, y, cv=5, random_state=42):
    """5-fold cross-validated R² (Q²)."""
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2",
                             n_jobs=-1, random_state=random_state)
    return float(scores.mean()), float(scores.std())
```

### 11.2 Integrate into `model/evaluate.py`

The `compute_metrics` function (or equivalent) should return:

```python
{
    "r2": ...,
    "mae": ...,
    "rmse": ...,
    "mare": ...,
    "coverage_5pct": ...,
    "coverage_10pct": ...,
    "ccc": ...,
    "q2_cv": ...,  # only for sklearn-compatible models
    "q2_cv_std": ...,
}
```

Update the comparison CSV and harness output to include all fields.

### 11.3 Re-run Harness on All Models

```bash
python -m model.evaluate --models gc_pcsaft rf nn chemberta --metrics all
```

Regenerate `model/saved/comparison_metrics.csv` with new columns.

### 11.4 Generate Figures

- `figures/11_metrics/coverage_bar_chart.png`: grouped bar chart of coverage@5% and coverage@10% for all models and all three targets
- `figures/11_metrics/ccc_vs_r2_scatter.png`: scatter of CCC vs R² across all model-target combinations; diagonal where CCC = R² is the bias-free reference
- `figures/11_metrics/mare_by_target.png`: MARE grouped by model and target

### 11.5 Update All Phase 2 Reports

All subsequent step reports (12–15) must include MARE and CCC in addition to R² and MAE.

## Acceptance Criteria (When to Move On)

- [ ] `model/metrics.py` has `mare`, `coverage_at_threshold`, `ccc`, `q2_cv` functions with unit tests
- [ ] Harness outputs new metrics for all registered models
- [ ] `model/saved/comparison_metrics.csv` updated with new columns
- [ ] Three figures saved to `figures/11_metrics/`
- [ ] Report written at `docs/reports/11_enhanced_metrics.md`
- [ ] All existing tests pass; add ≥ 5 new tests for metric functions
- [ ] Commit: `step 11: add MARE, coverage, CCC, and Q² metrics to harness`
