# Step 52: Log-Target Transform for RF and XGBoost

## Objective

Test whether training RF and XGBoost on log-transformed targets improves prediction accuracy, motivated by GNNePCSAFT's use of relative (percentage-based) loss. This is the cheapest way to approximate a relative loss function within tree ensembles whose internal split criterion is fixed at MSE.

## Motivation

The GNNePCSAFT benchmark (Step 48) revealed that a published GNN trained on the same Esper dataset achieves substantially better general accuracy (Esper epsilon/k R² = 0.84 vs RF 0.33). One of the key architectural differences is the loss function: GNNePCSAFT minimizes Huber loss on **relative percentage error**, while our RF and XGBoost minimize MSE on raw (or z-scored) target values.

MSE on raw targets is dominated by epsilon/k (range ~50-550 K) and barely penalizes m errors (range ~1-25). While RF trains separate per-target models (so cross-target scale doesn't directly interfere), within each target the MSE still weights absolute errors uniformly — a 10 K error matters the same whether the true value is 80 K or 500 K. For epsilon/k, where relative accuracy matters more than absolute accuracy for EOS calculations, this is suboptimal.

Log-transforming the targets before training makes the implicit MSE act like minimizing relative error in linear space:

```
MSE(log(y_pred), log(y_true)) ≈ Var(log(y_pred/y_true))
```

This is the classic variance-stabilizing transform for right-skewed, strictly positive targets — which all three PC-SAFT parameters are.

This experiment is fast (< 30 minutes) because it reuses the existing feature pipeline and only changes the target preprocessing.

## Dependencies

- Requires: trained RF and XGBoost models from Steps 01 and 47 (for baseline comparison)
- Requires: Esper dataset and feature pipeline from `model/data/`
- Requires: fluorinated validation set from `model/saved/gnn_fluorinated_validation_set.csv`
- No new packages needed (numpy, sklearn, xgboost already installed)

## Implementation Guide

### 52.1 — Establish baselines

Record the current production RF and XGBoost results for direct comparison. These should already be available from Steps 48 and 50 but recompute them in the same script for consistency:

```python
from model.data.load import load_data, split_data, TARGETS
from model.registry import _compute_features
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

df = load_data()
train_df, test_df = split_data(df)

X_train = _compute_features(train_df["smiles"].tolist())
X_test = _compute_features(test_df["smiles"].tolist())

# Baseline: train on raw targets (one model per target)
for target in TARGETS:
    y_train = train_df[target].values
    y_test = test_df[target].values
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    # Record R², MAE, MAPE
```

### 52.2 — Log-transformed RF and XGBoost

Train RF and XGBoost on `log(y)` for each target. Predict in log-space, then `exp()` back to original scale. All three PC-SAFT parameters are strictly positive (m >= 1, sigma >= 1.9, epsilon/k >= 50), so the log transform is always valid.

```python
for target in TARGETS:
    y_train_raw = train_df[target].values
    y_test_raw = test_df[target].values
    
    y_train_log = np.log(y_train_raw)
    
    # RF on log targets
    rf_log = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_log.fit(X_train, y_train_log)
    pred_log = rf_log.predict(X_test)
    pred_raw = np.exp(pred_log)
    
    # XGBoost on log targets
    xgb_log = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                           random_state=42, n_jobs=-1)
    xgb_log.fit(X_train, y_train_log)
    pred_log_xgb = xgb_log.predict(X_test)
    pred_raw_xgb = np.exp(pred_log_xgb)
    
    # Evaluate in original scale: R², MAE, MAPE
```

**Important**: Evaluate all metrics in **original (linear) scale**, not log-space. The point is to see whether the model's predictions in physical units improve, not whether log-space R² looks different.

Also compute MAPE (mean absolute percentage error), which is the metric most directly targeted by the log transform. If MAPE improves but MAE doesn't, the transform is redistributing error from small molecules (where relative errors were large) to large molecules (where absolute errors are now larger).

### 52.3 — Cross-validated comparison

The single-split comparison from 52.2 may be noisy. Run a 5-fold cross-validated comparison for all four model variants (RF-raw, RF-log, XGB-raw, XGB-log) on epsilon/k:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_ek = train_df["epsilon_k"].values

for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_ek[tr_idx], y_ek[val_idx]
    y_tr_log = np.log(y_tr)
    
    # RF raw
    rf_raw = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_raw.fit(X_tr, y_tr)
    pred_rf_raw = rf_raw.predict(X_val)
    
    # RF log
    rf_log = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_log.fit(X_tr, y_tr_log)
    pred_rf_log = np.exp(rf_log.predict(X_val))
    
    # Same for XGBoost...
    # Record per-fold R², MAE, MAPE for all four variants
```

Report mean ± std across folds. Use a paired comparison: for each fold, compute the delta (log_metric - raw_metric) and check whether the mean delta is consistently positive or negative.

### 52.4 — Fluorinated external validation

This is the decisive test. Train each variant on the full Esper training set and evaluate on the 15 fluorinated compounds:

```python
import pandas as pd

fluor = pd.read_csv("model/saved/gnn_fluorinated_validation_set.csv")
X_fluor = _compute_features(fluor["smiles"].tolist())

for target, lit_col in [("m", "m_lit"), ("sigma", "sigma_lit"), ("epsilon_k", "epsilon_k_lit")]:
    y_lit = fluor[lit_col].values
    
    # Train on full Esper train set
    y_train_raw = train_df[target].values
    
    rf_raw = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_raw.fit(X_train, y_train_raw)
    
    rf_log = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_log.fit(X_train, np.log(y_train_raw))
    
    pred_raw = rf_raw.predict(X_fluor)
    pred_log = np.exp(rf_log.predict(X_fluor))
    
    # Compare MAE and MAPE against literature values
```

If the log-transform helps on Esper but hurts on fluorinated, it is not deployment-relevant. Only report a recommendation to change if both internal and external metrics improve.

### 52.5 — Boiling point propagation (conditional)

If 52.4 shows a meaningful improvement in fluorinated epsilon/k MAE for either log-RF or log-XGB, propagate the predictions through the EOS to compute boiling-point MAE. Use the same `compute_boiling_point()` pipeline from Step 48.

If 52.4 shows no improvement or degradation, skip this section and note that the log-transform did not improve deployment-domain accuracy.

### 52.6 — Generate figures

Save all figures to `figures/52_log_target_transform/`.

**Required:**

1. **`esper_comparison_bar.png`** — Grouped bar chart: R², MAE, and MAPE for RF-raw, RF-log, XGB-raw, XGB-log on the Esper test set. One panel per metric, grouped by target. Error bars from 5-fold CV where available.

2. **`fluorinated_comparison.png`** — Grouped bar chart: epsilon/k MAE and MAPE for all four variants on the 15 fluorinated compounds.

3. **`cv_paired_delta.png`** — Box plot (or forest plot) showing the per-fold delta in epsilon/k R² between log and raw variants. A box clearly to one side of zero indicates a consistent effect; a box spanning zero indicates no meaningful difference.

### 52.7 — Write report

Write `docs/reports/52_log_target_transform.md` following the standard template:

1. **Title**: `# Results Report: Log-Target Transform (RF and XGBoost)`
2. **Motivation**: One paragraph connecting to the GNNePCSAFT relative-loss insight.
3. **Esper holdout results**: Table comparing RF-raw, RF-log, XGB-raw, XGB-log on all three targets. Highlight epsilon/k.
4. **Cross-validated results**: Mean ± std for epsilon/k across 5 folds. State whether the log-transform consistently helps, hurts, or makes no difference.
5. **Fluorinated external validation**: Table comparing all variants on the 15-compound set. This is the decisive result.
6. **Boiling point propagation** (if applicable): BP MAE comparison.
7. **Key findings**: Bullet points.
8. **Recommendation**: Whether to adopt the log-transform for production or not, with explicit evidence.
9. **Figures**: Reference all figure paths.
10. **Readiness check**: Checklist.

### 52.8 — Update supplementary materials

If the log-transform produces a meaningful improvement on both Esper and fluorinated validation, update the unified comparison table in `supplementary_information/model_comparison.md` with the new variant. If not, add a brief note to the "Where Other Models Might Be Valid" section:

> **Log-target transform**: Tested in Step 52 as a way to approximate GNNePCSAFT's relative-loss training. [Result summary].

## Key Files

| File | Purpose |
|------|---------|
| `model/data/load.py` | `load_data()`, `split_data()`, `TARGETS` |
| `model/registry.py` | `_compute_features()` for feature computation |
| `model/saved/gnn_fluorinated_validation_set.csv` | Fluorinated validation compounds with literature PC-SAFT parameters |
| `screening/hfo_screening.py` | `compute_boiling_point()` for EOS propagation (52.5, conditional) |
| `scripts/step52_log_target_transform.py` | Create: driver script |

## Artifacts

| File | Description |
|------|-------------|
| `scripts/step52_log_target_transform.py` | Driver script |
| `docs/reports/52_log_target_transform.md` | Results report |
| `figures/52_log_target_transform/` | All figures |
| `model/saved/step52_log_transform_results.json` | Full comparison results (Esper + fluorinated + CV) |

## Success Criteria

- [ ] RF and XGBoost trained on both raw and log-transformed targets
- [ ] Esper holdout comparison on all three targets with R², MAE, and MAPE
- [ ] 5-fold CV paired comparison on epsilon/k
- [ ] Fluorinated external validation for all four variants
- [ ] Boiling point propagation if fluorinated epsilon/k improves
- [ ] At least 3 figures saved to `figures/52_log_target_transform/`
- [ ] Report written at `docs/reports/52_log_target_transform.md`
- [ ] Supplementary materials updated with result
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- The log-target transform has been evaluated on both internal (Esper) and external (fluorinated) validation
- A clear recommendation is recorded: adopt, reject, or note as marginal
- If the transform helps: the production pipeline is updated (or flagged for update) with explicit evidence
- If the transform doesn't help: this is documented as evidence that the RF = 0.33 ceiling on epsilon/k is not simply a loss-function artifact

## Budget

30–45 minutes. The experiment involves 8 model fits on the Esper training set (4 variants × train + CV), 4 fluorinated evaluations, and basic plotting. No hyperparameter search is needed — use production defaults for both RF and XGBoost. The XGBoost hyperparameters should match those used in Step 47 or the production XGBoost config.
