# Step 52: Log-Target Transform for RF and XGBoost

## Objective

Test whether training RF and XGBoost on log-transformed targets improves prediction accuracy, motivated by GNNePCSAFT's use of relative (percentage-based) loss. This is the cheapest way to approximate a relative loss function within tree ensembles whose internal split criterion is fixed at MSE.

## Motivation

The GNNePCSAFT benchmark (Step 48) revealed that a published GNN trained on the same Esper dataset achieves substantially better general accuracy (Esper epsilon/k R² = 0.84 vs RF 0.33). One of the key architectural differences is the loss function: GNNePCSAFT minimizes Huber loss on **relative percentage error**, while our RF and XGBoost minimize MSE on raw (or z-scored) target values.

MSE on raw targets is dominated by epsilon/k (range ~50-550 K) and barely penalizes m errors (range ~1-25). While RF trains separate per-target models (so cross-target scale doesn't directly interfere), within each target the MSE still weights absolute errors uniformly — a 10 K error matters the same whether the true value is 80 K or 500 K. For epsilon/k, where relative accuracy matters more than absolute accuracy for EOS calculations, this is suboptimal.

Log-transforming the targets before training makes the implicit MSE act like minimizing relative error in linear space. For small relative errors:

```
log(y_pred) - log(y_true) = log(y_pred / y_true) ≈ (y_pred - y_true) / y_true
```

so `MSE(log ŷ, log y) ≈ mean((ŷ - y)²/y²)`, i.e. mean squared relative error. This approximation holds when |ŷ/y - 1| << 1; for molecules where the model errs by >30% (which occurs in the tail of epsilon/k), the correspondence weakens. The experiment should report the fraction of test molecules where relative error exceeds 30% so the approximation's validity can be assessed.

This is the classic variance-stabilizing transform for right-skewed, strictly positive targets — which all three PC-SAFT parameters are. A more general approach is the Box-Cox family (of which log is the λ=0 case), but log is the natural starting point because (a) it has a direct interpretation as relative-error minimization, (b) it requires no fitted parameter, and (c) the PC-SAFT parameter ranges span roughly one order of magnitude, where log is well-conditioned.

**Back-transform bias (Jensen's inequality):** A critical subtlety is that `exp(E[log ŷ])` is the geometric mean of the predictions, which is systematically ≤ the arithmetic mean. RF predictions in log-space are leaf-node averages of `log(y)`, so `exp(pred)` returns the geometric mean of training targets in that leaf — a negatively biased estimator of the conditional mean. This bias is small when within-leaf variance is small (which RF achieves via deep trees), but it should be quantified. Two standard corrections exist:

1. **Smearing estimator** (Duan 1983): multiply `exp(pred)` by the empirical mean of `exp(residuals)` from the training set.
2. **Variance correction**: add `0.5 * σ²_log` to the log-space prediction before exponentiating, where `σ²_log` is the estimated variance of log-space residuals (assumes normality).

The experiment should compute back-transformed predictions both with and without bias correction, and report the difference. If the correction is <1% of MAE, it can be safely ignored.

This experiment is fast (~1 hour) because it reuses the existing feature pipeline and only changes the target preprocessing. The added statistical diagnostics (heteroscedasticity check, bias correction, paired tests) are what make it rigorous rather than merely suggestive.

## Dependencies

- Requires: trained RF and XGBoost models from Steps 01 and 47 (for baseline comparison)
- Requires: Esper dataset and feature pipeline from `model/data/`
- Requires: fluorinated validation set from `model/saved/gnn_fluorinated_validation_set.csv`
- No new packages needed (numpy, sklearn, xgboost already installed)

## Implementation Guide

### 52.1 — Verify the heteroscedasticity assumption

The log-transform is justified only if prediction errors scale with target magnitude (multiplicative error structure). Before running the full comparison, verify this assumption on the production RF baseline:

1. **Residual-vs-fitted plot**: for each target, scatter `|y_test - ŷ|` against `y_test`. If errors fan out with increasing y (heteroscedastic), the log-transform has a theoretical basis. If errors are roughly constant (homoscedastic), the log-transform may hurt by distorting the error structure.
2. **Breusch-Pagan test** or rank correlation (Spearman) between `|residual|` and `y_test`. Report the test statistic and p-value for each target.
3. **Histogram of raw vs log-transformed targets**: verify that the log-transform actually reduces skewness. Report skewness and kurtosis for both.

Save the residual-vs-fitted plots to `figures/52_log_target_transform/heteroscedasticity_check.png`. If epsilon/k residuals show no meaningful heteroscedasticity (Spearman |r| < 0.1, p > 0.1), note this as a reason to expect limited benefit from the transform.

### 52.2 — Establish baselines

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

### 52.3 — Log-transformed RF and XGBoost

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
    
    # XGBoost on log targets — use the same hyperparameters as the production
    # XGBoost (load from saved model's best_params, or from the GridSearchCV
    # grid in model/xgb/xgb_model.py). The production model uses
    # MultiOutputRegressor(XGBRegressor(...)) with CV-selected params; for this
    # per-target experiment, use the per-target XGBRegressor directly with the
    # same n_estimators, max_depth, and learning_rate from the saved best_params.
    import joblib
    xgb_saved = joblib.load("model/saved/xgb/xgb_model.joblib")
    bp = xgb_saved["best_params"]
    xgb_log = XGBRegressor(
        n_estimators=bp.get("estimator__n_estimators", 200),
        max_depth=bp.get("estimator__max_depth", 6),
        learning_rate=bp.get("estimator__learning_rate", 0.1),
        subsample=bp.get("estimator__subsample", 0.8),
        colsample_bytree=bp.get("estimator__colsample_bytree", 0.8),
        random_state=42, n_jobs=-1
    )
    xgb_log.fit(X_train, y_train_log)
    pred_log_xgb = xgb_log.predict(X_test)
    pred_raw_xgb = np.exp(pred_log_xgb)
    
    # Evaluate in original scale: R², MAE, MAPE
```

**Important**: Evaluate all metrics in **original (linear) scale**, not log-space. The point is to see whether the model's predictions in physical units improve, not whether log-space R² looks different.

**Fairness caveat — hyperparameter mismatch**: The production RF and XGBoost hyperparameters were tuned (Steps 15, 50) to minimize MSE on raw targets. The optimal tree depth, number of estimators, and regularization for MSE on `log(y)` may differ. Using the same hyperparameters for both variants means any improvement from the log-transform is a lower bound (better params might help more), but any degradation might be partially recoverable with re-tuning. This is acknowledged as a limitation. A full hyperparameter sweep on the log-transformed targets is out of scope for this step (the budget is 30–45 minutes), but if the log-transform shows a promising but sub-threshold improvement, re-tuning is the natural follow-up.

Also compute MAPE (mean absolute percentage error) and MdAPE (median absolute percentage error). MAPE is the metric most directly targeted by the log transform, but it has a known asymmetry: under-predictions are bounded (0–100%) while over-predictions are unbounded. This means MAPE systematically favors models that under-predict, which is exactly what the Jensen's inequality bias produces. Therefore, **a MAPE improvement alone is not sufficient evidence that the log-transform helps** — it may simply reflect the negative bias being rewarded by an asymmetric metric.

To guard against this, also compute:
- **Symmetric MAPE** (sMAPE): `mean(|y - ŷ| / ((|y| + |ŷ|) / 2))`, which penalizes over- and under-predictions equally.
- **Median signed percentage error**: to check whether the log-transform introduces a systematic directional bias.

If MAPE improves but sMAPE doesn't, or if median signed error shifts negative, the improvement is an artifact of MAPE's asymmetry, not a genuine accuracy gain.

### 52.4 — Cross-validated comparison

The single-split comparison from 52.3 may be noisy. Run a **repeated 5-fold cross-validated comparison** for all four model variants (RF-raw, RF-log, XGB-raw, XGB-log) on **all three targets** (not just epsilon/k — selective CV on the target expected to benefit most is a form of outcome-driven analysis):

```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

for target in TARGETS:
    y_target = train_df[target].values
    
    for fold_idx, (tr_idx, val_idx) in enumerate(rkf.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_target[tr_idx], y_target[val_idx]
        y_tr_log = np.log(y_tr)
        
        # RF raw
        rf_raw = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_raw.fit(X_tr, y_tr)
        pred_rf_raw = rf_raw.predict(X_val)
        
        # RF log (with and without bias correction)
        rf_log = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_log.fit(X_tr, y_tr_log)
        pred_log_space = rf_log.predict(X_val)
        pred_rf_log_naive = np.exp(pred_log_space)
        
        # Smearing correction: exp(pred) * mean(exp(train_residuals))
        train_pred_log = rf_log.predict(X_tr)
        train_resid = y_tr_log - train_pred_log
        smear_factor = np.mean(np.exp(train_resid))
        pred_rf_log_corrected = pred_rf_log_naive * smear_factor
        
        # Same for XGBoost...
        # Record per-fold R², MAE, sMAPE for all variants
```

Report mean ± std across 15 folds (5 folds × 3 repeats). Use a **paired t-test** (or Wilcoxon signed-rank if normality of deltas is questionable) on the per-fold deltas: for each fold, compute `delta = log_metric - raw_metric` and test H0: mean(delta) = 0. With 15 paired observations, this provides reasonable power to detect a consistent effect. Report the paired 95% CI on the delta, not just a p-value.

### 52.5 — Fluorinated external validation

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

With only 15 compounds, any MAE difference has wide confidence intervals. Use a **paired Wilcoxon signed-rank test** on per-molecule absolute errors (raw vs log) for each variant to test whether the difference is statistically significant. Also report the **paired bootstrap 95% CI** on the MAE difference (2,000 resamples). If the CI spans zero, the difference is not meaningful regardless of the point estimate.

If the log-transform helps on Esper but hurts on fluorinated, it is not deployment-relevant. Only report a recommendation to change if both internal and external metrics improve and the fluorinated improvement is statistically significant (p < 0.05 on the paired test, or bootstrap CI excluding zero).

### 52.6 — Boiling point propagation (conditional)

If 52.5 shows a meaningful improvement in fluorinated epsilon/k MAE for either log-RF or log-XGB (paired bootstrap CI excluding zero), propagate the predictions through the EOS to compute boiling-point MAE. Use the same `compute_boiling_point()` pipeline from Step 48.

If 52.5 shows no improvement or degradation, skip this section and note that the log-transform did not improve deployment-domain accuracy.

### 52.7 — Generate figures

Save all figures to `figures/52_log_target_transform/`.

**Required:**

1. **`heteroscedasticity_check.png`** — 1×3 panel: |residual| vs y_true for each target from the baseline RF. Include Spearman r and p-value annotations. This is the assumption-validation figure.

2. **`target_distributions.png`** — 2×3 panel: histograms of raw and log-transformed targets for all three parameters, with skewness annotated.

3. **`esper_comparison_bar.png`** — Grouped bar chart: R², MAE, and sMAPE for RF-raw, RF-log, RF-log-corrected, XGB-raw, XGB-log, XGB-log-corrected on the Esper test set. One panel per metric, grouped by target. Error bars from repeated CV where available.

4. **`fluorinated_comparison.png`** — Grouped bar chart: epsilon/k MAE and sMAPE for all variants on the 15 fluorinated compounds. Include paired bootstrap 95% CI error bars on the MAE difference (raw vs log).

5. **`cv_paired_delta.png`** — Box plot (or forest plot) showing the per-fold delta in R² between log and raw variants for **all three targets** (not just epsilon/k). A box clearly to one side of zero indicates a consistent effect; a box spanning zero indicates no meaningful difference. Annotate with paired t-test p-values.

### 52.8 — Write report

Write `docs/reports/52_log_target_transform.md` following the standard template:

1. **Title**: `# Results Report: Log-Target Transform (RF and XGBoost)`
2. **Motivation**: One paragraph connecting to the GNNePCSAFT relative-loss insight.
3. **Assumption validation**: Heteroscedasticity test results (Spearman r, p-value) and target distribution skewness. State whether the log-transform has a theoretical basis for each target.
4. **Back-transform bias quantification**: Smearing factor magnitude and naive-vs-corrected prediction difference for each target.
5. **Esper holdout results**: Table comparing RF-raw, RF-log (naive), RF-log (corrected), XGB-raw, XGB-log (naive), XGB-log (corrected) on all three targets. Report R², MAE, and sMAPE. Highlight epsilon/k.
6. **Cross-validated results**: Mean ± std for **all three targets** across 15 folds (5×3 repeats). Paired t-test p-values and 95% CIs on deltas. State whether the log-transform consistently helps, hurts, or makes no difference.
7. **Fluorinated external validation**: Table comparing all variants on the 15-compound set with paired Wilcoxon p-values and bootstrap CIs. This is the decisive result.
8. **Boiling point propagation** (if applicable): BP MAE comparison.
9. **Key findings**: Bullet points.
10. **Recommendation**: Whether to adopt the log-transform for production or not, with explicit evidence. If the recommendation is "reject," note whether hyperparameter re-tuning is a warranted follow-up.
11. **Figures**: Reference all figure paths.
12. **Readiness check**: Checklist.

### 52.9 — Update supplementary materials

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

- [ ] Heteroscedasticity assumption verified (residual-vs-fitted + Spearman test for each target)
- [ ] Target distribution skewness reported for raw and log-transformed
- [ ] RF and XGBoost trained on both raw and log-transformed targets
- [ ] Back-transform bias quantified (smearing factor reported; naive vs corrected predictions compared)
- [ ] Esper holdout comparison on all three targets with R², MAE, and sMAPE
- [ ] Repeated 5-fold CV (5×3) paired comparison on **all three targets** with paired tests
- [ ] Fluorinated external validation for all variants with paired Wilcoxon and bootstrap CIs
- [ ] Boiling point propagation if fluorinated epsilon/k improves (paired CI excludes zero)
- [ ] At least 5 figures saved to `figures/52_log_target_transform/`
- [ ] Report written at `docs/reports/52_log_target_transform.md`
- [ ] Supplementary materials updated with result
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- The log-target transform has been evaluated on both internal (Esper) and external (fluorinated) validation
- A clear recommendation is recorded: adopt, reject, or note as marginal
- If the transform helps: the production pipeline is updated (or flagged for update) with explicit evidence
- If the transform doesn't help: this is documented as evidence against the specific hypothesis that MSE-on-raw-targets (vs. relative-error loss) explains a meaningful portion of the RF epsilon/k accuracy gap. It does **not** rule out all loss-function explanations — other formulations (Huber on relative error, quantile regression, asymmetric losses) remain untested — but it narrows the hypothesis space

## Budget

45–60 minutes. The experiment involves heteroscedasticity diagnostics (1 baseline RF fit), 8+ model fits on the Esper training set (4 variants × train + repeated CV with bias correction), 4+ fluorinated evaluations with paired statistics, and diagnostic plotting. No hyperparameter search is needed — use production defaults for both RF and XGBoost. The XGBoost hyperparameters should match those from the saved production model (`model/saved/xgb/xgb_model.joblib`). The added statistical rigor (smearing correction, paired tests, repeated CV) is worth the extra 15 minutes to pre-empt the most common reviewer objections.
