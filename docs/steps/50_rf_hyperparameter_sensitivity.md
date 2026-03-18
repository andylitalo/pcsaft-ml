# Step 50: Hyperparameter Sensitivity for Production RF

## Objective

Validate that the production Random Forest's hyperparameters are reasonable for the deployed screening workflow, quantify **local sensitivity around the production configuration**, and determine whether any alternative configuration delivers a materially better `epsilon_k` model under a production-aligned evaluation protocol.

## Motivation

The production RF uses `n_estimators=100, max_features="sqrt", max_depth=None, min_samples_leaf=1` — sklearn defaults with the exception of `n_estimators`. These were never systematically validated against alternatives on this dataset. Step 49 established that 100 trees is sufficient via OOB diagnostics; this step asks whether the other hyperparameters matter and whether a meaningfully better configuration exists.

This matters because:

1. **Credibility**: Reporting that you used sklearn defaults without checking is a fair criticism. Showing that performance is only weakly sensitive to nearby alternatives is a strong defense.
2. **The RF=XGBoost tie**: RF (0.33) and XGBoost (0.33) achieved identical epsilon/k R² on different hyperparameter strategies (defaults vs grid search). If RF's defaults happen to be near-optimal, this reinforces the "representation bottleneck" interpretation. If a better RF configuration exists, the interpretation changes.
3. **Feature importance stability**: If the top features change dramatically across CV folds, the model may be fitting unstable signals. Stable rankings increase confidence that the importance story is repeatable, though not necessarily mechanistic.

This step focuses exclusively on RF because it is the deployed production model. The other models' hyperparameters are less consequential since they are not used for screening.

## Dependencies

- Requires: Step 49 (RF OOB convergence from 49.4 provides the n_estimators validation; this step extends to other hyperparameters)
- Requires: trained RF model and Esper dataset from Step 01
- No new packages needed (sklearn, numpy, matplotlib)

**Generalization caveat**: The CV and held-out analyses in this step are designed to validate the production RF within the project's current random-split Esper protocol. They support internal robustness and deployment tuning discipline, but they should not be described as proof of scaffold-level chemistry generalization unless a chemistry-aware split is added separately.

## Implementation Guide

### 50.1 — Production-aligned RF hyperparameter search on Esper training data

Evaluate RF over a grid of the four most impactful hyperparameters using 5-fold cross-validation on the full Esper training set (1,440 molecules).

Critical requirement: **match the production estimator and target structure**. The deployed RF path in `model/train.py` trains one separate `RandomForestRegressor` per target. The primary search in this step should therefore be a **target-specific search for `epsilon_k`**, not a `MultiOutputRegressor` scored by average multi-output `R²`.

**Hyperparameter grid:**

| Hyperparameter | Values | Rationale |
|----------------|--------|-----------|
| `n_estimators` | [50, 100, 200, 500] | Ensemble size; Step 49 OOB analysis provides the convergence context |
| `max_features` | ["sqrt", "log2", 0.3, 0.5, None] | Features considered per split; controls bias-variance tradeoff in high-dim space |
| `max_depth` | [None, 10, 20, 30] | Tree depth; None = fully grown (low bias, high variance per tree) |
| `min_samples_leaf` | [1, 3, 5] | Leaf size; larger = more regularization |

Total: 4 x 5 x 4 x 3 = 240 configurations. For a target-specific `epsilon_k` search with 5-fold CV, that is 240 x 5 = 1,200 RF fits. At ~3-5 seconds per fit on the Esper dataset (~2,200 features, up to 500 trees), this is still substantial. To reduce runtime, consider dropping `n_estimators=500` after Step 49 if the OOB curve already shows a clear plateau by 300, reducing the grid to 180 configs.

**Implementation** in `scripts/step50_rf_hyperparameter_sensitivity.py`:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from model.data.load import TARGETS, load_data, split_data
from model.data.descriptors import build_features_with_names

# Load Esper train/test split used by production RF
df = load_data("esper")
train_df, test_df = split_data(df)
X_train, feature_names = build_features_with_names(train_df["smiles"].tolist())
y_train = train_df[TARGETS].values
y_ek = y_train[:, TARGETS.index("epsilon_k")]

param_grid = {
    "n_estimators": [50, 100, 200, 500],
    "max_features": ["sqrt", "log2", 0.3, 0.5, None],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 3, 5],
}

cv = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=1,
    refit=True,
    verbose=2,
    return_train_score=True,
)
cv.fit(X_train, y_ek)
```

Save the full CV results:

```python
import json
import numpy as np

cv_results = {}
for key, val in cv.cv_results_.items():
    if key == "params":
        cv_results[key] = [
            {k: (str(v) if v is None else v) for k, v in p.items()}
            for p in val
        ]
    elif hasattr(val, "tolist"):
        cv_results[key] = val.tolist()
    else:
        cv_results[key] = val

cv_results["best_params"] = {
    k: (str(v) if v is None else v) for k, v in cv.best_params_.items()
}
cv_results["best_score"] = float(cv.best_score_)

results_path = Path("model/saved/rf_hyperparam_search.json")
results_path.write_text(json.dumps(cv_results, indent=2) + "\n")
```

Save the full CV results, but also record the baseline production configuration's rank within the grid and the score delta from the best configuration.

Optional secondary analysis: run a separate descriptive multi-target search only if you want to compare "best for epsilon/k" vs "best averaged across targets." If included, present it as a secondary artifact rather than the decision-driving result.

If the best `epsilon_k`-specific configuration differs from the descriptive multi-target best, report both and use the `epsilon_k`-specific result for the production-vs-best comparison. This matches the production architecture, which trains separate RFs per target in `model/train.py`.

### 50.2 — One-at-a-time local sensitivity plots

For each hyperparameter, sweep it while holding the other three at production defaults (`n_estimators=100, max_features="sqrt", max_depth=None, min_samples_leaf=1`). For each configuration, compute 5-fold CV R² per target (m, sigma, epsilon_k separately).

```python
from sklearn.model_selection import cross_val_score, KFold

production_config = {
    "n_estimators": 100,
    "max_features": "sqrt",
    "max_depth": None,
    "min_samples_leaf": 1,
}

sweeps = {
    "n_estimators": [25, 50, 75, 100, 150, 200, 300, 500],
    "max_features": ["sqrt", "log2", 0.1, 0.2, 0.3, 0.5, 0.7, None],
    "max_depth": [5, 10, 15, 20, 25, 30, 40, None],
    "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15, 20],
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for param_name, param_values in sweeps.items():
    for target_idx, target_name in enumerate(TARGETS):
        y_col = y_train[:, target_idx]
        for val in param_values:
            config = production_config.copy()
            config[param_name] = val
            rf = RandomForestRegressor(random_state=42, n_jobs=-1, **config)
            scores = cross_val_score(rf, X_train, y_col, cv=kf, scoring="r2")
            # Record mean and std of scores
```

This produces one curve per target per hyperparameter (4 hyperparameters x 3 targets = 12 curves). The primary figure focuses on `epsilon_k`.

Save sweep results to `model/saved/rf_oat_sensitivity.json`.

State the claim narrowly: these plots show **local sensitivity near the production point**, not a proof that the full 4D search space is globally flat.

### 50.3 — Compare production vs best-found configuration with paired uncertainty

Extract the best configuration from the `epsilon_k` grid search (50.1). Compare it to the production configuration on the held-out Esper test set using a **paired bootstrap over the per-sample prediction delta**, not separate CI overlap.

```python
from sklearn.metrics import r2_score

# Train both configs on full training set
rf_production = RandomForestRegressor(
    n_estimators=100, max_features="sqrt", max_depth=None,
    min_samples_leaf=1, random_state=42, n_jobs=-1,
)
rf_production.fit(X_train, y_train)
pred_production = rf_production.predict(X_test)

rf_best = RandomForestRegressor(random_state=42, n_jobs=-1, **best_found_params)
rf_best.fit(X_train, y_train)
pred_best = rf_best.predict(X_test)

# Paired bootstrap on the metric delta
rng = np.random.RandomState(42)
delta_samples = []
for _ in range(2000):
    idx = rng.randint(0, len(y_test), size=len(y_test))
    prod_score = r2_score(y_test[idx, ek_idx], pred_production[idx, ek_idx])
    best_score = r2_score(y_test[idx, ek_idx], pred_best[idx, ek_idx])
    delta_samples.append(best_score - prod_score)

delta_lo, delta_hi = np.percentile(delta_samples, [2.5, 97.5])
```

**Decision rule:**
- If the paired 95% CI for the `epsilon_k` delta spans 0 and the effect size is small, treat the production configuration as not meaningfully worse on Esper.
- If the paired 95% CI is entirely above 0 and the gain is practically relevant, report the improvement magnitude. Do **not** auto-swap production without checking external fluorinated behavior (50.4).

### 50.4 — External validation of top RF candidates

Because RF is deployed for fluorinated screening, Esper CV alone is not enough to justify changing or affirming production settings. Evaluate the production configuration plus the top 3 `epsilon_k` configurations from 50.1 on the fluorinated validation workflow from Step 48:

- Predict PC-SAFT parameters on the fluorinated validation set
- Compute `epsilon_k` MAE/RMSE
- Propagate predictions through `compute_boiling_point()`
- Compare boiling-point MAE and EOS convergence rate

Decision logic:

- If Esper-improving configs do **not** match or improve fluorinated boiling-point behavior, keep production unchanged and report that the internal CV gain does not transfer to the deployment domain.
- If one config improves both Esper `epsilon_k` and fluorinated boiling-point behavior, flag it as a credible production replacement candidate.

Save this comparison to `model/saved/rf_external_config_comparison.json`.

### 50.5 — Feature importance stability

Run both Gini (impurity-based) importance and permutation importance across the 5 CV folds to assess stability.

```python
from sklearn.inspection import permutation_importance

kf = KFold(n_splits=5, shuffle=True, random_state=42)
gini_importances = []
perm_importances = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # Train separate RF for each target (to get per-target importances)
    for target_idx, target_name in enumerate(TARGETS):
        rf.fit(X_fold_train, y_fold_train[:, target_idx])
        gini_importances.append({
            "fold": fold_idx, "target": target_name,
            "importances": rf.feature_importances_.tolist(),
        })

        perm_result = permutation_importance(
            rf, X_fold_val, y_fold_val[:, target_idx],
            n_repeats=10, random_state=42, n_jobs=-1,
        )
        perm_importances.append({
            "fold": fold_idx, "target": target_name,
            "importances_mean": perm_result.importances_mean.tolist(),
            "importances_std": perm_result.importances_std.tolist(),
        })
```

For each target, identify the top 20 features by mean Gini importance across folds. Report:
- How many of the top 20 are consistent across all 5 folds (i.e., appear in every fold's top 20)
- The coefficient of variation (CV = std/mean) of importance scores for the top features
- Whether Morgan fingerprint bits or RDKit descriptors dominate the top features

**Limitation — correlated feature dilution**: With ~2,200 features, many are correlated (related Morgan substructures encoding similar neighborhoods, correlated RDKit descriptors like molecular weight and heavy atom count). Both Gini and permutation importance distribute importance across correlated groups — if two features carry identical signal, each gets ~50% of the importance that one alone would receive. This means:
- The "top 20 consistent across folds" metric can appear unstable even when the underlying signal is stable, because importance randomly shifts among correlated features across folds.
- If Gini importance favors Morgan FP bits while permutation importance favors RDKit descriptors (or vice versa), this is likely a measurement artifact of the different biases (Gini is biased toward high-cardinality/continuous features; permutation importance is unbiased but noisier), not a genuine difference in feature-type relevance.

Acknowledge this limitation in the report. Treat stable rankings as evidence of **repeatability**, not direct physical interpretability. Optionally, cluster highly correlated features (e.g., Spearman |ρ| > 0.9) and compute group-level importance as a robustness check.

Save to `model/saved/rf_feature_importance_stability.json`.

### 50.6 — RF data efficiency (learning curve vs training set size)

The project's central narrative claims "1,800 Esper molecules is enough for RF but not for GNN" and estimates a GNN crossover at 5,000-10,000 molecules. An MLE will ask for the RF learning curve to anchor this claim. This is cheap to compute (~10 min) and directly supports the "data quality > quantity" argument in the presentation.

Train the production RF configuration on progressively larger random subsets of the Esper training set and evaluate on the full held-out test set:

```python
from sklearn.model_selection import learning_curve

train_sizes = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]

rf = RandomForestRegressor(
    n_estimators=100, max_features="sqrt", max_depth=None,
    min_samples_leaf=1, random_state=42, n_jobs=-1,
)

train_sizes_abs, train_scores, test_scores = learning_curve(
    rf, X_train, y_ek,
    train_sizes=train_sizes,
    cv=5,
    scoring="r2",
    n_jobs=1,
    random_state=42,
)
```

Report:
- R² vs number of training molecules, with ±1 std band across CV folds
- Whether the curve has plateaued by 1,440 training molecules (the full Esper training set)
- If the curve is still rising, estimate the slope and extrapolate roughly how much more data would be needed for a meaningful gain
- **Crucially, contextualize the plateau against the experimental noise floor (aleatoric uncertainty).** PC-SAFT parameters are fitted to experimental vapor pressure and liquid density data. Because different literature sources fit parameters using different temperature ranges or objective functions, there is inherent noise in the "ground truth" labels. Estimate this noise floor (e.g., by comparing duplicate molecules in the raw literature, or citing typical inter-annotator agreement for PC-SAFT fitting) and plot it as a horizontal asymptote on the learning curve. This answers the MLE question: *"Is the model underfitting, or is $R^2 \approx 0.33$ simply the theoretical maximum given the label noise?"*

This figure answers a specific question the presentation raises but does not currently back with data: **is the production RF data-saturated, or would more experimental data improve it?** If the curve is flat by ~1,000 molecules, the representation bottleneck claim is supported. If it is still rising, more experimental data (not more model complexity) is the next lever.

Save results to `model/saved/rf_learning_curve.json`.

### 50.7 — Systematic error analysis on the Esper test set

The project has excellent domain-specific error analysis (chlorobutene deep-dive, fluorinated validation). But there is no systematic analysis of **where the production RF fails on the general Esper test set**. An MLE will want to see whether errors are random or correlated with molecular properties.

Using the production RF predictions on the Esper test set (361 molecules), compute:

1. **Residuals vs predicted value**: scatter plot of (predicted - actual) vs predicted for epsilon/k. Check for heteroscedasticity (errors that grow with predicted value).
2. **Error vs molecular weight**: scatter plot of |error| vs `MolWt` to check if large or small molecules are systematically worse.
3. **Error by functional group presence**: group test molecules by the presence of key substructures (fluorine atoms, ring systems, halogens, double bonds) and compare median |error| across groups. This identifies chemistry domains where the model is weakest.
4. **Worst-predicted molecules**: list the 10 molecules with the highest absolute epsilon/k error, along with their SMILES, predicted vs actual values, and any shared structural features. This provides concrete examples for Q&A.

Present this as a table and 2-3 diagnostic plots. The goal is not to fix the model but to demonstrate awareness of its failure modes — which is exactly what an MLE expects from a production deployment.

Save the per-molecule error table to `model/saved/rf_esper_error_analysis.csv`.

### 50.8 — Generate figures

Save all figures to `figures/50_rf_hyperparameter_sensitivity/`.

**Required:**

1. **`oat_sensitivity_epsilon_k.png`** — 2x2 grid of one-at-a-time sensitivity plots for `epsilon_k` R². Each panel sweeps one hyperparameter (x-axis) against CV R² (y-axis) with error bars (±1 std across folds). A vertical dashed line marks the production value. Use the caption language "local sensitivity" rather than "global robustness."

2. **`cv_heatmap_n_estimators_vs_max_depth.png`** — 2D heatmap of mean CV R² (epsilon/k target) with `n_estimators` on one axis and `max_depth` on the other, holding `max_features="sqrt"` and `min_samples_leaf=1`. Star marks the production configuration. This visualizes the interaction between the two most structurally meaningful hyperparameters.

3. **`feature_importance_stability.png`** — Box plot of importance scores for the top 20 features (epsilon/k target), with each box spanning the 5 CV folds. Features sorted by median importance. Color-code Morgan FP bits vs RDKit descriptors.

4. **`rf_learning_curve.png`** — R² vs number of training molecules for epsilon/k, with ±1 std band across CV folds. Vertical dashed line at the full training set size. Horizontal dashed line indicating the estimated experimental noise floor. This directly supports the "data quality > quantity" narrative.

5. **`error_analysis_residuals.png`** — 2-panel figure: (a) residuals vs predicted epsilon/k, (b) |error| vs molecular weight. Highlights systematic failure patterns.

**Optional:**

6. **`oat_sensitivity_all_targets.png`** — Same as (1) but showing m and sigma alongside epsilon/k (3 lines per panel, 4 panels).

7. **`production_vs_best_comparison.png`** — Bar chart comparing R² with paired-bootstrap 95% CIs for production vs best-found configuration on each target.

8. **`fluorinated_config_comparison.png`** — Bar chart or table plot comparing the production config and top Esper configs on fluorinated `epsilon_k` MAE and boiling-point MAE.

9. **`error_by_functional_group.png`** — Grouped bar chart of median |epsilon/k error| by functional group presence (fluorine, ring, halogen, etc.).

### 50.9 — Create the driver script

Create `scripts/step50_rf_hyperparameter_sensitivity.py` with sections:

```python
"""Step 50: RF Hyperparameter Sensitivity Analysis.

Usage:
    python scripts/step50_rf_hyperparameter_sensitivity.py [--section SECTION]

Sections:
    grid        Full grid search (50.1)
    oat         One-at-a-time sensitivity (50.2)
    compare     Production vs best config (50.3)
    external    Fluorinated validation of top configs (50.4)
    importance  Feature importance stability (50.5)
    learning    RF data efficiency learning curve (50.6)
    errors      Systematic error analysis (50.7)
    figures     Generate all figures (50.8)
    report      Write report (50.10)
    all         Run everything (default)
"""
```

### 50.10 — Write report

Write `docs/reports/50_rf_hyperparameter_sensitivity.md` following the standard template:

1. **Title**: `# Results Report: RF Hyperparameter Sensitivity (Esper, 1,801 molecules)`
2. **Summary**: One paragraph stating whether production hyperparameters are supported by the combined internal and external evidence.
3. **Grid search results**: Best `epsilon_k` configuration found, its CV R² vs production CV R², and the production config's rank. Table of top-5 configurations.
4. **Local sensitivity analysis**: For each hyperparameter, state whether performance is locally sensitive or insensitive near the production setting. Reference the OAT figure.
5. **Production vs best comparison**: Paired-bootstrap CI for the held-out Esper delta. State whether the difference is statistically and practically meaningful.
6. **External validation of candidate configs**: Fluorinated `epsilon_k` and boiling-point comparison for production plus top candidates. This section determines whether any Esper improvement is deployment-relevant.
7. **Feature importance stability**: How many of the top 20 features are consistent across folds. Which feature types dominate, with caveats about correlated features and importance bias.
8. **Data efficiency and Noise Floor**: RF learning curve — is the model data-saturated at 1,440 training molecules, or would more experimental data improve epsilon/k? Connect to the "data quality > quantity" narrative. Explicitly state the estimated experimental noise floor and whether the model has reached it.
9. **Error analysis**: Where does the production RF fail on the Esper test set? Systematic patterns by molecular properties, worst-predicted molecules, and implications for applicability domain.
10. **Key findings**: Bullet points.
11. **Figures**: Reference all figure paths.
12. **Deviations**: If grid search was reduced for time, document it.
13. **Readiness check**: Checklist.

### 50.11 — Update supplementary materials

After the report is written, update the convergence section added to `supplementary_information/model_comparison.md` in Step 49 to include a subsection on RF hyperparameter validation:

> **RF hyperparameter sensitivity**: The production configuration (`n_estimators=100, max_features="sqrt", max_depth=None, min_samples_leaf=1`) was evaluated with a production-aligned `epsilon_k` RF search on Esper, paired comparison on the held-out Esper test set, and fluorinated external validation of the top candidate configurations. Performance near the production point is [locally insensitive / locally sensitive] to [hyperparameters], and the production configuration [matches / trails] the best Esper candidate by [X] R² units on Esper and [Y] K on fluorinated boiling-point MAE. See `figures/50_rf_hyperparameter_sensitivity/oat_sensitivity_epsilon_k.png`.

## Key Files

| File | Purpose |
|------|---------|
| `model/data/load.py` | `load_data("esper")`, `split_data()`, `TARGETS` |
| `model/data/descriptors.py` | `build_features()` for computing RDKit + Morgan feature matrix |
| `model/train.py` | Reference: current RF training with default hyperparameters |
| `model/saved/rf_oob_convergence.json` | From Step 49: OOB convergence (provides n_estimators context) |
| `model/uncertainty.py` | Helper patterns for uncertainty reporting; paired comparison logic should be implemented in the step script |
| `scripts/step50_rf_hyperparameter_sensitivity.py` | Create: driver script |

## Artifacts

| File | Description |
|------|-------------|
| `scripts/step50_rf_hyperparameter_sensitivity.py` | Driver script |
| `docs/reports/50_rf_hyperparameter_sensitivity.md` | Results report |
| `figures/50_rf_hyperparameter_sensitivity/` | All figures |
| `model/saved/rf_hyperparam_search.json` | Full grid-search CV results (240 configs) |
| `model/saved/rf_oat_sensitivity.json` | One-at-a-time sweep results |
| `model/saved/rf_external_config_comparison.json` | Fluorinated validation for production and top candidate configs |
| `model/saved/rf_feature_importance_stability.json` | Per-fold feature importances |
| `model/saved/rf_learning_curve.json` | R² vs training set size for data efficiency analysis |
| `model/saved/rf_esper_error_analysis.csv` | Per-molecule error table with molecular properties |

## Success Criteria

- [ ] Primary grid search completed for `epsilon_k` using the same per-target RF structure as production
- [ ] One-at-a-time sensitivity plots show local sensitivity near the production configuration
- [ ] Production vs best-found comparison uses paired uncertainty estimates rather than CI-overlap heuristics
- [ ] Top Esper configurations are checked on the fluorinated external validation workflow before any production-change recommendation
- [ ] Feature importance stability assessed across 5 folds for all 3 targets
- [ ] RF learning curve (R² vs training set size) generated for epsilon/k
- [ ] Systematic error analysis on Esper test set with residual plots, error-by-property breakdown, and worst-predicted molecules
- [ ] At least 5 figures saved to `figures/50_rf_hyperparameter_sensitivity/`
- [ ] Report written at `docs/reports/50_rf_hyperparameter_sensitivity.md`
- [ ] `supplementary_information/model_comparison.md` updated with RF sensitivity results
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- The production RF hyperparameters are either supported as reasonable for the screening workflow or a better candidate has been identified and flagged with both internal and external evidence
- Any claim of robustness is limited to local sensitivity near the production point unless broader interaction analysis is performed
- Feature importance stability is framed as repeatability evidence, not proof of physical interpretability
- The sensitivity analysis is documented in supplementary materials and available for presentation Q&A
- If a significantly better configuration was found: decision to swap or keep is recorded with rationale

## Budget

3–4 hours. The original 60-minute estimate assumed ~1 second per RF fit, but RF with `n_estimators=500` on 1,440 samples × ~2,200 features takes 5–10 seconds per fit. The primary `epsilon_k` grid search still takes the bulk of runtime. OAT sweeps (50.2) add ~15 minutes. Fluorinated candidate validation (50.4) and feature importance with permutation (50.5) add additional runtime. Figures and report (50.6-50.8) take ~15 minutes.

To reduce runtime if needed: drop `n_estimators=500` from the primary grid once Step 49 has shown a clear OOB plateau, cutting the grid to 180 configs and saving ~25% wall-clock time. Skip the optional multi-target search before skipping the production-aligned `epsilon_k` search.
