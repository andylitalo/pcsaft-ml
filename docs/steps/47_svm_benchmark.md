# Step 47: SVM and Linear Baselines

## Purpose

The RF = XGBoost tie (R² ε/k = 0.33 vs 0.33) on the same RDKit + Morgan features suggests the learning algorithm may no longer be the dominant bottleneck. To probe that hypothesis, add two distinct algorithm families: **support vector regression** (SVR with RBF kernel) and a **linear baseline** (Ridge Regression).

This step should be framed as an **exploratory algorithm-diversity benchmark**, not a definitive proof. If a reasonably tuned SVR lands near RF/XGBoost on ε/k across repeated evaluation, that strengthens the representation-bottleneck hypothesis. 

Crucially, the Ridge Regression baseline answers a fundamental MLE question: *"Does the complexity of tree ensembles and kernel methods actually provide value over a simple linear model?"* By quantifying the exact delta between Ridge and RF, we prove the value of non-linear representations.

To count as evidence, the benchmark must report **distributional results across repeated training/evaluation runs**, not a single best run or a single split point estimate.

SVR is a natural choice because:

- RBF kernels implicitly map features to infinite-dimensional space, a fundamentally different inductive bias from tree-based partitioning
- SVR is well-established for QSAR regression (Svetnik et al. 2003 benchmarked SVMs alongside RF)
- scikit-learn's `SVR` is already available — no new dependencies needed

Ridge Regression is a natural choice because:
- It provides a regularized linear baseline that handles high-dimensional correlated features better than OLS
- It establishes the "dumb" baseline performance floor
- It explicitly tests whether the structure-property relationships are non-linear

Important caveat: **RBF SVR on scaled RDKit + Morgan features is only one kernel baseline**, not a full test of all SVM/kernel formulations used in chemoinformatics. Interpret negative results accordingly.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| 2 algorithm families on same features (bagging, boosting) | 4 families (+ kernel methods, + linear models) |
| "RF ≈ XGBoost" suggests bottleneck | "RF ≈ XGBoost ≈ SVR" strengthens the bottleneck argument if the comparison is statistically stable |
| SVM and Linear listed as "excluded" in model_comparison.md | SVM and Ridge tested with full metrics |
| No measure of non-linear value | Exact delta between Ridge and RF quantifies the value of non-linearity |

## Skills Demonstrated

- **Kernel methods**: SVR with RBF kernel, hyperparameter tuning (C, gamma, epsilon)
- **Linear baselines**: Ridge regression as a sanity check
- **Feature scaling**: StandardScaler pipeline (required for SVR and Ridge, unlike tree methods)
- **Multi-output regression**: `MultiOutputRegressor` wrapper for per-target SVR
- **Experimental design**: using algorithm diversity to diagnose representation limits

## Dependencies

- Requires: Esper dataset (same as RF/XGBoost baselines)
- New packages: **none** — `sklearn.svm.SVR` ships with scikit-learn

## Key Files

| File | Purpose |
|------|---------|
| `model/svm/svm_model.py` | SVR model class with `@register_model("svm")` |
| `model/linear/ridge_model.py` | Ridge regression model class with `@register_model("ridge")` |
| `scripts/train_step47.py` | Training and evaluation driver script |
| `model/data/descriptors.py` | `build_features_with_names()` — same feature pipeline as RF/XGBoost |
| `model/data/load.py` | `load_data()`, `split_data()`, `TARGETS` |
| `model/saved/comparison_metrics.csv` | Baseline metrics for RF, XGBoost, and other models |
| `model/evaluate.py` | Evaluation harness, `bootstrap_metric_ci` |

## Artifacts

| File | Description |
|------|-------------|
| `model/saved/step47_summary.json` | Per-split metrics, aggregate metrics, seeds, best hyperparameters |
| `model/saved/svm/svm_model.joblib` | Saved SVR model (from final or representative split) |
| `model/saved/linear/ridge_model.joblib` | Saved Ridge model |
| `model/saved/svm/cv_results.json` | Full GridSearchCV results per target (needed by Step 49 for SVM CV heatmap) |
| `figures/47_svm_benchmark/` | R² heatmap, bottleneck comparison chart |
| `docs/reports/47_svm_benchmark.md` | Results report |

## Implementation Guide

### 47.1 SVM and Ridge Model Classes

Create `model/svm/__init__.py` and `model/svm/svm_model.py`:

- `@register_model("svm")` decorator
- `Pipeline([StandardScaler(), SVR(kernel='rbf', max_iter=10000)])` — feature scaling is critical since SVR is not scale-invariant
- Prefer **one SVR per target** (separate `GridSearchCV` per target) so the hardest target (ε/k) is not under-tuned by averaging across easier outputs. If a shared multi-output hyperparameter set is retained, explicitly label it as a coarse screen and avoid strong claims from ε/k alone.
- `fit()`, `predict()`, `save()`, `load()` methods matching the XGBoost interface
- No `predict_with_uncertainty()` — SVM has no built-in uncertainty mechanism
- No `feature_importances()` — kernel methods are opaque

Create `model/linear/__init__.py` and `model/linear/ridge_model.py`:

- `@register_model("ridge")` decorator
- `Pipeline([StandardScaler(), Ridge()])`
- `GridSearchCV` over `alpha`: `[0.1, 1.0, 10.0, 100.0, 1000.0]`
- Same `fit()`, `predict()`, `save()`, `load()` methods

#### Hyperparameter grid (24 combinations)

| Parameter | Values | Notes |
|-----------|--------|-------|
| `C` | [1, 10, 100] | Regularization strength |
| `gamma` | ['scale', 'auto', 0.01, 0.001] | **All four values required.** `'auto'` = `1/n_features` ≈ 4.5e-4 for 2,200 features. `'scale'` = `1/(n_features * X.var())` can be extremely small on this feature space; the two explicit values (0.01, 0.001) may both overshoot. Omitting `'auto'` leaves a gap in the middle of the search range. |
| `epsilon` | [0.1, 0.01] | SVR tube width |

#### Hard requirements (these are easy to miss — do not skip)

1. **`max_iter=10000`**: Set explicitly on the `SVR()` constructor. The sklearn default (`-1`, unlimited) lets libsvm iterate indefinitely, and with 10 outer splits × 24 grid combos × 3–5 inner folds = 720–1,200 SVR fits, a single unconverged fit can stall the entire run for hours.

2. **Convergence warning capture**: Wrap the `GridSearchCV.fit()` call with `warnings.catch_warnings(record=True)` to intercept `sklearn.exceptions.ConvergenceWarning`. Count and log the number of unconverged fits. Report this count in the summary artifact and the step report. Do not silently swallow convergence warnings — an MLE reviewer will ask whether the solver converged.

3. **Grid boundary check**: After `GridSearchCV` completes, check whether the best hyperparameter for any target sits at the edge of the grid (e.g., best C = 100, best gamma = 0.001). If so, extend the grid in that direction (e.g., add C = 1000) and re-run for that target. Report whether the boundary was hit, whether extending it changed the result, and record the final grid in the summary artifact. This is required because a boundary-optimal result means the search space was too narrow.

4. **Save full `cv_results_`**: After each `GridSearchCV.fit()`, persist the full `cv.cv_results_` dict to `model/saved/svm/cv_results.json`. This is needed by Step 49 (convergence diagnostics) to produce the SVM CV heatmap. Without this artifact, Step 49 must re-run Step 47.

### 47.2 Training Script (`scripts/train_step47.py`)

Use a robust evaluation protocol rather than a single benchmark run.

Required protocol:

1. Load data via `load_data(source="esper")`
2. Build the same RDKit + Morgan feature pipeline used by RF/XGBoost
3. Run **repeated stratified outer splits** on Esper:
   - recommended default: 10 outer splits with distinct fixed seeds
   - stratify on binned `epsilon_k`, consistent with the existing split logic
4. Within each outer split, train SVR and Ridge with inner CV hyperparameter selection:
   - minimum: 3-fold inner CV
   - preferred if runtime is acceptable: 5-fold inner CV
5. Evaluate each outer-split model on its held-out fold/split and record MAE, RMSE, and R² per target
6. Summarize **distributions**, not winners:
   - mean and standard deviation across outer splits
   - 95% interval across outer splits, or bootstrap CIs on the aggregated outer-split results
   - number of outer splits completed
7. Load RF/XGBoost baseline metrics generated under the same repeated-evaluation protocol, or clearly label any mismatch if only single-split baselines exist
8. Generate updated comparison outputs and save a summary artifact containing per-split metrics, aggregate metrics, seeds, and best hyperparameters per run
9. Add at least a smoke test for the new SVM and Ridge wrappers and script entry point

Acceptable stronger alternative:

- **Nested CV** in place of repeated outer splits, provided the outer folds are used for final comparison and the report summarizes fold-level distributions rather than the best fold

Do **not** treat a single split, a single best run, or the highest observed ε/k R² as sufficient to claim the algorithm family no longer matters.

### 47.3 Evaluate and Compare

The key comparison is whether SVM achieves roughly the same ε/k performance as RF and XGBoost **within realistic sampling noise across repeated runs**, and how much better all three are compared to the Ridge baseline. Focus on ε/k first, since it is the hardest target and the basis of the bottleneck narrative.

Interpret outcomes conservatively:

- **Ridge significantly worse than RF/XGBoost/SVR**: quantifies the value of non-linear representations.
- **SVR ≈ RF ≈ XGBoost on ε/k, with overlapping intervals and similar repeated-split means**: strengthens the representation-bottleneck hypothesis.
- **SVR significantly better on ε/k**: suggests the RBF kernel captures structure that tree ensembles are not extracting from these features.
- **SVR significantly worse on ε/k**: suggests this SVR formulation struggles with the high-dimensional sparse feature space (2,048 binary Morgan bits), which is informative but does **not** by itself rule out all kernel methods.

Comparison rules:

- compare model families using aggregate statistics across all outer runs, not the best observed run
- report the spread of ε/k outcomes across splits/folds so split sensitivity is visible
- if RF/XGBoost comparison numbers come from older single-split artifacts, regenerate them under the same repeated protocol before making a strong three-way claim

Nice-to-have diagnostics:

- report train vs test scores to check overfitting
- report best hyperparameters and runtime
- note whether performance is sensitive to scaling choices or `gamma`
- if using shared multi-output tuning, compare against an ε/k-only tuned run to ensure the headline result is not an artifact of averaging across targets

### 47.4 Update `supplementary_information/model_comparison.md`

- Move SVM and Ridge from "What was excluded" to tested models with their own narrative section
- Add SVM and Ridge rows to the Esper test set comparison table
- Update the representation-bottleneck argument to cite three non-linear algorithm families and one linear baseline

### 47.5 Write the Report

Write `docs/reports/47_svm_benchmark.md` following the standard report template:

1. **Title**: `# Results Report: SVM and Linear Baselines (Esper Dataset, 1,801 molecules)`
2. **Evaluation protocol**: Describe the repeated-split or nested-CV design — number of outer splits, inner CV folds, stratification, seeds used. This section is essential because the protocol differs from earlier single-split reports.
3. **Per-target aggregate metrics table**: Mean ± std across outer splits for MAE, RMSE, R² on each of m, σ, ε/k. Include the 95% interval.
4. **Four-way comparison table**: Ridge vs RF vs XGBoost vs SVR on ε/k, with interval estimates. If RF/XGBoost baselines are from single-split artifacts, label the mismatch explicitly.
5. **Best hyperparameters**: Report which C, gamma, epsilon (for SVR) and alpha (for Ridge) were selected most frequently across splits. Flag any grid boundary hits and whether extending the grid changed the result.
6. **Convergence**: Number of unconverged SVR fits (if any) and whether they affected results.
7. **Interpretation**: State whether the result strengthens, weakens, or is ambiguous for the representation-bottleneck hypothesis. Quantify the value of non-linearity by comparing Ridge to RF. Use appropriately cautious language.
8. **Figures**: Reference paths in `figures/47_svm_benchmark/`.
9. **Deviations**: Anything that diverged from this guide and why. Omit if none.
10. **Readiness check**: Confirm the acceptance criteria below are met, as a checklist.

## Acceptance Criteria (When to Move On)

- [ ] `model/svm/svm_model.py` created with `@register_model("svm")` decorator
- [ ] `model/linear/ridge_model.py` created with `@register_model("ridge")` decorator
- [ ] SVR constructor uses `max_iter=10000` (not the default -1)
- [ ] Gamma grid includes all 4 values: `['scale', 'auto', 0.01, 0.001]`
- [ ] `ConvergenceWarning` count captured and reported in summary artifact
- [ ] Grid boundary check implemented: if best param is at grid edge, grid is extended and re-run
- [ ] Full `cv_results_` saved to `model/saved/svm/cv_results.json` (needed by Step 49)
- [ ] `scripts/train_step47.py` runs end-to-end and produces per-split or per-fold metrics plus aggregate summaries
- [ ] Step 47 uses repeated stratified outer splits or nested CV; a single split is not acceptable
- [ ] Step 47 reports mean, standard deviation, and interval estimates for ε/k comparison metrics across runs
- [ ] ε/k result is reported separately and interpreted as the primary benchmark target
- [ ] Best-run-only reporting is explicitly avoided in the script output and report narrative
- [ ] SVM and Ridge rows added to Esper test set comparison table in `model_comparison.md`
- [ ] Representation-bottleneck narrative updated with appropriately cautious language (`strengthens` / `is consistent with` rather than `proves` / `confirms`)
- [ ] R² heatmap regenerated with SVM and Ridge included
- [ ] Smoke test added for the SVM and Ridge wrappers
- [ ] Report written at `docs/reports/47_svm_benchmark.md`
- [ ] Commit: `step 47: benchmark SVR and Ridge against RF and XGBoost on same features`
