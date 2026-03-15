# Step 2C: Multi-Model Ensemble for PC-SAFT Prediction

## Purpose

Step 2B showed that a GNN trained on expanded data achieves R² = 0.76/0.77/0.73 on m/σ/ε/k — a major leap over the RF baseline. But the project now has three trained models (RF, GNN, ChemBERTa) that use fundamentally different molecular representations:

| Model | Representation | Inductive Bias |
|-------|---------------|----------------|
| RF | Morgan FP + RDKit descriptors (2,218-dim flat vector) | Substructure counting, global molecular properties |
| GNN | Molecular graph (atoms, bonds, connectivity) | Local chemical environments, topology |
| ChemBERTa | Tokenized SMILES (sequential text) | Character-level patterns, pretrained chemical knowledge |

Because each model encodes a different view of the same molecule, their errors are partially uncorrelated. Ensembling models with diverse error patterns is one of the most reliable ways to reduce prediction variance without collecting more data.

This step implements ensemble prediction in two phases:

1. **Phase 1 — Uncertainty-weighted combination.** Weight each model's prediction by 1/σ² from its own uncertainty estimate (MC Dropout for GNN and ChemBERTa; tree disagreement for RF). This is principled, requires no additional training, and naturally favors the more confident model on a per-molecule basis.

2. **Phase 2 — Stacking meta-learner** (only if Phase 1 leaves significant room for improvement). Train a lightweight Ridge regression on the predictions of RF, GNN, and ChemBERTa as input features. This learns which model to trust for different molecular types. Requires a held-out stacking fold.

The ensemble should be registered in the model registry so it can be evaluated, served, and compared like any other model.

## What It Adds Over Step 2B

| After Step 2B | After This Step |
|---------------|----------------|
| GNN is the single best model | Ensemble combines GNN + RF + ChemBERTa for lower variance |
| Each model's uncertainty is used independently | Uncertainties from all models are fused into a single, tighter estimate |
| Model choice is all-or-nothing | Best-of-each-model is exploited per molecule |
| One architecture's blind spots persist | Blind spots of one model can be covered by another |

## Skills Demonstrated

- **Model ensembling**: uncertainty-weighted prediction fusion, inverse-variance weighting
- **Stacking / meta-learning**: held-out fold predictions as meta-features, Ridge regression meta-learner
- **Heterogeneous ensemble design**: combining models with fundamentally different input representations
- **Graceful degradation**: ensemble handles missing constituent models (e.g., ChemBERTa unavailable if `transformers` not installed)
- **Uncertainty propagation**: combining uncertainty estimates from MC Dropout and tree disagreement into a single ensemble uncertainty

## Dependencies

- Requires: all Step 2B dependencies (`torch`, `torch-geometric`), plus trained RF, GNN, and optionally ChemBERTa models
- No new pip packages. Phase 2 stacking uses scikit-learn's `Ridge` (already installed)
- ChemBERTa is optional: the ensemble must work with just RF + GNN if `transformers` is not available

## Implementation Guide

### 1. Project structure

```text
model/
  ensemble/
    __init__.py
    weighted.py          # Phase 1: uncertainty-weighted combination
    stacking.py          # Phase 2: stacking meta-learner (if needed)

model/saved/
    ensemble_config.json  # which models are in the ensemble, weights, etc.
    stacking_meta.joblib  # Phase 2 only: saved Ridge meta-learner
```

### 2. Phase 1: Uncertainty-Weighted Ensemble (`model/ensemble/weighted.py`)

The core idea: for each molecule and each target, combine predictions from available models using inverse-variance weighting.

Given predictions ŷ₁, ŷ₂, ..., ŷₖ with estimated standard deviations σ₁, σ₂, ..., σₖ:

```
w_i = 1 / σ_i²
ŷ_ensemble = Σ(w_i × ŷ_i) / Σ(w_i)
σ_ensemble = 1 / √(Σ(w_i))
```

This is the minimum-variance unbiased linear combination under the assumption that the models' errors are uncorrelated — a reasonable approximation given the completely different representations.

Implementation requirements:

1. Accept a list of model names to include in the ensemble (default: all available). The constructor should attempt to load each model via the registry and silently skip models that fail to load (e.g., ChemBERTa when `transformers` is missing). Log a warning for skipped models.

2. Call `predict_with_uncertainty(smiles_list)` on each constituent model.

3. For each molecule × target, compute the inverse-variance-weighted mean and combined uncertainty.

4. Handle edge cases:
   - If a model returns `NaN` for a molecule (e.g., unparseable SMILES for GNN), exclude that model from the weighted average for that molecule.
   - If only one model produces a valid prediction, fall back to that model's prediction (no averaging).
   - If no models produce a valid prediction, return `NaN`.
   - Clamp individual σ values to a minimum of 1e-6 to avoid division by zero.

5. Store the ensemble configuration (which models were loaded, their names) so it can be saved and reloaded.

Skeleton:

```python
class WeightedEnsemble:
    """Uncertainty-weighted ensemble of PC-SAFT prediction models."""

    def __init__(self, model_names: list[str] | None = None):
        # Default: try all registered models
        # Load each via registry; skip failures with warning
        ...

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        # Calls predict_with_uncertainty on each model
        # Returns inverse-variance-weighted mean predictions
        ...

    def predict_with_uncertainty(
        self, smiles_list: list[str]
    ) -> dict[str, np.ndarray]:
        # Returns combined predictions + combined uncertainty
        ...
```

### 3. Register in model registry (`model/registry.py`)

Register the ensemble as `"ensemble"` in the model registry. The registry wrapper should:

- Accept a configurable list of constituent models (default: `["rf", "gnn"]` — the two models that are always available without optional heavy dependencies)
- `load()` loads all constituent models
- `predict()` and `predict_with_uncertainty()` delegate to the `WeightedEnsemble`

When loading, attempt all of `["rf", "gnn", "chemberta"]` and keep whichever load successfully. The ensemble must function with a minimum of 2 models.

Important: importing the ensemble class must not trigger eager imports of `torch_geometric` or `transformers`. Use lazy imports inside `load()`, consistent with the existing pattern for `GNNModel` and `ChemBERTaModel`.

### 4. Phase 2: Stacking Meta-Learner (`model/ensemble/stacking.py`)

Only implement Phase 2 if Phase 1 results show that:
- The uncertainty-weighted ensemble does not improve over the GNN alone on at least 2 of 3 targets, OR
- Per-molecule model disagreement is high (mean σ_ensemble > 0.8 × mean σ_best_single_model)

Stacking approach:

1. **Create stacking fold.** From the training set, hold out 20% as a stacking calibration set. Train RF, GNN, and ChemBERTa on the remaining 80%.

2. **Generate meta-features.** Predict on the stacking fold using each trained model. For each molecule, the meta-feature vector is [ŷ_rf_m, ŷ_rf_σ, ŷ_rf_εk, ŷ_gnn_m, ŷ_gnn_σ, ŷ_gnn_εk, ŷ_cb_m, ŷ_cb_σ, ŷ_cb_εk] — 9 features (3 models × 3 targets). Optionally include uncertainty estimates as additional features (18 features total).

3. **Train meta-learner.** Fit one `sklearn.linear_model.Ridge` per target on the stacking fold. Ridge regression is appropriate because:
   - With only 3–6 input features per target, overfitting risk is low
   - L2 regularization prevents any single model from being completely ignored
   - The learned weights are interpretable (how much the meta-learner trusts each model)

4. **Save and load.** Save the meta-learner and meta-feature configuration as `model/saved/stacking_meta.joblib`. Include metadata: which models were used, feature order, alpha hyperparameter.

5. **Uncertainty for stacking.** Use the meta-learner's residuals on the stacking fold to estimate prediction uncertainty (analogous to conformal prediction). For each target, the ensemble uncertainty is the (1 − α) quantile of |ŷ_meta − y_true| on the stacking fold.

Skeleton:

```python
class StackingEnsemble:
    """Stacking meta-learner over RF, GNN, and ChemBERTa predictions."""

    def __init__(self, model_names: list[str] | None = None):
        ...

    def fit(self, smiles_list: list[str], targets: np.ndarray):
        # 1. Get predictions from each constituent model
        # 2. Build meta-feature matrix
        # 3. Fit Ridge per target
        ...

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        ...

    def save(self, path: Path): ...
    def load(self, path: Path): ...
```

### 5. Training and evaluation script (`model/ensemble/train_ensemble.py`)

A single CLI entry point that:

1. Loads the test set (same split as all other models for comparability)
2. Runs Phase 1 (weighted ensemble) and reports metrics
3. Compares to GNN-only baseline
4. If Phase 2 is warranted (see criteria above), trains the stacking meta-learner and reports its metrics
5. Saves ensemble configuration and (if applicable) the stacking meta-learner

```bash
# Phase 1 only (default)
python -m model.ensemble.train_ensemble --phase 1

# Phase 1 + Phase 2
python -m model.ensemble.train_ensemble --phase 2

# Specify which models to include
python -m model.ensemble.train_ensemble --models rf gnn chemberta
```

### 6. Evaluation and comparison

Run the ensemble through the standard evaluation harness:

```bash
python -m model.evaluate --models gnn ensemble
```

Report the full metric suite from Step 11 (R², MARE, CCC, Coverage@5%, Coverage@10%) for both the ensemble and the GNN alone. The comparison table should make it easy to see whether ensembling helps.

### 7. Figures

Save to `figures/02c_ensemble/`:

- `ensemble_vs_gnn_parity.png`: Side-by-side parity plots (predicted vs true) for GNN alone and the ensemble on all three targets. Six panels (2 models × 3 targets).
- `model_weight_distribution.png`: Histogram or box plot showing the inverse-variance weights assigned to each model across all test molecules. This reveals which model dominates and whether the weighting is heterogeneous (good) or trivial (one model always dominates).
- `ensemble_improvement_by_target.png`: Bar chart of R² improvement from GNN → ensemble for each target.
- (Phase 2 only) `stacking_coefficients.png`: Bar chart of Ridge regression coefficients per target, showing how the meta-learner weights each model.

## Science Improvements

### Ensemble Uncertainty for Screening

The ensemble's combined uncertainty (σ_ensemble) should be tighter than any single model's uncertainty because it incorporates multiple independent views of each molecule. This directly addresses the Step 11 finding that 0/645 candidates cleared the `ci_high_conf` threshold: if the ensemble reduces mean CI width by even 20–30%, some candidates may now clear the bar.

After training the ensemble:

1. Re-run `predict_pcsaft_with_ci()` from Step 11 using the ensemble's uncertainty estimates.
2. Report updated `ci_compatible` and `ci_high_conf` counts for the 645-candidate shortlist.
3. If `ci_high_conf > 0`, highlight which candidates newly clear the bar and why.

### Per-Molecule Model Disagreement as an AD Signal

The disagreement between constituent models provides a natural applicability-domain signal distinct from the existing Isolation Forest approach. If RF predicts ε/k = 250 K and GNN predicts ε/k = 310 K for the same molecule, the molecule is likely in a region where at least one model is unreliable.

Define a disagreement score for each molecule × target:

```
disagreement = max(|ŷ_i - ŷ_j|) for all model pairs (i, j)
```

If disagreement exceeds 2× the ensemble uncertainty, flag the molecule. This is complementary to the fingerprint-based and embedding-based AD checks from Steps 02 and 4B.

## Evaluation & Success Criteria

### Metrics to compare

| Model | R² (m) | R² (σ) | R² (ε/k) | MARE (ε/k) | Cov@10% (ε/k) |
|-------|--------|--------|----------|------------|----------------|
| RF | 0.619 | 0.353 | 0.330 | 0.113 | 74.0% |
| GNN (all data) | 0.760 | 0.766 | 0.728 | — | — |
| Ensemble (Phase 1) | — | — | — | — | — |
| Ensemble (Phase 2) | — | — | — | — | — |

### What success looks like

**Phase 1 (uncertainty-weighted)**:
- R² improves over the GNN alone on at least 1 of 3 targets
- σ_ensemble is at least 10% smaller than σ_gnn on average (tighter uncertainty)
- Inverse-variance weights are heterogeneous (no single model gets >90% weight on >80% of molecules)
- The ensemble gracefully handles the case where ChemBERTa is unavailable (RF + GNN only)

**Phase 2 (stacking — only if Phase 1 is insufficient)**:
- Stacking improves over Phase 1 weighted ensemble on at least 2 of 3 targets
- Ridge coefficients are interpretable and consistent with known model strengths (e.g., GNN gets higher weight on ε/k)
- Stacking fold residuals provide a valid uncertainty estimate (empirical coverage ≥ 90% at nominal 95%)

### What to watch for

- **Trivial weighting**: If the GNN's uncertainty is always much smaller than RF's and ChemBERTa's, the "ensemble" collapses to just the GNN. This is not wrong — it means the GNN is genuinely better — but it limits the ensemble's value. If this happens, the ensemble still adds value through the disagreement-based AD signal.
- **Stale constituent models**: RF and ChemBERTa were trained on Esper only (1,801 mol), while the GNN was trained on all data (13,764 mol). The data mismatch is acceptable for Phase 1 (uncertainty weighting accounts for it) but may confound Phase 2 (the meta-learner could just learn to ignore the weaker models). Document this asymmetry.
- **NaN propagation**: If the GNN fails on a molecule that RF handles (or vice versa), the ensemble should fall back gracefully. Test with at least one molecule that fails for one model but succeeds for another.
- **Evaluation split**: Use the same test split as Steps 01–2B so R² values are directly comparable. Do not contaminate the test set with stacking-fold molecules.
- **Latency**: The ensemble requires running all constituent models. On CPU, GNN + RF + ChemBERTa inference per molecule is ~50ms + ~5ms + ~100ms ≈ 155ms. This is acceptable for batch screening but may need attention for the serving endpoint. Document inference time.

### When to move on

Phase 1 is complete when:

1. [  ] `model/ensemble/weighted.py` implements inverse-variance-weighted prediction
2. [  ] Ensemble is registered as `"ensemble"` in the model registry
3. [  ] `predict()` and `predict_with_uncertainty()` work with RF + GNN (minimum) or RF + GNN + ChemBERTa
4. [  ] Comparison table with R², MARE, CCC, Coverage@10% for GNN vs ensemble on test set
5. [  ] Weight distribution figure shows which model dominates and where
6. [  ] Updated `ci_compatible` / `ci_high_conf` counts for 645-candidate shortlist using ensemble uncertainties
7. [  ] Report written at `docs/reports/02c_model_ensemble.md`
8. [  ] At least 4 tests covering: normal operation, missing model fallback, NaN handling, single-valid-model fallback

Phase 2 (only if warranted) adds:

9. [  ] `model/ensemble/stacking.py` with Ridge meta-learner
10. [  ] Stacking trained on held-out fold; meta-features from RF + GNN (+ ChemBERTa if available)
11. [  ] Ridge coefficients reported and interpreted
12. [  ] Comparison table extended with stacking row

### Out of scope

- Retraining RF or ChemBERTa on the expanded dataset (would help but is a separate effort)
- Bayesian model combination or Bayesian model averaging (too complex for the current dataset size)
- Neural meta-learners (a small MLP on 9 features from 3 models would overfit with ~360 test molecules)
- Serving endpoint changes (the ensemble integrates via the registry; serving uses whatever model is configured)
- Hyperparameter search over ensemble composition (just use all available models)
