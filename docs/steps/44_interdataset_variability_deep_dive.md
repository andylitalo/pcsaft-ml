# Step 44: RF Combined Retrain (Esper + ML-SAFT)

## Objective

Train and evaluate a **combined-data Random Forest** on the deduplicated Esper + ML-SAFT
corpus, then compare it directly against an **Esper-only Random Forest** under identical
experimental conditions. This step answers the question that the earlier inter-dataset
analysis did not: **does adding ML-SAFT improve the production RF, leave it unchanged, or
degrade it because of cross-dataset parameter disagreement?**

The deliverable is not a descriptive overlap study. The deliverable is a controlled,
reproducible RF experiment with:

1. **Two RF models**:
   - Esper-only RF trained on `load_data("esper")`
   - Combined RF trained on `load_data("combined")` (Esper + ML-SAFT with Esper-priority
     deduplication)
2. **Shared evaluation protocol**:
   - fixed split seed
   - identical feature pipeline
   - identical RF hyperparameters
   - common evaluation slices
3. **Uncertainty comparison**:
   - tree-disagreement standard deviations
   - empirical 1-sigma and 2-sigma coverage
   - error-vs-uncertainty calibration

## Motivation

Step 10 established that Esper and ML-SAFT disagree on overlapping molecules. That is
useful context, but it is not enough to decide whether ML-SAFT should be included in the
RF training corpus. The production question is narrower and more practical:

- Does the combined RF improve on the original Esper-only RF?
- Does it help on genuinely new ML-SAFT-only chemistry?
- Does it preserve or hurt performance on the original Esper benchmark?
- Are its uncertainty estimates as good as, or better than, the Esper-only RF?

This step is therefore a late-stage, uncertainty-aware version of Step 01b. The dataset
variability analysis becomes supporting evidence only if the combined RF degrades or shows
source-specific calibration failures.

## Dependencies

- `model/data/esper_pcsaft.csv`
- `model/data/mlsaft_pcsaft.csv`
- `model/data/load.py`:
  - `load_data("esper")`
  - `load_data("combined")`
  - `split_data(..., random_state=42, stratify_bins=5)`
- `model/data/descriptors.py` or the active RF feature builder used by Step 01
- scikit-learn `RandomForestRegressor`
- Existing RF uncertainty convention: per-target tree standard deviation from
  `model.estimators_`
- Optional external set:
  `model/saved/gnn_fluorinated_validation_set.csv`

## Implementation Guide

### 44.1 Verify dataset composition

Use the project loader rather than rebuilding the merge logic manually:

```python
from model.data.load import load_data

df_esper = load_data("esper")
df_combined = load_data("combined")

print(f"Esper: {len(df_esper)}")
print(f"Combined: {len(df_combined)}")
print(f"Net new molecules: {len(df_combined) - len(df_esper)}")
```

Confirm that:

- `load_data("combined")` uses Esper-priority deduplication
- the combined dataset is roughly Esper (1,801) + ~125 net-new ML-SAFT molecules
- all target columns are present: `m`, `sigma`, `epsilon_k`

Do not hand-wave the data composition. Save a short provenance table that reports:

- raw Esper size
- raw ML-SAFT size
- deduplicated combined size
- overlap count
- net-new ML-SAFT molecules retained after deduplication

### 44.2 Freeze the split strategy

Use the same split convention everywhere in this step:

```python
from model.data.load import split_data

esper_train, esper_test = split_data(
    df_esper, test_size=0.2, random_state=42, stratify_bins=5
)
combined_train, combined_test = split_data(
    df_combined, test_size=0.2, random_state=42, stratify_bins=5
)
```

This step requires **two benchmark test sets**:

1. **Esper benchmark test set**: `esper_test`
2. **Combined benchmark test set**: `combined_test`

The combined benchmark test set must then be sliced into:

- **Esper-subset of combined test**
- **ML-SAFT-only subset of combined test**

Determine those subsets by InChI membership against the Esper corpus, not by SMILES string
equality. Save the test-set membership table so the evaluation is reproducible.

Critical rule: the only intended difference between the two RF models is training data
composition. The split seed, feature construction, and hyperparameters must remain fixed.

### 44.3 Train the paired RF models

Train one RF per target for both experiments, matching the existing project RF setup:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
```

Use the same feature pipeline as Step 01 for both models. Do not introduce any new
features, scaling strategy, or target transformation in this step.

Required models:

1. **Esper-only RF**
   - train on `esper_train`
   - save as `rf_esper_<target>.joblib` or equivalent clearly named artifacts
2. **Combined RF**
   - train on `combined_train`
   - save as `rf_combined_<target>.joblib` or equivalent clearly named artifacts

Do not reuse historical Step 01 weights. Retrain both models within the same script so the
comparison is fully controlled and reproducible.

### 44.4 Evaluate both RFs on common benchmark slices

Both models must be evaluated on the same prediction targets and the same slices:

1. **Esper benchmark test set**
2. **Combined benchmark test set**
3. **Esper subset of combined benchmark**
4. **ML-SAFT-only subset of combined benchmark**
5. **Optional fluorinated external validation set** if
   `model/saved/gnn_fluorinated_validation_set.csv` is available

For every model × slice × target, report:

| Target | MAE | RMSE | R² | Train samples | Test samples |
|--------|-----|------|----|---------------|--------------|
| m | | | | | |
| sigma | | | | | |
| epsilon_k | | | | | |

Interpretation requirements:

- The **Esper benchmark test set** is the “did we preserve the original RF?” check.
- The **ML-SAFT-only subset** is the “did the new data help where Esper had no coverage?”
  check.
- The **Combined benchmark test set** is the “overall generalization on the expanded corpus”
  check.

Do not claim improvement from a single slice in isolation. The report must discuss all
three.

### 44.5 Compare parameter-level uncertainty

Use the existing RF uncertainty convention: standard deviation across tree predictions for
each target.

For every model × slice × target, compute:

- mean predicted std
- mean absolute error
- std/error ratio
- empirical **1-sigma coverage**
- empirical **2-sigma coverage**

Coverage is defined per target as:

```text
1-sigma coverage = fraction(|y_pred - y_true| <= y_std)
2-sigma coverage = fraction(|y_pred - y_true| <= 2 * y_std)
```

This step must explicitly compare whether the combined RF is:

- better calibrated,
- similarly calibrated, or
- worse calibrated

than the Esper-only RF on the Esper benchmark and on ML-SAFT-only molecules.

### 44.6 Generate required figures

Save all figures to `figures/44_rf_combined_retrain/`.

Required figures:

1. **`parity_rf_esper_vs_combined.png`**
   - 2 rows × 3 columns
   - top row: Esper-only RF
   - bottom row: Combined RF
   - columns: m, sigma, epsilon_k
   - use the same axis limits within each target

2. **`slice_metrics_comparison.png`**
   - bar or dot plot comparing MAE/RMSE/R² across the required slices
   - highlight `epsilon_k` because it is most important downstream

3. **`uncertainty_calibration_rf_comparison.png`**
   - mean predicted std vs mean absolute error in bins
   - separate traces for Esper-only RF and Combined RF
   - at minimum for the Esper benchmark and ML-SAFT-only slice

4. **`coverage_comparison.png`**
   - 1-sigma and 2-sigma coverage by target and slice for both RFs

5. **`combined_test_source_breakdown.png`**
   - composition of the combined test set: Esper-derived vs ML-SAFT-only molecules

### 44.7 Run supporting overlap diagnostics only after the RF comparison

If the combined RF improves little or degrades on the Esper benchmark, compute a **supporting**
diagnostic on the Esper/ML-SAFT overlap to explain why. This is not the primary result.

The supporting diagnostic may include:

- overlap count by InChI
- per-molecule signed and absolute deltas for m, sigma, epsilon_k
- subgroup summaries for associating vs non-associating or halogenated vs non-halogenated

Only include these analyses to explain RF behavior. Do not let them replace the model
comparison.

### 44.8 Create the driver script

Create `scripts/step44_rf_combined_retrain.py` that:

1. loads Esper and combined datasets via `load_data`
2. creates fixed train/test splits with `random_state=42`, `stratify_bins=5`
3. trains the Esper-only RF and Combined RF with identical hyperparameters
4. predicts on all required evaluation slices
5. computes point metrics and uncertainty metrics
6. saves per-molecule predictions for both models
7. generates all required figures
8. optionally computes overlap diagnostics as supporting analysis

Save a machine-readable predictions table to
`model/saved/step44_rf_combined_predictions.csv` with at least:

`smiles, inchi, eval_set, eval_source, model_name, m_true, sigma_true, epsilon_k_true,
m_pred, sigma_pred, epsilon_k_pred, m_std, sigma_std, epsilon_k_std`

Save a metrics summary to `model/saved/step44_rf_combined_metrics.json` or `.csv`.

### 44.9 Write the Step 44 report

Write `docs/reports/44_rf_combined_retrain.md` following the standard report format.
Required sections:

1. **Title**:
   `# Results Report: RF Combined Retrain (Esper + ML-SAFT)`
2. **Model Performance**:
   the full metrics table for both RFs on the required slices
3. **Comparison to Baseline**:
   Esper-only RF vs Combined RF, with deltas
4. **Uncertainty Quality**:
   predicted stds, coverage, calibration discussion
5. **Slice Analysis**:
   Esper benchmark, combined benchmark, ML-SAFT-only subset, optional fluorinated external
6. **Supporting Diagnostics**:
   only if overlap variability is needed to explain the outcome
7. **Key Findings**
8. **Figures**
9. **Deviations**
10. **Readiness Check**

The report must end with a crisp recommendation:

- keep Esper-only RF,
- adopt combined RF, or
- keep Esper-only for production and use combined RF only for exploratory analyses

## Key Files

| File | Purpose |
|------|---------|
| `model/data/load.py` | Combined-data loading and fixed split logic |
| `model/data/esper_pcsaft.csv` | Esper training corpus |
| `model/data/mlsaft_pcsaft.csv` | ML-SAFT training corpus |
| `docs/steps/01b_rf_combined_retrain.md` | Prior backbone for this experiment |
| `docs/steps/03_evaluation_harness.md` | Uncertainty and calibration reporting conventions |
| `model/saved/gnn_fluorinated_validation_set.csv` | Optional external validation set |

## Artifacts

| File | Description |
|------|-------------|
| `scripts/step44_rf_combined_retrain.py` | Controlled two-model RF experiment |
| `docs/reports/44_rf_combined_retrain.md` | Results report |
| `figures/44_rf_combined_retrain/` | All Step 44 figures |
| `model/saved/step44_rf_combined_predictions.csv` | Per-molecule predictions and stds |
| `model/saved/step44_rf_combined_metrics.json` | Aggregate metrics and coverage tables |

## Success Criteria

- [ ] Esper-only RF and Combined RF are both retrained in the same script with identical
      hyperparameters
- [ ] Both models use the same feature pipeline and the same split seed
- [ ] Both models are evaluated on the Esper benchmark test set
- [ ] Both models are evaluated on the combined benchmark test set
- [ ] ML-SAFT-only molecules in the combined test set are isolated and reported separately
- [ ] Point metrics (MAE, RMSE, R²) are reported for every target on every required slice
- [ ] Uncertainty metrics (mean std, coverage, calibration) are reported for both RFs
- [ ] All required figures are saved to `figures/44_rf_combined_retrain/`
- [ ] A report states whether ML-SAFT should or should not be incorporated into the
      production RF
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- There is a clear answer to whether the combined RF helps, hurts, or leaves the RF
  unchanged
- The answer is based on both point metrics and uncertainty behavior
- Performance on the Esper benchmark and ML-SAFT-only molecules is separately documented
- Any overlap-variability analysis is clearly secondary and used only to explain the RF
  result
- The project has a defensible recommendation on whether to include ML-SAFT in RF training

## Budget

45-60 minutes. This is a real model-training and evaluation step, not just a descriptive
data audit. Most of the work is training two RFs, saving predictions, and comparing
uncertainty behavior across slices.
