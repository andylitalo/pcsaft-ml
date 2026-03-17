# Step 01b: RF Retrain on Esper + ML-SAFT Combined Data

*Branches from Step 01 (Morgan Fingerprints / RF on Esper)*

## Purpose

Step 01 established the current Random Forest baseline on the Esper dataset: 1,801 molecules, combined Morgan + RDKit features, 80/20 stratified split, and test R2 of roughly 0.62 / 0.35 / 0.33 for `m / sigma / epsilon_k` (see `docs/reports/01_morgan_fingerprints.md`).

Step 10 then integrated the ML-SAFT dataset (Felton et al. 2024, 870 molecules), verified the overlap with Esper, and concluded qualitatively that retraining on the combined corpus did not yield a meaningful lift on the Esper benchmark (see `docs/reports/10_mlsaft_integration.md`). What is still missing is a fully documented, uncertainty-aware, apples-to-apples comparison between:

- `RF_esper`: trained on Esper only
- `RF_combined`: trained on deduplicated Esper + ML-SAFT

This step formalizes that comparison. Its job is not to invent a new modeling direction, but to answer rigorously whether the ML-SAFT expansion changes RF performance enough to justify including it in the production training set used for screening.

## What It Adds Over Step 01 and Step 10

| Current documented state | After this step |
|---|---|
| Step 01 defines the Esper-only RF baseline | The baseline is re-run and compared directly against a combined-data RF under matched conditions |
| Step 10 summarizes "no meaningful lift" without a full controlled comparison table | The comparison is fully reported with benchmark definitions, uncertainty, and figures |
| Uncertainty exists elsewhere in the project but is not attached to this dataset-expansion decision | Bootstrap CIs and RF predictive uncertainty are reported for the Esper-only vs combined comparison |
| Combined corpus size is described inconsistently across docs | The report states the exact deduplication rule and the exact resulting counts used in the experiment |

## Dependencies

- Requires: Step 01 completed (`docs/reports/01_morgan_fingerprints.md` is the baseline reference)
- Requires: Step 10 completed (`mlsaft_pcsaft.csv` downloaded, `load_data("combined")` verified)
- Requires: `model/data/mlsaft_pcsaft.csv` (~870 molecules)
- Uses existing evaluation conventions from `docs/reports/03_evaluation_harness.md`
- Does not require: any new packages

## Implementation Guide

### 01b.1 Verify Dataset Construction and Count Conventions

Before training anything, document exactly how the combined corpus is built.

```python
from model.data.load import load_data

df_esper = load_data("esper")
df_combined = load_data("combined")

print(f"Esper: {len(df_esper)} molecules")
print(f"Combined: {len(df_combined)} molecules")
print(f"Net increase: {len(df_combined) - len(df_esper)} molecules")
```

The final report must explicitly state:

1. Which duplicate rule is used by `load_data("combined")` during this step.
2. Whether overlap counts are being reported by canonical SMILES or by InChI.
3. That Esper values take priority for duplicates when both datasets contain the same molecule.

Use the Step 10 conventions as the reference narrative:

- 736 overlaps by canonical SMILES
- 745 overlaps by InChI
- approximately 134 genuinely new molecules and approximately 1,935 total molecules under the Step 10 combined-data convention

If the loader now produces a different exact count, report the exact value from code and explain why. Do not mix counting conventions within the same report.

### 01b.2 Reconstruct the Canonical Esper Benchmark Split

The primary comparison in this step is not "combined test set vs Esper test subset." It is:

- train two different models
- evaluate both on the exact same held-out Esper benchmark

Recreate the Step 01 benchmark split using the Esper dataset only, with the same policy used in `docs/reports/01_morgan_fingerprints.md`:

```python
from model.data.load import load_data, split_data

df_esper = load_data("esper")
esper_train_df, esper_test_df = split_data(
    df_esper,
    test_size=0.2,
    random_state=42,
    stratify_bins=5,
)
```

This `esper_test_df` is the canonical benchmark set for the main comparison in this step. Both models must be evaluated on this same holdout.

### 01b.3 Train the Two RF Variants Under Matched Conditions

Use identical feature construction, target definitions, and RF hyperparameters for both models. The only intended experimental difference is the training corpus.

```python
from model.data.descriptors import build_features
from model.data.load import TARGETS, load_data
from sklearn.ensemble import RandomForestRegressor

df_esper = load_data("esper")
df_combined = load_data("combined")

# Reuse esper_test_df from 01b.2
esper_train_df = df_esper.loc[~df_esper.index.isin(esper_test_df.index)].copy()

# Train RF_esper on Esper train split only
X_train_esper = build_features(esper_train_df["smiles"].tolist())

# Train RF_combined on the full deduplicated combined corpus,
# excluding any molecules that belong to the canonical Esper holdout.
# Use the same identity key used for deduplication in load_data("combined")
# (for example, InChI if available), not a looser post hoc rule.
combined_train_df = df_combined.loc[
    ~df_combined["smiles"].isin(set(esper_test_df["smiles"]))
].copy()
X_train_combined = build_features(combined_train_df["smiles"].tolist())

X_test_esper = build_features(esper_test_df["smiles"].tolist())

for target in TARGETS:
    rf_esper = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_combined = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # fit, predict, evaluate, save
```

Critical details:

- Do not compare models trained with different features or different RF settings.
- Do not allow molecules from the canonical Esper holdout to leak into `RF_combined` through the merged dataset.
- The primary benchmark must remain the original Step 01 Esper holdout so the result can be compared directly to the established baseline.

### 01b.4 Primary and Supplemental Evaluations

#### Primary benchmark

Report all main conclusions from the canonical Esper holdout:

1. Evaluate `RF_esper` on `esper_test_df`.
2. Evaluate `RF_combined` on the same `esper_test_df`.
3. Compare metric deltas target-by-target.

This is the only apples-to-apples benchmark for deciding whether ML-SAFT should be added to RF training.

#### Supplemental benchmarks

These are useful, but secondary:

1. **Combined holdout**: Construct a natural 20% holdout from the deduplicated combined corpus and evaluate `RF_combined` on it. Label this as a combined-distribution generalization check, not as the primary baseline comparison.
2. **ML-SAFT-unique holdout subset**: If enough unique ML-SAFT-only molecules remain after deduplication and holdout construction, report their metrics separately. Treat this as directional evidence only because the sample size will be small.
3. **External fluorinated validation**: If the 15-compound fluorinated set from Step 38 is available (`model/saved/gnn_fluorinated_validation_set.csv`), evaluate both RF variants on it and compare boiling point MAE and `epsilon_k` MAE against the Step 38d RF reference.

The report should clearly label which results are primary and which are supplemental.

### 01b.5 Make Uncertainty Reporting Mandatory

Follow the conventions already established in `docs/reports/03_evaluation_harness.md`.

For every benchmark that is reported in the comparison table:

1. Report point estimates for `R2`, `MAE`, and `RMSE`.
2. Report 95% bootstrap confidence intervals for those aggregate metrics.
3. Report RF predictive uncertainty as the standard deviation across trees (for example, mean predicted std per target on the evaluated set).
4. Where feasible, add a simple uncertainty calibration summary, such as mean predicted std vs mean absolute error across bins, to show whether adding ML-SAFT materially changes RF uncertainty behavior.

Interpretation requirement:

- The final recommendation must distinguish a real performance change from a delta that falls inside bootstrap uncertainty.
- If predictive accuracy is flat but uncertainty calibration worsens, that still counts as a regression.

### 01b.6 Compare Applicability Domain Coverage

Applicability-domain analysis is useful supporting evidence, but it is not the primary decision criterion.

Refit the Tanimoto AD on both training sets and compare in-domain rates on the downstream screening candidates:

```python
from model.ad_tanimoto import TanimotoAD

ad_esper = TanimotoAD()
ad_esper.fit(esper_train_df["smiles"].tolist())

ad_combined = TanimotoAD()
ad_combined.fit(combined_train_df["smiles"].tolist())

candidates = [...]  # Load from screening results
esper_rates = [ad_esper.check(s) for s in candidates]
combined_rates = [ad_combined.check(s) for s in candidates]
```

If AD coverage improves while canonical Esper-holdout metrics do not, document that as a tradeoff rather than presenting it as a pure win.

### 01b.7 Generate Comparison Tables and Figures

Produce a comparison table that separates primary and supplemental evidence.

#### Primary comparison table

| Metric | RF_esper | RF_combined | Delta | 95% CI interpretation |
|---|---|---|---|---|
| R2(m) - Esper holdout | | | | |
| R2(sigma) - Esper holdout | | | | |
| R2(epsilon_k) - Esper holdout | | | | |
| MAE(m) - Esper holdout | | | | |
| MAE(sigma) - Esper holdout | | | | |
| MAE(epsilon_k) - Esper holdout | | | | |
| RMSE(m) - Esper holdout | | | | |
| RMSE(sigma) - Esper holdout | | | | |
| RMSE(epsilon_k) - Esper holdout | | | | |
| Mean std(m) - Esper holdout | | | | |
| Mean std(sigma) - Esper holdout | | | | |
| Mean std(epsilon_k) - Esper holdout | | | | |

#### Supplemental comparison table

| Metric | RF_esper | RF_combined | Delta |
|---|---|---|---|
| R2 / MAE / RMSE on fluorinated validation (if available) | | | |
| R2 / MAE / RMSE on combined holdout | n/a | | |
| R2 / MAE / RMSE on ML-SAFT-unique subset | n/a | | |
| AD in-domain rate on screening candidates | | | |

Save plots to `figures/01b_rf_combined_retrain/`. At minimum include:

- parity plots on the canonical Esper holdout for both RF variants
- a delta plot or bar chart for Esper-holdout metrics
- one uncertainty calibration figure comparing the two RF variants
- external-validation parity or summary plots if the fluorinated benchmark is used

## Expected Outcomes

**Hypothesis 1 (likely)**: Minimal change on the canonical Esper holdout. Step 10 already suggests that adding ML-SAFT does not materially improve RF performance on the Step 01 benchmark, and the net increase in genuinely new compounds is small.

**Hypothesis 2 (possible)**: Slight coverage gains with little or no accuracy gain. The ML-SAFT-unique molecules may widen chemical-space coverage enough to improve AD rates or some supplemental benchmark without moving the core Esper metrics.

**Hypothesis 3 (important guardrail)**: No material degradation in uncertainty behavior. Even if mean accuracy is unchanged, the combined model should not become less calibrated or less trustworthy on the canonical holdout or fluorinated external validation.

## Evaluation and Success Criteria

### What success looks like

- A controlled comparison where the only intended experimental difference is the training corpus
- A report that distinguishes the primary Esper-holdout benchmark from supplemental evaluations
- Explicit uncertainty reporting with bootstrap CIs and RF predictive-std summaries
- A clear recommendation on whether ML-SAFT should be included in RF training, supported by both metric deltas and uncertainty evidence
- Exact documentation of the duplicate-resolution rule and the resulting combined-corpus size used in the experiment

### When to consider this done

- [ ] `RF_esper` and `RF_combined` are trained with the same feature pipeline and hyperparameters
- [ ] Both models are evaluated on the canonical Step 01 Esper holdout
- [ ] Primary comparison table includes `R2`, `MAE`, `RMSE`, 95% bootstrap CIs, and mean RF predictive std for all three targets
- [ ] Supplemental results are clearly labeled: combined holdout, ML-SAFT-unique subset, fluorinated validation, and AD coverage where available
- [ ] Figures saved to `figures/01b_rf_combined_retrain/`, including at least one uncertainty-comparison figure
- [ ] Report at `docs/reports/01b_rf_combined_retrain.md` states the exact dataset-count convention used and gives a clear include/exclude recommendation for ML-SAFT in RF training
- [ ] Commit: `step 01b: retrain RF on Esper + ML-SAFT combined data`
