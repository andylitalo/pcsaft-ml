# Step 48: Model Selection Validation

> **Presentation priority: HIGH.** This is the single most important gap for MLE credibility. The current model selection claim ("RF is best for production") was made by comparing RF against GNN only on the deployment domain. XGBoost and chemprop — both statistically tied with RF on the Esper holdout — were never tested on the fluorinated validation set. Until this step is done, the supplementary material contains `—` placeholders in the fluorinated comparison table. An MLE reviewer who finds those during Q&A will correctly conclude the model selection is incomplete.

## Objective

Evaluate the competitive deployment candidates (RF, XGBoost, chemprop, GNN) through a **reproducible, apples-to-apples validation protocol** and make a data-driven model selection decision. The current RF selection was based on fluorinated validation against GNN only — XGBoost and chemprop were never tested on the deployment domain, and the evaluation provenance (shared test split / shared feature config) was not yet formalized for all models.

## Motivation

On the Esper test set (361 molecules, 80/20 stratified split), three models are statistically indistinguishable on epsilon/k MAE:

| Model    | epsilon/k MAE (K) | epsilon/k R² |
|----------|-------------------|--------------|
| RF       | 26.8              | 0.33         |
| XGBoost  | 27.0              | 0.33         |
| chemprop | 26.8              | 0.39         |

RF was selected in Step 38d based on fluorinated external validation where it achieved 8.2 K boiling point MAE vs GNN's 133.7 K. But XGBoost and chemprop — the two models closest to RF on general accuracy — were never evaluated on the fluorinated set or boiling point estimation. The model selection is therefore incomplete.

This step closes that gap by evaluating all four models under explicitly documented conditions on all three validation tiers, with paired statistical comparisons and deployment-domain metrics that reflect the actual screening task.

## Validation Hierarchy

The hierarchy is ordered by relevance to the deployment task (screening fluorinated HFO blowing agents):

**Tier 1 — Fluorinated external validation (15 refrigerants)**
The decisive tier. Tests the full deployment path: SMILES → predicted PC-SAFT parameters → boiling point via EOS. Boiling point MAE is the primary metric because the screening filter window is only ~35 K wide. Epsilon/k MAE is secondary because it dominates the HFO distance ranking (5:2:1 weighting). Because the external set is small (`n=15`), report uncertainty on Tier 1 itself: bootstrap 95% CIs for boiling point MAE/RMSE and paired per-compound error comparisons between models.

**Statistical power caveat**: With `n=15`, bootstrap CIs and paired permutation tests have low power — they can detect large effects (like the 16x RF-vs-GNN gap) but will fail to distinguish models that differ by, say, 5 K on boiling point MAE. Do not over-engineer the statistical machinery: if the effect is large, the per-compound error table speaks for itself; if the effect is small, state honestly that Tier 1 cannot distinguish the candidates at this sample size and let Tier 2 take over. Elaborate statistics on tiny samples can look like false rigor.

**Tier 2 — Esper holdout with paired bootstrap / paired permutation tests (361 molecules)**
General parameter prediction accuracy. Without uncertainty quantification, the current MAE values are too close to assign a winner. Do not use CI overlap as the significance test. Instead, compare models with paired resampling on the **same molecules** by bootstrapping the metric difference (for example, `MAE_modelA - MAE_modelB`) or by using a paired permutation test on per-molecule losses.

**Tier 3 — Cross-validated robustness check (5-fold Q² for tree models)**
Supporting analysis, not the primary model-selection criterion. Confirms that the single-split results are not an artifact of a lucky/unlucky draw. Only directly comparable for RF and XGBoost unless all other models are retrained under the same CV protocol.

**Generalization caveat**: The split logic in this step is still random-split / shuffled-CV validation on the available molecular corpus. That supports **internal reproducibility and model-selection discipline**, but it is not the same as demonstrating scaffold-level or chemotype-level generalization. If a presentation will make strong claims about out-of-family generalization, add a chemistry-aware split as future work or label the current evidence as internal validation only.

## Reproducibility Requirements

Before computing any metrics, freeze the evaluation provenance. This step is only scientifically valid if the compared models are evaluated on a known split and a known feature schema.

1. **Verify candidate artifacts**:
   - Saved model path
   - Training corpus (`esper` vs unified)
   - Test split identity (row count, molecule identities, random seed if known)
   - Feature schema for descriptor-based models (`feature_config.json`, `feature_names.joblib`)
   - Model interface (`predict()` input/output contract)
2. **Use one comparison cohort for any final deployment claim**:
   - Required for a non-provisional model-selection decision: retrain or reload all deployment candidates on the same frozen Esper split and, for RF/XGBoost, the same frozen descriptor schema.
   - If that cannot be done and existing saved artifacts must be mixed, the outcome may be reported only as a **provisional comparison**. Do not present it as the final production choice.
3. **Import model modules explicitly before `get_model()`**:
   - `model.registry` does not auto-import every model family.
   - Import `model.xgb`, `model.chemprop_model`, and any other external wrappers before calling `get_model("xgboost")` or `get_model("chemprop")`.
4. **Do not silently skip competitive models**:
   - If a required artifact is missing, that is a blocking provenance issue, not a reason to proceed to a final selection with partial evidence.
5. **Data leakage audit for the fluorinated validation set**:
   - Verify that none of the 15 fluorinated validation compounds appear in the Esper training set. Compare SMILES (canonicalized) or InChI keys between `gnn_fluorinated_validation_set.csv` and the training split. This is almost certainly clean (they are published refrigerants, not Esper molecules), but an MLE reviewer will expect the check to be documented. Record the result in the provenance metadata artifact.
6. **Near-duplicate / analog leakage sanity check for the fluorinated set**:
   - Go beyond exact identity. For each fluorinated validation compound, compute its nearest-neighbor similarity to the Esper training molecules under the same Morgan fingerprint representation used elsewhere in the project, and record the maximum Tanimoto similarity in the provenance artifact.
   - If any external compound has a very close training analog (for example, max similarity `> 0.85`), call that out explicitly in the report. The result may still be useful, but it should be described as a demanding analog-generalization check rather than a fully distant external test.
7. **Quantifying the Generalization Caveat (Esper Test Set)**:
   - The Esper test set is a random split, not a scaffold split. To quantify exactly how "in-domain" the test set is, compute the maximum Tanimoto similarity to the training set for every molecule in the test set.
   - Plot the **Tanimoto nearest-neighbor similarity distribution** between the Esper test set and the Esper train set. This replaces a qualitative caveat with quantitative evidence of the split's difficulty.

## Dependencies

- Saved models for RF, XGBoost, chemprop, and GNN (from Steps 01, 15, 02b)
- Fluorinated validation set: `model/saved/gnn_fluorinated_validation_set.csv`
- Boiling point computation: `screening/hfo_screening.py::compute_boiling_point()`
- Evaluation harness: `model/evaluate.py`
- Feature computation: `model/registry.py::_compute_features()`
- **GNNePCSAFT** (new): `pip install gnnepcsaft` — external pre-trained benchmark for PC-SAFT parameter prediction via GNNs. No model training required; inference only.

If the saved artifacts are not provenance-compatible, retraining on one frozen split is part of this step.

## Implementation Guide

### 48.1 Verify artifacts, import model modules, and load the fluorinated validation set

Load the fluorinated validation set from `model/saved/gnn_fluorinated_validation_set.csv`. This contains 15 fluorinated refrigerants with literature PC-SAFT parameters (m, sigma, epsilon_k) and experimental boiling points.

Load the five models. **First import the model families so they register with `model.registry`. Interface differences must be handled explicitly:**

| Model | Registry name | predict() input | predict() output |
|-------|--------------|-----------------|------------------|
| RF | `rf` | `smiles_list` | `dict[str, np.ndarray]` |
| XGBoost | `xgboost` | feature matrix `X` (numpy) | `np.ndarray` shape `(n, 3)` |
| chemprop | `chemprop` | `smiles_list` | `np.ndarray` shape `(n, 3)` |
| GNN | `gnn` | `smiles_list` | `dict[str, np.ndarray]` |
| GNNePCSAFT | external (no registry) | `smiles_list` | `dict[str, np.ndarray]` |

RF and GNN are loaded via `model.registry.get_model(name)` and accept SMILES directly, returning target-keyed dicts. XGBoost requires feature computation via `_compute_features(smiles_list)` first and returns a numpy array with columns `[m, sigma, epsilon_k]`. Chemprop accepts SMILES but also returns a numpy array. Write a thin prediction wrapper for each model that normalizes the output to a consistent `dict[str, np.ndarray]` format. Record the exact artifact paths and provenance in a small metadata block or JSON alongside the outputs.

GNNePCSAFT is a published, pre-trained model (`pip install gnnepcsaft`) that predicts *e*PC-SAFT parameters directly from SMILES using a GNN trained on literature databases. It does not go through the project's model registry — call it via its own inference API and normalize the output to the same `dict[str, np.ndarray]` format. **Important: GNNePCSAFT predicts ePC-SAFT parameters**, which include an association term for polar/hydrogen-bonding molecules. For non-associating fluorinated hydrocarbons (HFOs/HCFOs), the non-associating ePC-SAFT parameters (m, σ, ε/k) are numerically close but not identical to standard PC-SAFT values. Document this in the provenance artifact and treat any systematic offset as a known modeling difference, not a bug.

```python
import model.chemprop_model  # register chemprop
import model.xgb  # register xgboost
from model.registry import get_model, _compute_features
from model.data.load import TARGETS  # ["m", "sigma", "epsilon_k"]

def predict_gnnepcsaft(smiles_list: list[str]) -> dict[str, np.ndarray]:
    """Return {target: np.ndarray} from the pre-trained GNNePCSAFT model.

    GNNePCSAFT predicts ePC-SAFT parameters. For non-associating molecules
    (HFOs/HCFOs) the association term is zero and the remaining three
    parameters (m, sigma, epsilon_k) are directly comparable to PC-SAFT.
    """
    from gnnepcsaft.predict import predict_pcsaft  # adjust import path per installed version

    preds = predict_pcsaft(smiles_list)  # returns array or dict; normalize below
    # Normalize to {target: np.ndarray} matching project convention
    # Check gnnepcsaft docs for exact output format and column order
    return {
        "m": np.array(preds[:, 0]),
        "sigma": np.array(preds[:, 1]),
        "epsilon_k": np.array(preds[:, 2]),
    }

def predict_all_models(smiles_list):
    """Return {model_name: {target: np.ndarray}} for all five models."""
    results = {}

    # RF and GNN: SMILES → dict
    for name in ["rf", "gnn"]:
        model = get_model(name)
        model.load()
        results[name] = model.predict(smiles_list)

    # XGBoost: SMILES → features → array → dict
    xgb = get_model("xgboost")
    xgb.load()
    X = _compute_features(smiles_list)
    preds = xgb.predict(X)  # (n, 3) array
    results["xgboost"] = {t: preds[:, i] for i, t in enumerate(TARGETS)}

    # chemprop: SMILES → array → dict
    cp = get_model("chemprop")
    cp.load()
    preds = cp.predict(smiles_list)  # (n, 3) array
    results["chemprop"] = {t: preds[:, i] for i, t in enumerate(TARGETS)}

    # GNNePCSAFT: pre-trained external benchmark
    results["gnnepcsaft"] = predict_gnnepcsaft(smiles_list)

    return results
```

If any of the four project-trained models fails to load because the saved artifact is missing or provenance is unclear, stop and document the issue. Do not make a final model-selection claim from an incomplete candidate set. GNNePCSAFT does not carry a saved artifact path — its parameters are fixed at install time — but its package version must be recorded in the provenance metadata.

**Candidate scope note — ChemBERTa-2**: ChemBERTa-2 (`DeepChem/ChemBERTa-77M-MTR`) was evaluated for inclusion in this step and declined. The project already fine-tuned ChemBERTa-1 in Step 04 (R² = 0.27 on ε/k, 185x slower than RF). ChemBERTa-2's multi-task pre-training targets molecular property prediction and could yield a marginal improvement, but the fundamental constraint is unchanged: ~1,800 fine-tuning examples is insufficient for any SMILES transformer to surpass RDKit + Morgan fingerprints at this scale. Adding it would require hours of fine-tuning compute for a model that is not a deployment candidate. The existing ChemBERTa-1 result already represents the SMILES-transformer angle in the comparison.

### 48.2 Fluorinated external validation (Tier 1)

For each model, predict PC-SAFT parameters for the 15 fluorinated compounds and compare to literature values. This tier must be reported with uncertainty and with failure accounting because `n=15` is too small for point estimates alone.

**48.2a — Parameter accuracy**

For each model and each target (`m`, `sigma`, `epsilon_k`), compute:
- MAE, RMSE, R²
- Bootstrap 95% CI for MAE on the fluorinated set
- Per-compound signed error (for parity plots)

Present results as:

| Model       | m MAE | sigma MAE | epsilon/k MAE (K) | epsilon/k R² |
|-------------|-------|-----------|-------------------|--------------|
| RF          |       |           |                   |              |
| XGBoost     |       |           |                   |              |
| chemprop    |       |           |                   |              |
| GNN         |       |           |                   |              |
| GNNePCSAFT  |       |           |                   |              |

**48.2b — Boiling point accuracy**

For each model's predicted (`m`, `sigma`, `epsilon_k`), compute boiling point via `compute_boiling_point()` from `screening/hfo_screening.py`. Compare to experimental values in the validation set's `T_b_experimental_K` column.

For GNNePCSAFT, first check whether `compute_boiling_point()` (which calls the project's EOS backend) is compatible with ePC-SAFT parameters or requires a separate call. GNNePCSAFT integrates natively with the FeOs thermodynamic library; if the project's EOS backend is also FeOs-based, the same call should work. If not, document the EOS used and treat the boiling point comparison as approximate.

Report:
- Boiling point MAE (K)
- Boiling point RMSE (K)
- Bootstrap 95% CI for BP MAE and BP RMSE
- EOS convergence rate (fraction of compounds where the EOS solver finds a valid boiling point)
- `n_converged / n_total`
- Per-compound boiling point errors (for parity plots)

| Model       | BP MAE (K) | BP RMSE (K) | EOS convergence |
|-------------|-----------|-------------|-----------------|
| RF          |           |             |                 |
| XGBoost     |           |             |                 |
| chemprop    |           |             |                 |
| GNN         |           |             |                 |
| GNNePCSAFT  |           |             |                 |

The boiling point MAE is the single most important number in this step. A model with >50 K BP MAE is not useful for screening (the filter window is ~35 K). Treat non-converged EOS cases as deployment failures and discuss them explicitly; do not hide them behind MAE computed only on the converged subset.

**48.2c — Per-compound analysis**

Save a per-compound table with all model predictions, errors, and boiling points. This supports detailed parity plots and identifies whether failures are systemic or concentrated on specific compounds (e.g., heavily fluorinated vs lightly fluorinated, HFOs vs non-HFO refrigerants).

**48.2d — Paired model comparison on deployment errors**

For each pair of competitive models, compare absolute boiling point errors on the same 15 compounds:
- Bootstrap the paired difference in BP MAE (`MAE_A - MAE_B`)
- Report whether the 95% CI excludes 0
- Optionally add a paired permutation test or Wilcoxon signed-rank test on per-compound absolute errors

Tier 1 should not rely on an arbitrary "`2x better`" heuristic alone.

**48.2e — Downstream screening behavior**

The deployment task is screening, not only parameter regression. If the project will be presented as a **screening-ready system** rather than only a parameter-regression benchmark, this subsection is required rather than optional. Add a lightweight downstream check using the existing HFO screening pipeline:
- Rank correlation (Spearman) between candidate lists produced by each model
- Top-K overlap for the highest-ranked fluorinated candidates
- Pass/fail agreement for the key screening filters

**48.2f — Predictive Uncertainty Quantification (UQ) Calibration**

For models that provide built-in predictive uncertainty (e.g., RF via tree variance, chemprop via ensembles), calculate the **coverage probability** on the fluorinated set.
- For each molecule, compute the 95% prediction interval.
- Report the fraction of molecules where the true experimental boiling point falls within the 95% prediction interval.
- An MLE reviewer will expect to see whether the model's self-reported confidence is actually calibrated, or if it is overconfident/underconfident.

**48.2g — Inference Throughput Benchmark**

For a screening pipeline, throughput is a hard engineering constraint. A model that is 1% more accurate but 100x slower may not be viable for screening millions of candidates.
- For each model, measure the inference time on a batch of 1,000 molecules (or the full Esper test set).
- Include the time required for feature computation (e.g., RDKit + Morgan for RF/XGBoost, graph construction for GNNs).
- Report **throughput (molecules per second)** on a standard CPU (and GPU if applicable for GNN/chemprop).
- This provides a Pareto frontier (Accuracy vs. Latency) to justify the final model selection.

If this downstream screening comparison is deferred, state explicitly that the step supports **parameter-validation claims only**, not a full screening-readiness claim.

### 48.3 Esper holdout with paired uncertainty analysis (Tier 2)

Load the Esper test set from `model/saved/test_set.csv`. Verify it contains the standard 361-molecule Esper split and that the compared models are valid on that exact cohort. If the saved `test_set.csv` reflects a different corpus or ambiguous provenance, regenerate one frozen Esper split using `split_data()` from `model/data/load.py` with the standard parameters (`test_size=0.2`, `random_state=42`, `stratify_bins=5`) and evaluate every candidate model against that same frozen split.

For each model, predict on the test set and compute point metrics with bootstrap 95% CIs:

```python
from model.uncertainty import bootstrap_metric_ci
from sklearn.metrics import r2_score, mean_absolute_error

for target in TARGETS:
    for metric_name, metric_fn in [("r2", r2_score), ("mae", mean_absolute_error)]:
        ci = bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95)
```

Present as:

| Model       | epsilon/k R² [95% CI]          | epsilon/k MAE [95% CI]          |
|-------------|-------------------------------|---------------------------------|
| RF          | 0.33 [lo, hi]                 | 26.8 [lo, hi]                   |
| XGBoost     | 0.33 [lo, hi]                 | 27.0 [lo, hi]                   |
| chemprop    | 0.39 [lo, hi]                 | 26.8 [lo, hi]                   |
| GNN         | 0.41 [lo, hi]                 | 28.2 [lo, hi]                   |
| GNNePCSAFT  | [compute]                     | [compute]                       |

**Note on GNNePCSAFT Tier 2 interpretation**: GNNePCSAFT was trained on external PC-SAFT databases (not the Esper corpus). Its Esper holdout performance is therefore a test of generalization to a novel compound set, not a standard in-distribution test-set result. Label it clearly as "external benchmark — out-of-distribution on Esper" in the report. A high R² would suggest broad generalization; a low R² would not disqualify it if Tier 1 (the deployment domain) is strong.

Then add **paired comparisons** between the serious candidates (RF vs XGBoost, RF vs chemprop, XGBoost vs chemprop, and optionally each vs GNN and GNNePCSAFT):
- Bootstrap the metric difference on the same resampled molecules
- Report `delta_mae`, `delta_r2`, and their 95% CIs
- Avoid using overlap / non-overlap of marginal CIs as the significance rule

**UQ Calibration on Esper (Tier 2b)**
- For RF (and chemprop if ensembled), compute the 95% prediction interval for `epsilon_k` on the 361 Esper holdout molecules.
- Report the **coverage probability** (fraction of molecules where the true value falls within the interval).
- Report the **average interval width**. A model with 100% coverage but infinitely wide intervals is useless; calibration requires narrow intervals that still achieve ~95% coverage.

If the paired-difference CI includes 0, treat the models as statistically indistinguishable on that metric and let Tier 1 take precedence.

### 48.4 Cross-validated robustness check for tree models (Tier 3)

Adapt `compute_rf_cv_q2()` from `model/evaluate.py` (line 586) to also evaluate XGBoost with 5-fold CV:

```python
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# RF
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# XGBoost (use best hyperparameters from Step 15)
xgb = MultiOutputRegressor(XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
))
```

For each fold, train on the training fold, predict on the held-out fold, and compute R² per target. Report mean and std across folds:

| Model   | Q² m           | Q² sigma       | Q² epsilon/k   |
|---------|----------------|----------------|-----------------|
| RF      | mean ± std     | mean ± std     | mean ± std      |
| XGBoost | mean ± std     | mean ± std     | mean ± std      |

Chemprop and GNN Q² requires retraining from scratch per fold (GPU-dependent, ~20 min/fold for chemprop, longer for GNN). This is optional. If skipped, document why and note that Tier 3 is a **tree-only robustness analysis**, not a cross-model decision tier.

If XGBoost is included, load or record the actual saved hyperparameters used in Step 15 rather than inserting new hard-coded values unless retraining is intentional and documented.

### 48.5 Final model selection decision

Combine results from all three tiers into a unified decision.

**Decision criteria, applied in order:**

1. **Tier 1 first**: Prefer the model with the best fluorinated boiling point performance, provided its deployment-tier uncertainty and convergence behavior are acceptable.
2. **Require a predeclared Tier 1 win condition**: Treat Tier 1 as decisive only if **all** of the following hold:
   - the finalists were evaluated on the same frozen comparison cohort (otherwise the decision is provisional only)
   - the paired boiling-point error CI favors one model over the other(s)
   - the winning model improves boiling-point MAE by at least **5 K** over the closest competitor, which is a meaningful fraction of the ~35 K screening window
   - the winning model does not have worse EOS convergence than the closest competitor
3. **Automatic tie rule**: If any Tier 1 win condition above fails, treat Tier 1 as unable to separate the candidates at `n=15` and defer automatically to Tier 2.
4. **Tier 2 as tiebreaker**: If Tier 1 is too close to separate candidates, use Tier 2 paired-difference results on epsilon/k MAE and R².
5. **Tier 3 as supporting evidence only**: Use CV Q² to support confidence in tree-model robustness, not to overrule Tier 1.
6. **Practical considerations only after statistical tie**: If the competitive models remain tied after Tier 1 and Tier 2, choose based on: (a) inference speed, (b) uncertainty calibration quality, (c) interpretability, (d) deployment simplicity.

Document the decision with explicit evidence and reasoning, following the format of Step 38d section 38d.5 in `scripts/step38d_rf_vs_gnn_comparison.py`.

**Possible outcomes:**
- RF confirmed as best → current pipeline unchanged; model selection is now rigorous
- XGBoost wins → swap RF for XGBoost in serving and screening; update `serving/model_loader.py`
- chemprop wins → swap to chemprop; note heavier dependency (torch, chemprop)
- GNN wins on the actual deployment tier and survives external validation → justify why the prior fluorinated failure mode no longer applies
- GNNePCSAFT wins on Tier 1 → a strong finding, but note it is a pre-trained external model, not a deployable artifact trained on the project's data; discuss trade-offs (no retraining path on proprietary data, ePC-SAFT vs PC-SAFT alignment, black-box provenance) before recommending it for production
- Tie → document the tie; recommend the simpler model (RF or XGBoost) by parsimony

### 48.6 Create the driver script

Create `scripts/step48_model_selection_validation.py` with subsections:
- `section_48_1_load()` — verify provenance, import/register models, load validation data
- `section_48_2_fluorinated_validation()` — Tier 1
- `section_48_3_esper_bootstrap()` — Tier 2
- `section_48_4_cv_q2()` — Tier 3
- `section_48_5_decision()` — model selection
- `section_48_6_figures()` — all plots
- `section_48_7_report()` — write report markdown

Also write a small provenance artifact such as `model/saved/step48_model_selection_metadata.json` containing:
- model artifact paths
- dataset identifiers and counts
- test split hash or sorted SMILES fingerprint
- feature config / feature names hash for descriptor-based models
- software versions if available

### 48.7 Generate figures

Save all figures to `figures/48_model_selection_validation/`.

Required:

1. **`fluorinated_parity_all_models.png`** — parity plots (predicted vs literature) for epsilon/k on the fluorinated set, one panel per model (5 panels: RF, XGBoost, chemprop, GNN, GNNePCSAFT)
2. **`fluorinated_boiling_point_parity.png`** — predicted vs experimental boiling point for all models, overlaid or paneled (include GNNePCSAFT)
3. **`esper_bootstrap_ci_comparison.png`** — forest plot or grouped bar chart showing R² and MAE with 95% CIs for each model on each target (include GNNePCSAFT, labeled "external benchmark")
4. **`paired_difference_forest.png`** — forest plot of paired metric differences between the top candidate models
5. **`tier_summary_heatmap.png`** — heatmap with rows = models (5 rows including GNNePCSAFT), columns = key metrics across all tiers (fluorinated BP MAE, fluorinated epsilon/k MAE, Esper epsilon/k R², Q² epsilon/k), color-coded by rank. Annotate GNNePCSAFT row as "external benchmark" to distinguish it from project-trained models.

Optional:

6. **`per_compound_fluorinated_errors.png`** — grouped bar chart showing per-compound BP errors for each model, sorted by fluorination degree
7. **`esper_test_tanimoto_distribution.png`** — histogram of the maximum Tanimoto similarity to the training set for each molecule in the Esper test set, quantifying the structural leakage of the random split.
8. **`screening_rank_agreement.png`** — rank correlation / top-K overlap summary if downstream screening analysis is performed

### 48.8 Write the report

Write `docs/reports/48_model_selection_validation.md` following the standard report template:

1. Title: `# Results Report: Model Selection Validation (5-Model Comparison)`
2. Summary paragraph
3. **Candidate scope**: A short paragraph naming the five models evaluated (RF, XGBoost, chemprop, GNN, GNNePCSAFT) and explaining why ChemBERTa-2 was considered and declined — existing ChemBERTa-1 result already covers the SMILES-transformer angle; the fundamental bottleneck (~1,800 fine-tuning examples) is unchanged for v2; it is not a deployment candidate. Also state that GNNePCSAFT is an external pre-trained benchmark, not a project-trained model, and that it predicts ePC-SAFT parameters rather than PC-SAFT.
4. Tier 1 results table and interpretation (5 models, including GNNePCSAFT labeled as external benchmark)
5. Tier 1 paired BP error comparison and EOS convergence discussion
6. Tier 2 results table with CIs and paired significance assessment (GNNePCSAFT labeled "out-of-distribution on Esper")
7. Tier 3 results (if computed) or note that it was deferred (GNNePCSAFT excluded from Tier 3 — no retraining path)
8. Unified comparison table across all tiers, including Inference Throughput (molecules/sec)
9. Model selection decision with explicit criteria and evidence
10. Provenance / reproducibility note (what exact artifacts and split were compared; GNNePCSAFT version recorded)
11. Generalization Caveat: Quantitative assessment of test set leakage (Tanimoto similarity distribution)
12. Key findings (bullets)
13. Figures
14. Deviations from this guide
15. Readiness check

### 48.9 Update supplementary information

Update `supplementary_information/model_comparison.md`:
- Add XGBoost, chemprop, and GNNePCSAFT rows to the fluorinated validation table
- Update the "Why RF Was Chosen" section with the new evidence (whether confirming or revising the decision)
- Add a "Validation Hierarchy" section explaining the three-tier framework
- Add GNNePCSAFT to the "All Models Evaluated" table with a note that it is an external pre-trained benchmark and that its Esper holdout performance is out-of-distribution
- Add a sentence on ChemBERTa-2 in the "What was excluded and why" section
- If SVM remains excluded from deployment comparison, state explicitly that Step 47 was a representation-bottleneck benchmark rather than a serious production candidate evaluation

## Key Files

| File | Purpose |
|------|---------|
| `model/saved/gnn_fluorinated_validation_set.csv` | 15-compound fluorinated validation set |
| `model/saved/test_set.csv` | Esper holdout test set |
| `model/registry.py` | Model loading (`get_model`, `_compute_features`) |
| `model/evaluate.py` | Evaluation harness, `_compute_metrics`, `bootstrap_metric_ci` |
| `model/xgb/xgb_model.py` | XGBoost model (takes feature matrix, not SMILES) |
| `model/chemprop_model/chemprop_wrapper.py` | chemprop model (takes SMILES, returns array) |
| `screening/hfo_screening.py` | `compute_boiling_point()` for Tier 1 |
| `model/data/load.py` | `split_data()`, `TARGETS` |
| `scripts/step38d_rf_vs_gnn_comparison.py` | Pattern for decision logic and reporting |
| `gnnepcsaft` (PyPI package) | External pre-trained ePC-SAFT GNN benchmark; `pip install gnnepcsaft` |

## Artifacts

| File | Description |
|------|-------------|
| `scripts/step48_model_selection_validation.py` | Driver script |
| `docs/reports/48_model_selection_validation.md` | Results report |
| `figures/48_model_selection_validation/` | All figures |
| `model/saved/step48_model_selection.json` | Structured decision + evidence |
| `model/saved/step48_model_selection_metadata.json` | Evaluation provenance: artifacts, split, feature schema, GNNePCSAFT version |
| `model/saved/step48_fluorinated_all_models.csv` | Per-compound fluorinated predictions (5 models) |
| `model/saved/step48_esper_bootstrap.csv` | Bootstrap CI results (5 models) |

## Success Criteria

- [ ] All five models (RF, XGBoost, chemprop, GNN, GNNePCSAFT) evaluated on fluorinated validation set with parameter and boiling point metrics
- [ ] GNNePCSAFT labeled as "external pre-trained benchmark" (ePC-SAFT, not PC-SAFT) in all tables and figures
- [ ] ChemBERTa-2 "considered and declined" rationale documented in the report's Candidate Scope section
- [ ] Esper holdout metrics include bootstrap 95% CIs for at least R² and MAE on all targets
- [ ] GNNePCSAFT Esper holdout results labeled "out-of-distribution" in the report
- [ ] Systematic inference throughput benchmark (molecules/sec) included in the comparison table
- [ ] Predictive UQ calibration (coverage probability) reported for RF on the Esper holdout
- [ ] Serious candidate comparisons use paired metric-difference uncertainty rather than CI-overlap heuristics
- [ ] Data leakage audit: no fluorinated validation compound overlaps with the Esper training set (documented in provenance metadata)
- [ ] Near-duplicate audit documented: max training-set similarity recorded for each fluorinated validation compound
- [ ] Tanimoto similarity distribution between the Esper test and train sets plotted to quantify the random split leakage
- [ ] Evaluation provenance (split, artifacts, feature schema, GNNePCSAFT package version) is documented well enough to reproduce the comparison
- [ ] Any final deployment claim is based on one frozen comparison cohort; otherwise the report labels the decision as provisional
- [ ] Q² computed for RF and XGBoost (chemprop/GNN/GNNePCSAFT optional, documented if skipped; GNNePCSAFT cannot participate in CV without retraining — exclude and note)
- [ ] Unified comparison table spans all three tiers
- [ ] Model selection decision documented with explicit tier-by-tier evidence
- [ ] Downstream screening agreement metrics are reported if the project is being presented as a screening-ready system
- [ ] At least 5 figures saved to `figures/48_model_selection_validation/`
- [ ] Report written at `docs/reports/48_model_selection_validation.md`
- [ ] `supplementary_information/model_comparison.md` updated with new results including GNNePCSAFT
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- Every model that was competitive on the Esper holdout has been tested on the deployment domain (fluorinated compounds + boiling point)
- The model selection decision is supported by deployment-tier evidence plus paired uncertainty analysis, not just point estimates
- Anyone rerunning the step can identify exactly which artifacts, split, and feature schema were compared
- If the decision changes from RF, the serving configuration has been updated accordingly
- The project narrative about model selection can be stated in one sentence with data to back it up

## Budget

60-120 minutes. If the saved artifacts are provenance-compatible, this is mostly a prediction-and-comparison step. If the split or feature provenance is ambiguous, budget extra time to freeze a comparison cohort or retrain the directly competing models on one frozen setup. Tier 1 and paired-comparison logic are the highest priority; Tier 3 remains optional if time is short.
