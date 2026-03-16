# Step 37: GNN HFO Screening Re-Run

## Objective

Re-run the full HFO-centric screening workflow from Step 25 using the promoted GNN
model from Step 31 instead of the RF baseline. Preserve the original RF artifacts for
comparison, include all seven HFO filters now present in `apply_hfo_filters()`, and
regenerate the ranked fluorinated candidate list under a model that is more credible
for fluorinated screening chemistry.

This step also adopts the 5:2:1 (epsilon_k : sigma : m) parameter weighting recommended
by Step 9's thermodynamic validation, replacing the original 3:1:1 default.

This step is a **model-promotion rerun**, not the final decision gate for experimental
candidate selection. The outputs remain screening hypotheses until the validation,
uncertainty, EOS, and safety follow-up steps are complete.

## Motivation

Step 25 produced the first HFO-centric ranked library, but it was driven by an RF model
trained on Esper-only data and explicitly framed as an out-of-domain exploratory screen.
Step 31 materially changed the scientific picture: the unified GNN achieved
R^2 = 0.86 / 0.73 / 0.91 on fluorinated molecules and was recommended as the default
screening model for novel fluorinated chemistry.

Step 32 then strengthened the screening logic by adding fluorine mass fraction and
reactive-site filters, but those rules were not applied to a full rerun of the HFO
library. Step 37 closes both gaps at once:

1. Replace RF predictions with GNN predictions for the HFO screen
2. Recompute the filter funnel with the full seven-filter cascade
3. Preserve RF outputs for side-by-side comparison instead of silently overwriting them

### Parameter weighting change

Step 9 (thermodynamic validation) showed that epsilon_k and sigma dominate vapor pressure
behavior, and recommended revising the parameter weighting from the original 3:1:1
(epsilon_k : sigma : m) established in PLAN.md Q3 to **5:2:1**. This step adopts that
recommendation. The HFO distance metric now uses weights (5, 2, 1) for
(epsilon_k, sigma, m) respectively, which better reflects the physical importance of
each parameter for thermodynamic property prediction. The Step 25 RF results used 3:1:1,
so the ranking comparison between RF and GNN also captures the weighting change. The
report must discuss both effects.

## Dependencies

- Step 25 completed: `scripts/step25_hfo_centric_screening.py` and its artifacts exist
- Step 31 completed: `model/saved/gnn_pcsaft.pt` exists and `model.registry.get_model("gnn")` loads
- Step 32 completed: `screening/hfo_screening.py::apply_hfo_filters()` includes the fluorination filters
- Step 36 completed: GNN is available in the `pcsaft_predict` distributable package
- `screening/generate.py::generate_systematic_candidates()` exists
- `model/thermodynamic.py::compute_henrys_constant()` exists

## Implementation Guide

### 37.1 Create `scripts/step37_gnn_hfo_screening.py`

Create a new screening script rather than modifying `scripts/step25_hfo_centric_screening.py`
in place. Reuse the same high-level stages:

1. Generate candidates with the same enumeration settings as Step 25
2. Predict `(m, sigma, epsilon_k)` with the GNN through `model.registry.get_model("gnn")`
3. Compute boiling points via `screening.hfo_screening.batch_boiling_points()`
4. Compute SA scores
5. Apply `apply_hfo_filters()` with all seven active filters
6. Rank by HFO proximity using **5:2:1 weighting** (epsilon_k : sigma : m)
7. Compute Henry's constant for the top 50 surviving candidates
8. Save results to new output paths

Use the registry interface rather than `model.predict.predict_pcsaft()` so the script is
explicitly model-aware and consistent with the promoted screening path.

### 37.2 Adopt 5:2:1 parameter weighting

Update the HFO distance calculation to use weights (5, 2, 1) for (epsilon_k, sigma, m).
`screening.hfo_screening.hfo_parameter_distance()` already accepts a `weights` dict with
keys `m`, `sigma`, `epsilon_k` (default: all 1.0). Pass:

```python
weights={"epsilon_k": 5.0, "sigma": 2.0, "m": 1.0}
```

Do not change the function's default weights, which would silently alter any other caller.
The step script should pass the weights explicitly.

The script and report must:

- State the new weighting explicitly
- Cite Step 9's finding that Spearman rho = 0.372 (p = 1.34e-18) between parameter and
  property distance, and that 5:2:1 was the recommended revision
- Note that the RF-vs-GNN comparison includes both the model change and the weighting
  change; the report should discuss both effects. To partially disentangle them,
  optionally compute GNN rankings with the old 3:1:1 weights as well, so the report can
  attribute ranking shifts to the model vs. the weighting revision separately

### 37.3 Warning banner and framing

Retain the Step 25 warning banner, but update it for the GNN context:

- Cite Step 31 fluorinated metrics rather than RF Esper-only metrics
- State that the chemistry is still partially out-of-domain for final decision-making
- Explicitly note that the outputs are suitable for prioritization, not direct material selection

The report and stdout should say that Step 37 improves the **model basis** of the screen,
but does not by itself validate the final shortlist.

### 37.4 Preserve RF outputs and save new GNN artifacts

Do not overwrite the Step 25 RF artifacts. Save the GNN rerun to:

- `screening/results/hfo_gnn_ranked.csv`
- `screening/results/hfo_gnn_filter_funnel.csv`

The ranked CSV should include all key columns from Step 25 plus the new fluorination
metadata already available in the pipeline:

```text
smiles, m, sigma, epsilon_k, boiling_point_K, sa_score, n_fluorine,
backbone, cf3_count, hfo_distance, H_Pa, H_ratio
```

If practical, also add:

- `model_type` = `gnn`
- `filter_passed` boolean
- `filter_reason` or equivalent annotation for failures if tracked cheaply

### 37.5 Regenerate the full seven-filter funnel

The funnel CSV and report must track candidate counts through the full filter cascade.
The seven filters (matching `apply_hfo_filters()`) are:

1. Boiling point [288, 323] K
2. Has C=C bond
3. No chlorine
4. F count >= 2
5. SA score <= 4.5
6. Fluorine mass fraction >= 65%
7. No reactive fluorination sites

The funnel should also record the initial candidate count before any filter and the
final count after all filters pass, for a total of 9 rows in the funnel CSV.

Also record:

- Count removed by fluorine mass fraction filter
- Count removed by reactive site filter
- Fraction of survivors with `cf3_count >= 1`

### 37.6 Filter threshold sensitivity analysis

For the three most consequential filter thresholds, compute how many candidates
enter/leave the passing set under threshold perturbation:

1. **Boiling point window**: widen by +/- 5 K (i.e., [283, 328] K and [293, 318] K)
2. **SA score**: vary from 4.0 to 5.0 in steps of 0.5
3. **Fluorine mass fraction**: vary from 0.55 to 0.75 in steps of 0.05

Report:

- How many new candidates enter or exit the passing set under each perturbation
- Whether the top-10 candidates are robust to these threshold changes or are marginal
- A summary statement on filter sensitivity (e.g., "the T_b filter is the most selective;
  widening by 5 K adds N candidates, of which M are in the top 20 by distance")

This analysis strengthens the screening methodology by quantifying how much the results
depend on specific cutoff choices.

### 37.7 Keep the HFO reference and adopt the new ranking basis

Use `screening.hfo_screening.HFO_1336MZZ` as the reference compound. Apply the
5:2:1 weighting as the new default ranking metric.

The report must explicitly say:

- Step 37 is a **model-promotion and weighting-revision rerun**
- The weighting change is motivated by Step 9's thermodynamic validation
- The purpose is to quantify how much the candidate pool changes under both the model
  promotion and the revised weighting
- Further ranking changes (property-space re-ranking) are deferred to Step 40

### 37.8 Generate comparison figures

Save figures to `figures/37_gnn_hfo_screening/`.

Required figures:

1. `screening_funnel_gnn.png`
   - Seven-filter funnel for the GNN rerun
2. `rf_vs_gnn_parameter_histograms.png`
   - Side-by-side histograms of predicted `m`, `sigma`, `epsilon_k` on the shared candidate pool
3. `rf_vs_gnn_rank_shift.png`
   - Comparison of RF and GNN rankings for the top 20-50 overlapping candidates
4. `boiling_point_vs_hfo_distance_gnn.png`
   - Boiling point vs HFO distance, colored by `cf3_count` or fluorine count
5. `rf_vs_gnn_top20_comparison.png`
   - Compact visual table or bump-chart showing how the top candidates change between models
6. `filter_sensitivity.png`
   - Summary of how candidate counts change under threshold perturbation

### 37.9 Write the report

Write `docs/reports/37_gnn_hfo_screening.md`.

Follow the normal report template from `CLAUDE.md`, but add the following emphasis:

- Model-promotion context from Step 31
- 5:2:1 weighting rationale from Step 9
- Direct comparison to Step 25 RF outputs
- Impact of the Step 32 filters on the final funnel
- Filter sensitivity analysis results
- Explicit note that this is still not the final candidate decision gate

Required comparison sections:

1. **GNN context**: summarize the fluorinated-slice metrics from Step 31
2. **Weighting change**: explain the 5:2:1 revision and its effect on rankings
3. **RF vs GNN funnel**: how the pass counts change between models
4. **RF vs GNN top-20**: which candidates persist, rise, or disappear
5. **Step 32 filter impact**: how many candidates are excluded by each fluorination heuristic
6. **Filter sensitivity**: how robust the pass set is to threshold perturbation
7. **Limitations**: source mismatch caveat, lack of uncertainty propagation, no EOS re-ranking yet

## Artifacts

| File | Description |
|------|-------------|
| `scripts/step37_gnn_hfo_screening.py` | GNN-based HFO screening rerun |
| `screening/results/hfo_gnn_ranked.csv` | Ranked GNN candidate list |
| `screening/results/hfo_gnn_filter_funnel.csv` | Seven-stage GNN funnel |
| `figures/37_gnn_hfo_screening/` | RF-vs-GNN comparison and sensitivity figures |
| `docs/reports/37_gnn_hfo_screening.md` | Step report |

## Success Criteria

- [ ] `scripts/step37_gnn_hfo_screening.py` runs end-to-end without errors
- [ ] GNN predictions are loaded through `model.registry.get_model("gnn")`
- [ ] HFO distance uses `weights={"epsilon_k": 5, "sigma": 2, "m": 1}`, explicitly documented
- [ ] `screening/results/hfo_gnn_ranked.csv` is saved without overwriting Step 25 RF outputs
- [ ] `screening/results/hfo_gnn_filter_funnel.csv` includes the full seven-filter cascade
- [ ] Filter sensitivity analysis is reported for T_b, SA, and F mass fraction thresholds
- [ ] Comparison figures are saved to `figures/37_gnn_hfo_screening/`
- [ ] Report compares GNN outputs to Step 25 RF results and Step 31 promotion rationale
- [ ] Report discusses the combined effect of model change and weighting revision
- [ ] Report explicitly frames the output as a screening rerun, not a final shortlist
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- The GNN rerun exists as a reproducible set of artifacts separate from the RF baseline
- The seven-stage funnel is documented clearly enough to support later shortlist review
- The 5:2:1 weighting is adopted and its effect is quantified
- Filter sensitivity has been analyzed
- The top-ranked GNN candidates are available for follow-on validation, uncertainty, and EOS steps
- The report makes no claim that Step 37 alone is sufficient for experimental selection

## Budget

25 minutes for scripting, sensitivity analysis, figure regeneration, and report writing.
No model training is required; this is a pipeline rerun plus comparative analysis.
