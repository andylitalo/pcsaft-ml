# Step 45: Fluorinated RF Comparison and Cross-Dataset Context

## Objective

Take the **two RF models trained in Step 44** and stress-test them on the slices that
matter most for deployment: fluorinated compounds, halogenated chemistry, and
uncertainty-sensitive subsets. The primary question is:

**Does the Combined RF outperform, match, or underperform the Esper-only RF on
fluorinated and out-of-source molecules, and are its uncertainty estimates at least as
trustworthy?**

This step is a **follow-on evaluation step**, not a fresh standalone data-audit. The core
artifacts from Step 44 must already exist. SPT-vs-Esper and ML-SAFT overlap analyses are
included only as supporting diagnostics to explain the RF outcome.

## Motivation

Step 44 answers the top-level question about whether ML-SAFT should enter the RF training
corpus. Step 45 asks where that answer comes from.

This matters because the project’s highest-stakes evaluation slices are not the pooled
dataset averages. They are:

- fluorinated molecules relevant to HFO screening,
- chemically shifted subsets where uncertainty quality matters,
- and out-of-source molecules where the combined RF might add useful coverage.

If the Combined RF helps, the gain is most likely to appear on these slices. If it hurts,
the failure is also likely to appear here first, often as miscalibrated uncertainty before
it shows up as an obvious R² collapse.

## Dependencies

- Step 44 completed:
  - `scripts/step44_rf_combined_retrain.py`
  - `model/saved/step44_rf_combined_predictions.csv`
  - `model/saved/step44_rf_combined_metrics.json`
- `model/data/esper_pcsaft.csv`
- `model/data/mlsaft_pcsaft.csv`
- `model/data/spt_pcsaft.csv` for supporting context only
- `model/saved/gnn_fluorinated_validation_set.csv`
- RDKit for halogenation and fluorination-degree annotations
- Optional EOS utilities from the Step 38 workflow if boiling-point back-calculation is
  already implemented

## Implementation Guide

### 45.1 Load the Step 44 prediction tables and freeze the comparison

Do not retrain with new seeds or altered hyperparameters in this step. Reuse the exact two
RF models defined in Step 44:

1. **Esper-only RF**
2. **Combined RF**

All Step 45 analyses must be derived from those models and from the Step 44 split
definitions. This preserves a clean experimental chain:

- Step 44: train both RFs and establish the global result
- Step 45: explain that result on targeted slices

### 45.2 Head-to-head fluorinated validation

Evaluate both RFs on the external fluorinated validation set from Step 38 if available:

`model/saved/gnn_fluorinated_validation_set.csv`

For every molecule, save:

- true literature parameters if available
- RF predictions from both models
- RF standard deviations from both models
- absolute errors
- optional boiling-point predictions if the EOS workflow is already available

For every target, report:

| Model | MAE | RMSE | Mean std | 1-sigma coverage | 2-sigma coverage |
|-------|-----|------|----------|------------------|------------------|
| Esper-only RF | | | | | |
| Combined RF | | | | | |

If boiling-point ground truth is available, also report:

| Model | BP MAE (K) | BP RMSE (K) | EOS convergence |
|-------|------------|-------------|-----------------|
| Esper-only RF | | | |
| Combined RF | | | |

The external fluorinated set is the most important secondary decision point after Step 44.

### 45.3 Compare in-domain fluorinated and halogenated slices

Using the Step 44 benchmark predictions, create the following slices on the combined test
set and evaluate both RFs on each:

1. **Fluorinated**
2. **Chlorinated**
3. **Other halogenated or mixed-halogenated**
4. **Non-halogenated**

At minimum, report point metrics and uncertainty metrics for:

- fluorinated vs non-fluorinated
- halogenated vs non-halogenated

If sample counts allow, add fluorination-degree bins:

- light fluorination: F fraction < 0.2
- moderate fluorination: 0.2 ≤ F fraction < 0.4
- heavy fluorination: F fraction ≥ 0.4

The point is not to maximize subgroup count. The point is to determine whether the
Combined RF changes performance in the actual deployment chemistry.

### 45.4 Quantify gain/loss by evaluation slice

Compute explicit deltas between the two RFs:

```text
delta_MAE = MAE_combined_rf - MAE_esper_rf
delta_RMSE = RMSE_combined_rf - RMSE_esper_rf
delta_R2 = R2_combined_rf - R2_esper_rf
delta_coverage_1sigma = cov1_combined_rf - cov1_esper_rf
delta_coverage_2sigma = cov2_combined_rf - cov2_esper_rf
```

Required slices:

- Esper benchmark test set
- Combined benchmark test set
- Esper subset of combined benchmark
- ML-SAFT-only subset of combined benchmark
- Fluorinated external validation
- Fluorinated combined-test slice

This step must make it obvious where the Combined RF helps and where it hurts.

### 45.5 Link RF behavior to cross-dataset disagreement

Only after the head-to-head RF comparison is complete, use matched-pair dataset analysis to
explain the pattern.

For Esper/ML-SAFT overlap molecules:

1. match by InChI
2. compute signed and absolute deltas for `m`, `sigma`, `epsilon_k`
3. annotate chemical families:
   - fluorinated / chlorinated / other halogenated
   - associating vs non-associating
   - fluorination degree if relevant
4. test whether large Esper/ML-SAFT disagreement predicts:
   - larger Combined-RF error
   - larger Combined-vs-Esper RF delta
   - poorer coverage

This is the correct role of the old Step 44-style analysis: it is a mechanism probe for
RF performance, not the main result by itself.

### 45.6 Keep SPT-vs-Esper fluorinated analysis as supporting context only

Reproduce the corrected SPT-vs-Esper fluorinated matched-pair comparison if it helps
interpret the fluorinated validation outcome, but keep it explicitly secondary.

Allowed uses:

- contextualize whether fluorinated PC-SAFT datasets disagree at a level larger or smaller
  than the RF-vs-RF difference
- correct the older "~75 K" narrative where necessary
- explain why data-source mismatch was not the dominant cause of the GNN failure

Not allowed:

- substituting SPT-vs-Esper matched-pair statistics for the actual RF comparison
- presenting a dataset-only bias story without tying it back to RF performance and
  uncertainty

If reported, the SPT/Esper section must be titled as **supporting context** or
**secondary diagnostic**.

### 45.7 Reassess the project narrative

Use the Step 45 results to update the project narrative in a disciplined way:

1. **RF training-data decision**
   - Did adding ML-SAFT improve deployment-relevant RF behavior?
2. **Uncertainty decision**
   - Did the Combined RF preserve or improve uncertainty calibration?
3. **Narrative decision**
   - Does the cross-dataset story explain RF behavior materially, or is it mostly a side
     note relative to the RF’s empirical performance?

Any discussion of the GNN failure must respect the Step 38d conclusion that the dominant
problem was architectural limitation, not merely data-source mismatch.

### 45.8 Create the driver script

Create `scripts/step45_rf_fluorinated_comparison.py` that:

1. loads Step 44 predictions and metrics
2. evaluates both RFs on the fluorinated external validation set
3. computes halogenated and fluorinated subgroup metrics on the Step 44 benchmark outputs
4. compares uncertainty calibration and empirical coverage on those slices
5. performs the supporting Esper/ML-SAFT matched-pair analysis
6. optionally performs the supporting SPT/Esper fluorinated context analysis
7. saves a consolidated comparison table and figures

Save a machine-readable table to
`model/saved/step45_rf_fluorinated_comparison.csv` with at least:

`smiles, inchi, eval_set, subgroup, model_name, target, y_true, y_pred, y_std,
abs_error, in_1sigma, in_2sigma`

Save aggregate slice metrics to
`model/saved/step45_rf_fluorinated_metrics.json`.

### 45.9 Generate required figures

Save all figures to `figures/45_rf_fluorinated_comparison/`.

Required figures:

1. **`fluorinated_validation_parity.png`**
   - parity comparison for both RFs on the Step 38 fluorinated set
   - separate panels for `m`, `sigma`, `epsilon_k`

2. **`fluorinated_uncertainty_comparison.png`**
   - uncertainty vs error scatter or binned calibration curves
   - show both RFs on the same axes

3. **`slice_gain_loss_heatmap.png`**
   - heatmap of delta metrics (`delta_MAE`, `delta_RMSE`, `delta_R2`, coverage deltas)
   - rows: evaluation slices
   - columns: targets / uncertainty metrics

4. **`halogenated_slice_breakdown.png`**
   - grouped bar or dot plot for fluorinated / chlorinated / non-halogenated slices

5. **`overlap_disagreement_vs_rf_delta.png`**
   - supporting figure relating Esper/ML-SAFT disagreement magnitude to RF performance
     change or coverage shift

6. **`spt_esper_context.png`**
   - optional
   - include only if the SPT/Esper comparison materially informs the fluorinated RF result

### 45.10 Write the Step 45 report

Write `docs/reports/45_rf_fluorinated_comparison.md`. Required sections:

1. **Title**:
   `# Results Report: Fluorinated RF Comparison (Esper-only vs Combined)`
2. **Summary**:
   one paragraph stating whether the Combined RF helps on fluorinated and out-of-source
   slices
3. **Model Performance**:
   fluorinated external validation and combined-test subgroup metrics
4. **Comparison to Baseline**:
   explicit deltas vs the Esper-only RF
5. **Uncertainty Quality**:
   mean stds, coverage, and calibration findings
6. **Subgroup Analysis**:
   fluorinated, halogenated, and ML-SAFT-only slices
7. **Supporting Diagnostics**:
   Esper/ML-SAFT overlap, and SPT/Esper only if needed
8. **Narrative Implications**:
   what should change in project framing, if anything
9. **Key Findings**
10. **Figures**
11. **Deviations**
12. **Readiness Check**

The report must clearly state whether Step 44’s recommendation survives the fluorinated
stress test.

### 45.11 Propose downstream wording changes if needed

If the supporting SPT/Esper analysis is included, propose wording updates for:

- `docs/reports/31_unified_gnn_retrain.md`
- `docs/reports/38_CRITICAL_FINDING.md`
- `presentation/ml_for_chemistry_framing.md`

Do not edit those files in this step. Only propose revisions if the Step 45 results show
that the older source-mismatch framing is materially misleading.

## Key Files

| File | Purpose |
|------|---------|
| `model/saved/step44_rf_combined_predictions.csv` | Step 44 prediction table |
| `model/saved/step44_rf_combined_metrics.json` | Step 44 aggregate metrics |
| `model/saved/gnn_fluorinated_validation_set.csv` | External fluorinated validation set |
| `model/data/esper_pcsaft.csv` | Esper corpus |
| `model/data/mlsaft_pcsaft.csv` | ML-SAFT corpus |
| `model/data/spt_pcsaft.csv` | Supporting SPT context only |
| `docs/reports/31_unified_gnn_retrain.md` | Historical source-mismatch claim |
| `docs/reports/38_CRITICAL_FINDING.md` | Historical GNN-failure framing |
| `docs/reports/38d_rf_vs_gnn_comparison.md` | Architectural-limitation conclusion |

## Artifacts

| File | Description |
|------|-------------|
| `scripts/step45_rf_fluorinated_comparison.py` | Subgroup and fluorinated comparison script |
| `docs/reports/45_rf_fluorinated_comparison.md` | Results report |
| `figures/45_rf_fluorinated_comparison/` | All Step 45 figures |
| `model/saved/step45_rf_fluorinated_comparison.csv` | Per-molecule subgroup and uncertainty table |
| `model/saved/step45_rf_fluorinated_metrics.json` | Aggregate metrics and coverage deltas |

## Success Criteria

- [ ] Step 45 reuses the exact two RF models defined in Step 44
- [ ] Both RFs are compared on the fluorinated external validation set if available
- [ ] Both RFs are compared on fluorinated and halogenated benchmark slices
- [ ] Point metrics and uncertainty metrics are reported side by side for every required
      slice
- [ ] Gain/loss relative to the Esper-only RF is quantified explicitly
- [ ] Esper/ML-SAFT overlap analysis is present only as supporting explanation
- [ ] SPT/Esper analysis, if included, is clearly labeled as secondary context
- [ ] The report states whether the Combined RF recommendation survives the fluorinated
      stress test
- [ ] All required figures are saved to `figures/45_rf_fluorinated_comparison/`
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- The project knows whether the Combined RF improves deployment-relevant fluorinated
  behavior
- The project knows whether the Combined RF uncertainty is trustworthy on those slices
- Any cross-dataset narrative is subordinate to the actual RF evidence
- There is a clear statement on whether Step 44’s recommendation should be kept or revised

## Budget

45 minutes. This step reuses the Step 44 models but performs deeper slice analysis,
external fluorinated validation, and uncertainty comparison. Supporting dataset-diagnostic
work is secondary and should not dominate the implementation time.
