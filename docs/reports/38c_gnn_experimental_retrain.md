# Results Report: GNN Retrain on Experimental Data Only (Step 38c)

## Experiment Design

Step 38 revealed that the unified GNN (trained on Esper + ML-SAFT + SPT-PCSAFT, ~12,800 molecules) catastrophically fails on fluorinated compounds: 133.7 K boiling point MAE vs RF's 8.2 K. The root cause hypothesis was **source mismatch** -- SPT-PCSAFT data (87% of the unified corpus) introduces systematic bias in the learned PC-SAFT parameters.

This experiment tests a narrower hypothesis: **can the GNN architecture itself match RF when given only clean, experimentally fitted data?** We retrained the GNN exclusively on the Esper dataset (~1,801 molecules, all experimentally optimized PC-SAFT parameters), then re-validated on the same 15-compound fluorinated set from Step 38.

### Hyperparameter Adjustments for Small Data

The Esper dataset is 12x smaller than the unified corpus. To mitigate overfitting:
- Hidden dim: 128 (vs 256 unified)
- Layers: 3 (vs 4 unified)
- Dropout: 0.2 (vs 0.1 unified)
- Batch size: 32 (vs 64 unified)
- Weight decay: 1e-4 (vs 1e-5 unified)
- Patience: 25 (vs 15 unified)
- Max epochs: 200

Total trainable parameters: 195,075.

## Training Details

| Setting | Value |
|---------|-------|
| Dataset | Esper only (experimentally fitted) |
| Train / Test | 1,440 / 361 |
| Architecture | GINEConv (3-layer, 128-dim) |
| Parameters | 195,075 |
| Training time | 25.7 s (CPU, macOS) |
| Early stopping | Epoch 49 of 200 (patience 25) |
| Best val loss | ~0.72 (normalized MSE) |

Training converged quickly (49 epochs). The validation loss plateaued around epoch 24 and never improved thereafter, indicating limited model capacity on this small dataset.

## Esper Test-Set Metrics (GNN Esper-only)

| Parameter | MAE | RMSE | R2 (test) | Train | Test |
|-----------|-----|------|-----------|-------|------|
| m (segments) | 0.789 | 1.318 | 0.509 | 1,440 | 361 |
| sigma (A) | 0.224 | 0.343 | 0.279 | 1,440 | 361 |
| epsilon_k (K) | 29.57 | 51.61 | 0.321 | 1,440 | 361 |

The Esper-only GNN performs **worse** than the RF baseline on the Esper test set:

| Parameter | RF R2 | GNN (Esper) R2 | RF MAE | GNN (Esper) MAE |
|-----------|-------|----------------|--------|-----------------|
| m | 0.61 | 0.51 | 0.634 | 0.789 |
| sigma | 0.32 | 0.28 | 0.192 | 0.224 |
| epsilon_k | 0.27 | 0.32 | 26.1 | 29.6 |

The GNN is competitive on epsilon_k (slightly better R2) but worse on m and sigma. With only 1,801 molecules and 195K parameters, the GNN cannot learn representations as effective as RF's hand-crafted RDKit descriptors (170 features) at this sample size.

## Fluorinated Validation (Key Result)

### Boiling Point MAE

| Model | Boiling Point MAE (K) | EOS Convergence | Improvement vs Unified |
|-------|----------------------|-----------------|----------------------|
| **GNN (Esper-only)** | **142.3** | 7/15 (47%) | -8.6 K (worse) |
| GNN (unified) | 133.7 | 6/15 (40%) | -- |
| RF | 8.2 | 6/15 (40%) | -- |

The Esper-only GNN is **8.6 K worse** than the unified GNN. Removing SPT data did not help -- the GNN still catastrophically over-predicts all three PC-SAFT parameters for fluorinated compounds.

### Parameter-Level Errors (vs Literature, n=15)

| Parameter | GNN (Esper) MAE | GNN (Unified) MAE | RF MAE |
|-----------|----------------|--------------------|--------|
| m | 1.696 | 1.575 | 0.852 |
| sigma (A) | 0.409 | 0.397 | 0.072 |
| epsilon_k (K) | 72.4 | 77.6 | 14.3 |

All three parameters show systematic over-prediction (mean errors are positive for all), with epsilon_k errors exceeding 70 K. The GNN learns a general "organic molecule average" rather than correctly capturing the distinct parameter signatures of small, heavily fluorinated refrigerants.

### R2 on Fluorinated Set

| Parameter | GNN (Esper) R2 | GNN (Unified) R2 | RF R2 |
|-----------|---------------|-------------------|-------|
| m | -15.9 | -13.0 | -2.5 |
| sigma | -4.9 | -4.9 | 0.76 |
| epsilon_k | -20.1 | -25.1 | -0.47 |

Negative R2 values confirm predictions are worse than predicting the mean, indicating the fluorinated regime is truly out-of-distribution for the GNN.

## Uncertainty Calibration

| Parameter | 1sigma Coverage | Expected | 2sigma Coverage | Expected |
|-----------|----------------|----------|----------------|----------|
| m | 0% | 68% | 7% | 95% |
| sigma | 7% | 68% | 13% | 95% |
| epsilon_k | 7% | 68% | 13% | 95% |

MC Dropout uncertainty is severely underestimated. The mean uncertainty (sigma) for epsilon_k is 9.5 K, but the mean actual error is 72.4 K -- a 7.6x underestimation. This means the GNN "confidently" makes wrong predictions on fluorinated compounds.

## Screening Results Preview

| Stage | Count |
|-------|-------|
| Initial candidates | 10,700 |
| Boiling point [288-323K] | 168 |
| Has C=C bond | 10,700 |
| No chlorine | 2,558 |
| F count >= 2 | 10,132 |
| SA score <= 4.5 | 4,663 |
| Fluorine mass fraction >= 65% | 330 |
| No reactive fluorination sites | 3,511 |
| Final (all filters) | 23 |

23 candidates pass all filters (vs similar count for unified GNN). However, given the 142 K boiling point MAE, these rankings have no predictive value for fluorinated compounds.

## Verdict

**The GNN architecture fundamentally cannot match RF for small-molecule fluorinated compound prediction at this sample size.**

The problem is **not** SPT data contamination -- it is architectural:

1. **Insufficient training data**: 1,801 molecules (1,440 train) is too few for a 195K-parameter GNN to learn effective molecular representations. The RF succeeds because RDKit descriptors provide strong inductive bias at low data volumes.

2. **Message-passing limitations**: GINEConv message passing learns local substructure patterns. Fluorinated refrigerants (C1-C4, heavily halogenated) have unusual atom/bond patterns that are underrepresented in the Esper corpus (which is dominated by larger organic molecules).

3. **Small molecule regime**: PC-SAFT parameters for small, heavily fluorinated molecules (sigma ~2.8-3.5 A, m ~1.3-2.8) fall in a narrow, distinct region of parameter space that the GNN cannot extrapolate to from predominantly hydrocarbon training data.

4. **Uncertainty failure**: MC Dropout uncertainty (9.5 K sigma for epsilon_k) does not capture the true 72 K errors, making it unsuitable for uncertainty-aware screening.

## Readiness Check

- [x] GNN retrained on Esper-only data
- [x] Esper test set metrics computed (R2: 0.28-0.51)
- [x] Fluorinated validation: 142.3 K boiling point MAE (FAIL vs 10 K target)
- [x] Uncertainty calibration checked (FAIL: 0-7% at 1sigma)
- [x] HFO screening run (23 candidates, but rankings invalid)
- [x] Figures generated in `figures/38c_gnn_experimental_retrain/`
- [x] Results saved to `screening/results/hfo_gnn_esper_*`
- [ ] GNN matches RF on fluorinated set: **NO** -- GNN is architecturally limited

## Next Steps

1. **Use RF for all fluorinated screening** (Option A from Step 38 Critical Finding). The RF model achieves 8.2 K boiling point MAE -- 17x better than the best GNN variant.

2. **Do not invest in further GNN tuning** for this specific application. The root cause is data scarcity and distribution mismatch, not hyperparameter choice.

3. **Consider domain-specific pretraining** (e.g., pretraining on large molecular property datasets, then fine-tuning on PC-SAFT) if GNN use is desired in the future.

4. **Re-run Step 37 with RF** to produce valid screening rankings for downstream steps.

## Figures

See `figures/38c_gnn_experimental_retrain/` for:
- `parity_gnn_esper_vs_unified.png` -- Parameter parity plots with MC Dropout error bars
- `boiling_point_validation.png` -- Boiling point parity with 3-model comparison text box
- `uncertainty_calibration.png` -- Bar chart of observed vs expected coverage
- `screening_funnel_gnn_esper.png` -- Filter funnel for Esper-only GNN screening

## Deviations

- The step guide spec called for a `training_curve.png` figure showing per-epoch losses. The `train_gnn.py` function does not save per-epoch history to disk (only prints to stdout), so this figure was omitted. The early stopping at epoch 49 (of 200) with patience 25 indicates the validation loss plateaued around epoch 24.
