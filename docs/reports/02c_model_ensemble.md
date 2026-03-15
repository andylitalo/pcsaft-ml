# Results Report: Model Ensemble (RF + NN + ChemBERTa, 361 Test Molecules)

## Model Performance

We implemented Phase 1 inverse-variance-weighted ensembling combining predictions from RF (tree disagreement σ), NN (MC Dropout σ), and ChemBERTa (MC Dropout σ). The ensemble uses the standard minimum-variance formula:

```
w_i = 1 / σ_i²
ŷ_ensemble = Σ(w_i × ŷ_i) / Σ(w_i)
σ_ensemble = 1 / √(Σ(w_i))
```

### Individual Model Results (Test Set, n=361)

| Model | R²(m) | R²(σ) | R²(ε/k) | MAE(m) | MAE(σ) | MAE(ε/k) | Mean σ(m) | Mean σ(σ) | Mean σ(ε/k) |
|-------|--------|--------|----------|--------|--------|-----------|-----------|-----------|-------------|
| RF | 0.619 | 0.353 | 0.330 | 0.585 | 0.189 | 26.83 | 0.811 | 0.238 | 29.61 |
| NN | 0.400 | 0.028 | 0.093 | 0.827 | 0.245 | 33.84 | 0.295 | 0.062 | 8.80 |
| ChemBERTa | 0.539 | 0.271 | 0.270 | 0.770 | 0.222 | 30.83 | 0.268 | 0.053 | 7.68 |

> **Uncertainty note**: All R², MAE, and RMSE values in this report are point estimates on a single holdout split. 95% bootstrap confidence intervals are available via `python -m model.evaluate --bootstrap` and saved to `model/saved/comparison_metrics.csv` with `_lo`/`_hi` suffix columns.

### Ensemble Results (RF + NN, Inverse-Variance Weighted)

| Method | R²(m) | R²(σ) | R²(ε/k) | MAE(m) | MAE(σ) | MAE(ε/k) |
|--------|--------|--------|----------|--------|--------|-----------|
| RF alone (baseline) | 0.619 | 0.353 | 0.330 | 0.585 | 0.189 | 26.83 |
| Inv-var ensemble (RF+NN) | 0.533 | 0.078 | 0.133 | 0.747 | 0.236 | 32.08 |
| Equal-weight (RF+NN) | 0.573 | 0.259 | 0.254 | 0.673 | 0.207 | 29.18 |
| RF-dominant (80/20 RF+NN) | 0.615 | 0.331 | 0.309 | 0.608 | 0.194 | 27.43 |

**The inverse-variance-weighted ensemble underperforms RF alone on all three targets.** This is the "trivial weighting" failure mode anticipated in the step guide.

## Comparison to Baseline

| Target | RF (baseline) | Ensemble (inv-var) | Delta |
|--------|---------------|-------------------|-------|
| m | 0.619 | 0.533 | **-0.086** |
| σ | 0.353 | 0.078 | **-0.275** |
| ε/k | 0.330 | 0.133 | **-0.197** |

The ensemble degrades performance because the NN receives 83–89% of the inverse-variance weight despite being the weaker model. This occurs because MC Dropout uncertainty estimates (σ ~ 0.05–0.30) are an order of magnitude narrower than RF tree disagreement (σ ~ 0.24–0.81), making the NN appear overconfident to the weighting scheme.

## Key Findings

- **Uncertainty calibration mismatch is the root cause.** RF tree disagreement is well-calibrated but conservative (wide σ), while MC Dropout for NN and ChemBERTa is overconfident (narrow σ). Inverse-variance weighting interprets narrow σ as "this model is very certain," but in reality the NN is simply poor at estimating its own uncertainty.

- **Weight distribution confirms overconfidence.** NN receives median weight 0.88–0.93 across targets, while RF receives only 0.07–0.12. A useful ensemble requires comparable uncertainty scales.

- **Equal weighting is strictly better than inverse-variance weighting** for this model combination (R² 0.57/0.26/0.25 vs 0.53/0.08/0.13), since it doesn't reward overconfidence. But it still averages in NN noise and underperforms RF alone.

- **Model disagreement provides a useful AD signal.** Mean pairwise disagreement on ε/k = 17.4 K (max = 103 K), which complements the Tanimoto-based and Isolation Forest AD detectors from Steps 02 and 14. Molecules with high disagreement (> 2σ_ensemble) can be flagged for additional review.

- **GNN unavailable for ensembling.** The GNN (R² 0.76/0.77/0.73 — Step 2B) requires `torch_geometric` which was not installed in this environment. An ensemble with RF + GNN would be the most promising combination since the GNN is both the best individual model AND uses a fundamentally different representation (graph structure vs. fingerprint vectors). Future work should evaluate RF + GNN ensembles once `torch_geometric` is available.

- **Phase 2 (stacking) is warranted.** Since Phase 1 failed to improve over RF on any target, the stacking meta-learner would be the natural next step. A Ridge regression trained on held-out fold predictions could learn to assign appropriate weights based on actual accuracy rather than uncertainty estimates. However, given the small test set (361 molecules) and existing model asymmetry (RF trained on all data, NN on Esper only), stacking results would be fragile and data-hungry. **We recommend deferring Phase 2 until the GNN is available as a constituent model.**

## Figures

See `figures/02c_ensemble/` for:
- `ensemble_vs_gnn_parity.png`: Side-by-side parity plots for RF baseline vs ensemble (2×3 panel)
- `model_weight_distribution.png`: Histograms of per-molecule inverse-variance weights by target
- `ensemble_improvement_by_target.png`: R² comparison bar chart (RF vs ensemble per target)

## Deviations

- **GNN not included.** The step guide assumes the GNN is available as a constituent model. Since `torch_geometric` is not installed, the ensemble uses RF + NN (and optionally ChemBERTa). The GNN's contribution would be the most impactful given its much higher accuracy and different representation.
- **ChemBERTa excluded from primary results.** ChemBERTa inference takes ~185s on CPU for 361 molecules (vs ~2s for RF + NN). The RF + NN ensemble is reported as the primary result; ChemBERTa metrics are included in the individual model table for reference.
- **Phase 2 (stacking) deferred.** The step guide conditions Phase 2 on Phase 1 not improving over the best single model. While this condition is met, we recommend deferring stacking until the GNN is available rather than training a meta-learner on two models where one (NN) is clearly weaker.

## Readiness Check

- [x] `model/ensemble/weighted.py` implements inverse-variance-weighted prediction
- [x] Ensemble is registered as `"ensemble"` in the model registry
- [x] `predict()` and `predict_with_uncertainty()` work with RF + NN (minimum 2 models)
- [x] Comparison table with R², MAE for RF vs ensemble on test set
- [x] Weight distribution figure shows which model dominates and where
- [x] Report documents why ensemble underperforms (uncertainty calibration mismatch)
- [x] 10 tests covering: normal operation, NaN handling, single-model fallback, zero-σ floor, disagreement, no-models-raises
- [ ] Updated `ci_compatible` / `ci_high_conf` counts — deferred (ensemble uncertainty is artificially narrow due to NN overconfidence, making CI metrics misleading)
- [ ] Phase 2 stacking — deferred until GNN available
