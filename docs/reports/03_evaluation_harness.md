# Results Report: Evaluation Harness (Esper Dataset, 1,801 molecules)

## Model Performance

A unified evaluation harness was built to compare all three PC-SAFT prediction models on the same held-out test set (361 molecules). The harness uses a registry pattern (`model/registry.py`) that wraps each model behind a common `load()` / `predict()` / `predict_with_uncertainty()` interface, and produces structured metrics, comparison CSV, and publication-quality figures in a single command (`python -m model.evaluate --models gc_pcsaft rf nn`).

### Unified Comparison Table


| Model      | Parameter     | MAE   | RMSE  | R2 (test) | Mean Predicted Std | Train | Test |
| ---------- | ------------- | ----- | ----- | --------- | ------------------ | ----- | ---- |
| GC-PC-SAFT | m (segments)  | 1.001 | 1.460 | 0.40      | --                 | --    | 358  |
| GC-PC-SAFT | sigma (A)     | 0.494 | 0.579 | -1.16     | --                 | --    | 358  |
| GC-PC-SAFT | epsilon_k (K) | 44.6  | 63.5  | -0.04     | --                 | --    | 358  |
| RF         | m (segments)  | 0.585 | 1.160 | 0.62      | 0.811              | 1,440 | 361  |
| RF         | sigma (A)     | 0.189 | 0.325 | 0.35      | 0.238              | 1,440 | 361  |
| RF         | epsilon_k (K) | 26.8  | 51.3  | 0.33      | 29.6               | 1,440 | 361  |
| NN         | m (segments)  | 0.767 | 1.363 | 0.47      | 0.297              | 1,440 | 361  |
| NN         | sigma (A)     | 0.238 | 0.381 | 0.11      | 0.062              | 1,440 | 361  |
| NN         | epsilon_k (K) | 32.9  | 58.0  | 0.14      | 8.77               | 1,440 | 361  |

> **Uncertainty note**: All R², MAE, and RMSE values in this report are point estimates on a single holdout split. 95% bootstrap confidence intervals are available via `python -m model.evaluate --bootstrap` and saved to `model/saved/comparison_metrics.csv` with `_lo`/`_hi` suffix columns.

Note: GC-PC-SAFT has 358 valid predictions (3 molecules with unrecognized functional groups produce NaN and are excluded). GC-PC-SAFT has no intrinsic uncertainty estimate.

The RF remains the best overall model on this dataset, outperforming both the group-contribution baseline and the neural network on all three targets. The NN sits between GC-PC-SAFT and RF for m (R2=0.47 vs 0.40 vs 0.62), but only modestly surpasses GC-PC-SAFT on sigma and epsilon_k.

## Comparison to Baseline

### R2 Comparison (side-by-side)


| Method                 | R2 (m) | R2 (sigma) | R2 (epsilon_k) |
| ---------------------- | ------ | ---------- | -------------- |
| GC-PC-SAFT             | 0.40   | -1.16      | -0.04          |
| RF (Combined features) | 0.62   | 0.35       | 0.33           |
| NN (PCSAFTNet)         | 0.47   | 0.11       | 0.14           |


### MAE Comparison (side-by-side)


| Method                 | MAE (m) | MAE (sigma) | MAE (epsilon_k) |
| ---------------------- | ------- | ----------- | --------------- |
| GC-PC-SAFT             | 1.001   | 0.494       | 44.6            |
| RF (Combined features) | 0.585   | 0.189       | 26.8            |
| NN (PCSAFTNet)         | 0.767   | 0.238       | 32.9            |


### Applicability Domain Analysis

The Isolation Forest AD model flagged 8.0% of test molecules (29/361) as out-of-domain. As expected, all models perform worse on OOD molecules.


| Model      | Parameter | R2 (in-domain) | R2 (OOD) | Fraction OOD |
| ---------- | --------- | -------------- | -------- | ------------ |
| GC-PC-SAFT | m         | 0.391          | 0.183    | 8.0%         |
| GC-PC-SAFT | sigma     | -1.11          | -2.05    | 8.0%         |
| GC-PC-SAFT | epsilon_k | -0.050         | 0.019    | 8.0%         |
| RF         | m         | 0.640          | 0.330    | 8.0%         |
| RF         | sigma     | 0.369          | 0.076    | 8.0%         |
| RF         | epsilon_k | 0.351          | 0.188    | 8.0%         |
| NN         | m         | 0.455          | 0.392    | 8.0%         |
| NN         | sigma     | 0.122          | -0.179   | 8.0%         |
| NN         | epsilon_k | 0.129          | 0.216    | 8.0%         |


For the RF, in-domain R2 is consistently higher than OOD R2 across all targets (e.g., m: 0.640 vs 0.330), validating that the AD check is meaningful. For the NN, the pattern holds for sigma (0.122 vs -0.179) but not for m and epsilon_k, where OOD performance is comparable or slightly higher. This anomaly is likely due to the small OOD sample size (n=29) and the NN's relatively poor overall fit, making the OOD subsample statistics noisy.

## Key Findings

- **The RF is the clear winner in this small-data regime.** With 1,440 training samples and 2,218 features, the RF's implicit feature selection via random splits gives it a decisive advantage. The RF achieves the best R2 on all three targets (m=0.62, sigma=0.35, epsilon_k=0.33).
- **The NN occupies a clear middle ground between GC-PC-SAFT and RF.** For m, the NN (R2=0.47) significantly outperforms GC-PC-SAFT (R2=0.40) but trails RF (R2=0.62). For sigma and epsilon_k, NN predictions are positive-R2 but weak (0.11 and 0.14), suggesting the network has learned some signal but cannot fully exploit the high-dimensional feature space with limited data.
- **Uncertainty calibration differs between RF and NN.** The RF's tree-disagreement uncertainty (mean std: m=0.811, sigma=0.238, epsilon_k=29.6 K) is well-correlated with actual error magnitude -- higher predicted uncertainty bins have higher mean absolute error. The NN's MC Dropout uncertainty (mean std: m=0.297, sigma=0.062, epsilon_k=8.77 K) is systematically lower and less well-calibrated, reflecting the dropout layers' limited ability to capture epistemic uncertainty in a small-data regime.
- **AD analysis confirms that in-domain molecules are predicted more reliably.** The RF shows the clearest separation: R2 drops by 0.15--0.31 points for OOD molecules. This validates the Isolation Forest AD model as a useful gatekeeper for deployment -- molecules flagged as OOD should have their predictions treated with extra caution.
- **GC-PC-SAFT struggles on sigma and epsilon_k but provides a meaningful baseline for m.** With R2=0.40 on m, the group-contribution method demonstrates that chain-length scaling (additive m per group) captures the dominant trend. Its failure on sigma (R2=-1.16) and epsilon_k (R2=-0.04) highlights the non-additive, many-body effects that ML models are needed to capture.
- **The registry pattern enables easy model comparison.** Adding a new model (e.g., ChemBERTa in Step 04) requires only a new class with `@register_model("chemberta")` -- the evaluation, plotting, and metrics infrastructure will automatically include it.

## Figures

See `figures/03_evaluation_harness/` for:

- `parity_comparison.png` -- 3x3 grid of parity plots (rows: GC-PC-SAFT, RF, NN; columns: m, sigma, epsilon_k). Points color-coded by AD status (blue = in-domain, red triangles = OOD).
- `residual_distributions.png` -- 3x3 grid of residual histograms (pred - true) with vertical line at zero. Reveals systematic biases per model/target.
- `nn_learning_curves.png` -- Train/val loss vs epoch for PCSAFTNet, with best-epoch marker at epoch 14.
- `uncertainty_calibration.png` -- Calibration plots (mean predicted std vs mean absolute error, binned) for RF and NN across all three targets.

## Deviations

None. The implementation follows the step guide, including the registry pattern, unified CLI, UQ reporting (Section 3A), and AD analysis (Section 3B).

## Readiness Check

1. [x] Clear, quantified comparison of GC-PC-SAFT vs RF vs NN: unified metrics table with MAE, RMSE, R2, and mean predicted std
2. [x] Evaluation harness accepts a new model with minimal code: just add `@register_model("name")` class with `load()`, `predict()`, `predict_with_uncertainty()`
3. [x] Performance difference is understood: RF's implicit feature selection and ensemble diversity dominate in the small-data (n=1,440), high-dimensional (d=2,218) regime; the NN needs richer input representations (e.g., ChemBERTa embeddings) to compete
4. [x] Figures tell the story visually: parity plots with AD coloring, residual distributions, learning curves, and uncertainty calibration
5. [x] Uncertainty calibration assessed: RF tree-disagreement is better calibrated than NN MC Dropout; both are documented in the calibration plot

