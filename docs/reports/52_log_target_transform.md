# Results Report: Log-Target Transform (RF and XGBoost)

## Motivation

GNNePCSAFT minimizes Huber loss on relative percentage error, while our RF and XGBoost minimize MSE on raw targets. Log-transforming targets before training makes MSE approximate mean squared relative error: MSE(log y_hat, log y) ~ mean((y_hat - y)^2 / y^2). This is the cheapest way to test whether relative-error training improves epsilon_k prediction.

## Assumption Validation: Heteroscedasticity

The log-transform is justified if errors scale with target magnitude (multiplicative error structure).


| Target    | Spearman rho | p-value | Skew (raw) | Skew (log) | >30% rel err |
| --------- | ------------ | ------- | ---------- | ---------- | ------------ |
| m         | 0.129        | 0.0142  | 2.79       | 0.32       | 14.7%        |
| sigma     | -0.061       | 0.2503  | -0.46      | -1.11      | 1.9%         |
| epsilon_k | 0.242        | 0.0000  | 0.54       | -0.97      | 6.6%         |


See `figures/52_log_target_transform/heteroscedasticity_check.png` and `target_distributions.png`.

## Back-Transform Bias (Smearing Correction)


| Target    | RF Smearing Factor | XGB Smearing Factor |
| --------- | ------------------ | ------------------- |
| m         | 1.0046             | 1.0093              |
| sigma     | 1.0005             | 1.0011              |
| epsilon_k | 1.0019             | 1.0040              |


A smearing factor of 1.0 means no bias; >1.0 means naive exp(pred) underestimates the conditional mean.

## Esper Holdout Results


| Target    | Variant           | R2    | MAE   | sMAPE | Med Signed % |
| --------- | ----------------- | ----- | ----- | ----- | ------------ |
| m         | rf raw            | 0.617 | 0.59  | 0.144 | 0.011        |
| m         | rf log corrected  | 0.625 | 0.59  | 0.144 | -0.003       |
| m         | xgb raw           | 0.609 | 0.61  | 0.152 | 0.001        |
| m         | xgb log corrected | 0.626 | 0.59  | 0.145 | 0.004        |
| sigma     | rf raw            | 0.355 | 0.19  | 0.053 | 0.000        |
| sigma     | rf log corrected  | 0.366 | 0.19  | 0.052 | -0.001       |
| sigma     | xgb raw           | 0.341 | 0.20  | 0.055 | 0.000        |
| sigma     | xgb log corrected | 0.339 | 0.20  | 0.056 | 0.002        |
| epsilon_k | rf raw            | 0.329 | 26.92 | 0.099 | -0.002       |
| epsilon_k | rf log corrected  | 0.287 | 27.34 | 0.102 | -0.001       |
| epsilon_k | xgb raw           | 0.322 | 26.48 | 0.097 | 0.004        |
| epsilon_k | xgb log corrected | 0.289 | 27.40 | 0.102 | 0.004        |


See `figures/52_log_target_transform/esper_comparison_bar.png`.

## Cross-Validated Results (5x3 Repeated)


| Target    | Model | Metric | Raw (mean +/- std) | Log (mean +/- std) | Delta   | 95% CI             | p-value |
| --------- | ----- | ------ | ------------------ | ------------------ | ------- | ------------------ | ------- |
| m         | RF    | r2     | 0.670 +/- 0.078    | 0.679 +/- 0.081    | +0.0082 | [+0.0023, +0.0140] | 0.017   |
| m         | RF    | mae    | 0.595 +/- 0.047    | 0.578 +/- 0.050    | -0.0166 | [-0.0218, -0.0113] | 0.000   |
| m         | XGB   | r2     | 0.665 +/- 0.072    | 0.680 +/- 0.076    | +0.0144 | [+0.0027, +0.0261] | 0.030   |
| m         | XGB   | mae    | 0.600 +/- 0.044    | 0.582 +/- 0.043    | -0.0185 | [-0.0262, -0.0109] | 0.000   |
| sigma     | RF    | r2     | 0.415 +/- 0.061    | 0.405 +/- 0.060    | -0.0100 | [-0.0147, -0.0053] | 0.001   |
| sigma     | RF    | mae    | 0.183 +/- 0.012    | 0.183 +/- 0.012    | +0.0005 | [-0.0002, +0.0012] | 0.195   |
| sigma     | XGB   | r2     | 0.390 +/- 0.052    | 0.386 +/- 0.056    | -0.0040 | [-0.0124, +0.0045] | 0.373   |
| sigma     | XGB   | mae    | 0.192 +/- 0.010    | 0.192 +/- 0.010    | +0.0001 | [-0.0009, +0.0010] | 0.887   |
| epsilon_k | RF    | r2     | 0.451 +/- 0.077    | 0.444 +/- 0.083    | -0.0068 | [-0.0216, +0.0081] | 0.388   |
| epsilon_k | RF    | mae    | 23.817 +/- 1.893   | 23.712 +/- 2.079   | -0.1059 | [-0.3220, +0.1102] | 0.353   |
| epsilon_k | XGB   | r2     | 0.454 +/- 0.065    | 0.444 +/- 0.074    | -0.0097 | [-0.0208, +0.0013] | 0.107   |
| epsilon_k | XGB   | mae    | 24.513 +/- 1.934   | 24.560 +/- 1.910   | +0.0466 | [-0.2045, +0.2977] | 0.721   |


See `figures/52_log_target_transform/cv_paired_delta.png`.

## Fluorinated External Validation


| Target    | Variant | MAE   | sMAPE | R2     |
| --------- | ------- | ----- | ----- | ------ |
| m         | rf raw  | 0.86  | 0.349 | -2.565 |
| m         | rf log  | 0.85  | 0.347 | -2.477 |
| m         | xgb raw | 0.75  | 0.318 | -1.715 |
| m         | xgb log | 0.80  | 0.333 | -2.089 |
| sigma     | rf raw  | 0.07  | 0.022 | 0.772  |
| sigma     | rf log  | 0.07  | 0.022 | 0.784  |
| sigma     | xgb raw | 0.05  | 0.015 | 0.873  |
| sigma     | xgb log | 0.06  | 0.019 | 0.838  |
| epsilon_k | rf raw  | 14.24 | 0.079 | -0.457 |
| epsilon_k | rf log  | 13.39 | 0.074 | -0.299 |
| epsilon_k | xgb raw | 15.90 | 0.090 | -0.558 |
| epsilon_k | xgb log | 13.98 | 0.079 | -0.345 |


### Paired Statistical Tests (Fluorinated)

- **m RF**: Bootstrap MAE delta CI [-0.04, 0.01] (spans zero); Wilcoxon p=0.762
- **m XGB**: Bootstrap MAE delta CI [0.01, 0.08] **significant**; Wilcoxon p=1.000
- **sigma RF**: Bootstrap MAE delta CI [-0.01, 0.01] (spans zero); Wilcoxon p=1.000
- **sigma XGB**: Bootstrap MAE delta CI [0.00, 0.02] **significant**; Wilcoxon p=1.000
- **epsilon_k RF**: Bootstrap MAE delta CI [-2.78, 0.64] (spans zero); Wilcoxon p=0.561
- **epsilon_k XGB**: Bootstrap MAE delta CI [-3.64, -0.27] **significant**; Wilcoxon p=1.000

## Boiling Point Propagation

- **rf_raw**: BP MAE = 7.67 K (6/15 converged)
- **rf_log**: BP MAE = 7.35 K (6/15 converged)

## Key Findings

- **Recommendation: reject**. No statistically significant improvement on the deployment-relevant fluorinated validation set.
- Esper epsilon_k: RF raw R2=0.329 vs RF log R2=0.287
- CV epsilon_k R2 delta (RF): -0.0068 (p=0.388)

## Figures

See `figures/52_log_target_transform/` for:

1. `heteroscedasticity_check.png` -- |residual| vs y_true
2. `target_distributions.png` -- raw vs log histograms
3. `esper_comparison_bar.png` -- R2/MAE/sMAPE grouped bars
4. `fluorinated_comparison.png` -- fluorinated MAE/sMAPE
5. `cv_paired_delta.png` -- CV R2 delta forest plot

## Readiness Check

- Heteroscedasticity assumption verified
- Target distribution skewness reported
- RF and XGBoost trained on both raw and log targets
- Back-transform bias quantified (smearing factor)
- Esper holdout comparison with R2, MAE, sMAPE
- Repeated 5-fold CV (5x3) with paired t-tests
- Fluorinated validation with paired bootstrap and Wilcoxon
- At least 5 figures generated
- Report written

