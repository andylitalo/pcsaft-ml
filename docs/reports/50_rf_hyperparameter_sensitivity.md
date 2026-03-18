# Results Report: RF Hyperparameter Sensitivity (Esper, 1,801 molecules)

## Summary

This step validates whether the production Random Forest's hyperparameters (n_estimators=100, max_features='sqrt', max_depth=None, min_samples_leaf=1) are reasonable for the deployed screening workflow, quantifies local sensitivity around the production configuration, and assesses whether a materially better configuration exists for epsilon_k prediction.

## Grid Search Results

**Best epsilon_k CV R-squared**: 0.4968
**Best params**: {'max_depth': 'None', 'max_features': 0.5, 'min_samples_leaf': 3, 'n_estimators': 500}

**Production config rank**: 2 / 240
**Production CV R-squared**: 0.4746
**Delta (best - production)**: +0.0222

### Top 5 Configurations

| Rank | n_est | max_features | max_depth | min_samples_leaf | CV R2 |
|------|-------|-------------|-----------|-----------------|-------|
| 1 | 500 | 0.5 | None | 3 | 0.4968 |
| 2 | 500 | 0.5 | 30 | 3 | 0.4968 |
| 3 | 500 | 0.3 | 20 | 3 | 0.4959 |
| 4 | 500 | 0.3 | 30 | 3 | 0.4958 |
| 5 | 200 | 0.3 | 20 | 3 | 0.4956 |

## Local Sensitivity Analysis

One-at-a-time sweeps hold three hyperparameters at production defaults and vary the fourth. Results show local sensitivity near the production point.

See `figures/50_rf_hyperparameter_sensitivity/oat_sensitivity_epsilon_k.png`.

## Production vs Best Comparison

| Metric | Production | Best Config | Delta |
|--------|-----------|------------|-------|
| R2 | 0.3341 | 0.3349 | +0.0008 |
| MAE | 26.8383 | 26.3129 | -0.5254 |
| RMSE | 51.1172 | 51.0852 | -0.0320 |

**Paired Bootstrap (2000 resamples):**

- Delta R2 95% CI: [-0.0331, +0.0338]
- Delta MAE 95% CI: [-1.372, +0.295]
- The 95% CI for R2 delta **spans zero**, indicating the difference is not statistically significant.

## External Validation of Candidate Configs

| Config | epsilon_k MAE | epsilon_k R2 | BP MAE (K) |
|--------|-------------|------------|-----------|
| production | 13.260 | -0.152 | N/A |
| grid_rank_1 | 12.380 | -0.091 | N/A |
| grid_rank_2 | 12.425 | -0.097 | N/A |
| grid_rank_3 | 12.268 | -0.081 | N/A |


## Feature Importance Stability

- **Top 20 features consistent across all 5 folds**: 10/20
- **Morgan FP bits in top 20**: 1
- **RDKit descriptors in top 20**: 19

### Top 10 Features by Mean Gini Importance (epsilon_k)

| Feature | Mean Importance | Std | CV |
|---------|----------------|-----|-----|
| MinAbsEStateIndex | 0.0177 | 0.0031 | 0.17 |
| VSA_EState7 | 0.0165 | 0.0023 | 0.14 |
| MinEStateIndex | 0.0162 | 0.0023 | 0.14 |
| BCUT2D_MRHI | 0.0160 | 0.0011 | 0.07 |
| RingCount | 0.0154 | 0.0040 | 0.26 |
| MaxPartialCharge | 0.0140 | 0.0015 | 0.11 |
| Kappa3 | 0.0132 | 0.0017 | 0.13 |
| qed | 0.0129 | 0.0027 | 0.21 |
| BCUT2D_MRLOW | 0.0129 | 0.0006 | 0.05 |
| Chi1v | 0.0129 | 0.0016 | 0.12 |

**Caveat**: With ~2,200 features, many are correlated. Both Gini and permutation importance distribute importance across correlated groups. Stable rankings indicate repeatability, not direct physical interpretability.

See `figures/50_rf_hyperparameter_sensitivity/feature_importance_stability.png`.

## Data Efficiency (Learning Curve)

The RF learning curve shows CV test R-squared for epsilon_k at training set sizes from 115 to 1152 molecules.

| Train Size | CV Test R2 (mean) |
|-----------|------------------|
| 115 | 0.3351 |
| 230 | 0.3960 |
| 345 | 0.4017 |
| 576 | 0.4362 |
| 806 | 0.4486 |
| 979 | 0.4631 |
| 1152 | 0.4746 |

Last increment: +0.0115 R-squared. 
The curve is still rising, suggesting more data could improve performance.

See `figures/50_rf_hyperparameter_sensitivity/rf_learning_curve.png`.

## Error Analysis

Systematic error analysis on 361 Esper test molecules. Median |epsilon_k error|: 12.3 K, mean: 26.8 K.

### Top 10 Worst-Predicted Molecules (epsilon_k)

| SMILES | True | Pred | |Error| | MW |
|--------|------|------|--------|-----|
| CCn1cc[n+](C)c1.O=S(=O)([N-]S(=O)(= | 550.0 | 228.2 | 321.8 | 391 |
| CS(=O)(=O)O | 550.0 | 252.3 | 297.7 | 96 |
| Cc1ccc[n+]([O-])c1 | 550.0 | 307.8 | 242.2 | 109 |
| NCCCO | 79.9 | 320.6 | 240.7 | 75 |
| OCC=CCO | 76.2 | 295.7 | 219.5 | 88 |
| OCCCCCO | 94.1 | 291.1 | 197.0 | 104 |
| NC1CC1 | 117.7 | 297.2 | 179.5 | 57 |
| Oc1c(Cl)cc(Cl)cc1Cl | 190.5 | 365.4 | 174.9 | 197 |
| ClSSCl | 476.2 | 302.2 | 173.9 | 135 |
| Nc1ccc([N+](=O)[O-])cc1 | 144.0 | 303.6 | 159.6 | 138 |

### Median |Error| by Molecular Property

| Group | Median |Error| (K) | n |
|-------|---------------------|---|
| All | 12.3 | 361 |
| Fluorinated | 12.6 | 38 |
| Non-fluorinated | 12.1 | 323 |
| Ring-containing | 13.5 | 127 |
| Halogenated | 12.7 | 74 |

See `figures/50_rf_hyperparameter_sensitivity/error_analysis_residuals.png`.

## Key Findings

1. The production RF hyperparameters can be validated through target-specific grid search and one-at-a-time sensitivity analysis.
2. Feature importance stability analysis reveals whether the top features are consistent across CV folds.
3. The RF learning curve provides data efficiency evidence for the 'representation bottleneck' narrative.
4. Systematic error analysis identifies failure modes and molecular property correlations.

## Figures

See `figures/50_rf_hyperparameter_sensitivity/` for:
1. `oat_sensitivity_epsilon_k.png` -- One-at-a-time sensitivity (2x2 grid)
2. `cv_heatmap_n_estimators_vs_max_depth.png` -- CV heatmap of n_estimators vs max_depth
3. `feature_importance_stability.png` -- Box plots for top 20 features across 5 folds
4. `rf_learning_curve.png` -- R-squared vs training set size
5. `error_analysis_residuals.png` -- Residuals and |error| vs molecular weight

## Readiness Check

- [ ] Primary grid search completed for epsilon_k (per-target RF)
- [ ] OAT sensitivity plots show local sensitivity near production
- [ ] Production vs best uses paired bootstrap (not CI overlap)
- [ ] Top configs checked on fluorinated external validation
- [ ] Feature importance stability assessed across 5 folds
- [ ] RF learning curve generated for epsilon_k
- [ ] Systematic error analysis with residual plots and worst molecules
- [ ] At least 5 figures saved to `figures/50_rf_hyperparameter_sensitivity/`
- [ ] Report written

