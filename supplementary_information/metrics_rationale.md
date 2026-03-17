# Evaluation Metrics: Definitions and Rationale

## Mean Absolute Error (MAE)

```
MAE = (1/n) * sum(|y_pred - y_true|)
```

MAE is the primary metric because it is directly interpretable in physical units: an MAE of 26.8 K on epsilon/k means the model's predictions are off by about 27 K on average. This is meaningful when the screening filter window for boiling point is only 35 K wide -- an epsilon/k error of that magnitude can flip a candidate from "pass" to "fail."

## Coefficient of Determination (R^2)

```
R^2 = 1 - SS_res / SS_tot = 1 - sum((y_true - y_pred)^2) / sum((y_true - y_mean)^2)
```

R^2 measures the fraction of variance explained by the model. Values near 1.0 indicate the model captures most of the variation in the target; values near 0 indicate predictions are no better than the mean; negative values indicate the model is actively worse than predicting the mean.

**Why R^2 is useful here**: PC-SAFT parameters have very different scales and variances across chemical classes. R^2 allows comparison across parameters (m, sigma, epsilon/k) and across models on a common scale.

**Why R^2 is insufficient alone**: The inter-dataset variability analysis (Step 10) showed that independently fitted PC-SAFT parameters disagree by 1-3% at the median but with heavy tails (P95: 31% for epsilon/k, 48% for m). This means even "perfect" predictions face a noise floor from parameter non-uniqueness. An R^2 of 0.33 on epsilon/k sounds low, but must be interpreted against this noise floor.

## Mean Absolute Relative Error (MARE)

```
MARE = (1/n) * sum(|y_pred - y_true| / |y_true|)
```

MARE provides scale-invariant error comparison across parameters with different magnitudes. An epsilon/k MAE of 27 K is large relative to a sigma MAE of 0.19 Angstroms, but MARE puts them on the same footing.

## Lin's Concordance Correlation Coefficient (CCC)

```
CCC = (2 * rho * sigma_pred * sigma_true) / (sigma_pred^2 + sigma_true^2 + (mu_pred - mu_true)^2)
```

CCC extends Pearson's correlation by penalizing both scale differences and systematic bias. A model with high R^2 but systematic over-prediction would have lower CCC. This was particularly important for detecting the SPT-PCSAFT systematic bias: high R^2 on the in-distribution test set masked a ~75 K systematic offset for fluorinated compounds.

## Why These Metrics Together

No single metric tells the full story:
- MAE gives the practical error budget
- R^2 gives the variance-explained context
- MARE enables cross-parameter comparison
- CCC catches systematic bias that R^2 can miss

The most important lesson from this project was that **external validation metrics** (boiling point MAE on 15 independent fluorinated refrigerants) mattered far more than any of these in-distribution metrics for model selection.
