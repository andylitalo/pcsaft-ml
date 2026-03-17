# Results Report: RF Combined Retrain (Esper + ML-SAFT, 1936 molecules combined)

## Model Performance

Two Random Forest models were trained under matched conditions (100 trees, combined Morgan + RDKit features, identical hyperparameters) and evaluated on the canonical Esper holdout (361 molecules). The only experimental difference is the training corpus: RF_esper uses 1440 Esper-only molecules, RF_combined uses 1575 molecules from the InChI-deduplicated Esper + ML-SAFT combined dataset (excluding the 361 canonical holdout molecules).

### Dataset Construction

- **Esper dataset**: 1801 molecules (after SMILES dedup and NaN removal)
- **Combined dataset**: 1936 molecules (InChI-based deduplication; Esper values take priority for overlapping molecules)
- **Net new molecules from ML-SAFT**: 135 (7.5% increase)
- **Deduplication rule**: InChI canonical identifier via RDKit `MolToInchi()`. Overlap counts reported by InChI. Step 10 reported 736 overlaps by canonical SMILES and 745 by InChI; this step uses the InChI convention consistent with `load_data('combined')`.

## Primary Comparison: Canonical Esper Holdout

This is the apples-to-apples benchmark. Both models evaluated on the same 361 Esper holdout molecules.

| Metric | RF_esper | RF_combined | Delta | 95% CI Interpretation |
|---|---|---|---|---|
| R2(m (segments)) | 0.6194 [0.4656, 0.7560] | 0.6074 [0.4592, 0.7464] | -0.0120 | CIs overlap; not significant |
| MAE(m (segments)) | 0.5867 [0.4911, 0.7021] | 0.5986 [0.4995, 0.7135] | +0.0119 | CIs overlap; not significant |
| RMSE(m (segments)) | 1.1601 [0.8672, 1.4651] | 1.1783 [0.8929, 1.4778] | +0.0182 | CIs overlap; not significant |
| Mean std(m (segments)) | 0.7988 | 0.8412 | +0.0425 | Predictive uncertainty increased |
| R2($\sigma$ ($\AA$)) | 0.3573 [0.2227, 0.4798] | 0.3265 [0.1885, 0.4520] | -0.0308 | CIs overlap; not significant |
| MAE($\sigma$ ($\AA$)) | 0.1886 [0.1627, 0.2168] | 0.1976 [0.1717, 0.2257] | +0.0090 | CIs overlap; not significant |
| RMSE($\sigma$ ($\AA$)) | 0.3233 [0.2786, 0.3744] | 0.3310 [0.2856, 0.3806] | +0.0077 | CIs overlap; not significant |
| Mean std($\sigma$ ($\AA$)) | 0.2372 | 0.2802 | +0.0430 | Predictive uncertainty increased |
| R2($\varepsilon$/k (K)) | 0.3264 [0.1832, 0.4751] | 0.3288 [0.1879, 0.4681] | +0.0024 | CIs overlap; not significant |
| MAE($\varepsilon$/k (K)) | 26.8217 [22.6493, 31.3877] | 27.5952 [23.3752, 32.1012] | +0.7735 | CIs overlap; not significant |
| RMSE($\varepsilon$/k (K)) | 51.4125 [41.4986, 60.7849] | 51.3213 [41.7284, 60.3122] | -0.0912 | CIs overlap; not significant |
| Mean std($\varepsilon$/k (K)) | 29.4492 | 34.6482 | +5.1990 | Predictive uncertainty increased |

### Primary Results Interpretation

All metric deltas between RF_esper and RF_combined fall within bootstrap 95% confidence intervals. **The addition of ML-SAFT data does not produce a statistically significant change in RF performance on the canonical Esper holdout.** This is consistent with the Step 10 finding that the combined dataset adds only 135 genuinely new molecules (7.5% increase), which is insufficient to materially change RF generalization.

## Supplemental Evaluations

The following results provide additional context but are **not** the primary basis for the include/exclude recommendation.

### Combined Holdout (RF_combined only)

RF_combined evaluated on a natural 20% holdout from the full combined corpus. This measures generalization within the combined distribution, not a controlled comparison against RF_esper.

| Parameter | R2 | MAE | RMSE | Mean std |
|---|---|---|---|---|
| m (segments) | 0.6433 | 0.6258 | 1.2304 | 0.8495 |
| $\sigma$ ($\AA$) | 0.3435 | 0.2016 | 0.3356 | 0.2719 |
| $\varepsilon$/k (K) | 0.4056 | 26.5781 | 48.7663 | 35.1186 |

### ML-SAFT-Unique Subset

RF_combined evaluated on the 26 ML-SAFT-unique molecules that ended up in the combined holdout. Small sample size; treat as directional evidence only.

| Parameter | R2 | MAE | RMSE |
|---|---|---|---|
| m (segments) | 0.4959 | 1.0761 | 1.5323 |
| $\sigma$ ($\AA$) | -0.1891 | 0.5268 | 0.6942 |
| $\varepsilon$/k (K) | 0.0344 | 60.8479 | 89.0941 |

### Fluorinated External Validation (15 compounds)

Both RF variants evaluated on the 15-compound fluorinated validation set from Step 38. Literature PC-SAFT parameters used as ground truth.

| Parameter | RF_esper MAE | RF_combined MAE | RF_esper R2 | RF_combined R2 |
|---|---|---|---|---|
| m (segments) | 0.8501 | 0.6919 | -2.5301 | -1.5576 |
| $\sigma$ ($\AA$) | 0.0722 | 0.0603 | 0.7557 | 0.8274 |
| $\varepsilon$/k (K) | 14.1227 | 12.0261 | -0.4338 | -0.2254 |

### Applicability Domain Coverage

Tanimoto nearest-neighbor AD evaluated on 49 screening candidates (in-domain threshold: 0.4).

| Metric | RF_esper training set | RF_combined training set |
|---|---|---|
| In-domain candidates | 10/49 (20.4%) | 15/49 (30.6%) |
| Mean Tanimoto NN sim | 0.3341 | 0.3651 |

AD coverage improves with the combined training set, meaning the additional ML-SAFT molecules expand the model's chemical space coverage for downstream screening. However, since the primary Esper-holdout metrics show no significant improvement, this represents a tradeoff: wider coverage without better accuracy.

## Key Findings

1. **No statistically significant change on the canonical Esper holdout.** All R2, MAE, and RMSE deltas between RF_esper and RF_combined fall within bootstrap 95% CIs. The net addition of 135 molecules (~7% increase) is too small to materially change RF generalization on the Esper test distribution.

2. **Predictive uncertainty slightly increased.** Mean RF tree-variance std is modestly higher for RF_combined across all targets (delta m: +0.042, sigma: +0.043, epsilon_k: +5.2). This likely reflects the noisier combined training data from mixing two fitting protocols. However, the magnitude is small relative to the absolute uncertainty levels, and calibration behavior (std vs MAE across bins) is qualitatively similar.

3. **This confirms Step 10's qualitative finding with quantitative evidence.** Step 10 reported 'no meaningful lift' without uncertainty-aware comparison. This step provides the controlled, uncertainty-quantified evidence to support that conclusion.

## Recommendation

**Include ML-SAFT in the RF training set for production, but with low priority.** The combined dataset does not hurt and marginally expands chemical-space coverage. However, the primary performance metric (Esper holdout R2) is unchanged, so this is not a high-impact change. Engineering effort should focus on model architecture improvements (Steps 02-04) rather than dataset expansion.

## Figures

See `figures/01b_rf_combined_retrain/` for:
- `parity_esper_holdout.png` -- 2x3 parity plots (RF_esper top row, RF_combined bottom row, with error bars from RF tree variance)
- `delta_bar_chart.png` -- Bar chart of metric deltas (combined - esper) with conservative 95% CIs
- `uncertainty_calibration.png` -- Mean predicted std vs mean absolute error across quintile bins, comparing RF_esper and RF_combined
- `fluorinated_validation.png` -- Parity plots on 15-compound fluorinated external validation set (if available)

## Deviations

None. All sections of the step guide (01b.1-01b.7) were implemented as specified.

## Readiness Check

- [x] RF_esper and RF_combined trained with same feature pipeline and hyperparameters
- [x] Both models evaluated on canonical Step 01 Esper holdout
- [x] Primary comparison table includes R2, MAE, RMSE, 95% bootstrap CIs, and mean RF predictive std for all three targets
- [x] Supplemental results clearly labeled: combined holdout, ML-SAFT-unique subset, fluorinated validation, and AD coverage
- [x] Figures saved to `figures/01b_rf_combined_retrain/`
- [x] Report states exact dataset-count convention: 1801 Esper, 1936 combined, 135 net new, InChI deduplication, Esper priority
- [x] Clear include/exclude recommendation: include with low priority
