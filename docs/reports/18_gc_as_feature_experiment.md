# Results Report: GC-PC-SAFT as Input Feature Experiment (Esper Dataset, 1,801 molecules)

## Model Performance

This experiment tested whether using GC-PC-SAFT predicted values (m, σ, ε/k) as additional input features to Random Forest improves performance over the baseline Morgan + RDKit feature set. The hypothesis was that even though GC-PC-SAFT predictions are poor on their own (R² = 0.14 on ε/k), they might encode physically motivated group-additivity information that molecular fingerprints miss.

All models use Random Forest with 100 trees, stratified 80/20 train/test split on binned epsilon_k, and the same random seed for reproducibility.

### Ablation Study Results

| Feature Set | R² (m) | R² (σ) | R² (ε/k) | MAE (m) | MAE (σ) | MAE (ε/k) |
|-------------|--------|--------|----------|---------|---------|-----------|
| Morgan + RDKit (baseline) | 0.619 | 0.353 | 0.330 | 0.585 | 0.189 | 26.8 |
| Morgan + RDKit + GC-PC-SAFT | 0.624 | 0.357 | 0.293 | 0.586 | 0.186 | 27.2 |
| GC-PC-SAFT only (reference) | 0.516 | 0.229 | 0.143 | 0.770 | 0.230 | 36.3 |

> **Uncertainty note**: All R², MAE, and RMSE values in this report are point estimates on a single holdout split. 95% bootstrap confidence intervals are available via `python -m model.evaluate --bootstrap` and saved to `model/saved/comparison_metrics.csv` with `_lo`/`_hi` suffix columns.

### Delta Analysis: Augmented vs Baseline

| Parameter | ΔR² | ΔMAE |
|-----------|-----|------|
| m | +0.0042 | +0.0009 |
| σ | +0.0047 | -0.0023 |
| ε/k | -0.0372 | +0.3983 |

## Key Findings

1. **GC-PC-SAFT features provide negligible benefit for m and σ.** Adding the 3 GC features improves R² by ~0.004 for both m and σ — a statistically insignificant change. The MAE for σ improves by 0.002 Å (about 1%), but m MAE is unchanged.

2. **GC-PC-SAFT features hurt ε/k prediction.** Surprisingly, the augmented model performs worse on epsilon_k (R² drops from 0.330 to 0.293, MAE increases from 26.8 to 27.2 K). This suggests the GC features add noise rather than signal for dispersion energy prediction. The RF may be overfitting to the poor GC predictions of ε/k.

3. **GC-PC-SAFT alone is a weak baseline.** The GC-only model achieves R² of 0.52/0.23/0.14 for m/σ/ε/k, confirming that the simplified group-contribution scheme is far inferior to data-driven models. This is expected: the GC implementation neglects cross-interaction terms, ring strain corrections, and association effects beyond simple group additivity.

4. **The hypothesis was wrong for this dataset.** The experiment was motivated by the idea that GC-PC-SAFT captures physical priors (group additivity, ring corrections) that fingerprints miss. However, the results show that Morgan fingerprints + RDKit descriptors already encode these structural patterns more effectively than the simplified GC scheme. Adding GC features adds no new information.

5. **Random Forest is feature-saturated.** With 2,218 features and 1,440 training samples, the RF is already operating in an over-parameterized regime. Adding 3 more features (even if they were informative) is unlikely to improve performance. A more interesting test would be GC features with a simpler baseline (e.g., RDKit-only, ~170 features).

6. **GC predictions fail for small molecules.** During featurization, 28 molecules in the Esper dataset produced "No recognized groups" warnings (e.g., C, N#N, O=O, [Ar]), indicating the GC scheme cannot handle monatomic or diatomic molecules. These were filled with zeros. This limitation is documented in `model/gc_pcsaft.py` but is a practical concern for production use.

## Comparison to Baseline/Previous Step

The baseline for this step is the Morgan + RDKit RF from Step 01 (R² = 0.62/0.35/0.33 for m/σ/ε/k). The augmented model achieves nearly identical performance, confirming that GC features are redundant with the existing feature set.

This result is consistent with the GC-PC-SAFT baseline evaluation in Step 01, which showed that the simplified GC scheme is far inferior to ML models. The current experiment extends this finding: even when GC predictions are used as input features (rather than standalone predictions), they provide no value.

## Figures

No figures were generated for this step. The ablation table above fully summarizes the results.

## Deviations

None. The implementation followed the step guide exactly.

## Recommendations

1. **Do not deploy the augmented model.** The +GC feature set offers no improvement and actually hurts ε/k prediction. The baseline Morgan + RDKit model should remain the production configuration.

2. **Consider GC features with neural networks.** Random Forest may not be expressive enough to learn useful interactions between GC features and fingerprints. A neural network with attention or cross-feature interactions could potentially extract value from GC priors. However, this is a low-priority experiment given the current results.

3. **Improve the GC-PC-SAFT implementation.** The simplified group-contribution scheme used here is known to be suboptimal (see `model/gc_pcsaft.py` docstring). A full GC-PC-SAFT implementation with cross-interaction terms and proper ring corrections might produce more informative features. However, this would require significant engineering effort and access to published GC parameters.

4. **Test GC features on out-of-domain molecules.** GC-PC-SAFT might be more valuable for molecules far from the training set, where data-driven models have no nearby examples. An applicability domain analysis (Step 14) could test whether GC features improve OOD predictions.

## Readiness Check

- [x] `build_features()` supports `use_gc_pcsaft=True`
- [x] RF retrained with GC features; metrics compared to baseline
- [x] Ablation table shows baseline vs +GC vs GC-only
- [x] Report documents findings and recommendations
- [x] All 142 tests pass (added 4 new tests for GC featurization)
- [x] Ruff clean (no style violations in modified code)

## Conclusion

This experiment demonstrates that even poor models can fail to provide value when used as input features. The simplified GC-PC-SAFT predictions add no information beyond what Morgan fingerprints and RDKit descriptors already encode. The baseline Morgan + RDKit feature set remains the recommended configuration.

The negative result is scientifically valuable: it rules out a plausible hypothesis and documents that group-contribution priors are redundant with learned molecular representations for this dataset.
