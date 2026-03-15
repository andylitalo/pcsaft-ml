# Model Card: Ensemble (RF + GNN) for PC-SAFT Prediction

Following the model card framework proposed by Mitchell et al. (2019) "Model Cards for Model Reporting."

---

## Model Details

**Model Name**: PC-SAFT Inverse-Variance Weighted Ensemble

**Model Version**: v1.0 (2026-03-15)

**Model Type**: Ensemble meta-model combining Random Forest and Graph Neural Network predictions

**Model Architecture**:
- **Base Model 1**: Random Forest (100 trees, 2,150 RDKit+Morgan features)
- **Base Model 2**: Graph Neural Network (4-layer GIN, 964K params)
- **Ensemble Method**: Inverse-variance weighting
  - Weight per model: w_i = 1 / σ²_i (where σ_i is model's uncertainty estimate)
  - Final prediction: ŷ = (w_RF · ŷ_RF + w_GNN · ŷ_GNN) / (w_RF + w_GNN)
  - Ensemble uncertainty: σ_ensemble = 1 / √(w_RF + w_GNN)

**Ensemble Rationale**:
- **Hypothesis**: Combining 2D descriptor-based RF with graph-based GNN should leverage complementary information (topological vs. geometric features)
- **Expectation**: Ensemble should outperform individual models, especially for molecules where one model has high uncertainty

**Developers**: ML-Chem project team (2025-2026)

**Contact**: GitHub repository issues

**License**: MIT License

---

## Intended Use

**Primary Intended Uses**:
1. **Robust prediction**: Combine RF and GNN to reduce variance in final predictions
2. **Uncertainty-aware screening**: Use ensemble uncertainty to prioritize confident predictions
3. **Method comparison**: Benchmark ensemble performance vs. individual models

**Primary Intended Users**:
- Researchers testing ensemble methods for molecular property prediction
- Chemical engineers requiring uncertainty quantification for risk assessment

**Out-of-Scope Uses**:
- **Production deployment**: **Ensemble underperforms RF baseline** — do NOT use in practice (see Limitations)
- **High-stakes prediction**: Ensemble uncertainty poorly calibrated due to GNN overconfidence
- Same out-of-scope uses as RF and GNN (associating compounds, direct engineering design, etc.)

---

## Training Data

**Ensemble Training**: No direct training — ensemble weights computed from pre-trained RF and GNN uncertainties

**RF Training Data**: Esper et al. (1,801 molecules, experimental)

**GNN Training Data**: Mixed provenance (13,764 molecules: 72% experimental, 18% literature, 10% model-generated)

**Data Mismatch Issue**:
- RF and GNN trained on different datasets (1,801 vs. 13,764 molecules)
- GNN training includes 10% model-generated data (potential circular dependencies)
- No joint calibration of RF and GNN uncertainties on common validation set

---

## Evaluation Results

**Test Set Performance** (n=356 molecules, Esper et al. holdout):

| Parameter | MAE | RMSE | R² | Mean Target Value | Relative Error (%) |
|-----------|-----|------|-----|-------------------|--------------------|
| m (segments) | 0.651 | 0.912 | 0.533 | 2.78 | 23.4% |
| σ (Å) | 0.223 | 0.318 | 0.078 | 3.65 | 6.1% |
| ε/k (K) | 31.24 | 44.56 | 0.133 | 242.1 | 12.9% |

**Critical Finding: Ensemble UNDERPERFORMS RF Baseline**

| Parameter | R² (RF) | R² (GNN) | R² (Ensemble) | Ensemble vs RF |
|-----------|---------|----------|---------------|----------------|
| m | 0.620 | 0.762 | **0.533** | **-0.087** ❌ |
| σ | 0.354 | 0.774 | **0.078** | **-0.276** ❌ |
| ε/k | 0.332 | 0.731 | **0.133** | **-0.199** ❌ |

→ **ENSEMBLE IS WORSE THAN USING RF ALONE** for all three parameters.

**Root Cause Analysis** (see `docs/reports/17_ensemble_evaluation.md`):
1. **GNN overconfidence**: GNN reports low uncertainties (σ_GNN ~ 0.1-0.2) even for extrapolated predictions
2. **Inverse-variance weighting failure**: Overconfident GNN gets high weight (w_GNN >> w_RF), dominating ensemble
3. **Train-test mismatch**: GNN trained on different dataset than test set (13,764 vs. 1,801 molecules)
4. **Poor uncertainty calibration**: Neither RF nor GNN uncertainties well-calibrated; combining uncalibrated uncertainties amplifies error

**Effective Weights** (averaged across test set):
- w_RF / (w_RF + w_GNN): 0.18 (RF contributes only 18% to ensemble)
- w_GNN / (w_RF + w_GNN): 0.82 (GNN dominates due to low reported uncertainties)

→ Ensemble is ~82% GNN, but test set is from Esper (RF's training set), so GNN is extrapolating.

**Performance by Molecular Class**:

| Molecular Class | n_test | R²(m) | R²(σ) | R²(ε/k) |
|----------------|--------|-------|-------|---------|
| Hydrocarbons | 185 | 0.59 | 0.11 | 0.17 |
| HFOs (C2-C4) | 45 | 0.51 | 0.05 | 0.11 |
| Cyclic hydrocarbons | 60 | 0.54 | 0.08 | 0.14 |
| Ethers | 35 | 0.47 | -0.02 | 0.08 |
| Other | 31 | 0.42 | -0.05 | 0.05 |

→ Negative R² for σ and ε/k in some classes — worse than predicting mean.

---

## Limitations and Biases

**Critical Limitations**:
1. **❌ Ensemble underperforms individual models**: R² drops by 0.09-0.28 vs. RF baseline
2. **❌ GNN overconfidence**: Inverse-variance weighting assigns 82% weight to GNN, despite GNN extrapolating on Esper test set
3. **❌ Uncalibrated uncertainties**: Neither RF tree variance nor GNN ensemble std are well-calibrated
4. **❌ No joint calibration**: RF and GNN uncertainties never calibrated on common validation set
5. **❌ Dataset mismatch**: RF trained on 1,801 molecules (Esper), GNN on 13,764 (mixed provenance) → different error characteristics

**Why Ensemble Failed**:
- **Inverse-variance weighting assumption violated**: Requires accurate uncertainty estimates; GNN uncertainties are systematically underestimated
- **Alternative weighting schemes not tested**: Equal weighting (w_RF = w_GNN = 0.5) would perform better (estimated R²=0.55 for m)
- **No uncertainty recalibration**: Should have used isotonic regression to calibrate RF/GNN uncertainties on validation set

**Dataset Biases**:
- Inherits biases from both RF (refrigerant-centric Esper data) and GNN (mixed-provenance data with model-generated samples)
- Exacerbates GNN's circular training issue by upweighting model-generated predictions

**Environmental Impact**:
- Inference: 2× slower than single model (must run both RF and GNN)
- Training: None (no ensemble-specific training)

---

## Ethical Considerations

**Transparency Failure**:
- Ensemble provides illusion of improved uncertainty quantification, but uncertainties are poorly calibrated
- Users may over-trust low ensemble uncertainties, leading to risky decisions

**Reproducibility**:
- Ensemble predictions fully deterministic given RF and GNN models
- Code in `model/ensemble.py`

**Downstream Risks**:
- Worse predictions than RF baseline could lead to false prioritization of unsuitable molecules
- **CRITICAL**: Do NOT use ensemble for any production applications

---

## Recommendations

**⚠️ DO NOT USE THIS ENSEMBLE IN PRODUCTION**

**Recommended Actions**:
1. **Use RF baseline for Esper-like molecules**: If test molecules similar to Esper training set, use RF (R²=0.62 for m)
2. **Use GNN for novel chemistries**: If test molecules outside Esper training set, use GNN (R²=0.76 for m on GNN test set)
3. **Discard inverse-variance weighting**: GNN overconfidence makes this weighting scheme fail
4. **Try equal weighting**: Simple average (w_RF = w_GNN = 0.5) likely outperforms inverse-variance
5. **Recalibrate uncertainties**: Use isotonic regression or temperature scaling on validation set before ensembling

**Future Improvements**:
1. **Joint training**: Train RF and GNN on same dataset to ensure comparable error characteristics
2. **Uncertainty calibration**: Use Platt scaling or isotonic regression to calibrate RF/GNN uncertainties before weighting
3. **Stacking**: Train meta-learner to combine RF and GNN predictions (e.g., linear regression or XGBoost stacker)
4. **Equal weighting**: Test simple average as baseline before sophisticated weighting
5. **Conformal prediction**: Use conformal inference to generate calibrated prediction sets
6. **Alternative GNN**: Replace GNN with less overconfident model (e.g., Bayesian GNN, Monte Carlo dropout)

---

## Lessons Learned

**Key Takeaways**:
1. **Uncertainty calibration is critical for ensembling**: Inverse-variance weighting requires well-calibrated uncertainties
2. **Dataset mismatch causes issues**: Training RF and GNN on different datasets leads to incompatible error patterns
3. **Overconfidence is insidious**: GNN's low reported uncertainties upweight poor extrapolations
4. **Simple baselines often win**: RF baseline (R²=0.62) outperforms sophisticated ensemble (R²=0.53)
5. **Ablation studies are essential**: Should have tested equal weighting before assuming inverse-variance is best

**What Worked**:
- Ensemble framework is modular and extensible
- Uncertainty propagation math is correct (1 / √(w_RF + w_GNN))
- Individual models (RF, GNN) perform well on their respective test sets

**What Failed**:
- Assumption that GNN uncertainties reflect true prediction error
- No joint calibration on common validation set
- Inverse-variance weighting without verifying uncertainty calibration

---

## Maintenance and Updates

**Model Versioning**: v1.0 is deprecated for production use

**Update Plan**:
- **v1.1**: Implement equal weighting (w_RF = w_GNN = 0.5)
- **v1.2**: Add isotonic regression uncertainty calibration
- **v2.0**: Joint training on unified dataset (experimental + literature, no model-generated)

**Issue Reporting**: GitHub Issues

---

## References

- Mitchell, M., et al. (2019). "Model Cards for Model Reporting." *ACM FAT* Conference.
- Dietterich, T. (2000). "Ensemble Methods in Machine Learning." *MCS*.
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.
- Project repository: `model/ensemble.py`, `docs/reports/17_ensemble_evaluation.md`

---

**Last Updated**: 2026-03-15
**Model Version**: v1.0 (DEPRECATED)
**Card Version**: 1.0
