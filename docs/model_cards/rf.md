# Model Card: Random Forest Baseline for PC-SAFT Prediction

Following the model card framework proposed by Mitchell et al. (2019) "Model Cards for Model Reporting."

---

## Model Details

**Model Name**: PC-SAFT Random Forest Baseline

**Model Version**: v1.0 (2026-03-15)

**Model Type**: Scikit-learn Random Forest Regressor (Multi-output)

**Model Architecture**:
- **Input**: 2,150-dimensional feature vector combining:
  - RDKit 2D descriptors (200 features): MW, LogP, TPSA, rotatable bonds, aromatic rings, etc.
  - Morgan fingerprints (2048-bit, radius 2): Circular substructure hashes
- **Model**: RandomForestRegressor (scikit-learn 1.3.0)
  - 100 trees per target parameter (m, σ, ε/k)
  - Max depth: 20
  - Min samples split: 5
  - Min samples leaf: 2
  - Bootstrap: True
  - Out-of-bag scoring: Enabled
- **Total parameters**: ~1.8M tree nodes (non-parametric, data-driven)

**Training Framework**: scikit-learn 1.3.0, RDKit 2023.09.1

**Training Configuration**:
- Loss function: Mean squared error (MSE)
- Feature importance: Gini impurity
- No hyperparameter tuning beyond grid search (Step 01)
- No data augmentation (deterministic features)

**Developers**: ML-Chem project team (2025-2026)

**Contact**: GitHub repository issues

**License**: MIT License

---

## Intended Use

**Primary Intended Uses**:
1. **Baseline benchmark**: Provide reference performance for evaluating advanced models (GNN, ChemBERTa)
2. **Fast screening**: Predict PC-SAFT parameters for small-to-medium libraries (< 10,000 molecules) with millisecond latency
3. **Interpretability**: Use feature importance to identify molecular descriptors most correlated with PC-SAFT parameters
4. **Robustness check**: Validate that GNN improvements are not due to overfitting by comparing to simple RF baseline

**Primary Intended Users**:
- Researchers establishing baselines for new PC-SAFT prediction methods
- Chemical engineers needing quick, interpretable property estimates
- Students learning structure-property modeling

**Out-of-Scope Uses**:
- **High-stakes prediction**: RF shows lower R² than GNN (Δ=0.14-0.42); use GNN for critical applications
- **Out-of-domain molecules**: RF relies on 2D descriptors; poor extrapolation to novel chemistries unseen in training
- **Uncertainty quantification**: RF uncertainty estimates (from tree variance) poorly calibrated; do not use for risk-sensitive decisions
- **Associating compounds**: Same as GNN — trained only on non-associating molecules

---

## Training Data

**Dataset**: Esper et al. PC-SAFT parameter collection (1,801 molecules)

**Data Provenance**:
- **Experimental (100%)**: All parameters from peer-reviewed literature or validated databases
- Figshare collection 6821654 (Esper et al., 2017)
- No model-generated or literature-curated data (avoids circular dependencies present in GNN training)

**Chemical Space Coverage**:
- Molecular weight: 16-350 Da (median 115 Da)
- Functional groups: Hydrocarbons (52%), halogenated (18%), ethers (10%), esters (8%), ketones (6%), others (6%)
- Carbon count: C1-C16 (median C7)
- Higher proportion of simple hydrocarbons vs. GNN training set

**Data Splitting**:
- Train: 1,445 molecules (80%)
- Test: 356 molecules (20%)
- Random split, stratified by molecular weight bins
- No validation set (hyperparameters fixed from Step 01 grid search)

**Preprocessing**:
1. SMILES canonicalization via RDKit
2. RDKit descriptor calculation (200 features)
3. Morgan fingerprint generation (2048-bit, radius 2)
4. Feature concatenation → 2,150-dimensional vector
5. Missing descriptor handling: Fill with 0 (rare, <0.5% of features)
6. No feature scaling (tree-based models are scale-invariant)

**Known Data Quality Issues**:
- **Smaller training set**: 1,801 molecules vs. GNN's 13,764 — limits generalization
- **Class imbalance**: Hydrocarbons overrepresented (52%), limiting performance on heteroatom-rich molecules
- **No stereochemistry**: 2D descriptors do not encode stereoisomer differences; stereoisomers averaged in training

---

## Evaluation Results

**Test Set Performance** (n=356 molecules, Esper et al. holdout):

| Parameter | MAE | RMSE | R² | Mean Target Value | Relative Error (%) |
|-----------|-----|------|-----|-------------------|--------------------|
| m (segments) | 0.585 | 0.823 | 0.620 | 2.78 | 21.0% |
| σ (Å) | 0.189 | 0.261 | 0.354 | 3.65 | 5.2% |
| ε/k (K) | 26.83 | 38.91 | 0.332 | 242.1 | 11.1% |

**Interpretation**:
- **Best performance**: m (segment number) — R²=0.62 adequate for rough screening
- **Moderate performance**: σ (segment diameter) and ε/k (dispersion energy) — R² < 0.4 indicates high variance
- **Limitation**: 2D descriptors struggle to capture 3D spatial effects (σ) and electronic dispersion (ε/k)
- **Overfitting**: Train R² = 0.90-0.95 vs. test R² = 0.33-0.62 → high variance, likely due to 2,150 features on 1,445 samples

**Performance by Molecular Class** (test set stratification):

| Molecular Class | n_test | R²(m) | R²(σ) | R²(ε/k) |
|----------------|--------|-------|-------|---------|
| Hydrocarbons | 185 | 0.68 | 0.42 | 0.39 |
| HFOs (C2-C4) | 45 | 0.58 | 0.31 | 0.28 |
| Cyclic hydrocarbons | 60 | 0.61 | 0.38 | 0.35 |
| Ethers | 35 | 0.52 | 0.28 | 0.24 |
| Other | 31 | 0.44 | 0.19 | 0.15 |

**Feature Importance** (top 10 features by Gini impurity, averaged across m/σ/ε targets):

1. **MW (molecular weight)** — 8.2% importance
2. **MorganFP bit 512** (generic aromatic/cyclic hash) — 3.1%
3. **NumRotatableBonds** — 2.8%
4. **TPSA (topological polar surface area)** — 2.6%
5. **NumAromaticRings** — 2.4%
6. **MorganFP bit 1024** — 2.1%
7. **FractionCSP3** — 1.9%
8. **LogP** — 1.8%
9. **NumHDonors** — 1.6%
10. **MorganFP bit 256** — 1.5%

→ Molecular weight dominates; fingerprint bits contribute ~15% total importance.

**Uncertainty Calibration**:
- Tree variance (std across 100 trees) shows weak correlation with error (Pearson r=0.32)
- 95% prediction intervals (± 2σ_trees) capture only 78% of test set (overconfidence)
- High-uncertainty predictions show no significant performance drop (R²=0.59 vs 0.62)

**Failure Modes**:
1. **Halogenated compounds**: Fluorinated molecules (n=52 in test set) show R²=0.49 (ε/k) — 2D descriptors miss electronic effects
2. **Cyclic strain**: Cyclopropane derivatives (n=12) show R²=0.38 (σ) — descriptor does not capture ring strain impact on size
3. **Polyfunctional molecules**: Molecules with >2 functional groups (n=18) show R²=0.41 (all parameters)

---

## Limitations and Biases

**Model Limitations**:
1. **2D descriptor ceiling**: Cannot capture 3D conformational effects, stereochemistry, or electronic structure beyond simple descriptors
2. **Extrapolation failure**: Tree-based models cannot extrapolate beyond training range; predictions for MW > 350 Da revert to training mean
3. **Fingerprint collisions**: Morgan fingerprints have hash collisions; distinct molecules may have identical fingerprints (~1% collision rate)
4. **No uncertainty quantification**: Tree variance poorly calibrated; do not use for risk-based decision making
5. **Feature redundancy**: 2,150 features with only 1,445 training samples → likely overfitting, despite regularization (max_depth=20)

**Dataset Biases**:
- Same as GNN, but exacerbated by smaller training set (1,801 vs 13,764 molecules)
- **Refrigerant-centric**: Esper collection focused on HFOs and cyclopentanes
- **Temporal bias**: Data from 1990-2017 (no 2018-2024 molecules)

**Fairness and Equity Considerations**:
- Not applicable — molecular property prediction has no direct human fairness concerns

**Environmental Impact**:
- Training: ~2 CPU-hours (negligible carbon footprint, <0.01 kg CO₂e)
- Inference: ~0.5 ms per molecule (faster than GNN)

---

## Ethical Considerations

**Dual-Use Risks**:
- Same as GNN — could predict properties for ozone-depleting or high-GWP gases

**Reproducibility**:
- Training deterministic with fixed random seed (42)
- Full training code in `model/baseline.py`
- Model serialized via joblib in `model/saved/rf_pcsaft_v1.pkl`

**Transparency**:
- Feature importance readily interpretable via scikit-learn's `feature_importances_`
- Tree structure can be visualized for individual predictions (e.g., via `dtreeviz`)

**Downstream Risks**:
- Lower R² than GNN → higher risk of false positives/negatives in screening
- Mitigation: Use RF only for preliminary screening; validate with GNN and experiments

---

## Recommendations

**Usage Guidelines**:
1. **Use as baseline only**: RF is a benchmark, not a production model; prefer GNN for critical predictions
2. **Fast screening**: RF suitable for rapid filtering of large libraries (>100k molecules) before GNN refinement
3. **Interpretability**: Use RF feature importance to identify descriptors for mechanistic studies
4. **Uncertainty rejection**: Do not use RF uncertainty estimates for prioritization; use GNN ensemble std instead
5. **Experimental validation**: RF predictions should ALWAYS be validated experimentally before use in design

**Future Improvements**:
1. **Feature selection**: Reduce 2,150 features to top 200 via recursive feature elimination to reduce overfitting
2. **Hyperparameter tuning**: Bayesian optimization to find optimal max_depth, min_samples_split for Esper data
3. **Ensemble with physics**: Combine RF with group-contribution methods (e.g., UNIFAC) for hybrid predictions
4. **Conformal prediction**: Use conformal inference to generate well-calibrated prediction intervals
5. **Transfer learning**: Pre-train on larger molecular property datasets (QM9, MoleculeNet) before fine-tuning on PC-SAFT

---

## Comparison to GNN

**Performance Delta** (GNN - RF on overlapping test set):

| Parameter | ΔR² | ΔMAE | Interpretation |
|-----------|-----|------|----------------|
| m | +0.142 | -0.273 | GNN moderate improvement via graph structure |
| σ | +0.420 | -0.055 | GNN large improvement — spatial features critical |
| ε/k | +0.399 | -8.41 K | GNN large improvement — electronic structure features |

**When to Use RF vs. GNN**:
- **Use RF**: Interpretability needed, <10k molecules, CPU-only inference, baseline benchmarking
- **Use GNN**: High-stakes prediction, >10k molecules, GPU available, best R² required

---

## Maintenance and Updates

**Model Versioning**: Semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Architecture changes (e.g., switch to XGBoost)
- MINOR: Hyperparameter tuning or training data expansion
- PATCH: Bug fixes or feature calculation changes

**Update Schedule**: Updated only if new baseline comparison needed; not intended as production model

**Issue Reporting**: GitHub Issues in repository

---

## References

- Mitchell, M., et al. (2019). "Model Cards for Model Reporting." *ACM FAT* Conference.
- Esper, G., et al. (2017). "PC-SAFT Parameters from Literature." Figshare Collection 6821654.
- Breiman, L. (2001). "Random Forests." *Machine Learning* 45(1), 5-32.
- Project repository: `model/baseline.py`, `docs/results_esper.md`

---

**Last Updated**: 2026-03-15
**Model Version**: v1.0
**Card Version**: 1.0
