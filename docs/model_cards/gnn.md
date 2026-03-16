# Model Card: Graph Neural Network for PC-SAFT Prediction

Following the model card framework proposed by Mitchell et al. (2019) "Model Cards for Model Reporting."

---

## Model Details

**Model Name**: PC-SAFT GNN Predictor

**Model Version**: v1.0 (2026-03-15)

**Model Type**: Graph Neural Network (Graph Isomorphism Network, GIN architecture)

**Model Architecture**:
- **Input**: Molecular graph representation (atoms as nodes, bonds as edges)
- **Encoder**: 4 layers of Graph Isomorphism Network (GIN) convolutions
  - Hidden dimension: 256
  - Edge features: 4 (bond type, aromaticity, ring membership, conjugation)
  - Node features: 11 (atom type, degree, formal charge, hybridization, aromaticity, etc.)
- **Pooling**: Global mean pooling over nodes
- **Decoder**: 2-layer MLP per target parameter (256 → 128 → 1)
- **Total parameters**: ~964,000 trainable parameters

**Training Framework**: PyTorch 2.0, PyTorch Geometric 2.3

**Training Configuration**:
- Loss function: Multi-task MSE (separate head per parameter)
- Optimizer: AdamW (lr=1e-3, weight decay=1e-4)
- Batch size: 64
- Epochs: 100 (with early stopping, patience=20)
- Data augmentation: Random SMILES enumeration during training

**Developers**: ML-Chem project team (2025-2026)

**Contact**: GitHub repository issues

**License**: CC-BY-NC-SA 4.0 (model weights; trained on SPT-PCSAFT data which is CC-BY-NC-SA 4.0). Source code is MIT.

---

## Intended Use

**Primary Intended Uses**:
1. **Refrigerant screening**: Predict PC-SAFT parameters for novel halogenated hydrocarbons to identify promising blowing agent candidates
2. **Property estimation**: Provide initial PC-SAFT estimates for molecules lacking experimental data
3. **Virtual library enrichment**: Annotate large combinatorial libraries with thermodynamic parameters for prioritization

**Primary Intended Users**:
- Computational chemists conducting refrigerant screening
- Chemical engineers performing early-stage process design
- Researchers studying structure-property relationships in fluorinated compounds

**Out-of-Scope Uses**:
- **Direct engineering design**: Model predictions should NOT be used for final refrigerant formulation, charge calculations, or safety assessments without experimental validation
- **Regulatory submissions**: Predictions are not suitable for environmental compliance (GWP, ODP) or safety filings
- **Associating compounds**: Model trained only on non-associating molecules; unsuitable for alcohols, amines, acids, or hydrogen-bonding systems
- **Exotic chemistries**: Poor performance expected on organosilicon, organometallic, ionic liquids, or high-molecular-weight polymers

---

## Training Data

**Dataset**: Mixed-provenance PC-SAFT parameter collection (13,764 molecules)

**Data Provenance**:
- **Experimental (72%)**: Esper et al. Figshare collection (1,801 molecules), Dortmund Data Bank exports (4,200 molecules), Thol et al. literature compilation (3,850 molecules)
- **Literature-curated (18%)**: Parameters extracted from academic publications (2,500 molecules)
- **Model-generated (10%)**: Predictions from prior models re-used as training data (1,400 molecules) — **potential for circular dependencies**

**Provenance Labeling**: Training data tagged by source; model performance stratified by provenance in evaluation reports.

**Chemical Space Coverage**:
- Molecular weight: 16-400 Da (median 120 Da)
- Functional groups: Hydrocarbons (45%), halogenated (25%), ethers (12%), esters (8%), others (10%)
- Carbon count: C1-C20 (median C8)
- Training set dominated by simple hydrocarbons and refrigerants; sparse coverage of heteroatoms (N, S, P)

**Data Splitting**:
- Train: 11,012 molecules (80%)
- Validation: 1,376 molecules (10%)
- Test: 1,376 molecules (10%)
- Split stratified by molecular weight and halogen count to ensure representative test set

**Preprocessing**:
1. SMILES canonicalization via RDKit
2. Molecular graph construction (atoms → nodes, bonds → edges)
3. Feature calculation: Node features (11 per atom), edge features (4 per bond)
4. Outlier removal: Parameters outside 3σ from log-transformed mean removed (48 molecules)
5. No explicit deduplication beyond InChIKey uniqueness

**Known Data Quality Issues**:
- **Provenance mixing**: 10% of training data is model-generated, potentially introducing bias
- **Parameter correlations**: Some literature sources report only m and σ, with ε/k estimated via group contribution
- **Stereoisomer handling**: Stereoisomers treated as distinct molecules, even if parameters are averaged across isomers in source

---

## Evaluation Results

**Test Set Performance** (n=1,376 molecules):

| Parameter | MAE | RMSE | R² | Mean Target Value | Relative Error (%) |
|-----------|-----|------|-----|-------------------|--------------------|
| m (segments) | 0.312 | 0.485 | 0.762 | 2.84 | 11.0% |
| σ (Å) | 0.134 | 0.201 | 0.774 | 3.68 | 3.6% |
| ε/k (K) | 18.42 | 28.75 | 0.731 | 245.7 | 7.5% |

**Interpretation**:
- **Best performance**: σ (segment diameter) — GNN captures spatial features from graph structure
- **Moderate performance**: m (segment number) and ε/k (dispersion energy) — R² > 0.7 suitable for screening
- **Improvement over Random Forest**: GNN outperforms RF baseline by ΔR²=+0.14 (m), +0.42 (σ), +0.40 (ε/k)

**Performance by Molecular Class** (test set stratification):

| Molecular Class | n_test | R²(m) | R²(σ) | R²(ε/k) |
|----------------|--------|-------|-------|---------|
| Hydrocarbons | 620 | 0.81 | 0.82 | 0.78 |
| HFOs (C2-C4) | 180 | 0.73 | 0.76 | 0.71 |
| HCFOs | 120 | 0.70 | 0.72 | 0.68 |
| Cyclic hydrocarbons | 220 | 0.78 | 0.80 | 0.75 |
| Ethers | 140 | 0.65 | 0.68 | 0.62 |
| Other | 96 | 0.52 | 0.58 | 0.49 |

**Uncertainty Calibration**:
- Ensemble standard deviations (from 5-fold cross-validation) correlate with prediction error (Pearson r=0.68)
- 95% prediction intervals capture 91% of test set (slight underconfidence)
- High-uncertainty predictions (σ_pred > 0.5 for m) show R²=0.48 vs R²=0.82 for low-uncertainty

**Failure Modes**:
1. **Heavy halogens**: Chlorinated/brominated molecules (n=42 in test set) show R²=0.53 (ε/k) — training set dominated by fluorinated compounds
2. **Polyfunctional molecules**: Molecules with >2 heteroatom types (n=18) show R²=0.41 (σ)
3. **Large molecules**: MW > 300 Da (n=32) show R²=0.57 (m) — extrapolation beyond training distribution

---

## Limitations and Biases

**Model Limitations**:
1. **Associating compounds**: Model trained exclusively on non-associating PC-SAFT parameters; cannot predict association parameters (ε_AB, κ_AB) for hydrogen-bonding systems
2. **Extrapolation risk**: Performance degrades for molecules dissimilar to training set (Tanimoto < 0.3 to nearest neighbor)
3. **Stereochemistry**: Model encodes stereoisomer information in graph structure but shows minimal performance difference between stereoisomers (ΔR² < 0.02)
4. **Uncertainty underestimation**: Ensemble uncertainties calibrated on validation set; may underestimate error for out-of-distribution molecules
5. **No physical constraints**: Model does not enforce thermodynamic consistency (e.g., van der Waals inequality, critical point constraints)

**Dataset Biases**:
1. **Refrigerant-centric**: Training data skewed toward C2-C6 fluorinated hydrocarbons due to Esper/Thol sources
2. **Geography bias**: European/US sources dominate; potential underrepresentation of molecules studied primarily in Asian literature
3. **Temporal bias**: Training data from 1990-2024; recent novel chemistries (e.g., HFOs published 2020-2024) underrepresented
4. **Molecular weight bias**: 85% of training data has MW < 200 Da; model unreliable for large molecules or polymers
5. **Circular training**: 10% of training data is model-generated, potentially amplifying errors from prior models

**Fairness and Equity Considerations**:
- Not applicable — model operates on molecular structures, not human data
- Environmental justice: If used to design low-GWP refrigerants, could reduce climate impact disproportionately affecting vulnerable communities

**Environmental Impact**:
- Training: ~8 GPU-hours on NVIDIA A100 (estimated 1.2 kg CO₂e)
- Inference: ~0.05 ms per molecule (negligible carbon footprint)

---

## Ethical Considerations

**Dual-Use Risks**:
- Model could predict properties for ozone-depleting substances (CFCs) or high-GWP gases; no built-in safeguards to prevent misuse
- Recommendation: Pair with automated screening to flag Montreal Protocol-banned substances

**Reproducibility**:
- Model training deterministic with fixed random seed (42)
- Full training code, hyperparameters, and data splits available in repository (`model/gnn/`)
- Model checkpoints saved in `model/saved/gnn_pcsaft_v1.pt`

**Transparency**:
- Graph attention weights can be visualized to interpret model focus on specific molecular substructures
- Uncertainty estimates provided for all predictions to quantify confidence

**Downstream Risks**:
- Over-reliance on predictions without experimental validation could lead to costly synthesis of unsuitable molecules
- Mitigation: Always report uncertainty estimates alongside predictions; recommend experimental validation before scale-up

---

## Recommendations

**Usage Guidelines**:
1. **Check applicability domain**: Use Tanimoto similarity to training set (`tanimoto_nn` in prediction output); flag molecules with similarity < 0.3
2. **Inspect uncertainty**: Reject predictions with ensemble std > 0.5 (m), 0.3 (σ), 40 K (ε/k) for high-stakes applications
3. **Validate top candidates**: Experimentally measure PC-SAFT parameters for top 5-10 molecules before process design
4. **Cross-check with physics**: Use teqp to compute derived properties (boiling point, vapor pressure) and verify physical plausibility
5. **Update model**: Retrain with new experimental data every 6-12 months to incorporate latest literature

**Future Improvements**:
1. **Active learning**: Prioritize experimental measurements for high-uncertainty, high-value molecules to improve model
2. **Physics-informed training**: Incorporate thermodynamic constraints (e.g., critical point relations) as auxiliary losses
3. **Uncertainty quantification**: Use Bayesian neural networks or ensemble diversity to better calibrate prediction intervals
4. **Multi-fidelity learning**: Train on both high-quality experimental data and lower-quality simulation data with fidelity weighting
5. **Extend to association**: Add separate heads to predict ε_AB and κ_AB for hydrogen-bonding compounds

---

## Maintenance and Updates

**Model Versioning**: Semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Architecture changes or training data provenance changes
- MINOR: Hyperparameter tuning or training data expansion
- PATCH: Bug fixes or post-processing changes

**Update Schedule**: Quarterly re-evaluation against new literature data; annual retraining if new data > 10% of training set

**Issue Reporting**: GitHub Issues in repository

**Contact**: ml-chem-team@example.com (placeholder)

---

## References

- Mitchell, M., et al. (2019). "Model Cards for Model Reporting." *ACM FAT* Conference.
- Esper, G., et al. (2017). "PC-SAFT Parameters from Literature." Figshare Collection 6821654.
- Xu, K., et al. (2019). "How Powerful are Graph Neural Networks?" *ICLR*.
- Project repository: `model/gnn/`, `docs/reports/15_gnn_training.md`

---

**Last Updated**: 2026-03-15
**Model Version**: v1.0
**Card Version**: 1.0
