# Changelog

All notable changes to pcsaft-predict will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-15

### Added
- Initial public release
- **Models**:
  - GNN model (Graph Isomorphism Network): R² = 0.76/0.77/0.73 on m/σ/ε/k, trained on 13,764 molecules (Esper + ML-SAFT + SPT-PCSAFT)
  - Random Forest baseline: R² = 0.62/0.35/0.33 on m/σ/ε/k, trained on 1,801 molecules (Esper)
  - Ensemble model: Inverse-variance-weighted RF + GNN combination
  - PyTorch Multi-Task Neural Network (PCSAFTNet): baseline MLP architecture
  - Fine-tuned ChemBERTa-zinc-base-v1: transformer-based prediction
- **Uncertainty quantification**:
  - MC Dropout uncertainty for GNN predictions
  - RF tree variance for Random Forest predictions
  - Ensemble standard deviation across models
- **Applicability domain**:
  - Tanimoto nearest-neighbor similarity checking
  - In-domain flagging (threshold: Tanimoto > 0.3)
  - Training set composition analysis
- **Thermodynamic validation**:
  - Boiling point computation from PC-SAFT parameters via teqp
  - Henry's constant estimation at infinite dilution in water
  - Vapor pressure and density calculations at 298 K
  - EOS-based property validation against experimental data
- **Screening capabilities**:
  - Novel predictions dataset: 4,612 fluorinated/chlorinated cycloalkenes
  - HFO-centric screening with boiling point and Henry's constant ranking
  - Cyclopentane-like candidate identification
  - Application presets for blowing agents, refrigerants, and solvents
- **Package features**:
  - Installable `pcsaft-predict` package with simple API
  - Streamlit portal with interactive visualization
  - Batch prediction support
  - SMILES validation and canonicalization
  - Comprehensive documentation and examples
- **Infrastructure**:
  - Docker containerization
  - Kubernetes deployment manifests
  - Kubeflow training pipeline
  - FastAPI serving endpoint (archived)
  - GitHub Actions CI/CD
  - Comprehensive test suite

### Models
- `gnn`: 4-layer Graph Isomorphism Network (PCSAFTGraphNet), 964K parameters, trained on Esper + ML-SAFT + SPT-PCSAFT
- `rf`: Random Forest (100 trees per target), trained on Esper dataset
- `ensemble`: Inverse-variance-weighted combination of RF + GNN
- `nn`: PyTorch MLP (PCSAFTNet), 3-layer feed-forward network, trained on Esper
- `chemberta`: Fine-tuned ChemBERTa-zinc-base-v1, transformer architecture, trained on Esper

### Data
- Training set: 13,764 unique molecules with experimental PC-SAFT parameters
  - Esper et al. (2017): 1,801 molecules from Figshare Collection 6821654
  - Felton et al. (2024) ML-SAFT: ~870 molecules from GitHub repository
  - Winter et al. (2025) SPT-PCSAFT: ~13,643 molecules from arXiv 2309.12404
- Test set: 1,376 molecules (held-out, disjoint from training)
- Novel predictions library: 4,612 molecules (model-generated, version 1.0)

### Documentation
- Model cards for all 5 models (GNN, RF, Ensemble, NN, ChemBERTa)
- Dataset datasheet following Gebru et al. (2021) framework
- CITATION.cff for academic citation
- Comprehensive README with usage examples
- CONTRIBUTING.md with development guidelines
- 30 step-by-step reports documenting development process

### Performance Highlights
- **GNN (default model)**:
  - m: MAE = 0.31 segments, R² = 0.76
  - σ: MAE = 0.13 Å, R² = 0.77
  - ε/k: MAE = 18.4 K, R² = 0.73
- **Ensemble (best overall)**:
  - Matches GNN performance with improved uncertainty estimates
  - Effective for molecules with moderate Tanimoto similarity (0.3-0.7)
- **RF (fastest inference)**:
  - Suitable for CPU-only environments
  - 10x faster than GNN for batch predictions

### Known Limitations
- Predictions are extrapolations for novel fluorinated molecules (0% within applicability domain)
- Training set dominated by simple hydrocarbons; exotic functional groups unreliable
- No associating compound support (alcohols, amines, acids not suitable)
- Ensemble uncertainty may underestimate true error, especially for ε/k
- Boiling point validation shows systematic bias for high-boiling compounds (>350 K)

### Breaking Changes
None (initial release)

---

## [Unreleased]

### Planned for v1.1.0
- [ ] Add thermodynamic properties (BP, VP, density) for all 4,612 molecules
- [ ] Package GNN and ensemble models (currently only RF is packaged)
- [ ] Add CLI entry point for batch predictions
- [ ] Improve uncertainty calibration via conformal prediction
- [ ] Add model performance dashboard to portal

### Planned for v2.0.0
- [ ] Re-train models with expanded training set (20,000+ molecules)
- [ ] Add associating compound support (alcohols, amines)
- [ ] Expand chemical space to C7-C8 hydrocarbons and sulfur-containing compounds
- [ ] Implement active learning for targeted experimental validation
- [ ] Add transfer learning support for related property predictions

---

[1.0.0]: https://github.com/example/pcsaft-predict/releases/tag/v1.0.0
