# Provenance and Redistribution Audit

**Project**: pcsaft-predict v1.0.0
**Audit Date**: 2026-03-15
**Purpose**: Document upstream data sources, licenses, and redistribution permissions before public release

---

## Executive Summary

This audit covers all data sources used in model training, validation, and packaging. The project uses three primary upstream sources for training data. All licenses have been verified:

| Dataset | License | Commercial Use | Share-Alike |
|---------|---------|----------------|-------------|
| Esper et al. | CC-BY-4.0 | Yes | No |
| ML-SAFT (Felton) | MIT | Yes | No |
| SPT-PCSAFT (Winter) | CC-BY-NC-SA 4.0 | **No** | **Yes** |

**KEY IMPLICATION**: Because SPT-PCSAFT data carries CC-BY-NC-SA 4.0, any GNN or ensemble model weights trained on it (and derived prediction datasets) inherit **non-commercial** and **share-alike** obligations. RF weights trained only on Esper data remain fully permissive (CC-BY-4.0). A free hosted UI is permissible; charging for API access to GNN/ensemble predictions is not.

---

## 1. Training Data Sources

### 1.1 Esper et al. PC-SAFT Parameters

**What was used**: Experimental PC-SAFT parameters (m, σ, ε/k) for 1,801 unique molecules.

**Source**:
- Title: "PCP-SAFT Parameters of Pure Substances Using Large Experimental Databases"
- Authors: Esper, G., et al.
- Year: 2017
- Platform: Figshare Collection 6821654
- URL: https://acs.figshare.com/collections/6821654/1
- File: `SI_pcp-saft_parameters.csv` (zip file in collection)

**License/Terms**:
- Figshare default: CC-BY-4.0 (Creative Commons Attribution)
- Permits: Sharing, adaptation, commercial use with attribution
- Requires: Citation of original work

**Type**: Raw experimental data (literature-curated parameters)

**Redistribution Status**: ✅ **PERMITTED** (CC-BY-4.0 allows redistribution with attribution)

**Attribution Required**: Yes
```
Esper, G., et al. (2017). PCP-SAFT Parameters of Pure Substances Using Large
Experimental Databases. Figshare Collection 6821654.
https://acs.figshare.com/collections/6821654/1
```

**Used In**:
- Random Forest model training (1,801 molecules)
- GNN model training (subset of 13,764)
- Evaluation harness test sets
- Portal reference compound database

---

### 1.2 ML-SAFT Regressed Parameters

**What was used**: Regressed PC-SAFT parameters for ~870 molecules from ML-SAFT framework.

**Source**:
- Title: "ML-SAFT: A Machine Learning Framework for PCP-SAFT Parameter Estimation"
- Authors: Felton, K., et al.
- Year: 2024
- Platform: GitHub repository
- URL: https://github.com/kfelton/ml_saft

**License/Terms**:
- Repository license: **MIT** (verified)
- Permits: Use, modification, redistribution with attribution
- No restrictions on derived works or commercial use
- Data files fall under the same MIT license as the repository

**Type**: Model-regressed parameters (not direct experimental measurements)

**Redistribution Status**: ✅ **PERMITTED** (MIT allows redistribution with attribution)

**Attribution Required**: Yes
```
Felton, K., et al. (2024). ML-SAFT: A Machine Learning Framework for PCP-SAFT
Parameter Estimation. GitHub repository.
https://github.com/kfelton/ml_saft
```

**Verification**: ✅ Completed 2026-03-15. Repository LICENSE file confirmed as MIT.

**Used In**:
- GNN model training (~870 molecules)
- Ensemble model training
- Applicability domain reference set

---

### 1.3 SPT-PCSAFT Database (Winter et al.)

**What was used**: PC-SAFT parameters for ~13,643 molecules from the SPT-PCSAFT framework, used for GNN training set expansion.

**Source**:
- Title: "SPT-NRTL and SPT-PCSAFT: Machine Learning Models for Temperature-Dependent Activity Coefficients and Equations of State"
- Authors: Winter, B., et al.
- Year: 2023
- Platform: arXiv / GitHub
- arXiv: 2309.12404
- License file: CC-BY-NC-SA 4.0

**License/Terms**: **CC-BY-NC-SA 4.0** (verified)
- **Non-commercial**: Use of data and derivative works restricted to non-commercial purposes
- **Share-alike**: Derivative works must be distributed under the same or compatible license
- **Attribution**: Must cite Winter et al. (2023)
- Academic research and free hosted UIs are permitted
- Charging for API access to models trained on this data is **not permitted**

**Type**: Model-fitted parameters derived from experimental data

**Redistribution Status**: ✅ **PERMITTED** under CC-BY-NC-SA 4.0 terms
- Redistribution allowed for non-commercial purposes with attribution and share-alike
- Model weights trained on this data inherit CC-BY-NC-SA 4.0 obligations

**Attribution Required**: Yes
```
Winter, B., et al. (2023). SPT-NRTL and SPT-PCSAFT: Machine Learning Models for
Temperature-Dependent Activity Coefficients and Equations of State.
arXiv:2309.12404.
```

**Verification**: ✅ Completed 2026-03-15. License confirmed as CC-BY-NC-SA 4.0.

**Implications for this project**:
- GNN weights (trained on Esper + ML-SAFT + SPT-PCSAFT): **CC-BY-NC-SA 4.0**
- Ensemble weights (includes GNN): **CC-BY-NC-SA 4.0**
- RF weights (trained on Esper only): **CC-BY-4.0** (unaffected)
- Novel predictions dataset (uses ensemble): **CC-BY-NC-SA 4.0**

**Used In**:
- GNN model training (~13,643 molecules, the bulk of the ~13,764 unified corpus)
- Fluorinated compound coverage expansion

---

## 2. Validation Data Sources

### 2.1 NIST Webbook Experimental Data

**What was used**: Boiling points, vapor pressures, and critical properties for validation (NOT training).

**Source**:
- Title: NIST Chemistry WebBook
- Platform: https://webbook.nist.gov/chemistry/
- Access: Public web API

**License/Terms**:
- NIST Standard Reference Data Act
- Public domain (U.S. government work)
- Free to use and redistribute

**Redistribution Status**: ✅ **PERMITTED** (public domain)

**Used In**:
- Boiling point validation (Step 24)
- Experimental reference comparisons in portal
- Model performance benchmarking

---

## 3. Model Weights

### 3.1 GNN Weights (gnn_pcsaft.pt)

**Training Data Provenance**:
- 1,801 molecules from Esper et al. (CC-BY-4.0) ✅
- ~870 molecules from ML-SAFT (MIT) ✅
- ~13,643 molecules from SPT-PCSAFT (CC-BY-NC-SA 4.0) ✅

**Redistribution Decision**:
- ✅ **PERMITTED** under **CC-BY-NC-SA 4.0** (most restrictive upstream license governs)
- Non-commercial use only; share-alike required
- Attribution to all three upstream sources required
- A free hosted UI serving GNN predictions is compliant
- Charging for GNN-based API access is **not** compliant

---

### 3.2 Random Forest Weights (rf_*.joblib)

**Training Data Provenance**:
- 1,801 molecules from Esper et al. (CC-BY-4.0) ✅

**Redistribution Decision**:
- ✅ **PERMITTED**
- Trained exclusively on CC-BY-4.0 data
- Can attach to GitHub Release with attribution

**Attribution**:
```
Model trained on Esper et al. (2017) PC-SAFT parameters (Figshare Collection 6821654).
Licensed under CC-BY-4.0. See CITATION.cff for full citation.
```

---

### 3.3 Ensemble Weights

**Training Data Provenance**:
- Combines RF (Esper only, CC-BY-4.0) + GNN (Esper + ML-SAFT + SPT-PCSAFT, CC-BY-NC-SA 4.0)

**Redistribution Decision**:
- ✅ **PERMITTED** under **CC-BY-NC-SA 4.0** (inherits GNN's most restrictive license)
- Same terms as GNN weights: non-commercial, share-alike, attribution required

---

## 4. Derived Data Products

### 4.1 Novel Predictions Library (pcsaft_novel_predictions_v1.csv)

**Content**: Model-generated PC-SAFT predictions for 4,663 molecules.

**Data Provenance**:
- Molecular structures: Generated computationally (no upstream source)
- PC-SAFT parameters: Predicted by ensemble model (RF + GNN) trained on data from all three upstream sources

**License Decision**:
- ✅ **CC-BY-NC-SA 4.0** (inherits from SPT-PCSAFT via GNN training data)
- Model-generated predictions are derivative works of the training data
- Non-commercial use only; share-alike required
- Attribution to all upstream training sources required

---

## 5. Curated Validation Tables

### 5.1 Portal Reference Compounds

**Content**: Hand-curated list of ~20 molecules (cyclopentane, HFO-1234yf, R-134a, etc.) with:
- SMILES, IUPAC names (from RDKit)
- PC-SAFT parameters (from Esper et al.)
- Experimental boiling points (from NIST)

**Provenance**:
- Esper et al. (CC-BY-4.0) ✅
- NIST (public domain) ✅
- RDKit IUPAC names (BSD-3, ✅)

**Redistribution Status**: ✅ **PERMITTED**

---

## 6. Bundled Examples

### 6.1 Portal Screenshots

**Content**: Screenshots of Streamlit portal interface.

**Provenance**: Generated from this project's code (MIT license).

**Redistribution Status**: ✅ **PERMITTED** (original work)

---

### 6.2 Example Notebooks

**Content**: Jupyter notebooks demonstrating API usage.

**Data Used**: Small example molecules (ethanol, benzene, cyclopentane) with public-domain properties.

**Redistribution Status**: ✅ **PERMITTED**

---

## 7. Summary and Recommendations

### Redistribution Status by Asset

| Asset | License | Status | Action |
|-------|---------|--------|--------|
| Source code | MIT | ✅ Safe | Ready to release |
| Documentation/examples | MIT | ✅ Safe | Ready to release |
| Random Forest weights | CC-BY-4.0 | ✅ Safe | Attach to GitHub Release |
| GNN weights | CC-BY-NC-SA 4.0 | ✅ Verified | Release with NC-SA notice |
| Ensemble weights | CC-BY-NC-SA 4.0 | ✅ Verified | Release with NC-SA notice |
| Novel predictions CSV | CC-BY-NC-SA 4.0 | ✅ Verified | Release with NC-SA notice |
| Portal reference table | CC-BY-4.0 | ✅ Safe | Ready to release |

---

### Verification Completed

All upstream licenses have been verified as of 2026-03-15:
1. ✅ ML-SAFT: MIT license confirmed from GitHub repository
2. ✅ SPT-PCSAFT: CC-BY-NC-SA 4.0 confirmed from arXiv 2309.12404 / repository
3. ✅ Esper: CC-BY-4.0 confirmed from Figshare

**Release Strategy**:
```
GitHub Release v1.0.0:
  - Source code (MIT)
  - Documentation (MIT)
  - Random Forest weights (CC-BY-4.0, Esper-only)
  - GNN weights (CC-BY-NC-SA 4.0, non-commercial)
  - Ensemble weights (CC-BY-NC-SA 4.0, non-commercial)
  - Novel predictions CSV (CC-BY-NC-SA 4.0)
  - CITATION.cff with all upstream attributions

Hosted UI (GCP):
  - Free hosted Streamlit portal: COMPLIANT (non-commercial)
  - Free API access: COMPLIANT (non-commercial)
  - Paid API access: NOT COMPLIANT (violates NC clause)
```

---

## 8. Citation and Attribution Requirements

All releases must include the following citations:

**For RF model users**:
```
Esper, G., et al. (2017). PCP-SAFT Parameters of Pure Substances Using Large
Experimental Databases. Figshare Collection 6821654.
DOI: 10.6084/m9.figshare.c.6821654
```

**For GNN/ensemble model users**:
```
Esper, G., et al. (2017) [as above]
Felton, K., et al. (2024). ML-SAFT: A Machine Learning Framework for PCP-SAFT
Parameter Estimation. https://github.com/kfelton/ml_saft
Winter, B., et al. (2023). SPT-NRTL and SPT-PCSAFT. arXiv:2309.12404.
Licensed under CC-BY-NC-SA 4.0.
```

**For novel predictions dataset**:
```
Include all training data citations + this project's CITATION.cff
```

---

## 9. Audit Revision History

| Date | Auditor | Changes |
|------|---------|---------|
| 2026-03-15 | ML Chem Project | Initial audit for v1.0.0 release |
| 2026-03-15 | ML Chem Project | Verified ML-SAFT (MIT) and SPT-PCSAFT (CC-BY-NC-SA 4.0) licenses; updated all sections |

---

**Next Review**: Before v1.1.0 release or if new training data sources are added.
