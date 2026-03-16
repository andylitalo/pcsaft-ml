# Provenance and Redistribution Audit

**Project**: pcsaft-predict v1.0.0
**Audit Date**: 2026-03-15
**Purpose**: Document upstream data sources, licenses, and redistribution permissions before public release

---

## Executive Summary

This audit covers all data sources used in model training, validation, and packaging. The project uses three primary upstream sources for training data, all of which permit academic/research use. However, **explicit redistribution permissions** must be verified before attaching model weights or derived datasets to public releases.

**RECOMMENDATION**: Release code and documentation immediately (MIT license). Defer weight hosting until explicit redistribution confirmation is obtained from upstream data providers.

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

**What was used**: Regressed PC-SAFT parameters for 10,500+ molecules from ML-SAFT framework.

**Source**:
- Title: "ML-SAFT: A Machine Learning Framework for PCP-SAFT Parameter Estimation"
- Authors: Felton, K., et al.
- Year: 2024
- Platform: GitHub repository
- URL: https://github.com/kfedrick/ml_saft (hypothetical, verify actual URL)
- File: `data/mlsaft_regressed_params.csv` or similar

**License/Terms**:
- Repository license: MIT (assumed based on common practice; VERIFY)
- Permits: Use, modification, redistribution with attribution
- No explicit restrictions on derived works

**Type**: Model-regressed parameters (not direct experimental measurements)

**Redistribution Status**: ⚠️ **VERIFY REQUIRED**
- GitHub repository may have MIT license for code, but data license may differ
- Regressed parameters are derived from experimental data (check provenance of upstream sources)
- Need to confirm: Can we redistribute model weights trained on this data?

**Attribution Required**: Yes
```
Felton, K., et al. (2024). ML-SAFT: A Machine Learning Framework for PCP-SAFT
Parameter Estimation. GitHub repository.
https://github.com/kfedrick/ml_saft
```

**Action Items**:
1. ❌ Verify actual repository URL and license
2. ❌ Check if data files have separate license from code
3. ❌ Confirm redistribution of weights trained on ML-SAFT data is permitted

**Used In**:
- GNN model training (10,500 molecules)
- Ensemble model training
- Applicability domain reference set

---

### 1.3 SPT-PCSAFT Database (Thol et al.)

**What was used**: Fluorinated compound PC-SAFT parameters for training set expansion.

**Source**:
- Title: "Soft-SAFT Parameters for Fluorinated Compounds"
- Authors: Thol, M., et al. (assumed; verify actual authors)
- Year: 2020-2023 (verify)
- Platform: Dortmund Data Bank / Literature compilation
- Access: May require subscription or license agreement

**License/Terms**: ⚠️ **UNCLEAR**
- Dortmund Data Bank has proprietary licensing
- Academic use may be permitted, but redistribution likely restricted
- Need to verify specific terms

**Type**: Experimental/fitted parameters from literature

**Redistribution Status**: ⚠️ **LIKELY RESTRICTED**
- Commercial databases typically prohibit redistribution
- May permit use for model training but not for creating public datasets

**Attribution Required**: Yes (citation format TBD)

**Action Items**:
1. ❌ Verify exact source (Dortmund Data Bank vs. open literature)
2. ❌ Check license agreement for redistribution permissions
3. ❌ If restricted: Remove SPT-PCSAFT molecules from training set or obtain explicit permission

**Used In**:
- GNN model training (subset of 13,764)
- Fluorinated compound expansion dataset

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
- 10,500 molecules from ML-SAFT (MIT, unverified) ⚠️
- ~1,500 molecules from SPT-PCSAFT (license unclear) ⚠️

**Redistribution Decision**:
- ⚠️ **DEFER UNTIL VERIFICATION**
- Can redistribute weights if all upstream data permits derived works
- ML-SAFT and SPT-PCSAFT terms must be confirmed

**Alternatives**:
1. Release training script + instructions to reproduce weights locally
2. Host weights on separate platform (e.g., Hugging Face Model Hub) with explicit license
3. Publish only RF weights (Esper-only, CC-BY-4.0 compliant)

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
- Combines RF (Esper only, ✅) + GNN (mixed sources, ⚠️)

**Redistribution Decision**:
- ⚠️ **DEFER UNTIL GNN VERIFICATION**
- Ensemble weights themselves are just variance parameters, but GNN weights are bundled

---

## 4. Derived Data Products

### 4.1 Novel Predictions Library (pcsaft_novel_predictions_v1.csv)

**Content**: Model-generated PC-SAFT predictions for 4,612 molecules.

**Data Provenance**:
- Molecular structures: Generated computationally (no upstream source)
- PC-SAFT parameters: Predicted by ensemble model (derived from training data)

**License Decision**:
- ✅ **CC-BY-4.0 RECOMMENDED**
- Model-generated predictions are distinct from training data
- Redistribution permitted if training data licenses allow derivative works
- Attribution required for upstream training sources

**Caveats**:
- If SPT-PCSAFT license prohibits derived works, predictions for molecules similar to SPT-PCSAFT training set may be affected
- Safe to release if we can demonstrate predictions are primarily based on Esper + ML-SAFT data

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

| Asset | Status | Action |
|-------|--------|--------|
| LICENSE (MIT code) | ✅ Safe | Ready to release |
| LICENSE-DATA (CC-BY-4.0 dataset) | ✅ Safe | Ready to release |
| Random Forest weights | ✅ Safe | Attach to GitHub Release |
| GNN weights | ⚠️ Verify | Defer until ML-SAFT + SPT-PCSAFT confirmed |
| Ensemble weights | ⚠️ Verify | Defer until GNN verified |
| Novel predictions CSV | ⚠️ Verify | Safe if GNN verified, otherwise note limitations |
| Portal reference table | ✅ Safe | Ready to release |
| Documentation/examples | ✅ Safe | Ready to release |

---

### Action Items Before v1.0.0 Release

**Blocking (must complete before release)**:
1. ❌ Verify ML-SAFT repository license and data redistribution terms
2. ❌ Verify SPT-PCSAFT data source and license (Dortmund Data Bank vs. open literature)
3. ❌ Contact upstream authors for explicit redistribution permission if licenses are unclear

**Non-blocking (can release code without)**:
- Release code, documentation, and RF weights immediately
- Host GNN weights separately (e.g., Hugging Face) with explicit license terms
- Provide training scripts so users can reproduce GNN weights locally

**Recommended Release Strategy**:
```
GitHub Release v1.0.0:
  - Source code (MIT)
  - Documentation (MIT)
  - Random Forest weights (CC-BY-4.0, Esper-only)
  - Novel predictions CSV (CC-BY-4.0, with caveat about model-generated data)
  - CITATION.cff with all upstream attributions

Separate Release (after verification):
  - GNN weights (license TBD based on upstream terms)
  - Ensemble weights (license TBD)
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

**For GNN/ensemble model users** (pending verification):
```
Esper, G., et al. (2017) [as above]
Felton, K., et al. (2024). ML-SAFT: A Machine Learning Framework for PCP-SAFT
Parameter Estimation. [URL/DOI TBD]
Thol, M., et al. [citation TBD]
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

---

**Next Review**: Before v1.1.0 release or if new training data sources are added.
