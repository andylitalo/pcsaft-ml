# Fluorinated Refrigerant External Validation

This document explains the 15-molecule fluorinated refrigerant validation that produced the project's critical finding: in-distribution test metrics dramatically overstated real-world performance, and only external validation against independently published experimental data distinguished model quality.

---

## The External Validation Set

The validation set comprises 15 fluorinated refrigerants with **independently published, literature-optimized PC-SAFT parameters**. These are real-world molecules with experimental data—not from Esper or SPT-PCSAFT, the project's primary training sources.

The set was selected specifically to test the deployment domain: fluorinated compounds for blowing agent screening. Molecules include commercial refrigerants (e.g., HFO-1234yf, R-32, R-134a) and related C1–C4 fluorinated species. Boiling points are known experimentally, providing property-level validation beyond parameter accuracy: predicted PC-SAFT parameters can be used in the equation of state to compute boiling points, which are then compared against experimental values.

---

## The Critical Finding (Step 38)

Step 38 compared the GNN (unified dataset) and RF (Esper-only) models on this 15-molecule set.

### GNN (unified dataset, 13,764 molecules)

| Metric | Value |
|--------|-------|
| epsilon/k MAE | 77.6 K |
| Boiling point MAE | 133.7 K |
| epsilon/k R² | -25.08 |
| EOS convergence | 40% (6/15) |
| MC Dropout uncertainty | 7.6x miscalibrated: 0–6.7% of errors within 1-sigma (expected: 68%) |

### RF (Esper-only, 1,801 molecules)

| Metric | Value |
|--------|-------|
| epsilon/k MAE | 14.3 K |
| Boiling point MAE | 8.2 K |
| epsilon/k R² | -0.47 |
| EOS convergence | 40% (6/15) |

**RF advantage: 16.4x on boiling point MAE.**

---

## Why the GNN Failed

1. **SPT contamination**: The GNN was trained on 87% SPT-PCSAFT data. SPT predicts epsilon/k roughly 75 K higher than experimental fits for fluorinated molecules. The GNN learned SPT's biased distribution rather than physically correct parameter values.

2. **Source-stratified evaluation**: On SPT test data, GNN R² (epsilon/k) ≈ 0.90; on Esper test data, R² ≈ 0.36. The strong test-set performance was driven by the majority SPT source, masking poor generalization to experimental parameters.

3. **Architectural limitation**: Retraining the GNN on Esper-only data (Step 38c) still failed—boiling point MAE = 142 K, worse than the unified GNN. With only ~210 fluorinated molecules in Esper, the GNN's 195K parameters cannot learn useful representations for this chemical class. The failure is not only data source but model capacity vs. data scale.

4. **Why RF succeeds**: Hand-crafted RDKit descriptors encode electronegativity, polarizability, and related chemical knowledge that the GNN must infer from data it does not have in sufficient quantity. RF benefits from this built-in inductive bias for fluorinated compounds.

---

## The Lesson

In-distribution test-set performance can dramatically overstate real-world utility. The same GNN achieved R² ≈ 0.91 on the in-distribution fluorinated test set and R² = -25.08 on the external validation set. Only validation against independently published experimental data exposed this gap.

MC Dropout uncertainty was confidently wrong: the model produced tight uncertainty intervals around incorrect predictions, with 0–6.7% of errors falling within 1-sigma instead of the expected 68%. This highlights that uncertainty estimates can be both overconfident and miscalibrated when the model has learned a biased or incorrect data distribution.

---

## Figure Reference

See `figures/38_gnn_fluorinated_validation/boiling_point_parity.png` and `figures/38_gnn_fluorinated_validation/error_distribution.png`.
