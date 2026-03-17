# Feature Engineering for PC-SAFT Parameter Prediction: Supplementary Information

This document explains the feature engineering approaches used and considered for predicting PC-SAFT parameters (m, σ, ε/k) from molecular structure.

---

## RDKit 2D Descriptors

RDKit (open-source cheminformatics) provides ~200 descriptors via `MoleculeDescriptors.MolecularDescriptorCalculator`. Representative examples: `MolWt`, `ExactMolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`, `RingCount`, `HeavyAtomCount`, `FractionCSP3`, `NumRotatableBonds`, plus E-state indices and partial-charge statistics.

Cleaning drops zero-variance columns and highly correlated pairs (|r| > 0.95), yielding ~170 features. This encodes chemical domain knowledge (size, polarity, topology) that the model need not learn from data.

**Implementation**: `model/data/descriptors.py` — `_get_descriptor_calculator()`, `compute_descriptors()` (14–44); `clean_descriptors()` (47–75).

---

## Morgan Fingerprints (ECFP4-equivalent)

Morgan fingerprints are circular: each atom's local environment is hashed iteratively to radius r. Used: `radius=2`, `nBits=2048` (ECFP4). Features are `morgan_0` … `morgan_2047`; each bit indicates presence of a substructural motif. Bit semantics are hash-derived (no fixed mapping per index).

**Implementation**: `model/data/descriptors.py` — `compute_morgan_fingerprints()` (78–108); `AllChem.GetMorganFingerprintAsBitVect()`.

---

## Combined Feature Space

Production concatenates RDKit descriptors and Morgan fingerprints (~2,200 dims). RDKit supplies interpretable global properties; Morgan supplies local substructure coverage. Together they give strong inductive bias: the RF maps features to parameters without rediscovering chemistry.

### Ablation (Step 01)

| Method | R² (m) | R² (σ) | R² (ε/k) | MAE (m) | MAE (σ) | MAE (ε/k, K) |
|--------|--------|--------|----------|---------|---------|--------------|
| GC-PC-SAFT (no ML) | 0.40 | -1.16 | -0.04 | 1.001 | 0.494 | 44.6 |
| RF (RDKit only) | 0.61 | 0.34 | 0.31 | 0.634 | 0.192 | 26.1 |
| RF (Morgan only) | 0.43 | 0.17 | 0.17 | 0.736 | 0.227 | 30.9 |
| RF (Combined) | **0.62** | **0.35** | **0.33** | **0.585** | **0.189** | 26.8 |

Morgan-only RF performs substantially worse than RDKit-only (R² = 0.17 vs 0.31 on ε/k) because trees split most efficiently on continuous descriptors like `MolWt` and `LogP`, not on 2,048 sparse binary bits. RDKit-only is strong but misses substructure-level detail that Morgan encodes. The combined set wins on all three parameters — modestly in R² (+0.01–0.02) but notably in MAE for m (0.634 → 0.585). The XGBoost tie on the same combined features (R² = 0.33 vs 0.33 on ε/k) confirmed the bottleneck is the feature representation, not the tree algorithm.

---

## GC-PC-SAFT as Optional Features (Considered, Not Default)

`build_features()` supports `use_gc_pcsaft=True`, appending three features: `gc_m`, `gc_sigma`, `gc_epsilon_k` from a simplified group-contribution PC-SAFT scheme (`model/gc_pcsaft.py`). These are rough additivity estimates over functional groups (CH3, CH2, F, Cl, OH, etc.).

An ablation (Report 18) showed negligible benefit for m and σ (ΔR² ≈ +0.004) and slight harm for ε/k (ΔR² ≈ −0.04). The RF is feature-saturated; GC priors overlap with Morgan + RDKit. Production uses `use_gc_pcsaft=False`.

---

## 3D Conformer Features (Not Used)

3D features (bond angles, dihedrals, interatomic distances) would capture conformational effects. Parameter dependence:

- **m**: Topology-dependent; effectively invariant to conformation.
- **σ**: Segment diameter is an effective average; PC-SAFT fits treat it as conformation-invariant.
- **ε/k**: Partially conformer-dependent (dipole/quadrupole orientation); 2D descriptors capture the dominant electronic contributions.

Excluded for cost (conformer generation via ETKDG), pipeline complexity, and data scale (~1,800 molecules) where 2D suffices. 3D refinement is a future option.

---

## ChemBERTa CLS Embeddings as RF Features (Not Tested)

A natural idea: extract the 768-dimensional CLS embedding from fine-tuned ChemBERTa and feed it to RF, combining pretrained chemical knowledge with RF's low-data robustness. This was not tested. Three reasons suggest it would not outperform RDKit + Morgan:

1. **RF is axis-aligned; CLS embeddings are distributed.** Each RF split partitions data along a single feature. RDKit descriptors work well because `MolWt` or `LogP` alone is informative. CLS embeddings encode chemical information in correlated combinations of dimensions — "dispersion energy" lives in a linear combination of dozens of axes, not in any single one. RF would need many deep splits to reconstruct what one matrix multiply captures in the regression head.

2. **Ceiling from the regression head.** ChemBERTa's own two-layer MLP — the best possible consumer of those embeddings — reached R² = 0.27 on ε/k. That is the upper bound on what the CLS representation contains for this task. Giving the same embeddings to a worse consumer (axis-aligned RF) cannot exceed that ceiling.

3. **Signal dilution under concatenation.** Appending 768 correlated embedding dims to 2,200 existing features that RF already splits on efficiently would likely be drowned out. The GC-PC-SAFT ablation (Report 18) showed the same pattern: adding 3 weakly informative features to a feature-saturated RF produced ΔR² ≈ 0 or slightly negative.

This remains a plausible but low-priority ablation. A variant worth testing in future work would be to PCA the CLS embeddings down to ~50 uncorrelated dimensions and concatenate with RDKit + Morgan, reducing the axis-alignment penalty while retaining whatever unique signal the embeddings carry.

---

## Why These Features Work for PC-SAFT

Parameters relate to topology, size, and electronics: m to mass/connectivity, σ to effective volume, ε/k to polarity and electronegativity. RDKit encodes these; Morgan adds substructure. In a low-data regime, hand-crafted features outperform learned representations because the model need not rediscover fundamentals.
