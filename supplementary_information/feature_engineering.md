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

## Why These Features Work for PC-SAFT

Parameters relate to topology, size, and electronics: m to mass/connectivity, σ to effective volume, ε/k to polarity and electronegativity. RDKit encodes these; Morgan adds substructure. In a low-data regime, hand-crafted features outperform learned representations because the model need not rediscover fundamentals.
