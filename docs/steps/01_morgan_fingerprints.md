# Step 1: Morgan Fingerprint Feature Module

## Purpose

The current pipeline uses ~200 RDKit 2D descriptors as input features. These capture global molecular properties (molecular weight, LogP, topological indices) but miss **local substructural patterns** -- the specific arrangement of atoms in a neighborhood. Morgan fingerprints (also called extended-connectivity fingerprints, ECFP) encode exactly this: each bit represents the presence of a particular circular substructure of a given radius.

Combining Morgan fingerprints with selected RDKit descriptors gives the downstream neural network two complementary views of each molecule:

- **Morgan FP (2048 bits)**: "What substructures are present?" -- local chemical environment
- **RDKit descriptors (~100 after cleaning)**: "What are the global molecular properties?" -- size, polarity, shape

This step does not change the model architecture yet; it builds the feature pipeline that Steps 2–4 consume.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| ~200 RDKit 2D descriptors only | 2048-bit Morgan FP + ~100 cleaned RDKit descriptors |
| Misses local substructure patterns | Captures atom neighborhoods at radius 2 |
| Feature set coupled to `clean_descriptors()` logic | Modular feature builder that returns a single NumPy array |

Even using the existing Random Forest, adding Morgan fingerprints typically improves R² by 2–5% on molecular property prediction tasks because the RF can now split on "does substructure X exist?" in addition to global descriptor values.

## Skills Demonstrated

- **RDKit cheminformatics**: `AllChem.GetMorganFingerprintAsBitVect`, understanding radius and bit-vector size trade-offs
- **Feature engineering**: Combining heterogeneous feature types (binary fingerprints + continuous descriptors) into a single matrix
- **NumPy / Pandas**: Efficient array construction and alignment

## Implementation Guide

### 1. Add Morgan fingerprint computation to `model/data/descriptors.py`

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def compute_morgan_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprints as a dense bit array.

    Returns shape (n_molecules, n_bits). Rows for invalid SMILES are all zeros.
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps)
```

### 2. Create a combined feature builder

```python
def build_features(
    smiles_list: list[str],
    use_morgan: bool = True,
    use_rdkit: bool = True,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
) -> np.ndarray:
    """Build a combined feature matrix from SMILES.

    Returns a 2D NumPy array of shape (n_molecules, n_features).
    """
    parts = []
    if use_morgan:
        parts.append(compute_morgan_fingerprints(smiles_list, morgan_radius, morgan_bits))
    if use_rdkit:
        desc_df = compute_descriptors(smiles_list)
        desc_df = clean_descriptors(desc_df)
        parts.append(desc_df.fillna(0).values.astype(np.float32))
    return np.hstack(parts)
```

### 3. Handle feature name tracking

For interpretability and downstream use, track which columns came from which source:

```python
def build_features_with_names(smiles_list, ...):
    # ... same logic ...
    names = []
    if use_morgan:
        names += [f"morgan_{i}" for i in range(morgan_bits)]
    if use_rdkit:
        names += desc_df.columns.tolist()
    return features, names
```

### 4. Update `model/train.py` to accept a `--features` flag

Add `--features {morgan,rdkit,combined}` so you can train the RF baseline on different feature sets and compare.

### 5. Normalization considerations

- Morgan FP bits are already 0/1 -- no scaling needed for tree models, but neural networks (Step 2) benefit from having all features on similar scales.
- Save a `StandardScaler` fitted on training data alongside the model artifacts. Apply it to the RDKit descriptor portion only (scaling binary fingerprints is unnecessary but harmless).

## Evaluation & Success Criteria

### Metrics to check

Run the existing `model/evaluate.py` with each feature set and compare:

| Feature Set | R² (m) | R² (σ) | R² (ε/k) | MAE (m) | MAE (σ) | MAE (ε/k) |
|-------------|--------|--------|----------|---------|---------|-----------|
| RDKit only (baseline) | — | — | — | — | — | — |
| Morgan only | — | — | — | — | — | — |
| Combined | — | — | — | — | — | — |

### What success looks like

- `compute_morgan_fingerprints` returns correct shapes and handles invalid SMILES gracefully
- `build_features` produces a combined array with no NaN values
- RF trained on combined features matches or improves on the RDKit-only baseline (expect +2–5% R² on ε/k, which is the hardest target)
- All existing tests still pass (`pytest tests/`)
- New unit tests cover Morgan FP computation (valid SMILES, invalid SMILES, known substructure presence)

### When to move to Step 2

You're ready for Step 2 when:

1. The feature pipeline is modular (`build_features` works with any combination of feature types)
2. You have a baseline comparison table (RF on each feature set)
3. The combined feature vector feeds cleanly into a NumPy array (no DataFrames needed downstream -- this is what PyTorch expects)
