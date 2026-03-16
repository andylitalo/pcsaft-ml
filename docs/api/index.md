# API Reference

Complete API documentation for the `pcsaft_predict` package.

## Core Functions

### predict()

```python
pcsaft_predict.predict(smiles, model="rf") -> pd.DataFrame
```

Predict PC-SAFT parameters from SMILES strings.

**Parameters**

- `smiles` (str | list[str]): Single SMILES string or list of SMILES strings to predict
- `model` (str, optional): Model name to use. Default: `"rf"`. Available: `"rf"` (Random Forest)

**Returns**

- `pd.DataFrame`: DataFrame with columns:
  - `smiles` (str): Input SMILES string
  - `m` (float): Number of segments (dimensionless)
  - `sigma` (float): Segment diameter in Angstroms (Å)
  - `epsilon_k` (float): Dispersion energy in Kelvin (K)
  - `in_domain` (bool): Applicability domain flag (True if Tanimoto ≥ 0.4 to training set)
  - `tanimoto_nn` (float): Maximum Tanimoto similarity to nearest training molecule

**Example**

```python
import pcsaft_predict

# Single molecule
df = pcsaft_predict.predict("CCO")  # ethanol
print(df)
#    smiles      m  sigma  epsilon_k  in_domain  tanimoto_nn
# 0     CCO  2.123  3.456      234.5       True         0.85

# Batch prediction
candidates = ["c1ccccc1", "CC(C)C", "FC1=CCCC1"]
df = pcsaft_predict.predict(candidates)
print(df[["smiles", "m", "sigma", "epsilon_k"]])
#         smiles      m  sigma  epsilon_k
# 0    c1ccccc1  2.567  3.648      287.4
# 1      CC(C)C  1.987  3.234      198.2
# 2   FC1=CCCC1  2.345  3.512      256.3
```

**Notes**

- Invalid SMILES (unparseable by RDKit) return NaN for all parameters
- The `in_domain` flag uses Tanimoto similarity threshold of 0.4 (conservative)
- Molecules with `in_domain=False` are extrapolations and should be used cautiously

---

### predict_with_uncertainty()

```python
pcsaft_predict.predict_with_uncertainty(smiles, model="rf", n_forward=30) -> pd.DataFrame
```

Predict PC-SAFT parameters with uncertainty estimates from model disagreement.

**Parameters**

- `smiles` (str | list[str]): Single SMILES string or list of SMILES strings
- `model` (str, optional): Model name. Default: `"rf"`
- `n_forward` (int, optional): Number of forward passes for uncertainty estimation. Default: 30
  - For Random Forest: Unused (uncertainty comes from tree variance)
  - For Neural Networks: Number of MC dropout forward passes
  - For Ensembles: Number of bootstrap resamples

**Returns**

- `pd.DataFrame`: DataFrame with columns:
  - `smiles` (str): Input SMILES
  - `m` (float): Mean prediction for m
  - `sigma` (float): Mean prediction for σ
  - `epsilon_k` (float): Mean prediction for ε/k
  - `m_std` (float): Standard deviation for m (uncertainty estimate)
  - `sigma_std` (float): Standard deviation for σ
  - `epsilon_k_std` (float): Standard deviation for ε/k
  - `in_domain` (bool): Applicability domain flag
  - `tanimoto_nn` (float): Nearest-neighbor Tanimoto similarity

**Example**

```python
import pcsaft_predict

df = pcsaft_predict.predict_with_uncertainty(["CCO", "c1ccccc1"])
print(df[["smiles", "m", "m_std", "epsilon_k", "epsilon_k_std", "in_domain"]])
#       smiles      m  m_std  epsilon_k  epsilon_k_std  in_domain
# 0       CCO  2.123  0.045      234.5           12.3       True
# 1  c1ccccc1  2.567  0.062      189.2           15.8       True

# Identify high-uncertainty predictions
high_unc = df[df["epsilon_k_std"] > 20.0]
print(f"Found {len(high_unc)} high-uncertainty molecules")
```

**Interpreting Uncertainty**

Uncertainty estimates represent **model disagreement**, not prediction error:

- **Random Forest**: Standard deviation across 100 tree predictions (aleatoric + epistemic)
- **Neural Network**: Standard deviation across MC dropout forward passes (epistemic only)
- **Ensemble**: Standard deviation across ensemble members (epistemic)

**Typical uncertainty ranges** (RF model):

| Parameter | In-domain (σ) | Out-of-domain (σ) |
|-----------|---------------|-------------------|
| m         | 0.05-0.15     | 0.15-0.40         |
| σ (Å)     | 0.03-0.10     | 0.10-0.25         |
| ε/k (K)   | 8-15          | 15-30             |

**Decision thresholds**:
- `epsilon_k_std > 20 K`: High risk, consider experimental validation before use
- `in_domain=False` AND `epsilon_k_std > 15 K`: Very high risk, out-of-distribution extrapolation

---

### list_models()

```python
pcsaft_predict.list_models() -> list[str]
```

Return list of available model names.

**Returns**

- `list[str]`: List of registered model names

**Example**

```python
import pcsaft_predict

models = pcsaft_predict.list_models()
print(models)
# ['rf']

# Test all available models
for model_name in models:
    df = pcsaft_predict.predict("CCO", model=model_name)
    print(f"{model_name}: m={df['m'].iloc[0]:.3f}")
# rf: m=2.123
```

**Available Models**

Currently packaged models:

- `"rf"`: Random Forest baseline (Morgan fingerprints + RDKit descriptors)
  - Performance: R²(m)=0.62, R²(σ)=0.35, R²(ε/k)=0.33
  - Fast inference (~10 ms per molecule)
  - Uncertainty via tree variance

**Coming Soon** (in development):

- `"gnn"`: Graph Neural Network (GIN architecture)
  - Performance: R²(m)=0.76, R²(σ)=0.77, R²(ε/k)=0.73
  - Slower inference (~100 ms per molecule)
  - Uncertainty via MC dropout

- `"ensemble"`: Inverse-variance weighted ensemble (RF + GNN)
  - Performance: R²(m)=0.76, R²(σ)=0.77, R²(ε/k)=0.73
  - Best overall accuracy
  - Uncertainty via ensemble disagreement

---

### load_model()

```python
pcsaft_predict.load_model(name: str) -> Model
```

Load a model instance for advanced usage (batch prediction, custom pipelines).

**Parameters**

- `name` (str): Model name (e.g., `"rf"`)

**Returns**

- `Model`: Loaded model object with `.predict()` and `.predict_with_uncertainty()` methods

**Raises**

- `KeyError`: If model name is not registered

**Example**

```python
import pcsaft_predict

# Load model once for repeated predictions
model = pcsaft_predict.load_model("rf")

# Use model directly
batch1 = ["CCO", "c1ccccc1"]
batch2 = ["CC(C)C", "FC1=CCCC1"]

preds1 = model.predict(batch1)
preds2 = model.predict(batch2)

print(preds1)
# {'m': array([2.123, 2.567]), 'sigma': array([3.456, 3.648]), ...}
```

**When to use `load_model()`**

- **High-throughput prediction**: Load model once, reuse for many batches (avoids repeated deserialization)
- **Custom pipelines**: Integrate with custom feature computation or post-processing
- **Model inspection**: Access model internals (e.g., feature importances, tree structures)

**Model Interface**

All models implement:

```python
class Model:
    def load(self) -> None:
        """Load model weights from disk."""
        ...

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        """Predict parameters without uncertainty.

        Returns
        -------
        dict with keys: "m", "sigma", "epsilon_k"
        """
        ...

    def predict_with_uncertainty(
        self, smiles_list: list[str], n_forward: int = 30
    ) -> dict[str, np.ndarray]:
        """Predict parameters with uncertainty.

        Returns
        -------
        dict with keys: "m", "sigma", "epsilon_k", "m_std", "sigma_std", "epsilon_k_std"
        """
        ...
```

---

## Applicability Domain

### TanimotoAD (internal)

The applicability domain module (`pcsaft_predict._ad.TanimotoAD`) is used internally but not exposed in the public API. It computes Tanimoto similarity between input molecules and the training set.

**Methodology**

1. Compute Morgan fingerprints (radius=2, 2048 bits) for input molecule
2. Compare to all training molecules using Tanimoto similarity
3. Return maximum similarity score (`tanimoto_nn`)
4. Flag as `in_domain=True` if `tanimoto_nn ≥ 0.4`

**Threshold selection**

The 0.4 threshold is conservative:

- Tanimoto ≥ 0.4: Structurally similar to at least one training molecule
- Tanimoto < 0.4: Out-of-distribution, expect higher prediction error
- Tanimoto < 0.3: Very high risk, model has not seen similar chemistry

**Training set coverage**

- **Total molecules**: 13,764
- **Fluorinated compounds**: ~10% (mostly saturated HFCs, not HFOs)
- **Molecular weight range**: 30-250 Da (85% < 200 Da)
- **Chemical classes**: Alkanes, aromatics, ethers, ketones, refrigerants

**Improving applicability domain**

If you have molecules consistently flagged as out-of-domain:

1. Add experimental data for representative molecules (see [CONTRIBUTING.md](../../CONTRIBUTING.md#adding-training-data))
2. Retrain models with expanded dataset
3. Re-evaluate on your use case

---

## Feature Computation

### build_features() (internal)

The `pcsaft_predict._features.build_features()` function computes molecular descriptors:

**Feature types**:

1. **RDKit 2D descriptors** (200 features)
   - Molecular weight, LogP, TPSA, H-bond donors/acceptors
   - Topological indices (Kier, Hall, Balaban)
   - Electrotopological state indices

2. **Morgan fingerprints** (2048 bits)
   - Radius 2 (ECFP4 equivalent)
   - Bit vectors capturing substructure patterns

**Total dimensionality**: 2,248 features

**Preprocessing**:
- RDKit descriptors: StandardScaler normalization (zero mean, unit variance)
- Morgan fingerprints: No scaling (binary features)

---

## Data Types and Constants

### PC-SAFT Parameters

The package predicts three PC-SAFT parameters for non-associating compounds:

| Parameter | Symbol | Unit | Physical Meaning | Typical Range |
|-----------|--------|------|------------------|---------------|
| **m** | m | dimensionless | Number of spherical segments in chain | 1.0 - 10.0 |
| **σ** | σ | Angstrom (Å) | Diameter of each segment | 2.5 - 5.0 |
| **ε/k** | ε/k | Kelvin (K) | Dispersion energy (depth of Lennard-Jones well) | 100 - 500 |

**Not supported**:

- Association parameters (ε_AB, κ_AB) for hydrogen-bonding compounds
- Dipole moment (μ) for polar compounds
- Quadrupole moment (Q) for highly polar/aromatic compounds

**Associating compounds** (alcohols, amines, carboxylic acids) will return predictions for (m, σ, ε/k), but these are **3-parameter fits** that do not capture hydrogen bonding. Use with extreme caution.

---

## Error Handling

### Invalid SMILES

```python
import pcsaft_predict

# Invalid SMILES returns NaN
df = pcsaft_predict.predict("INVALID_SMILES")
print(df["m"].iloc[0])
# nan

# Check for errors
df = pcsaft_predict.predict(["CCO", "INVALID", "c1ccccc1"])
valid = df[df["m"].notna()]
print(f"Valid predictions: {len(valid)}/3")
# Valid predictions: 2/3
```

### Model Not Found

```python
import pcsaft_predict

try:
    df = pcsaft_predict.predict("CCO", model="nonexistent")
except KeyError as e:
    print(f"Error: {e}")
# Error: "Unknown model 'nonexistent'. Available: ['rf']"
```

### Missing Model Files

If model files are not found in the cache directory:

```python
import pcsaft_predict

try:
    model = pcsaft_predict.load_model("rf")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Reinstall the package: pip install --force-reinstall pcsaft-predict")
```

---

## Performance Considerations

### Batch Size

For optimal performance:

- **Small batches (< 100 molecules)**: Call `predict()` directly
- **Medium batches (100-10,000 molecules)**: Use `load_model()` to avoid repeated deserialization
- **Large batches (> 10,000 molecules)**: Chunk into batches of 1,000-5,000 and process in parallel

**Example: Parallel batch processing**

```python
import pcsaft_predict
from concurrent.futures import ProcessPoolExecutor

# Load large dataset
with open("candidates.smi") as f:
    all_smiles = [line.strip() for line in f]

# Chunk into batches
batch_size = 1000
batches = [all_smiles[i:i+batch_size] for i in range(0, len(all_smiles), batch_size)]

# Process in parallel
def predict_batch(smiles_batch):
    return pcsaft_predict.predict(smiles_batch)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(predict_batch, batches))

# Combine results
import pandas as pd
df_all = pd.concat(results, ignore_index=True)
```

### Memory Usage

Approximate memory footprint:

- **RF model**: ~150 MB (loaded into memory)
- **Feature computation**: ~8 KB per molecule (2,248 features × 4 bytes)
- **Batch of 10,000 molecules**: ~80 MB features + 150 MB model = ~230 MB total

---

## Version History

### v1.0.0 (2026-03-15)

Initial release:

- Random Forest model (m, σ, ε/k prediction)
- Applicability domain via Tanimoto similarity
- Uncertainty quantification via tree variance
- 4,663 novel predictions for fluorinated refrigerants

**Planned for v1.1.0**:

- Graph Neural Network (GNN) model
- Ensemble model (RF + GNN)
- Thermodynamic property prediction (boiling point, vapor pressure)
- Streamlit portal for interactive exploration
