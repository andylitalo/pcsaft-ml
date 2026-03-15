# pcsaft-predict

Predict PC-SAFT equation-of-state parameters from molecular SMILES strings using machine learning.

## Installation

```bash
pip install pcsaft-predict
```

For thermodynamic property calculations (requires teqp):

```bash
pip install pcsaft-predict[thermo]
```

## Quick Start

```python
import pcsaft_predict

# Predict parameters for ethanol
df = pcsaft_predict.predict("CCO")
print(df)
#    smiles      m  sigma  epsilon_k  in_domain  tanimoto_nn
# 0     CCO  2.123  3.456      234.5       True         0.85

# Batch prediction
candidates = ["CCO", "c1ccccc1", "CCCCC"]
df = pcsaft_predict.predict(candidates)

# Prediction with uncertainty
df = pcsaft_predict.predict_with_uncertainty("CCO")
print(df[["smiles", "m", "m_std", "in_domain"]])
#   smiles      m  m_std  in_domain
# 0    CCO  2.123  0.045       True
```

## Features

- **Simple API**: Single function call for predictions
- **Uncertainty Quantification**: RF tree variance estimates model uncertainty
- **Applicability Domain**: Tanimoto similarity flags out-of-domain molecules
- **Thermodynamic Properties**: Optional boiling point and Henry's constant calculations
- **Type Hints**: Full type annotations for IDE support

## Parameters

PC-SAFT parameters are returned in a pandas DataFrame:

- `m`: Number of segments (dimensionless)
- `sigma`: Segment diameter (Angstrom)
- `epsilon_k`: Dispersion energy (K)

Additional columns:

- `in_domain`: Boolean flag (True = in applicability domain)
- `tanimoto_nn`: Maximum Tanimoto similarity to training set
- `m_std`, `sigma_std`, `epsilon_k_std`: Uncertainty estimates (if using `predict_with_uncertainty`)

## Model Information

The default model is a Random Forest ensemble trained on:

- **Training set**: 1,445 molecules from Esper/ML-SAFT/fluorinated datasets
- **Features**: 2048-bit Morgan fingerprints (radius=2) + 170 RDKit 2D descriptors
- **Performance** (R² on test set):
  - m (segments): 0.61
  - sigma (diameter): 0.32
  - epsilon_k (energy): 0.27

## Advanced Usage

### Filter by Applicability Domain

```python
df = pcsaft_predict.predict(smiles_list)
reliable = df[df["in_domain"]]
```

### Filter by Uncertainty

```python
df = pcsaft_predict.predict_with_uncertainty(smiles_list)
confident = df[df["m_std"] < 0.05]
```

### Thermodynamic Properties

```python
from pcsaft_predict._thermo import compute_boiling_point

T_b = compute_boiling_point(m=2.5, sigma=3.6, epsilon_k=250.0)
print(f"Boiling point: {T_b:.1f} K")
```

## Model Weights

Model weights are expected in:

1. `$PCSAFT_PREDICT_CACHE_DIR` (if environment variable set)
2. `~/.cache/pcsaft_predict/` (default)
3. `<repo>/model/saved/` (development mode)

Required files:

- `rf_m.joblib`, `rf_sigma.joblib`, `rf_epsilon_k.joblib`
- `feature_names.joblib`, `rdkit_scaler.joblib`
- `tanimoto_ad.joblib` (optional, for AD checking)

## Citation

If you use this package in your research, please cite:

```
@software{pcsaft_predict,
  title = {pcsaft-predict: Machine Learning for PC-SAFT Parameter Prediction},
  year = {2024},
  url = {https://github.com/example/pcsaft-predict}
}
```

## License

MIT
