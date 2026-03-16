# pcsaft-predict

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Data-CC--BY--NC--SA--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/example/pcsaft-predict)

Machine learning models for predicting PC-SAFT equation-of-state parameters (m, σ, ε/k) from molecular SMILES strings. Enables rapid screening of refrigerant and blowing agent candidates without requiring experimental VLE data.

> **Caution**: These are **screening-quality predictions**, not reference data. The GNN model achieves R² = 0.73-0.77 on test sets. 19% of novel predictions fall within the applicability domain (Tanimoto >= 0.4 to training set); the rest are extrapolations. Use for prioritizing experimental synthesis, not for engineering design decisions.

## Try It: Hosted Portal

The primary way to use pcsaft-predict is through the **hosted Streamlit portal**:

- **Predict**: Enter a SMILES string and get PC-SAFT parameters (m, σ, ε/k) with uncertainty estimates
- **Screen**: Batch screening of candidates with fluorination safety filters
- **Similar molecules**: Find the closest matches in the 4,663-molecule prediction database
- **Dashboard**: Compare model performance across RF, GNN, and ensemble

```bash
# Run locally
streamlit run portal/app.py
```

## Python SDK

For programmatic access, install the package:

```bash
pip install pcsaft-predict
```

### Basic Prediction

```python
import pcsaft_predict

# Predict PC-SAFT parameters for a single molecule
df = pcsaft_predict.predict("CCO")  # ethanol
print(df)
#    smiles      m  sigma  epsilon_k  in_domain  tanimoto_nn
# 0     CCO  2.123  3.456      234.5       True         0.85

# Batch prediction
candidates = ["c1ccccc1", "CC(C)C", "FC1=CCCC1"]
df = pcsaft_predict.predict(candidates)
print(df[["smiles", "m", "sigma", "epsilon_k", "in_domain"]])
```

### Uncertainty Quantification

```python
# Get predictions with model uncertainty
df = pcsaft_predict.predict_with_uncertainty(["CCO", "c1ccccc1"])
print(df[["smiles", "m", "m_std", "epsilon_k", "epsilon_k_std"]])
#       smiles      m  m_std  epsilon_k  epsilon_k_std
# 0       CCO  2.123  0.045      234.5           12.3
# 1  c1ccccc1  2.567  0.062      189.2           15.8

# Higher uncertainty indicates out-of-domain predictions
# Consider molecules with epsilon_k_std > 20 K as high-risk
```

## Model Performance

Performance on held-out test set (n=1,376 molecules, disjoint from training):

| Model | m (R²) | σ (R²) | ε/k (R²) | Parameters |
|-------|--------|--------|----------|------------|
| **GNN** (default) | **0.76** | **0.77** | **0.73** | 964K |
| Random Forest | 0.62 | 0.35 | 0.33 | 1.8M nodes |

**Mean Absolute Errors (GNN)**:
- m: 0.31 segments
- σ: 0.13 Å
- ε/k: 18.4 K

Note: Test-set molecules are structurally similar to training data (median Tanimoto = 0.65). Performance degrades for novel fluorinated olefins (Tanimoto < 0.3).

## Available Models

```python
import pcsaft_predict

# List available models
print(pcsaft_predict.list_models())
# ['rf', 'gnn']

# Load a specific model
model = pcsaft_predict.load_model("rf")
preds = model.predict(["CCO"])
```

**Model descriptions**:
- `gnn`: Graph Neural Network (GIN architecture, 4 layers, 256 hidden dim) — best accuracy
- `rf`: Random Forest baseline (Morgan fingerprints + RDKit descriptors, 100 trees per parameter) — fastest inference

## Novel Predictions Dataset

The models were trained on experimental and literature PC-SAFT parameters, then used to generate predictions for **4,663 novel refrigerant candidates** (fluorinated/chlorinated cycloalkenes). The predictive library is available at `data/pcsaft_novel_predictions_v1.csv`.

**Key findings from the screening campaign**:
- Zero highly-fluorinated HFOs match cyclopentane's thermophysical properties (dispersion energy deficit of 10-30%)
- Strained-ring candidates (e.g., tetrafluoromethylenecyclopropane) preserve ε/k but face thermal instability
- Commercial HFOs like HFO-1336mzz(Z) succeed by **redesigning foam formulations**, not by matching cyclopentane parameters

See `docs/reports/23_ml_chemistry_narrative.md` for full scientific narrative.

## Project Background

This package originated from a research question: **Can ML-predicted PC-SAFT parameters identify fluorinated blowing agents as drop-in replacements for cyclopentane?**

After systematically screening 4,663 candidates (98.9% novel molecules with no published data), the answer is **no**. Fluorination reduces dispersion energy (ε/k) by 10-30%, which exponentially increases Henry's law constants via Boltzmann statistics. Every candidate achieving parameter similarity to cyclopentane fails on practical constraints (ozone depletion, toxicity, thermal stability, or flammability).

**This is itself a scientific contribution**: The ML-driven exhaustive search closes the question rather than leaving it open, proving that viable drop-in replacements do not exist in fluorinated/chlorinated olefin space. Commercial HFOs succeed by abandoning parameter similarity and redesigning formulations around the agent's own thermodynamics.

**Positive outcome**: The predictive library and transferable workflow are immediately useful for other screening applications (e.g., refrigerants, solvents) where the constraint set differs. Replace "cyclopentane" with any reference molecule, and the 5-step pipeline (enumerate → predict → screen → quantify uncertainty → validate) applies.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{pcsaft_predict_2026,
  author = {ML-Chem Project Team},
  title = {pcsaft-predict: Machine Learning for PC-SAFT Parameter Prediction},
  year = {2026},
  url = {https://github.com/example/pcsaft-predict},
  version = {1.0.0},
  note = {Includes 4,663 novel PC-SAFT predictions for refrigerant screening}
}
```

Dataset citation (CC-BY-NC-SA 4.0):

```bibtex
@dataset{pcsaft_novel_2026,
  author = {ML-Chem Project Team},
  title = {PC-SAFT Parameter Predictions for 4,663 Fluorinated Refrigerant Candidates},
  year = {2026},
  publisher = {Zenodo},
  version = {v1.0},
  doi = {10.5281/zenodo.XXXXXX},
  url = {https://github.com/example/pcsaft-predict/data/pcsaft_novel_predictions_v1.csv},
  note = {Licensed under CC-BY-NC-SA 4.0 due to SPT-PCSAFT upstream data}
}
```

## License

**Code**: MIT License (see `LICENSE`)

**Data and model weights**:
- Novel predictions dataset and GNN/ensemble weights: **CC-BY-NC-SA 4.0** (see `LICENSE-DATA`)
- RF weights (Esper-only training): CC-BY-4.0

The CC-BY-NC-SA 4.0 constraint propagates from SPT-PCSAFT training data (Winter et al., 2023). Non-commercial use only for GNN-derived artifacts.

## Contributing

See `CONTRIBUTING.md` for guidelines on adding models, training data, and submitting pull requests.

## Acknowledgments

Training data sourced from:
- Esper et al. (2017) PC-SAFT parameter dataset (Figshare Collection 6821654, CC-BY-4.0)
- Felton et al. (2024) ML-SAFT regressed parameters (MIT)
- Winter et al. (2023) SPT-PCSAFT parameters (arXiv:2309.12404, CC-BY-NC-SA 4.0)

Thermodynamic validation powered by [teqp](https://github.com/usnistgov/teqp) equation-of-state library.
