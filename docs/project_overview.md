# ML-Driven Blowing Agent Screening via PC-SAFT Parameter Prediction

## Goal

Screen candidate blowing agents, including HFOs and other low-GWP chemistries, as drop-in replacements for cyclopentane in polyurethane foam blowing, using machine learning to predict thermodynamic (PC-SAFT) parameters from molecular structure.

## Architecture

```
SMILES → RDKit 2D Descriptors → Random Forest → PC-SAFT Parameters (m, σ, ε/k)
```

Two main components:

1. **ML Model** (`model/`): Trains Random Forest regressors on known PC-SAFT parameters to predict m, σ, and ε/k from molecular structure.
2. **Screening Pipeline** (`screening/`): Generates candidate blowing-agent chemistries, filters by synthesizability and patent freedom, then ranks them by PC-SAFT similarity to cyclopentane.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train model (uses fallback dataset of ~30 molecules)
python -m model.train

# Evaluate model
python -m model.evaluate

# Run screening pipeline (use --skip-patents for offline)
python -m screening.run_screening --skip-patents

# Run tests
pytest tests/
```

## Dependencies

- `rdkit-pypi`: Molecular descriptors, SMILES parsing, stereoisomer enumeration
- `scikit-learn`: Random Forest models
- `pandas`, `numpy`: Data handling
- `matplotlib`: Parity plots
- `requests`: PubChem API, Figshare download
- `joblib`: Model serialization

## Data Sources

- **Fallback** (`model/data/pcsaft_data.csv`): ~30 curated molecules, included in repo
- **Esper Dataset** (1,842 molecules): Download via `python -m model.data.download_esper`
