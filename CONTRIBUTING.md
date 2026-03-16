# Contributing to pcsaft-predict

Thank you for considering contributing to pcsaft-predict! This document provides guidelines for adding models, training data, and submitting improvements.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Adding a New Model](#adding-a-new-model)
3. [Adding Training Data](#adding-training-data)
4. [Running Tests](#running-tests)
5. [Code Style](#code-style)
6. [Pull Request Process](#pull-request-process)

---

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/example/pcsaft-predict.git
cd pcsaft-predict

# Install with development dependencies
uv sync --extra dev

# Or using pip
pip install -e ".[dev]"
```

### 2. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check code style
ruff check .
```

---

## Adding a New Model

To add a new PC-SAFT prediction model (e.g., a Graph Neural Network or ensemble):

### 1. Create Model Wrapper

Create a new module in `pcsaft_predict/` (e.g., `_gnn.py`) implementing the model interface:

```python
# pcsaft_predict/_gnn.py
import numpy as np
from pcsaft_predict._features import build_features
from pcsaft_predict._weights import get_model_path, check_model_exists

class GNNModel:
    """Graph Neural Network model for PC-SAFT prediction."""

    def __init__(self):
        self._model = None
        self._loaded = False

    def load(self):
        """Load model weights from cache directory."""
        if self._loaded:
            return

        model_path = get_model_path("gnn_model.pt")
        if not check_model_exists("gnn_model.pt"):
            raise FileNotFoundError(f"GNN model not found: {model_path}")

        # Load model (PyTorch, TensorFlow, etc.)
        # self._model = torch.load(model_path)
        self._loaded = True

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        """Predict PC-SAFT parameters.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys: m, sigma, epsilon_k
        """
        if not self._loaded:
            self.load()

        # Implement prediction logic
        # return {"m": ..., "sigma": ..., "epsilon_k": ...}
        raise NotImplementedError

    def predict_with_uncertainty(
        self, smiles_list: list[str], n_forward: int = 30
    ) -> dict[str, np.ndarray]:
        """Predict with uncertainty estimates.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys: m, sigma, epsilon_k, m_std, sigma_std, epsilon_k_std
        """
        if not self._loaded:
            self.load()

        # Implement uncertainty quantification (e.g., MC dropout, ensembles)
        # return {"m": ..., "m_std": ..., "sigma": ..., "sigma_std": ..., ...}
        raise NotImplementedError
```

### 2. Register Model

Add the model to the registry in `pcsaft_predict/_registry.py`:

```python
from pcsaft_predict._gnn import GNNModel

# Add to MODELS dictionary
MODELS["gnn"] = GNNModel
```

### 3. Package Model Weights

Model weights should be distributed via the package manifest. Add model files to `pcsaft_predict/data/`:

```bash
# Create data directory
mkdir -p pcsaft_predict/data

# Copy trained model files
cp model/saved/gnn_model.pt pcsaft_predict/data/
```

Update `pcsaft_predict/MANIFEST.txt` to include the new model:

```
include pcsaft_predict/data/gnn_model.pt
```

### 4. Add Tests

Create tests in `tests/test_models.py`:

```python
import pytest
import pcsaft_predict

def test_gnn_prediction():
    """Test GNN model predictions."""
    df = pcsaft_predict.predict("CCO", model="gnn")
    assert df.shape == (1, 6)
    assert "m" in df.columns
    assert df["m"].iloc[0] > 0  # Sanity check

def test_gnn_uncertainty():
    """Test GNN uncertainty quantification."""
    df = pcsaft_predict.predict_with_uncertainty("CCO", model="gnn")
    assert "m_std" in df.columns
    assert df["m_std"].iloc[0] > 0  # Uncertainty should be positive
```

### 5. Document Performance

Create a model card in `docs/model_cards/<model_name>.md` following the template in `docs/model_cards/rf.md`. Include:

- Model architecture
- Training data details
- Performance metrics (R², MAE, RMSE on test set)
- Limitations and ethical considerations
- Intended use cases

---

## Adding Training Data

To expand the training dataset with new experimental or literature PC-SAFT parameters:

### 1. Data Format

New data must be provided as a CSV with these columns:

- `smiles` (str): Canonical SMILES string
- `m` (float): Number of segments
- `sigma` (float): Segment diameter in Angstroms
- `epsilon_k` (float): Dispersion energy in Kelvin
- `source` (str): Data provenance (e.g., "DOI:10.1021/xyz", "Lab notebook XYZ")
- `temperature_K` (float, optional): Reference temperature for parameters

Example:

```csv
smiles,m,sigma,epsilon_k,source
CCO,2.3827,3.1771,198.24,DOI:10.1021/ie8007857
c1ccccc1,2.4653,3.6478,287.35,DDB:ethylbenzene
```

### 2. Validation Pipeline

Before adding data to the training set, run validation checks:

```bash
# Validate SMILES syntax and uniqueness
python scripts/validate_data.py --input new_data.csv

# Check for duplicates against existing training set
python scripts/check_duplicates.py --input new_data.csv --reference data/pcsaft_training.csv
```

### 3. Append to Training Set

```bash
# Combine with existing data
python -m model.data.merge_datasets \
    --existing data/pcsaft_training.csv \
    --new new_data.csv \
    --output data/pcsaft_training_v2.csv
```

### 4. Retrain Models

```bash
# Retrain all models with new data
python -m model.train --data data/pcsaft_training_v2.csv --output model/saved/

# Evaluate on held-out test set
python -m model.evaluate --data data/pcsaft_training_v2.csv
```

### 5. Update Package Version

After retraining, bump the package version in `pyproject.toml` and `pcsaft_predict/__init__.py`:

```python
__version__ = "1.1.0"  # Increment minor version for new data
```

### 6. Document Changes

Update `CHANGELOG.md` with data additions:

```markdown
## [1.1.0] - 2026-04-15

### Added
- 150 new experimental PC-SAFT parameters from [source]
- Improved R² for ε/k from 0.73 to 0.75 on test set

### Changed
- Retrained all models with expanded dataset (n=13,914 total)
```

---

## Running Tests

### Basic Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v

# Run with coverage report
pytest tests/ --cov=pcsaft_predict --cov-report=html
```

### Slow Tests

Some tests require downloading models or running expensive computations. These are marked with `@pytest.mark.slow`:

```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Run only slow tests
pytest tests/ -v -m "slow"
```

### Integration Tests

Test the full prediction pipeline:

```bash
# Test batch prediction
pytest tests/test_integration.py::test_batch_prediction -v

# Test uncertainty propagation
pytest tests/test_integration.py::test_uncertainty_quantification -v
```

---

## Code Style

This project follows strict code style conventions enforced by [ruff](https://github.com/astral-sh/ruff).

### Style Requirements

- **Line length**: 99 characters maximum
- **Python version**: 3.10+ (use modern type hints)
- **Imports**: Sorted and grouped (E, F, I rules)
- **Type hints**: Required for all public functions
- **Docstrings**: NumPy-style docstrings for all public APIs

### Auto-Formatting

```bash
# Check code style
ruff check .

# Auto-fix style issues
ruff check --fix .

# Format code
ruff format .
```

### Pre-Commit Hook (Optional)

```bash
# Install pre-commit hook
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/add-gnn-model
```

### 2. Make Changes

Follow the guidelines above for code style, tests, and documentation.

### 3. Verify Tests Pass

```bash
# Run tests locally
pytest tests/ -v

# Check code style
ruff check .
```

### 4. Commit with Clear Messages

```bash
git add pcsaft_predict/_gnn.py tests/test_gnn.py
git commit -m "Add GNN model with MC dropout uncertainty

Implements Graph Isomorphism Network (GIN) architecture for PC-SAFT
prediction. Achieves R²=0.76/0.77/0.73 on m/σ/ε_k (test set n=1,376).
Uncertainty via MC dropout (n_forward=30)."
```

### 5. Push and Open PR

```bash
git push origin feature/add-gnn-model
```

Open a pull request on GitHub with:

- **Title**: Short description (< 70 chars)
- **Description**: What changed, why, and key metrics/results
- **Checklist**:
  - [ ] Tests pass (`pytest tests/ -v`)
  - [ ] Code style verified (`ruff check .`)
  - [ ] Documentation updated (model card, README, CHANGELOG)
  - [ ] Version bumped if needed

### 6. Address Review Comments

Maintainers will review your PR and may request changes. Address feedback by:

```bash
# Make changes
git add <files>
git commit -m "Address review feedback: ..."
git push origin feature/add-gnn-model
```

### 7. Merge

Once approved, a maintainer will merge your PR. The new model/data will be included in the next release.

---

## Questions?

For questions or discussions:

- **Issues**: [GitHub Issues](https://github.com/example/pcsaft-predict/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/pcsaft-predict/discussions)
- **Email**: ml-chem-team@example.com

Thank you for contributing to pcsaft-predict!
