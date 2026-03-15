# Step 26: Extract Installable Python Package (`pcsaft-predict`)

## Objective

Extract the prediction core into a standalone, pip-installable Python package so
that a user can `pip install pcsaft-predict` and call `predict(["CCO"])` without
cloning the repository or understanding the project's internal structure.

## Motivation

The project currently has 6 registered models (GC-PC-SAFT, RF, NN, GNN, ChemBERTa,
ensemble), a model registry with a clean `get_model()` / `predict()` /
`predict_with_uncertainty()` interface, applicability domain checking, boiling point
computation, and Henry's constant estimation. But all of this is embedded in a
project-specific directory structure (`model/registry.py`, `model/gnn/`,
`model/saved/`, `screening/hfo_screening.py`) that requires cloning the entire repo.

No usable, open-source PC-SAFT parameter prediction tool exists. The three closest
competitors — Winter et al. (SPT-PCSAFT), Felton et al. (ML-SAFT), and Habicht
et al. — all published results or datasets but not a callable prediction tool.

The two target audiences:

- **Audience A (Engineers)**: Want `predict(smiles_list)` → DataFrame with m, σ, ε/k,
  uncertainties, AD flags. Don't care about the MLOps stack.
- **Audience B (Researchers)**: Want trained weights, training code, and evaluation
  scripts. Care about reproducibility, benchmarking, and **training-label provenance**
  (experimental fits vs literature regressions vs model-generated labels).

This step serves both by creating a minimal package for Audience A and keeping the
full repo functional for Audience B.

## Dependencies

- Steps 01-04 (model training): at least RF and GNN models trained and saved
- Step 14 (applicability domain): Tanimoto AD available
- Step 24 (`screening/hfo_screening.py`): boiling point computation
- No dependency on Steps 05-08 (serving/infra) — the package bypasses FastAPI entirely

## Implementation Guide

### 26.1 Create package directory structure

```
pcsaft_predict/
├── __init__.py          # Public API: predict, predict_with_uncertainty, list_models
├── _registry.py         # Simplified model registry (no serving dependencies)
├── _features.py         # Feature computation (Morgan FP + RDKit descriptors)
├── _gnn.py              # GNN architecture + graph featurizer (self-contained)
├── _rf.py               # RF model wrapper
├── _ensemble.py         # Ensemble wrapper
├── _ad.py               # Applicability domain (Tanimoto nearest-neighbor)
├── _thermo.py           # Boiling point and Henry's constant from PC-SAFT params
├── _weights.py          # Weight download/cache manager
├── py.typed             # PEP 561 marker for type checking
└── data/
    └── .gitkeep         # Weights downloaded here on first use
```

### 26.2 Public API design (`__init__.py`)

Expose exactly these functions at the top level:

```python
from pcsaft_predict._registry import predict, predict_with_uncertainty, list_models, load_model

__all__ = ["predict", "predict_with_uncertainty", "list_models", "load_model"]
__version__ = "1.0.0"
```

#### `predict(smiles, model="gnn")` → `pandas.DataFrame`

- Input: `list[str]` of SMILES, or a single SMILES string
- Output: DataFrame with columns: `smiles`, `m`, `sigma`, `epsilon_k`, `in_domain`, `tanimoto_nn`
- Default model: `"gnn"` (best benchmark performance)
- Lazy-loads model weights on first call

Important documentation requirement: if `"gnn"` is the default, the package
README and top-level docstring must state the training provenance of that model
(for example: experimentally fitted labels only, mixed literature-fitted labels,
or inclusion of upstream model-generated labels such as SPT-PCSAFT). Do not let
"best model" imply "best for every scientific use case."

#### `predict_with_uncertainty(smiles, model="gnn", n_forward=30)` → `pandas.DataFrame`

- Same as `predict()` plus columns: `m_std`, `sigma_std`, `epsilon_k_std`
- Uses MC Dropout (GNN, NN, ChemBERTa) or tree variance (RF)

#### `list_models()` → `list[str]`

- Returns available model names: `["gnn", "rf", "nn", "chemberta", "ensemble"]`

At minimum, also expose model provenance in package documentation. Optional:
extend `list_models(detailed=True)` to return a compact metadata table including
training source, uncertainty method, and intended use.

#### `load_model(name)` → model wrapper

- Returns the model wrapper object for advanced use (e.g., accessing embeddings)

### 26.3 Extract and simplify model code

For each model, create a self-contained wrapper in `pcsaft_predict/` that:
1. Imports only from standard libraries, torch, torch_geometric, rdkit, numpy, pandas, scikit-learn
2. Does NOT import from `model/`, `serving/`, `portal/`, `screening/`, or `pipeline/`
3. Loads weights from a local cache directory (`~/.cache/pcsaft_predict/` or `pcsaft_predict/data/`)
4. Carries minimal provenance metadata: training data source, whether labels are
   experimental/literature-fitted/model-generated, and the uncertainty method used

Key extractions from existing code:

| Source file | Target file | What to extract |
|-------------|-------------|-----------------|
| `model/registry.py` lines 77-89 (`_compute_features`) | `_features.py` | Feature computation with Morgan FP + RDKit |
| `model/registry.py` lines 273-381 (`GNNModel`) | `_gnn.py` | GNN wrapper + architecture import |
| `model/gnn/architecture.py` | `_gnn.py` | `PCSAFTGraphNet` class (inline) |
| `model/gnn/graph_featurizer.py` | `_gnn.py` | `smiles_to_graph()` function (inline) |
| `model/registry.py` lines 122-170 (`RFModel`) | `_rf.py` | RF wrapper |
| `model/registry.py` lines 531-565 (`EnsembleModel`) | `_ensemble.py` | Ensemble wrapper |
| `screening/hfo_screening.py` | `_thermo.py` | `compute_boiling_point()`, `compute_henrys_constant()` |

The GNN is the priority — it's the best benchmarked model and has no scikit-learn
dependency for inference (only torch + torch_geometric + rdkit). If its training
set includes model-generated labels, that must be disclosed prominently in the
package docs and model metadata.

### 26.4 Weight management (`_weights.py`)

Implement a download-on-first-use pattern:

```python
WEIGHT_URLS = {
    "gnn": "https://github.com/<org>/<repo>/releases/download/v1.0.0/gnn_pcsaft.pt",
    "rf_m": "https://github.com/<org>/<repo>/releases/download/v1.0.0/rf_m.joblib",
    # ...
}
CACHE_DIR = Path.home() / ".cache" / "pcsaft_predict"

def ensure_weights(model_name: str) -> Path:
    """Download model weights if not cached. Return path to weights file."""
```

- Check `CACHE_DIR` first; download from GitHub Releases if missing
- SHA-256 checksum verification after download
- Support `PCSAFT_PREDICT_CACHE_DIR` environment variable override
- Support offline mode: if weights are pre-placed in cache, no network needed
- Print a one-time message on first download: "Downloading GNN weights (45 MB)..."

### 26.5 Create package `pyproject.toml`

Separate from the project root `pyproject.toml`. Place at `pcsaft_predict/pyproject.toml`
(or at repo root with `packages = ["pcsaft_predict"]`).

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pcsaft-predict"
version = "1.0.0"
description = "Predict PC-SAFT equation-of-state parameters from molecular SMILES"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "torch>=2.0",
    "torch_geometric>=2.4",
    "rdkit",
    "joblib>=1.3",
    "scipy>=1.10",
]

[project.optional-dependencies]
hf = ["transformers>=4.30"]  # Only needed for ChemBERTa model
thermo = ["teqp>=0.22"]      # Only needed for boiling point / Henry's constant
all = ["transformers>=4.30", "teqp>=0.22"]

[project.urls]
Homepage = "https://github.com/<org>/pcsaft-predict"
Documentation = "https://github.com/<org>/pcsaft-predict#readme"
```

### 26.6 Create tests (`tests/test_pcsaft_predict.py`)

Smoke tests for the public API:

1. `test_predict_single_smiles`: `predict("CCO")` returns DataFrame with 1 row, correct columns
2. `test_predict_batch`: `predict(["CCO", "c1ccccc1", "CCCCC"])` returns 3 rows
3. `test_predict_with_uncertainty`: extra columns present (`m_std`, `sigma_std`, `epsilon_k_std`)
4. `test_list_models`: returns list containing at least `"gnn"` and `"rf"`
5. `test_invalid_smiles`: `predict(["invalid"])` returns row with NaN values, no crash
6. `test_predict_rf_model`: `predict("CCO", model="rf")` uses RF model
7. `test_ad_flag_present`: `in_domain` and `tanimoto_nn` columns exist in output

Mark tests that require model weights as `@pytest.mark.slow` so CI can skip them
when weights aren't available.

### 26.7 Verify coexistence with project

The `pcsaft_predict/` package must coexist with the existing `model/` package.
Both should be importable from the repo root:

```python
# Existing project code (unchanged)
from model.registry import get_model

# New standalone package
from pcsaft_predict import predict
```

The key difference: `pcsaft_predict` is self-contained and installable separately;
`model/` is the full project code with all dependencies.

## Key Outputs

| Artifact | Path |
|----------|------|
| Package source | `pcsaft_predict/` |
| Package pyproject.toml | `pcsaft_predict/pyproject.toml` (or root-level with package config) |
| Tests | `tests/test_pcsaft_predict.py` |
| Report | `docs/reports/26_extract_installable_package.md` |

## Acceptance Criteria (When to Move On)

- [ ] `from pcsaft_predict import predict` works in a Python session
- [ ] `predict(["CCO", "c1ccccc1"])` returns a DataFrame with m, σ, ε/k, in_domain, tanimoto_nn
- [ ] `predict_with_uncertainty(["CCO"])` returns DataFrame with uncertainty columns
- [ ] `list_models()` returns at least `["gnn", "rf"]`
- [ ] Invalid SMILES produce NaN rows, not crashes
- [ ] Weight download/cache mechanism works (mock in tests, real in integration)
- [ ] Package builds: `pip install -e ./pcsaft_predict` succeeds
- [ ] Tests in `tests/test_pcsaft_predict.py` pass (≥ 7 tests)
- [ ] Package documentation or metadata clearly states each model's training provenance
- [ ] Default-model messaging does not imply universal scientific superiority outside the evaluated benchmark setting
- [ ] No modifications to existing `model/`, `serving/`, `portal/` code
- [ ] Report at `docs/reports/26_extract_installable_package.md`
