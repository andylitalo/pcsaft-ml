# Results Report: GNN Distributable Package

## Overview

Successfully added GNN (Graph Isomorphism Network) support to the `pcsaft_predict` standalone package. The promoted screening model from Step 31 is now available through the same public API as the RF baseline. Users can predict PC-SAFT parameters using `pcsaft_predict.predict(smiles, model="gnn")` with optional dependencies on `torch` and `torch_geometric`.

## Implementation Summary

### Package Changes

Added three new modules to `pcsaft_predict/`:

1. `_gnn_architecture.py` — `PCSAFTGraphNet` architecture (copied from `model/gnn/architecture.py`)
2. `_gnn_featurizer.py` — SMILES-to-graph conversion functions (`smiles_to_graph`, `atom_features`, `bond_features`)
3. `_gnn.py` — `GNNModel` wrapper class with `load()`, `predict()`, and `predict_with_uncertainty()` methods

Modified:

4. `_registry.py` — Added conditional GNN registration with `try/except ImportError`
5. `__init__.py` — Updated docstring to include GNN usage example

### GNN Model Interface

The GNN model wrapper follows the same interface as the RF model:

```python
import pcsaft_predict

# Basic prediction
df = pcsaft_predict.predict("C(F)=C(F)F", model="gnn")
# Returns: DataFrame with columns [smiles, m, sigma, epsilon_k, in_domain, tanimoto_nn]

# Prediction with uncertainty (MC Dropout)
df = pcsaft_predict.predict_with_uncertainty(["CCO", "c1ccccc1"], model="gnn", n_forward=30)
# Returns: DataFrame with additional columns [m_std, sigma_std, epsilon_k_std]
```

The implementation:

- Loads the GNN checkpoint from `model/saved/gnn_pcsaft.pt` (or cache directory)
- Reconstructs the `PCSAFTGraphNet` from saved model configuration
- Denormalizes predictions using saved target scalers
- Handles invalid SMILES gracefully (returns NaN, same as RF)
- Implements MC Dropout uncertainty via the model's `predict_with_uncertainty()` method

### Optional Dependency Handling

The GNN model requires `torch>=2.0` and `torch_geometric>=2.4`, which are heavy dependencies that RF-only users should not be forced to install. The implementation uses a lazy-import pattern:

1. **Conditional registration**: `_registry.py` imports `GNNModel` inside a `try/except` block. If `torch` or `torch_geometric` are not installed, the GNN model is silently excluded from the registry.

2. **Lazy imports in GNNModel**: All PyTorch imports happen inside the `load()` and `predict()` methods, not at module level.

3. **Clear error messages**: If a user calls `predict(smiles, model="gnn")` without torch installed, they receive:
   ```
   ImportError: GNN model requires torch and torch_geometric.
   Install with: pip install torch torch_geometric
   ```

4. **Graceful degradation**: `list_models()` returns `["rf"]` when torch is unavailable and `["rf", "gnn"]` when it is.

### Applicability Domain (AD) Consistency

The existing `tanimoto_ad.joblib` in the package was built from the Esper+ML-SAFT training set (used for the RF model). The GNN was trained on a unified dataset (Esper + ML-SAFT + SPT-PCSAFT, 13,764 molecules), which is a superset of the RF training data.

**Decision**: Continue using the existing AD model for both RF and GNN. This is the conservative choice:

- The AD fingerprints represent the union of Esper + ML-SAFT (the majority of the GNN training set)
- SPT-PCSAFT molecules (the additional ~1,800 in the GNN training set) will be flagged as out-of-domain, but this is acceptable because:
  - The AD is an alert, not a hard cutoff
  - Users evaluating domain should primarily care about similarity to well-characterized molecules
  - Flagging additional training data as OOD is conservative and safe

If per-model AD becomes a requirement in the future, the package could load `tanimoto_ad_gnn.joblib` when `model="gnn"` is specified.

### Comparison to Internal Registry

The package GNN produces identical results to the internal registry GNN:

```python
# Package predictions for ["CCO", "c1ccccc1", "C(F)=C(F)F"]
[[  2.68917274   3.05074525 227.42904663]
 [  1.48111606   3.79364967 322.70617676]
 [  2.76497269   2.92176867 156.93811035]]

# Registry predictions for the same molecules
[[  2.68917274   3.05074525 227.42904663]
 [  1.48111606   3.79364967 322.70617676]
 [  2.76497269   2.92176867 156.93811035]]

Max absolute difference: 0.0
```

The uncertainty estimates are stochastic (MC Dropout), so they vary between runs, but the mean predictions are deterministic and match exactly.

## Test Coverage

Added comprehensive test suite in `tests/test_pcsaft_predict_gnn.py`:

1. `test_gnn_in_list_models` — Verify `"gnn"` appears in `list_models()` when torch is available
2. `test_gnn_predict_single` — Predict one SMILES, check output columns
3. `test_gnn_predict_batch` — Predict a list, verify shapes
4. `test_gnn_predict_with_uncertainty` — Verify `_std` columns are present and non-negative
5. `test_gnn_invalid_smiles` — Verify NaN handling for invalid input
6. `test_gnn_fluorinated_molecule` — Test on fluorinated olefin (GNN's strength)
7. `test_gnn_predict_consistency` — Verify deterministic behavior across calls

All tests use `pytest.importorskip("torch")` to skip gracefully on environments without torch.

**Test results**:

```
tests/test_pcsaft_predict.py::test_predict_single_smiles PASSED
tests/test_pcsaft_predict.py::test_predict_batch PASSED
tests/test_pcsaft_predict.py::test_predict_with_uncertainty PASSED
tests/test_pcsaft_predict.py::test_list_models PASSED
tests/test_pcsaft_predict.py::test_invalid_smiles PASSED
tests/test_pcsaft_predict.py::test_predict_rf_model PASSED
tests/test_pcsaft_predict.py::test_ad_flag_present PASSED
tests/test_pcsaft_predict_gnn.py::test_gnn_in_list_models PASSED
tests/test_pcsaft_predict_gnn.py::test_gnn_predict_single PASSED
tests/test_pcsaft_predict_gnn.py::test_gnn_predict_batch PASSED
tests/test_pcsaft_predict_gnn.py::test_gnn_predict_with_uncertainty PASSED
tests/test_pcsaft_predict_gnn.py::test_gnn_invalid_smiles PASSED
tests/test_pcsaft_predict_gnn.py::test_gnn_fluorinated_molecule PASSED
tests/test_pcsaft_predict_gnn.py::test_gnn_predict_consistency PASSED

============================== 14 passed in 4.67s ===============================
```

All existing tests continue to pass, confirming backward compatibility.

## Key Findings

1. **Seamless integration**: The GNN model integrates cleanly into the existing package architecture with no changes to the public API beyond adding `model="gnn"` as an option.

2. **Identical predictions**: The package GNN produces bit-for-bit identical predictions to the internal registry GNN, confirming correct implementation of checkpoint loading, graph featurization, and denormalization.

3. **Optional dependency pattern works**: Conditional registration via `try/except ImportError` allows the package to support both lightweight RF-only installs and full GNN installs without code duplication or version forks.

4. **Uncertainty estimation**: MC Dropout uncertainty via `predict_with_uncertainty()` works correctly and returns reasonable uncertainty estimates (e.g., m_std ~0.15-0.16, sigma_std ~0.02-0.03, epsilon_k_std ~3-4 for common molecules).

5. **AD conservatism**: Using the RF training set AD for the GNN is conservative (flags some GNN training data as OOD) but safe. The alternative (building a unified AD) would require rebuilding the AD artifact and re-validating the Tanimoto threshold.

## Artifacts

| File | Description |
|------|-------------|
| `pcsaft_predict/_gnn.py` | GNN model wrapper for the package |
| `pcsaft_predict/_gnn_architecture.py` | PCSAFTGraphNet architecture (inference only) |
| `pcsaft_predict/_gnn_featurizer.py` | SMILES-to-graph conversion for inference |
| `pcsaft_predict/_registry.py` | Updated with conditional GNN registration |
| `pcsaft_predict/__init__.py` | Updated docstring with GNN example |
| `tests/test_pcsaft_predict_gnn.py` | GNN-specific package tests (7 tests) |

## Readiness Check

- [x] `pcsaft_predict.predict("C(F)=C(F)F", model="gnn")` returns valid predictions
- [x] `pcsaft_predict.predict_with_uncertainty(["C(F)=C(F)F"], model="gnn")` returns predictions with `_std` columns
- [x] `"gnn"` appears in `pcsaft_predict.list_models()` when torch is installed
- [x] Package degrades gracefully (clear ImportError) when torch is not installed
- [x] GNN predictions from the package match `model.registry.get_model("gnn").predict()` within floating-point tolerance (exact match achieved)
- [x] All existing tests pass
- [x] `ruff check .` passes

## Next Steps

The distributable package now exposes both RF and GNN models through the same API. External users can install and use the promoted GNN model for fluorinated screening. Steps 37-42 (screening workflows) can reference the package interface as an alternative to internal imports, improving portability and reproducibility.

The GNN's advantage over RF is particularly pronounced for fluorinated chemistry:

- Fluorinated slice R² (GNN): m=0.86, sigma=0.73, epsilon_k=0.91
- Fluorinated slice R² (RF): m=-0.15, sigma=0.30, epsilon_k=0.24

Users can now access this improved performance through the public package with minimal dependency overhead.
