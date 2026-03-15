# Step 15: Improved Models Benchmark

## Purpose

The RF (R² ε/k = 0.33, σ = 0.35) is the Phase 1 champion. Two model families are expected to improve on this:

- **XGBoost**: gradient boosting with built-in regularization, explicit handling of feature interactions, and typically +0.03–0.10 R² over RF on tabular regression tasks
- **chemprop (D-MPNN)**: message-passing graph neural network that operates directly on molecular graphs; the current SOTA for molecular property prediction from structure; has outperformed descriptor-based methods on most MoleculeNet benchmarks

If chemprop beats RF by ≥ 0.05 R² on ε/k (the most impactful target), it is promoted as the new default model in `serving/model_loader.py`.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| 4 models in harness (GC, RF, NN, ChemBERTa) | 6 models (+ XGBoost, chemprop) |
| RF is the champion | Champion possibly updated to chemprop |
| No graph-based model | D-MPNN explores chemical-graph features unavailable to descriptor methods |

## Skills Demonstrated

- **Gradient boosting**: XGBoost hyperparameter tuning, feature importance
- **Graph neural networks**: chemprop D-MPNN, multi-task regression on SMILES
- **Model selection**: rigorous champion/challenger protocol with a defined threshold

## Dependencies

- Requires: combined dataset from Step 10 (or Esper if Step 10 not completed)
- New packages: `xgboost`, `chemprop`

```toml
[project.optional-dependencies]
models2 = ["xgboost>=2.0", "chemprop>=2.0"]
```

Install: `uv sync --extra dev --extra models2`

## Implementation Guide

### 15.1 XGBoost Regressor (`model/xgb/`)

Create `model/xgb/__init__.py` and `model/xgb/xgb_model.py`:

```python
"""XGBoost-based PC-SAFT parameter predictor."""
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from model.registry import register_model


@register_model("xgboost")
class XGBoostPCSAFT:
    """Multi-output XGBoost regressor for PC-SAFT parameters."""

    PARAM_GRID = {
        "estimator__n_estimators": [200, 500],
        "estimator__max_depth": [4, 6],
        "estimator__learning_rate": [0.05, 0.1],
        "estimator__subsample": [0.8],
        "estimator__colsample_bytree": [0.8],
    }

    def __init__(self):
        self.model_ = None
        self.feature_names_ = None

    def fit(self, X_train, y_train, feature_names=None):
        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        wrapped = MultiOutputRegressor(base)
        cv = GridSearchCV(wrapped, self.PARAM_GRID, cv=5, scoring="r2",
                          n_jobs=-1, refit=True, verbose=1)
        cv.fit(X_train, y_train)
        self.model_ = cv.best_estimator_
        self.feature_names_ = feature_names
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_with_uncertainty(self, X, n_estimators_sample: int = 50):
        """Bootstrap uncertainty via random subsample of XGBoost trees."""
        preds = np.array([
            est.predict(X) for est in self.model_.estimators_
        ])
        return preds.mean(axis=0), preds.std(axis=0)

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
```

### 15.2 Train XGBoost

```bash
python -m model.train --source combined --model xgboost --output model/saved/xgb/
```

Log best hyperparameters, feature importances (top 20), and all metrics.

### 15.3 chemprop D-MPNN

chemprop v2.x has a `train` CLI and a Python API. Use the Python API for integration with the harness:

```python
"""chemprop D-MPNN wrapper for PC-SAFT parameter prediction."""
from chemprop import data, models, nn, train
from model.registry import register_model


@register_model("chemprop")
class ChempropPCSAFT:
    """D-MPNN (chemprop) for multi-target PC-SAFT prediction."""

    def fit(self, smiles_list, y_train):
        # Build chemprop dataset
        # Use default D-MPNN architecture (message_passing + ffn)
        # Train with AdamW, cosine LR schedule, early stopping
        ...

    def predict(self, smiles_list):
        ...

    def predict_with_uncertainty(self, smiles_list, n_ensemble=5):
        # Train 5 models with different random seeds; mean + std
        ...
```

Train with:
- Default message-passing depth = 3
- Hidden size = 300
- 20 epochs with early stopping (patience=5)
- Multi-task output (3 targets: m, σ, ε/k)

If chemprop's Python API changes significantly from this sketch, adapt to the installed version's API.

### 15.4 Evaluate and Compare

```bash
python -m model.evaluate --models gc_pcsaft rf nn chemberta xgboost chemprop --metrics all
```

Full 6-model comparison table (R², MARE, CCC, coverage@10%) for all three targets.

### 15.5 Promotion Decision

If `chemprop_epsilon_k_r2 - rf_epsilon_k_r2 >= 0.05`:
- Update `serving/model_loader.py` default from `"rf"` to `"chemprop"`
- Update `serving/config.py` `DEFAULT_MODEL` setting
- Re-run serving tests

Otherwise: document the result, keep RF as default.

### 15.6 Generate Figures

- `figures/15_models/radar_ccc.png`: radar chart with one axis per model-target combination; 5 models × 3 targets shown as a single chart
- `figures/15_models/r2_heatmap.png`: heatmap (models × targets) with R² values
- `figures/15_models/parity_best_model.png`: parity plots for the best model on all three targets
- `figures/15_models/xgb_feature_importance.png`: top 20 XGBoost feature importances

## Acceptance Criteria (When to Move On)

- [ ] `model/xgb/xgb_model.py` registered in harness; model saved to `model/saved/xgb/`
- [ ] chemprop D-MPNN registered; model saved to `model/saved/chemprop/`
- [ ] Full 6-model comparison table with R², MARE, CCC; compared to Phase 1 RF baseline
- [ ] Promotion decision documented with rationale
- [ ] If promoted: serving tests pass with new default model
- [ ] Four figures saved to `figures/15_models/`
- [ ] Report written at `docs/reports/15_improved_models.md`
- [ ] Add ≥ 8 tests for XGBoost and chemprop wrappers
- [ ] All existing tests pass
- [ ] Commit: `step 15: benchmark XGBoost and chemprop against RF baseline`
