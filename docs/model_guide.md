# Model Guide

## Feature Engineering

### RDKit 2D Descriptors (~200 features)

We use the full set of RDKit 2D molecular descriptors, which encode:
- **Topological**: Wiener index, Balaban J, Kappa shape indices
- **Electronic**: Gasteiger charges, TPSA, LogP
- **Constitutional**: Molecular weight, atom counts, ring counts
- **Connectivity**: Chi connectivity indices, Hall-Kier alpha

After cleaning (removing zero-variance and highly correlated features), ~80–120 descriptors remain.

### Why 2D Descriptors?

- Fast to compute (no 3D conformer generation needed)
- Capture molecular size, shape, and electronics
- Well-suited for Random Forests which handle mixed-scale features
- PC-SAFT parameters are primarily molecular-level properties, well-captured by 2D features

## Model Architecture

### Three Independent Random Forests

We train one `RandomForestRegressor` per target:
- `rf_m.joblib` → segment number (m)
- `rf_sigma.joblib` → segment diameter (σ)
- `rf_epsilon_k.joblib` → dispersion energy (ε/k)

Default hyperparameters: `n_estimators=100, random_state=42`.

### Why Random Forest?

- Robust to feature scaling (no normalization needed)
- Handles nonlinear relationships between descriptors and PC-SAFT parameters
- Built-in feature importance for interpretability
- Low risk of overfitting with default settings
- Fast training on datasets of this size

## Training Details

```bash
# Train with auto data source selection
python -m model.train

# Train with Esper dataset specifically
python -m model.train --data esper

# Train with hyperparameter tuning
python -m model.train --tune
```

## Expected Performance

### With Fallback Data (~30 molecules)
Small dataset, so expect overfitting. Useful for pipeline testing only.

### With Esper Data (~1,842 molecules)
Expected ranges based on literature:
- m: R² ≈ 0.90–0.95, MAE ≈ 0.2–0.4 segments
- σ: R² ≈ 0.80–0.90, MAE ≈ 0.1–0.2 Å
- ε/k: R² ≈ 0.75–0.85, MAE ≈ 10–20 K

## Evaluation

```bash
python -m model.evaluate
```

Outputs MAE, RMSE, R² per parameter and saves parity plots to `model/saved/parity_plots.png`.
