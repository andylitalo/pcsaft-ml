# Results Report

This document summarizes both training runs. Detailed reports and figures for each are in separate files.

## Training Runs

| Run | Dataset | Train / Test | Features | Report | Figures |
|-----|---------|-------------|----------|--------|---------|
| Fallback | 34 hand-curated molecules | 27 / 7 | 68 | [results_fallback.md](results_fallback.md) | [figures/fallback/](../figures/fallback/) |
| Esper | 1,801 Esper et al. molecules | 1,410 / 356 | 170 | [results_esper.md](results_esper.md) | [figures/esper/](../figures/esper/) |

## Model Performance Comparison

| Parameter | Fallback R² (n=7) | Esper R² (n=356) |
|-----------|--------------------|-------------------|
| m (segments) | 0.68 | 0.61 |
| σ (Å) | 0.52 | 0.32 |
| ε/k (K) | 0.77 | 0.27 |

The fallback R² values were inflated by having only 7 test points. The Esper dataset provides a more realistic evaluation:

- **m** is the best-predicted parameter (R²=0.61)--topological descriptors correlate well with chain length.
- **σ** and **ε/k** are harder to predict from 2D descriptors alone (R²=0.27-0.32). These parameters depend on 3D molecular shape and polarizability, which RDKit 2D descriptors only partially capture.
- High train R² (~0.90-0.95) vs lower test R² indicates overfitting with 170 features on 1,410 training samples.

## Screening Summary

Both runs screen the same 63 HFO candidates through SA Score filtering (49 pass) and PC-SAFT similarity ranking against cyclopentane (m=2.37, σ=3.71 Å, ε/k=288.8 K).

No candidates meet the strict closeness criteria (σ < 2%, ε/k < 5%) in either run. The top-ranked candidates are fluorinated cyclopropane derivatives, which are physically smaller and have weaker dispersion interactions than cyclopentane.

## How to Reproduce

### Prerequisites

- Python ≥ 3.10
- pip

### Steps

```bash
# 1. Install the package and dev dependencies
pip install -e ".[dev]"

# 2. Download the Esper dataset (~1,801 molecules after dedup)
python -m model.data.download_esper

# 3. Train on the Esper dataset (with hyperparameter tuning)
python -m model.train --data esper --tune

# 4. Evaluate on held-out test set
python -m model.evaluate

# 5. Run the screening pipeline (offline, skipping patent API)
python -m screening.run_screening --skip-patents

# 6. Run the test suite
pytest tests/ -v
```

### Predict for Specific Molecules

```bash
python -m model.predict "C1CCCC1" "CC=C" "FC(=CF)C(F)F"
```
