# Results Report: Esper Dataset (1,801 molecules)

## Model Performance

Three independent Random Forest regressors trained on the Esper dataset (1,801 molecules, 80/20 split, 170 RDKit 2D descriptors after cleaning). GridSearchCV hyperparameter tuning was applied.

| Parameter | MAE | RMSE | R² (test) | Train samples | Test samples |
|-----------|-----|------|-----------|---------------|--------------|
| m (segments) | 0.634 | 1.416 | 0.61 | 1,410 | 356 |
| σ (Å) | 0.192 | 0.330 | 0.32 | 1,410 | 356 |
| ε/k (K) | 26.1 | 49.8 | 0.27 | 1,410 | 356 |

The segment number (m) is the best-predicted parameter at R²=0.61, while σ and ε/k are harder to predict from 2D molecular descriptors alone. The high train R² (~0.90-0.95) versus lower test R² indicates overfitting, expected with tree-based models on 170-dimensional descriptor spaces.

Key limitations: RDKit 2D descriptors capture topological and electronic features but miss 3D conformational effects that influence segment diameter and dispersion energy. More advanced representations (e.g., molecular fingerprints, graph neural networks, or 3D descriptors) could improve σ and ε/k predictions.

## Screening Results

The pipeline was run with patent checking skipped (`--skip-patents`):

| Stage | Candidates In | Candidates Out |
|-------|---------------|----------------|
| 1. HFO generation | -- | 63 |
| 2. SA Score ≤ 4.5 | 63 | 49 |
| 3. Patent check | skipped | -- |
| 4. PC-SAFT ranking | 49 | 49 (ranked) |

### Top Candidates

Cyclopentane reference: m=2.37, σ=3.71 Å, ε/k=288.8 K

| Rank | SMILES | SA Score | m | σ (Å) | ε/k (K) | Distance | σ diff | ε/k diff |
|------|--------|----------|------|--------|---------|----------|--------|----------|
| 1 | F[C@@H]1C[C@@H]1F | 3.76 | 2.91 | 3.29 | 268.8 | 0.265 | 11.5% | 6.9% |
| 2 | F[C@H]1[C@@H](F)[C@H]1F | 3.26 | 3.01 | 3.24 | 225.2 | 0.371 | 12.6% | 22.0% |
| 3 | FC1(F)CC1(F)F | 3.31 | 3.01 | 3.31 | 220.7 | 0.376 | 10.9% | 23.6% |
| 4 | F[C@@H]1CC1(F)F | 3.69 | 3.09 | 3.26 | 228.1 | 0.391 | 12.1% | 21.0% |
| 5 | F/C=C/C(F)F | 4.24 | 2.74 | 3.34 | 185.4 | 0.403 | 9.9% | 35.8% |

### Comparison to Closeness Criteria

Using the empirical rules of thumb:
- **σ < 2% difference**: No candidates meet this criterion. Best σ match is ~9.9% off.
- **ε/k < 5% difference**: No candidates meet this criterion. Best ε/k match is ~6.9% off.

All top-ranked candidates are fluorinated cyclopropane or small fluorinated alkene derivatives. Their predicted σ values (3.2-3.3 Å) are consistently below cyclopentane's 3.71 Å, reflecting smaller molecular size. Their ε/k values (170-270 K) are below cyclopentane's 288.8 K, indicating weaker dispersion interactions.

These results are physically reasonable: small fluorinated C₃-C₄ molecules inherently have different PC-SAFT parameters than a C₅ cycloalkane. However, the model's moderate predictive accuracy (R²=0.27-0.61) means the predicted parameter gaps may be larger or smaller than shown. The screening ranking should be treated as a prioritization guide rather than a quantitative filter.

## Parity Plots

See `figures/esper/` for parity plots from this run.
