# Results Report: Fallback Dataset (34 molecules)

## Model Performance

Three independent Random Forest regressors were trained on the fallback dataset (34 molecules, 80/20 train/test split, 68 RDKit 2D descriptors after cleaning).

| Parameter | MAE | RMSE | R² (test) | Train samples | Test samples |
|-----------|-----|------|-----------|---------------|--------------|
| m (segments) | 0.316 | 0.476 | 0.68 | 27 | 7 |
| σ (Å) | 0.107 | 0.143 | 0.52 | 27 | 7 |
| ε/k (K) | 12.8 | 17.1 | 0.77 | 27 | 7 |

These numbers reflect the small training set (27 train, 7 test molecules). The R² values are inflated by the small test set. Even so, the model captures the right trends--a cyclopentane self-prediction returns m=2.43, σ=3.74, ε/k=283.4 versus the true values of m=2.37, σ=3.71, ε/k=288.8 (errors of 2.5%, 0.8%, and 1.9% respectively).

## Screening Results

The pipeline was run with patent checking skipped (`--skip-patents`):

| Stage | Candidates In | Candidates Out |
|-------|---------------|----------------|
| 1. HFO generation | -- | 63 |
| 2. SA Score ≤ 4.5 | 63 | 49 |
| 3. Patent check | skipped | -- |
| 4. PC-SAFT ranking | 49 | 49 (ranked) |

### Top Candidates

| Rank | SMILES | SA Score | m | σ (Å) | ε/k (K) | Distance | σ diff | ε/k diff |
|------|--------|----------|---|--------|---------|----------|--------|----------|
| 1 | FC1(F)CC1(F)F | 3.31 | 2.24 | 3.52 | 225.4 | 0.232 | 5.2% | 22.0% |
| 2 | F[C@@H]1CC1(F)F | 3.69 | 2.23 | 3.45 | 226.9 | 0.232 | 7.1% | 21.4% |
| 5 | F[C@@H]1C[C@@H]1F | 3.76 | 2.25 | 3.46 | 224.7 | 0.238 | 6.8% | 22.2% |
| 8 | F[C@H]1[C@@H](F)[C@H]1F | 3.26 | 2.24 | 3.46 | 223.0 | 0.244 | 6.8% | 22.8% |

### Comparison to Closeness Criteria

- **σ < 2% difference**: No candidates meet this criterion. The best σ match is ~5% off.
- **ε/k < 5% difference**: No candidates meet this criterion. The best ε/k match is ~22% off.

All top-ranked candidates are fluorinated cyclopropane derivatives. Their predicted ε/k values cluster around 220-230 K, significantly below cyclopentane's 288.8 K.

## Parity Plots

See `figures/fallback/` for parity plots from this run.
