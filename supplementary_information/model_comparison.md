# Model Comparison: Full Results and Production Rationale

This document provides the complete model comparison results across all evaluation sets and explains the rationale for selecting Random Forest (RF) for production deployment.

---

## All Models Evaluated

### On Esper Test Set (n=361, 80/20 split of 1,801 molecules)

| Model | m R^2 | sigma R^2 | epsilon/k R^2 | m MAE | sigma MAE | epsilon/k MAE (K) |
|-------|-------|-----------|---------------|-------|-----------|-------------------|
| GC-PC-SAFT | 0.40 | -1.16 | -0.04 | -- | -- | -- |
| RF | 0.62 | 0.35 | 0.33 | 0.59 | 0.19 | 26.8 |
| NN (PyTorch) | 0.47 | 0.11 | 0.14 | -- | -- | -- |
| ChemBERTa | 0.53 | 0.25 | 0.27 | 0.77 | 0.23 | 31.2 |
| XGBoost | 0.64 | 0.36 | 0.33 | -- | -- | -- |
| Chemprop D-MPNN | 0.54 | 0.33 | 0.39 | 0.69 | 0.21 | 26.8 |
| GINEConv (Esper) | -- | -- | 0.41 | -- | -- | -- |

### External Fluorinated Validation (15 refrigerants with published PC-SAFT params)

| Model | epsilon/k MAE (K) | Boiling Point MAE (K) | epsilon/k R^2 |
|-------|-------------------|----------------------|---------------|
| RF (Esper, 1,801 mol) | 14.3 | 8.2 | -0.47 |
| GINEConv (Esper) | -- | 142.3 | -- |
| GNN (unified, 13,764 mol) | 77.6 | 133.7 | -25.08 |

### On Unified Dataset (Esper + ML-SAFT + SPT-PCSAFT)

| Model | m R^2 | sigma R^2 | epsilon/k R^2 |
|-------|-------|-----------|---------------|
| GNN (unified) | 0.79 | 0.77 | 0.73 |
| GNN (Esper subset only) | -- | -- | 0.36 |
| GNN (SPT subset only) | -- | -- | 0.90 |

---

## Why RF Was Chosen for Production

1. **Best on the deployment domain**: RF boiling point MAE = 8.2 K on 15 fluorinated refrigerants; GNN was 16.4x worse at 133.7 K. The entire screening filter window is only 35 K wide.

2. **Data quality > quantity > architecture**: On the same ~1,900 Esper molecules, architectural differences between RF, XGBoost, chemprop, and GNN produced ±0.08 R^2 differences. With 7x more training data, GNN gained +0.40 R^2 on epsilon/k -- but the larger dataset carried systematic bias.

3. **Built-in uncertainty**: 100-tree ensemble provides per-prediction tree-variance uncertainty (though it overestimates actual error by ~6x for well-represented chemistry and must be calibrated locally).

4. **Fast inference**: ~2 seconds for 361 molecules vs 185 seconds for ChemBERTa.

5. **Interpretable features**: RDKit descriptors provide explainable feature importances.

---

## Where Other Models Might Be Valid

- **GNN (GINEConv, chemprop)**: Both consistently outperformed RF on epsilon/k (+0.06-0.08 R^2) on the same data. With a larger, clean experimental dataset (>5,000-10,000 molecules), GNNs should surpass RF. The crossover is data-limited, not architecture-limited.

- **XGBoost**: Matched RF exactly (0.33 vs 0.33 on epsilon/k); gradient boosting added no signal over bagging on these features. Could be preferred if feature interaction effects become important with richer data.

- **ChemBERTa**: Lower accuracy but requires no feature engineering. With pre-training on larger chemical corpora and more fine-tuning data, SMILES-based models could become competitive.

- **GC-PC-SAFT**: Zero data needed. Useful as a baseline or for novel functional groups with no training data at all.
