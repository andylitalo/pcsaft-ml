# Step 10: ML-SAFT Dataset Integration

## Purpose

All Phase 1 models trained on the Esper dataset (1,801 molecules). The ML-SAFT dataset (Heid et al. 2023 / Felton et al. 2024) provides additional PC-SAFT parameter fits from literature, covering a broader structural space including more fluorinated and heteroatom-containing molecules. Integrating it expands training data, improves coverage of the chemical space most relevant to blowing agents, and reduces the OOD rate for novel candidates.

The scaffolding is already in place: `model/data/download_mlsaft.py` and `load_data(source="combined")` in `model/data/load.py`. This step executes the integration and measures the impact on model performance.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| 1,801 molecules (Esper only) | Combined dataset (Esper + ML-SAFT, deduped by InChI) |
| Esper-biased toward common industrials | Broader structural coverage |
| RF R² ε/k = 0.33 | Expected improvement; document actual delta |

## Skills Demonstrated

- **Data pipeline integration**: Merging heterogeneous literature datasets with proper deduplication
- **Experimental design**: Controlled comparison (same train/test split strategy, same model architecture)
- **Scientific rigor**: Reporting null results if the uplift is smaller than expected

## Dependencies

- Requires: network access to download ML-SAFT CSV
- Requires: `model/data/download_mlsaft.py` (already exists)
- Requires: `load_data("combined")` path in `model/data/load.py` (already exists)
- Does not require: any new packages

## Implementation Guide

### 10.1 Download the ML-SAFT Dataset

```bash
cd /path/to/ml_chem
python model/data/download_mlsaft.py
```

Expected output: `model/data/mlsaft_pcsaft.csv`. Verify row count and column schema (should have at minimum: SMILES, m, sigma, epsilon_k).

### 10.2 Verify Combined Loader

```python
from model.data.load import load_data
df = load_data("combined")
print(f"Combined: {len(df)} molecules")
print(df[["m", "sigma", "epsilon_k"]].describe())
```

Confirm: (a) no NaN targets, (b) InChI deduplication ran, (c) Esper values win on conflicts.

### 10.3 Retrain RF (Combined Features) on Combined Data

Use the same feature pipeline and hyperparameters as Step 01. Only the data source changes.

```bash
python -m model.train --source combined --model rf --output model/saved/rf_combined_v2/
```

Log:
- Number of training samples
- Number of test samples (same random seed + stratification as before)
- R², MAE, RMSE for m/σ/ε/k

### 10.4 Retrain ChemBERTa on Combined Data

```bash
python -m model.hf.train_chemberta --source combined --output model/saved/chemberta_v2/
```

### 10.5 Generate Comparison Figure

Bar chart: R² for m/σ/ε/k, four bars per target: RF(Esper), RF(Combined), ChemBERTa(Esper), ChemBERTa(Combined).

Save to `figures/10_mlsaft/r2_comparison_esper_vs_combined.png`.

### 10.6 Update state.yaml

Add step 10 entry with combined dataset size and delta R² metrics.

## Acceptance Criteria (When to Move On)

- [ ] `model/data/mlsaft_pcsaft.csv` exists and has ≥ 1,000 rows
- [ ] Combined dataset has ≥ 2,500 molecules after deduplication
- [ ] RF and ChemBERTa retrained on combined data; metrics compared to Esper-only baseline
- [ ] Figure saved to `figures/10_mlsaft/`
- [ ] Report written at `docs/reports/10_mlsaft_integration.md`
- [ ] All existing tests pass
- [ ] Commit: `step 10: integrate ML-SAFT dataset and retrain`
