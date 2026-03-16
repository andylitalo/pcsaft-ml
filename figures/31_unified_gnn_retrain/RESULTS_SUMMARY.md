# Step 31 Results Summary

## Training Configuration

- **Dataset**: Unified (Esper + ML-SAFT + SPT-PCSAFT)
- **Total molecules**: 13,764 (after InChI deduplication)
- **Train/test split**: 11,011 / 2,753 (80% / 20%, stratified on epsilon_k)
- **GNN architecture**: 4-layer GIN, hidden_dim=128, dropout=0.1
- **Training**: 50 epochs max, early stopping at epoch 39 (patience=10)
- **Hardware**: CPU (macOS, no GPU)

## Test Set Composition

| Source | Count | Percentage |
|--------|-------|------------|
| SPT-PCSAFT | 2,409 | 87.5% |
| Esper | 317 | 11.5% |
| ML-SAFT | 27 | 1.0% |
| **Total** | **2,753** | **100%** |

## Overall Performance (n=2,753)

| Model | m (R²) | σ (R²) | ε/k (R²) | m (MAE) | σ (MAE) | ε/k (MAE) |
|-------|--------|--------|----------|---------|---------|-----------|
| RF (Esper-trained) | -0.44 | 0.30 | 0.24 | 0.79 | 0.21 | 23.4 |
| **GNN (unified)** | **0.79** | **0.77** | **0.73** | **0.28** | **0.10** | **10.0** |
| Ensemble (equal) | 0.51 | 0.68 | 0.64 | 0.49 | 0.14 | 15.0 |
| Ensemble (inv-var) | 0.80 | 0.78 | 0.74 | 0.28 | 0.10 | 9.8 |

## Performance by Subset

### SPT-PCSAFT (n=2,409, 87.5%)

| Model | m (R²) | σ (R²) | ε/k (R²) |
|-------|--------|--------|----------|
| RF | -1.13 | 0.24 | 0.01 |
| **GNN** | **0.90** | **0.90** | **0.90** |

### Esper (n=317, 11.5%)

| Model | m (R²) | σ (R²) | ε/k (R²) |
|-------|--------|--------|----------|
| **RF** | **0.88** | **0.77** | **0.78** |
| GNN | 0.59 | 0.03 | 0.36 |

### Fluorinated (n=297, 10.8%) - **Product Chemistry**

| Model | m (R²) | σ (R²) | ε/k (R²) |
|-------|--------|--------|----------|
| RF | -0.15 | 0.30 | 0.24 |
| **GNN** | **0.86** | **0.73** | **0.91** |

## Key Insights

1. **GNN dominates on the unified dataset** (R² = 0.79/0.77/0.73 overall)
2. **RF fails on SPT-PCSAFT** (R² = -1.13/0.24/0.01) because it was trained on Esper only
3. **GNN excels on fluorinated molecules** (R² = 0.86/0.73/0.91), which are 90% SPT-PCSAFT
4. **Data source mismatch** explains Esper poor performance:
   - Esper fluorinated: ε/k = 208.7 ± 45.5 K (experimental)
   - SPT-PCSAFT: ε/k = 284.4 ± 34.9 K (model-predicted)
   - ~75 K systematic offset

## Recommendation

**Switch HFO screening to GNN** as it performs best on:
- Fluorinated molecules (target chemistry for blowing agents)
- SPT-PCSAFT data (87% of available corpus)
- Novel molecule prediction (no experimental bias)

Keep RF available for Esper-like queries where experimental parameters are preferred.
