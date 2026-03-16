# Step 31: Unified Dataset and GNN Retraining

## Objective

Retrain the GNN on the full unified dataset (Esper + ML-SAFT + SPT-PCSAFT, ~13,764 molecules) using a shared train/test split with RF. This resolves the dataset mismatch that caused GNN to underperform on Esper-like molecules and the ensemble to fail. After retraining, switch the HFO screening pipeline from RF to GNN.

## Motivation

Three interconnected problems were identified in the current pipeline:

1. **`load_data("all")` was broken**: It fell through to a fallback CSV (~1,800 molecules) instead of loading the full corpus. This has been fixed in `model/data/load.py` (Phase A).
2. **Dataset mismatch**: RF was trained on Esper (1,801 molecules), GNN on a different pool. The shared test set is Esper-only, putting GNN at a disadvantage. GNN reports R² = 0.73-0.77 on its own test split but degrades on the Esper test set.
3. **Ensemble failure**: Inverse-variance weighting gave GNN 82% weight despite GNN being out-of-distribution on Esper. This caused the ensemble to underperform RF (Report 28, Step 02c).

After retraining on a unified dataset that includes Esper, GNN should perform well on Esper-like molecules, the ensemble should work correctly, and the screening pipeline can use the best model.

## Dependencies

- Phase A item A1 completed: `load_data("all")` returns Esper + ML-SAFT + SPT-PCSAFT
- Phase A item A4 completed: upstream licenses verified (ML-SAFT = MIT, SPT-PCSAFT = CC-BY-NC-SA 4.0)
- `model/gnn/train_gnn.py` exists with `--source all` support
- `model/data/load.py::split_data()` exists

## Implementation Guide

### 31.1 Verify `load_data("all")` produces the full corpus

```python
from model.data.load import load_data
df = load_data("all")
print(f"Total molecules: {len(df)}")
assert len(df) > 10000, f"Expected ~13,764, got {len(df)}"
print(f"Columns: {list(df.columns)}")
print(f"Source distribution (if available):")
```

Record the exact count for the report. Expected: ~13,764 after InChI deduplication.

### 31.2 Implement unified train/test split

Modify `model/gnn/train_gnn.py` (or add a shared utility) so that:

1. The train/test split is computed once and saved to `model/saved/test_set.csv`
2. Both GNN and RF use the same test set for evaluation
3. The split uses `split_data()` from `model/data/load` with `test_size=0.2, random_state=42, stratify_bins=5`

The saved test set must contain at minimum: `smiles`, `m`, `sigma`, `epsilon_k`.

If `model/saved/test_set.csv` already exists (from a previous RF run), check whether it's a subset of the full dataset. If not, regenerate it from the full corpus.

### 31.3 Retrain GNN

```bash
uv run python -m model.gnn.train_gnn --source all --epochs 100
```

Training configuration should remain the same as the original GNN:
- 4-layer GIN, hidden dim 256
- AdamW optimizer, lr=1e-3, weight decay=1e-4
- Early stopping with patience=20
- Batch size 64

Save the new weights to `model/saved/gnn_pcsaft.pt` (overwriting the old weights).

### 31.4 Re-evaluate all models on the unified test set

Load the saved test set and evaluate RF, GNN, and ensemble:

```python
from model.registry import get_model
from model.data.load import load_data, split_data

# Load test set
test_df = pd.read_csv("model/saved/test_set.csv")

for model_name in ["rf", "gnn"]:
    model = get_model(model_name)
    model.load()
    predictions = model.predict(test_df["smiles"].tolist())
    # Compute R², MAE, RMSE for each target
```

Record metrics in a comparison table:

| Model | m (R²) | σ (R²) | ε/k (R²) | m (MAE) | σ (MAE) | ε/k (MAE) |
|-------|--------|--------|----------|---------|---------|-----------|
| RF (Esper only) | ... | ... | ... | ... | ... | ... |
| GNN (unified) | ... | ... | ... | ... | ... | ... |
| Ensemble (equal) | ... | ... | ... | ... | ... | ... |
| Ensemble (inv-var) | ... | ... | ... | ... | ... | ... |

### 31.5 Re-calibrate ensemble

Test two ensemble strategies:

1. **Equal-weighted**: `ŷ = 0.5 * ŷ_RF + 0.5 * ŷ_GNN`
2. **Inverse-variance**: `ŷ = (w_RF * ŷ_RF + w_GNN * ŷ_GNN) / (w_RF + w_GNN)` where `w_i = 1/σ²_i`

With GNN now trained on a superset that includes Esper, its uncertainty estimates should be more calibrated for Esper-like molecules. The inverse-variance ensemble should now work correctly.

If inverse-variance still underperforms, fall back to equal weighting and document why.

### 31.6 Switch HFO screening to GNN

Update `screening/hfo_screening.py`:

1. Replace calls to `model.predict.predict_pcsaft()` (RF-only) with calls through the model registry: `model.registry.get_model("gnn")`
2. If the screening function accepts a `model_type` parameter, add one and default to `"gnn"`
3. Update `validate_boiling_points_rf()` to also support GNN (rename to `validate_boiling_points()` with a `model_type` parameter)

### 31.7 Update model cards and state.yaml

- Update `docs/model_cards/gnn.md` with new metrics (training data size, R² on unified test set)
- Update `docs/model_cards/ensemble.md` with new ensemble performance
- Update `state.yaml` with step 31 status and metrics

## Artifacts

| File | Description |
|------|-------------|
| `model/saved/gnn_pcsaft.pt` | Retrained GNN weights |
| `model/saved/test_set.csv` | Unified test set |
| `docs/reports/31_unified_gnn_retrain.md` | Step report with comparison tables |
| `docs/model_cards/gnn.md` | Updated model card |
| `docs/model_cards/ensemble.md` | Updated model card |
| `figures/31_unified_gnn_retrain/` | Parity plots, learning curves |

## Success Criteria

- [ ] `load_data("all")` returns >= 10,000 molecules
- [ ] GNN R² >= 0.73 on the unified test set for all three parameters
- [ ] Ensemble outperforms or matches best individual model
- [ ] HFO screening runs with GNN predictions (no errors)
- [ ] `model/saved/test_set.csv` exists and is used by both RF and GNN evaluation
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] `ruff check .` passes
- [ ] Report documents all metrics and comparison tables

## When to Move On

- GNN R² on unified test >= 0.73 for all targets
- Ensemble does not degrade below best individual model
- HFO screening produces ranked candidates using GNN
- Model cards updated with new metrics

## Budget

45 minutes (includes GNN training time). If GPU is unavailable, reduce epochs to 50 or use a smaller hidden dimension (128) as proof-of-concept.
