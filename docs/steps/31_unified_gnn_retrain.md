# Step 31: Unified Dataset and GNN Retraining

## Objective

Retrain the GNN on the full unified dataset (Esper + ML-SAFT + SPT-PCSAFT, ~13,764 molecules) using a shared persisted train/test split with RF, then evaluate it on both the overall corpus and the fluorinated screening regime that matters for the product. This resolves the dataset mismatch that caused GNN to underperform on Esper-like molecules and the ensemble to fail. Switch the HFO screening pipeline from RF to GNN only if the retrained model passes both the shared benchmark and the target-domain checks below.

## Motivation

Three interconnected problems were identified in the current pipeline:

1. **`load_data("all")` was broken**: It fell through to a fallback CSV (~1,800 molecules) instead of loading the full corpus. This has been fixed in `model/data/load.py` (Phase A).
2. **Dataset mismatch**: RF was trained on Esper (1,801 molecules), GNN on a different pool. The shared test set is Esper-only, putting GNN at a disadvantage. GNN reports R² = 0.73-0.77 on its own test split but degrades on the Esper test set.
3. **Ensemble failure**: Inverse-variance weighting gave GNN 82% weight despite GNN being out-of-distribution on Esper. This caused the ensemble to underperform RF (Report 28, Step 02c).

After retraining on a unified dataset that includes Esper, GNN should perform well on Esper-like molecules, the ensemble should work correctly, and the screening pipeline can use the best model. However, because the hosted portal will be used on fluorinated and partially out-of-domain chemistry, promotion to the default model should not rely on a single IID-style random split alone.

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
4. The saved test set retains enough metadata to support slice analysis, ideally including `source`, `inchi`, and any available fluorination/class labels in addition to `smiles`, `m`, `sigma`, `epsilon_k`

The saved test set must contain at minimum: `smiles`, `m`, `sigma`, `epsilon_k`. If `source` is present in the unified dataframe, persist it as well so the report can break out Esper / ML-SAFT / SPT performance rather than reporting only pooled metrics.

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

Also record at least two product-relevant slices, even if sample sizes are modest:

1. **Fluorinated / refrigerant-like subset**: molecules containing F, or a curated HFO/HCFO-like subset if labels already exist
2. **Source-aware subset**: metrics by `source` (Esper, ML-SAFT, SPT-PCSAFT) if the metadata survives deduplication

If the slice counts are too small for stable R², report MAE/RMSE plus sample counts and state that explicitly.

### 31.5 Add a chemistry-aware robustness check

Add one non-IID validation view before promoting the model to production default:

1. **Scaffold or grouped split**: group by Bemis-Murcko scaffold, InChI skeleton, or another chemistry-aware grouping so close analogs are not split between train and test
2. **Fluorinated holdout**: if a curated fluorinated subset exists, hold out that subset and compare RF vs GNN specifically on it
3. **Uncertainty sanity check**: compare error vs uncertainty on the held-out slice to confirm the GNN is not confidently wrong on fluorinated/OOD molecules

This does not need to replace the primary shared test set. It is an additional gate to answer the product question: "is the model trustworthy enough to serve by default on the chemistry the portal is likely to see?"

### 31.6 Re-calibrate ensemble

Test two ensemble strategies:

1. **Equal-weighted**: `ŷ = 0.5 * ŷ_RF + 0.5 * ŷ_GNN`
2. **Inverse-variance**: `ŷ = (w_RF * ŷ_RF + w_GNN * ŷ_GNN) / (w_RF + w_GNN)` where `w_i = 1/σ²_i`

With GNN now trained on a superset that includes Esper, its uncertainty estimates should be more calibrated for Esper-like molecules. The inverse-variance ensemble should now work correctly, but only if uncertainty tracks actual error better than before.

If inverse-variance still underperforms, fall back to equal weighting and document why. Do not assume inverse-variance weighting is production-ready merely because the training corpus is larger.

### 31.7 Decide whether to switch HFO screening to GNN

Update `screening/hfo_screening.py` only if the checks above support promotion:

1. Replace calls to `model.predict.predict_pcsaft()` (RF-only) with calls through the model registry: `model.registry.get_model("gnn")`
2. If the screening function accepts a `model_type` parameter, add one and default to `"gnn"`
3. Update `validate_boiling_points_rf()` to also support GNN (rename to `validate_boiling_points()` with a `model_type` parameter)

If the unified GNN wins on pooled metrics but fails the fluorinated/robustness checks, keep RF as the screening default for now and document the reason in the report and model cards.

### 31.8 Update model cards, reports, and `state.yaml`

- Update `docs/model_cards/gnn.md` with new metrics (training data size, R² on unified test set)
- Update `docs/model_cards/ensemble.md` with new ensemble performance
- Reconcile any now-stale claims in release-facing docs and summaries that currently describe the "all data" GNN as already validated or production-ready
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
- [ ] GNN R² >= 0.73 on the unified pooled test set for all three parameters
- [ ] Report includes fluorinated/product-relevant slice metrics with sample counts
- [ ] At least one chemistry-aware robustness check (scaffold/grouped split or fluorinated holdout) is reported
- [ ] Uncertainty/error behavior is checked on the held-out slice so overconfident failure modes are visible
- [ ] Ensemble outperforms or matches the best individual model, or is explicitly rejected with justification
- [ ] HFO screening switches to GNN only if the target-domain checks support it
- [ ] `model/saved/test_set.csv` exists and is used by both RF and GNN evaluation
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] `ruff check .` passes
- [ ] Report documents pooled metrics, slice metrics, and the model-promotion decision

## When to Move On

- GNN meets the pooled benchmark and does not show a fluorinated/OOD failure mode large enough to block product use
- The promotion decision (`gnn` default vs keep `rf` default) is explicit and justified in the report
- Ensemble behavior is calibrated well enough to use, or explicitly deferred
- Model cards and release-facing docs are updated so portal/API messaging matches reality

## Budget

45 minutes (includes GNN training time). If GPU is unavailable, reduce epochs to 50 or use a smaller hidden dimension (128) as proof-of-concept.
