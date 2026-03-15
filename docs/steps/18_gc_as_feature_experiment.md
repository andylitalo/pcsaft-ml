# Step 18: GC-PC-SAFT as Input Feature Experiment

## Purpose

GC-PC-SAFT predictions are poor on their own (R² = −1.16 on σ, −0.04 on ε/k), but they encode physically motivated group-additivity information that molecular fingerprints miss. This step tests whether using GC-PC-SAFT predicted values as additional input features to the RF/NN improves performance — bootstrapping physical priors into the ML model.

This is a low-cost experiment that tests the principle: even a poor model's predictions can be informative features if they capture variance that the other features don't.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| RF uses Morgan FP + RDKit descriptors only | RF uses Morgan FP + RDKit + GC-PC-SAFT predictions (3 extra features) |
| GC-PC-SAFT evaluated only as a standalone baseline | GC-PC-SAFT used as a feature, potentially improving ML performance |

## Skills Demonstrated

- **Feature engineering**: Using domain-knowledge-based predictions as ML features
- **Ablation study**: Controlled comparison with/without GC features
- **Scientific reasoning**: Connecting physics-based and data-driven approaches

## Dependencies

- Requires: Steps 01 (feature pipeline), 03 (evaluation harness)
- Requires: GC-PC-SAFT prediction function in `model/registry.py`

## Implementation Guide

### 18.1 Add GC Features to Feature Pipeline

In `model/data/descriptors.py`, add a `use_gc_pcsaft` flag to `build_features()`:

```python
def build_features(smiles_list, ..., use_gc_pcsaft=False):
    # ... existing Morgan FP + RDKit code ...
    if use_gc_pcsaft:
        gc_model = get_model("gc_pcsaft")
        gc_preds = gc_model.predict(smiles_list)
        gc_features = gc_preds[["m", "sigma", "epsilon_k"]].values
        features = np.hstack([features, gc_features])
    return features
```

### 18.2 Train RF with GC Features

Train the same RF configuration but with the 3 additional GC-PC-SAFT features.

### 18.3 Ablation Table

| Feature set | R² (m) | R² (σ) | R² (ε/k) |
|-------------|--------|--------|----------|
| Morgan + RDKit (baseline) | 0.62 | 0.35 | 0.33 |
| Morgan + RDKit + GC-PC-SAFT | ? | ? | ? |
| GC-PC-SAFT only (reference) | 0.40 | −1.16 | −0.04 |

If the +GC row shows improvement, the GC predictions add information that fingerprints alone don't capture.

## Acceptance Criteria

- [ ] `build_features()` supports `use_gc_pcsaft=True`
- [ ] RF retrained with GC features; metrics compared to baseline
- [ ] Ablation table in report
- [ ] Report at `docs/reports/18_gc_as_feature_experiment.md`
