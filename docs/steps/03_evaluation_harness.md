# Step 3: Evaluation Harness

## Purpose

Build a unified evaluation framework that puts all models (Random Forest, PyTorch NN, and later ChemBERTa) on an equal footing. The current `model/evaluate.py` only evaluates the RF. This step creates a standardized comparison pipeline with publication-quality plots and a structured metrics output.

Having a solid evaluation harness matters because the whole narrative of this project is "I upgraded from RF to NN to Transformer and here's exactly how each step improved predictions." Without rigorous, apples-to-apples comparison, that story falls apart.

## What It Adds Over Step 2

| After Step 2 | After This Step |
|--------------|----------------|
| Separate evaluation code per model type | Single `evaluate.py` that handles any registered model |
| Only parity plots | Parity plots + learning curves + residual distributions + metrics table |
| Metrics printed to stdout | Structured JSON/CSV output for all models |
| No visualization of training dynamics | Training loss and validation loss curves for NN models |

## Skills Demonstrated

- **Experiment reproducibility**: Fixed seeds, identical splits, saved metrics
- **scikit-learn metrics API**: MAE, RMSE, R², applied uniformly
- **matplotlib**: Multi-panel publication-quality figures
- **Software engineering**: Registry pattern for pluggable model evaluation

## Implementation Guide

### 1. Model registry pattern

Create `model/registry.py` that maps model names to load + predict functions:

```python
MODELS = {}

def register_model(name):
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator

@register_model("rf")
class RFModel:
    def load(self, path):
        ...
    def predict(self, X) -> dict[str, np.ndarray]:
        return {"m": ..., "sigma": ..., "epsilon_k": ...}

@register_model("nn")
class NNModel:
    def load(self, path):
        ...
    def predict(self, X) -> dict[str, np.ndarray]:
        ...
```

### 2. Unified evaluation script (`model/evaluate.py` rewrite)

```bash
python -m model.evaluate --models rf nn --data esper
```

For each registered model:
1. Load saved artifacts
2. Compute predictions on the held-out test set
3. Calculate MAE, RMSE, R² per target
4. Store results in a structured dict

### 3. Comparison metrics table

Output a CSV and print a formatted table:

```
model/saved/comparison_metrics.csv

model,  target,    mae,    rmse,   r2
rf,     m,         0.312,  0.421,  0.923
rf,     sigma,     0.147,  0.198,  0.856
rf,     epsilon_k, 14.23,  19.87,  0.801
nn,     m,         0.287,  0.389,  0.934
nn,     sigma,     0.132,  0.181,  0.879
nn,     epsilon_k, 12.56,  17.43,  0.847
```

### 4. Parity plots (multi-model)

One row per model, one column per target (3 columns). All panels share the same axis limits for each target so visual comparison is immediate:

```
         m              σ              ε/k
RF    [parity]      [parity]       [parity]
NN    [parity]      [parity]       [parity]
```

Save as `figures/parity_comparison.png`.

### 5. Residual distribution plots

For each model and target, plot a histogram of (y_pred - y_true). Useful for spotting systematic bias (e.g., NN consistently underpredicts ε/k for large molecules):

```python
fig, axes = plt.subplots(n_models, 3)
for i, model_name in enumerate(models):
    for j, target in enumerate(TARGETS):
        residuals = preds[target] - y_true[target]
        axes[i, j].hist(residuals, bins=30, edgecolor="k")
        axes[i, j].axvline(0, color="r", linestyle="--")
```

### 6. Learning curves (NN only)

Plot training loss and validation loss vs. epoch. Save the loss history during training (Step 2) as a JSON file:

```python
{
    "train_loss": [0.45, 0.32, 0.28, ...],
    "val_loss": [0.48, 0.35, 0.30, ...],
    "lr": [1e-3, 1e-3, 5e-4, ...]
}
```

Then plot:

```python
fig, ax = plt.subplots()
ax.plot(history["train_loss"], label="Train")
ax.plot(history["val_loss"], label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
```

Save as `figures/nn_learning_curves.png`.

### 7. Per-molecule error analysis (optional but impressive)

Identify the 10 worst-predicted molecules for each model. Check if they share structural features (e.g., "NN struggles with long-chain alcohols"). This becomes talking-point material in an interview.

## Evaluation & Success Criteria

### What success looks like

- Running `python -m model.evaluate --models rf nn` produces:
  - A printed comparison table
  - `model/saved/comparison_metrics.csv`
  - `figures/parity_comparison.png`
  - `figures/residual_distributions.png`
  - `figures/nn_learning_curves.png` (if NN history exists)
- The figures are clean, labeled, and suitable for a technical presentation
- The metrics CSV has a consistent schema that Step 4 (ChemBERTa) can append to

### Quality bar for figures

- Axis labels include units (e.g., "Predicted ε/k (K)")
- Parity line is visible and correctly positioned
- Font size is readable at presentation scale (12pt+)
- Color coding distinguishes models clearly
- Legend is present and unambiguous

### When to move to Step 4

You're ready for Step 4 when:

1. You have a clear, quantified comparison of RF vs. NN
2. The evaluation harness can accept a new model with minimal code (just register it)
3. You can articulate the performance difference and hypothesize why (e.g., "the NN captures nonlinear interactions between Morgan FP bits that the RF misses")
4. Your figures tell the story visually
