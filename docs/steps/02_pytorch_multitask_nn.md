# Step 2: PyTorch Multi-Task Neural Network

## Purpose

Replace the three independent Random Forest regressors with a single PyTorch neural network that predicts all three PC-SAFT parameters (m, σ, ε/k) simultaneously. The architecture uses a **shared trunk** that learns a common molecular representation, then branches into **task-specific heads** for each parameter.

This design exploits the physical correlation between PC-SAFT parameters: larger molecules tend to have higher segment counts (m) *and* different dispersion energies (ε/k). Sharing lower layers forces the network to learn features that are useful across all three targets, which acts as a regularizer -- especially important with only ~1,800 training samples.

## What It Adds Over Step 1

| After Step 1 | After This Step |
|--------------|----------------|
| Random Forest with combined features | PyTorch neural network with combined features |
| Three independent models | Single multi-task model (shared representation) |
| No gradient-based optimization | Full training loop with LR scheduling, early stopping |
| Limited capacity for feature interactions | Deep nonlinear feature interactions via hidden layers |
| No uncertainty estimation path | Architecture supports MC Dropout uncertainty (Step 5) |

## Skills Demonstrated

- **PyTorch `nn.Module`**: Custom architecture with shared and branching layers
- **`torch.utils.data.Dataset` / `DataLoader`**: Proper data loading with batching and shuffling
- **Training loop engineering**: Epoch loop, gradient clipping, loss logging
- **Learning rate scheduling**: `ReduceLROnPlateau` or cosine annealing with warm restarts
- **Early stopping**: Patience-based stopping on validation loss to prevent overfitting
- **Multi-task loss balancing**: Weighted sum of per-target losses to handle different scales

## Implementation Guide

### 1. Model architecture (`model/nn/architecture.py`)

```python
import torch
import torch.nn as nn

class PCSAFTNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        trunk_dims: list[int] = [512, 256, 128],
        head_dims: list[int] = [64],
        dropout: float = 0.2,
    ):
        super().__init__()

        # Shared trunk
        trunk_layers = []
        prev_dim = input_dim
        for dim in trunk_dims:
            trunk_layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Task-specific heads
        self.heads = nn.ModuleDict()
        for target in ["m", "sigma", "epsilon_k"]:
            head_layers = []
            hprev = prev_dim
            for hdim in head_dims:
                head_layers += [nn.Linear(hprev, hdim), nn.ReLU()]
                hprev = hdim
            head_layers.append(nn.Linear(hprev, 1))
            self.heads[target] = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.trunk(x)
        return {target: head(shared).squeeze(-1) for target, head in self.heads.items()}
```

### 2. Dataset class (`model/nn/dataset.py`)

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class PCSAFTDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: dict[str, np.ndarray]):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.y.items()}
```

### 3. Training loop (`model/nn/trainer.py`)

Key design decisions:

- **Loss**: Weighted sum of per-target MSE losses. Weight ε/k higher since it has the largest scale (~100–400 K vs. 1–5 for m).

```python
target_weights = {"m": 1.0, "sigma": 1.0, "epsilon_k": 0.01}
# 0.01 for epsilon_k because its MSE is ~10,000x larger due to scale
```

Alternatively, normalize targets to zero mean / unit variance before training and use equal weights. This is cleaner and recommended.

- **Optimizer**: AdamW with weight decay 1e-4
- **LR Schedule**: `ReduceLROnPlateau(patience=10, factor=0.5)` watching validation loss
- **Early stopping**: Stop if validation loss hasn't improved for 20 epochs
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

```python
def train_model(
    model, train_loader, val_loader,
    lr=1e-3, weight_decay=1e-4, max_epochs=200, patience=20,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = sum(
                nn.functional.mse_loss(preds[t], y_batch[t])
                for t in preds
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = sum(
                    nn.functional.mse_loss(preds[t], y_batch[t])
                    for t in preds
                )
                val_loss += loss.item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model
```

### 4. Data splitting strategy

Use the same `random_state=42` split as the RF baseline so results are directly comparable. Within the training set, carve out 10% as a validation set for early stopping:

```
Full data → 80% train / 20% test  (same split as RF)
            ↓
         train → 90% fit / 10% val  (for early stopping)
```

### 5. Target normalization

Normalize each target to zero mean and unit variance using training-set statistics. Save the `StandardScaler` so you can invert predictions back to physical units at evaluation time.

### 6. CLI integration

Update `model/train.py` to accept `--model {rf,nn}`:

```bash
python -m model.train --model nn --data esper
python -m model.train --model rf --data esper   # existing behavior
```

### 7. Model saving

Save `model.state_dict()` (not the full model) plus the feature scaler and target scalers:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {"input_dim": ..., "trunk_dims": ..., ...},
    "feature_scaler": feature_scaler,
    "target_scalers": target_scalers,
}, "model/saved/nn_pcsaft.pt")
```

## Science Improvements (Tier 1+2)

### 2A. Uncertainty Quantification

For a screening application, knowing confidence matters as much as the point prediction. Make uncertainty quantification a deliverable in this step, not just a future mention.

**Implementation**:
- **NN (MC Dropout)**: Add a `predict_with_uncertainty(X, n_forward=30)` method to PCSAFTNet that enables dropout at inference time, runs `n_forward` stochastic forward passes, and returns per-target mean and standard deviation. This is straightforward: call `model.train()` to keep dropout active, then `model.eval()` after.
- **RF (tree disagreement)**: Add `predict_with_uncertainty()` to the RF wrapper that computes variance across `model.estimators_` predictions. This is free -- scikit-learn already stores individual tree predictions.
- Save uncertainty estimates alongside point predictions in the model output dict: `{"m": ..., "sigma": ..., "epsilon_k": ..., "m_std": ..., "sigma_std": ..., "epsilon_k_std": ...}`

### 2B. Applicability Domain Check

Standard QSAR practice (Tropsha 2010, OECD guidelines) requires checking whether a test molecule is "inside" the training distribution. Without this, the model may produce confident but meaningless predictions for novel chemistries.

**Implementation**:
- Train an `IsolationForest` (or compute leverage scores) on the training set feature matrix (Morgan FP + RDKit descriptors from Step 01's `build_features`)
- Add `in_domain(X) -> np.ndarray[bool]` to the model interface
- Molecules flagged as out-of-domain get a warning column in predictions
- Save the AD model alongside the trained NN: `model/saved/ad_model.joblib`
- The AD check is feature-space based, so it works identically for RF and NN (both consume the same feature matrix)

## Evaluation & Success Criteria

### Metrics to compare

| Model | R² (m) | R² (σ) | R² (ε/k) | MAE (m) | MAE (σ) | MAE (ε/k) |
|-------|--------|--------|----------|---------|---------|-----------|
| GC-PC-SAFT | — | — | — | — | — | — |
| RF (RDKit only) | — | — | — | — | — | — |
| RF (combined) | — | — | — | — | — | — |
| NN (combined) | — | — | — | — | — | — |

### What success looks like

- The NN matches or exceeds the RF on at least 2 of 3 targets (ε/k is typically where NNs shine due to its more complex relationship with structure)
- Training converges smoothly: loss curves show monotonic decrease without instability
- Validation loss tracks training loss without large divergence (no severe overfitting)
- The model trains in under 5 minutes on CPU for the Esper dataset (~1,800 molecules)
- All model artifacts save and reload correctly
- `predict_with_uncertainty()` returns reasonable std values (not all zeros, not all huge)
- AD check correctly flags molecules structurally distant from training set (test with a few exotic SMILES)

### What to watch for

- **Overfitting**: If train R² is 0.99 but test R² is 0.70, increase dropout or reduce trunk width
- **Underfitting**: If both train and test R² are low, try wider layers or more epochs
- **Loss scale issues**: If one target dominates the loss, switch to normalized targets
- **UQ calibration**: MC Dropout std should be larger for out-of-domain molecules than in-domain ones

### When to move to Step 3

You're ready for Step 3 when:

1. The NN trains to convergence with stable loss curves
2. You have side-by-side metrics for RF vs. NN
3. The model saves and loads correctly for inference
4. You understand the performance gap (or improvement) and can articulate why
5. Both RF and NN have `predict_with_uncertainty()` methods
6. AD model is trained and saved alongside the NN artifacts
