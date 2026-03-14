"""Training loop for the PCSAFTNet multi-task neural network."""

import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from model.nn.architecture import TARGETS, PCSAFTNet
from model.nn.dataset import PCSAFTDataset

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved"


def train_nn(
    X_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    feature_config: dict,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    trunk_dims: list[int] | None = None,
    head_dims: list[int] | None = None,
    dropout: float = 0.2,
    random_state: int = 42,
) -> PCSAFTNet:
    """Train a multi-task PCSAFTNet on the given features and targets.

    Normalizes targets to zero mean / unit variance, trains with AdamW +
    ReduceLROnPlateau + early stopping, and saves all artifacts to
    ``model/saved/``.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape ``(n_train, n_features)``.
    y_train : dict[str, np.ndarray]
        Mapping from target names to 1D target arrays.
    feature_config : dict
        Feature configuration dict (saved alongside the model).
    lr : float
        Initial learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience (epochs without improvement).
    batch_size : int
        Training batch size.
    val_fraction : float
        Fraction of training data to use for validation.
    trunk_dims : list[int] | None
        Hidden dimensions for shared trunk.
    head_dims : list[int] | None
        Hidden dimensions for task heads.
    dropout : float
        Dropout probability.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    PCSAFTNet
        Trained model (loaded with best weights).
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # --- Feature normalization ---
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X_train).astype(np.float32)

    # --- Target normalization ---
    target_scalers: dict[str, StandardScaler] = {}
    y_scaled: dict[str, np.ndarray] = {}
    for target in TARGETS:
        scaler = StandardScaler()
        y_scaled[target] = scaler.fit_transform(
            y_train[target].reshape(-1, 1)
        ).ravel().astype(np.float32)
        target_scalers[target] = scaler

    # --- Train / val split (within the training data) ---
    idx = np.arange(len(X_scaled))
    idx_fit, idx_val = train_test_split(
        idx, test_size=val_fraction, random_state=random_state
    )

    X_fit = X_scaled[idx_fit]
    X_val = X_scaled[idx_val]
    y_fit = {t: y_scaled[t][idx_fit] for t in TARGETS}
    y_val = {t: y_scaled[t][idx_val] for t in TARGETS}

    train_ds = PCSAFTDataset(X_fit, y_fit)
    val_ds = PCSAFTDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    logger.info(
        "Train/val split: %d fit, %d val (%.0f%% val)",
        len(idx_fit),
        len(idx_val),
        val_fraction * 100,
    )

    # --- Model ---
    input_dim = X_scaled.shape[1]
    model = PCSAFTNet(
        input_dim=input_dim,
        trunk_dims=trunk_dims,
        head_dims=head_dims,
        dropout=dropout,
    )
    logger.info("Model: %s", model)

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None
    best_epoch = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, max_epochs + 1):
        # -- Train --
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = sum(
                nn.functional.mse_loss(preds[t], y_batch[t]) for t in TARGETS
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)

        # -- Validate --
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                preds = model(x_batch)
                loss = sum(
                    nn.functional.mse_loss(preds[t], y_batch[t]) for t in TARGETS
                )
                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / max(n_val_batches, 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
                epoch,
                avg_train_loss,
                avg_val_loss,
                current_lr,
            )

        # -- Early stopping --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d (best=%d, val_loss=%.4f)",
                    epoch,
                    best_epoch,
                    best_val_loss,
                )
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    final_lr = optimizer.param_groups[0]["lr"]
    logger.info(
        "Training complete. Best epoch: %d, best val loss: %.4f",
        best_epoch,
        best_val_loss,
    )

    # --- Save artifacts ---
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    # Build target scaler info as plain dicts (serializable)
    target_scaler_info = {}
    for target in TARGETS:
        sc = target_scalers[target]
        target_scaler_info[target] = {
            "mean": float(sc.mean_[0]),
            "std": float(sc.scale_[0]),
        }

    # Build feature scaler info
    feature_scaler_info = {
        "mean": feature_scaler.mean_.tolist(),
        "std": feature_scaler.scale_.tolist(),
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "target_scalers": target_scaler_info,
        "feature_scaler": feature_scaler_info,
        "feature_config": feature_config,
    }
    model_path = SAVED_DIR / "nn_pcsaft.pt"
    torch.save(checkpoint, model_path)
    logger.info("Saved model checkpoint to %s", model_path)

    # Save training history
    history = {
        "epochs": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_epoch": best_epoch,
        "final_lr": final_lr,
    }
    history_path = SAVED_DIR / "nn_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    logger.info("Saved training history to %s", history_path)

    return model
