"""Training script for the GNN PC-SAFT parameter predictor.

Usage:
    python -m model.gnn.train_gnn [--source combined] [--epochs 100] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

from model.data.load import TARGETS, load_data, split_data
from model.gnn.architecture import PCSAFTGraphNet
from model.gnn.graph_featurizer import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    MoleculeGraphDataset,
)

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved"


def _compute_target_scalers(
    train_targets: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-target mean/std for normalization."""
    scalers = {}
    for i, name in enumerate(TARGETS):
        col = train_targets[:, i]
        scalers[name] = {"mean": float(col.mean()), "std": float(col.std())}
    return scalers


def _normalize_targets(
    targets: np.ndarray,
    scalers: dict[str, dict[str, float]],
) -> np.ndarray:
    normed = np.zeros_like(targets)
    for i, name in enumerate(TARGETS):
        normed[:, i] = (targets[:, i] - scalers[name]["mean"]) / scalers[name]["std"]
    return normed


def train_gnn(
    source: str = "combined",
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    patience: int = 15,
    weight_decay: float = 1e-5,
) -> dict:
    """Train GNN on PC-SAFT data and save to model/saved/.

    Returns
    -------
    dict
        Metrics on the test set.
    """
    print(f"Loading data (source={source})...")
    df = load_data(source)

    # Use persistent test set if it exists and matches the corpus
    test_set_path = SAVED_DIR / "test_set.csv"
    if test_set_path.exists():
        existing_test = pd.read_csv(test_set_path)
        # Check if the test set is compatible (has source column for unified corpus)
        if "source" in existing_test.columns and "source" in df.columns:
            print(f"Using existing test set from {test_set_path}")
            test_smiles_set = set(existing_test["smiles"])
            train_df = df[~df["smiles"].isin(test_smiles_set)].reset_index(drop=True)
            test_df = df[df["smiles"].isin(test_smiles_set)].reset_index(drop=True)
            print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        else:
            # Old test set from Esper only — regenerate
            print("Regenerating test set for unified corpus...")
            train_df, test_df = split_data(df)
            # Save the new test set with source metadata
            test_df.to_csv(test_set_path, index=False)
            print(f"Saved unified test set to {test_set_path}")
            print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    else:
        # No test set exists — create one
        print("Creating new test set...")
        train_df, test_df = split_data(df)
        # Save test set with all available metadata
        test_df.to_csv(test_set_path, index=False)
        print(f"Saved test set to {test_set_path}")
        print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()
    train_targets = train_df[TARGETS].values.astype(np.float32)
    test_targets = test_df[TARGETS].values.astype(np.float32)

    # Normalize targets
    target_scalers = _compute_target_scalers(train_targets)
    train_targets_norm = _normalize_targets(train_targets, target_scalers)
    test_targets_norm = _normalize_targets(test_targets, target_scalers)

    print("Building molecular graphs...")
    train_dataset = MoleculeGraphDataset(train_smiles, train_targets_norm)
    test_dataset = MoleculeGraphDataset(test_smiles, test_targets_norm)
    print(f"Graphs built: {len(train_dataset)} train, {len(test_dataset)} test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PCSAFTGraphNet(
        node_dim=ATOM_FEATURE_DIM,
        edge_dim=BOND_FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_targets=len(TARGETS),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # --- Eval ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                val_loss += criterion(pred, batch.y).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"lr={current_lr:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # --- Final evaluation ---
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.append(pred.cpu().numpy())
            all_true.append(batch.y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    # Denormalize
    for i, name in enumerate(TARGETS):
        all_preds[:, i] = (
            all_preds[:, i] * target_scalers[name]["std"] + target_scalers[name]["mean"]
        )
        all_true[:, i] = (
            all_true[:, i] * target_scalers[name]["std"] + target_scalers[name]["mean"]
        )

    metrics = {}
    print("\n=== GNN Test Results ===")
    for i, name in enumerate(TARGETS):
        r2 = r2_score(all_true[:, i], all_preds[:, i])
        mae = mean_absolute_error(all_true[:, i], all_preds[:, i])
        rmse = np.sqrt(mean_squared_error(all_true[:, i], all_preds[:, i]))
        metrics[name] = {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}
        print(f"  {name:12s}: R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # --- Save ---
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "node_dim": ATOM_FEATURE_DIM,
            "edge_dim": BOND_FEATURE_DIM,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "num_targets": len(TARGETS),
        },
        "target_scalers": target_scalers,
        "metrics": metrics,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "source": source,
    }
    save_path = SAVED_DIR / "gnn_pcsaft.pt"
    torch.save(checkpoint, save_path)
    print(f"\nModel saved to {save_path}")

    metrics_path = SAVED_DIR / "gnn_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train GNN for PC-SAFT")
    parser.add_argument(
        "--source",
        default="combined",
        choices=["esper", "combined", "all", "fallback", "spt_pcsaft"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train_gnn(
        source=args.source,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
