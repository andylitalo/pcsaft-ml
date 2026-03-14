"""Evaluate trained PC-SAFT prediction models.

Usage:
    python -m model.evaluate
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.data.descriptors import compute_descriptors
from model.data.load import TARGETS

SAVED_DIR = Path(__file__).parent / "saved"


def evaluate():
    """Load saved models, evaluate on held-out test set, print metrics and save parity plots."""
    # Load feature names and test set
    feature_names = joblib.load(SAVED_DIR / "feature_names.joblib")
    test_df = pd.read_csv(SAVED_DIR / "test_set.csv")

    print(f"Evaluating on {len(test_df)} test molecules\n")

    # Compute descriptors for test set
    X_test = compute_descriptors(test_df["smiles"].tolist())[feature_names]
    valid_mask = X_test.notna().all(axis=1)
    X_test = X_test[valid_mask]
    test_df = test_df.iloc[valid_mask.values]

    if len(X_test) == 0:
        print("No valid test molecules after descriptor computation.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    units = {"m": "segments", "sigma": "Å", "epsilon_k": "K"}

    for i, target in enumerate(TARGETS):
        model = joblib.load(SAVED_DIR / f"rf_{target}.joblib")
        y_true = test_df[target].values
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"{target}:")
        print(f"  MAE  = {mae:.4f} {units[target]}")
        print(f"  RMSE = {rmse:.4f} {units[target]}")
        print(f"  R²   = {r2:.4f}")
        print()

        # Parity plot
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5, s=40)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
        ax.set_xlabel(f"True {target} ({units[target]})")
        ax.set_ylabel(f"Predicted {target} ({units[target]})")
        ax.set_title(f"{target}: R²={r2:.3f}, MAE={mae:.3f}")
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plot_path = SAVED_DIR / "parity_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Parity plots saved to {plot_path}")


if __name__ == "__main__":
    evaluate()
