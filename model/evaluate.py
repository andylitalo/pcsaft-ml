"""Evaluate trained PC-SAFT prediction models.

Usage:
    python -m model.evaluate
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.data.descriptors import build_features_with_names
from model.data.load import TARGETS

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent / "saved"


def _load_feature_config() -> dict:
    """Load feature configuration from saved artifacts.

    Falls back to rdkit-only mode for backward compatibility with models
    trained before feature_config.json was introduced.
    """
    config_path = SAVED_DIR / "feature_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    # Backward compatibility: assume rdkit-only
    logger.info("No feature_config.json found; assuming rdkit-only features")
    return {
        "features": "rdkit",
        "use_morgan": False,
        "use_rdkit": True,
        "morgan_radius": 2,
        "morgan_bits": 2048,
    }


def _get_rdkit_names_from_saved() -> list[str] | None:
    """Extract the RDKit descriptor column names from saved feature_names.

    Returns the non-Morgan feature names (i.e., RDKit descriptor names)
    from the saved feature_names artifact, or None if not available.
    """
    fnames_path = SAVED_DIR / "feature_names.joblib"
    if not fnames_path.exists():
        return None
    feature_names = joblib.load(fnames_path)
    rdkit_names = [n for n in feature_names if not n.startswith("morgan_")]
    return rdkit_names if rdkit_names else None


def evaluate():
    """Load saved models, evaluate on held-out test set, print metrics and save parity plots."""
    config = _load_feature_config()
    use_morgan = config.get("use_morgan", False)
    use_rdkit = config.get("use_rdkit", True)

    # Load test set
    test_df = pd.read_csv(SAVED_DIR / "test_set.csv")

    print(f"Evaluating on {len(test_df)} test molecules")
    print(f"Feature mode: {config.get('features', 'unknown')}\n")

    # Get saved RDKit descriptor names for consistent feature computation
    rdkit_names = _get_rdkit_names_from_saved() if use_rdkit else None

    # Compute features using the same config as training
    X_test, _ = build_features_with_names(
        test_df["smiles"].tolist(),
        use_morgan=use_morgan,
        use_rdkit=use_rdkit,
        morgan_radius=config.get("morgan_radius", 2),
        morgan_bits=config.get("morgan_bits", 2048),
        rdkit_names=rdkit_names,
    )

    # Handle any non-finite values
    valid_mask = np.isfinite(X_test).all(axis=1)
    X_test = X_test[valid_mask]
    test_df = test_df.iloc[valid_mask]

    if len(X_test) == 0:
        print("No valid test molecules after feature computation.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    units = {"m": "segments", "sigma": "\u00c5", "epsilon_k": "K"}

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
        print(f"  R\u00b2   = {r2:.4f}")
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
        ax.set_title(f"{target}: R\u00b2={r2:.3f}, MAE={mae:.3f}")
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plot_path = SAVED_DIR / "parity_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Parity plots saved to {plot_path}")


if __name__ == "__main__":
    evaluate()
