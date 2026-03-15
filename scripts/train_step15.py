"""Step 15: Train XGBoost and chemprop D-MPNN, compare to baseline.

Usage:
    python scripts/train_step15.py [--source esper] [--chemprop-epochs 20]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

import model.chemprop_model  # noqa: F401, E402
import model.xgb  # noqa: F401, E402
from model.data.descriptors import build_features_with_names  # noqa: E402
from model.data.load import TARGETS, load_data, split_data  # noqa: E402

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "15_improved_models"
SAVED_DIR = Path(__file__).resolve().parent.parent / "model" / "saved"

TARGET_LABELS = {"m": "m (segments)", "sigma": "\u03c3 (\u00c5)", "epsilon_k": "\u03b5/k (K)"}


def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R2 for valid (non-NaN) entries."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
    }


def train_xgboost(X_train, y_train, feature_names):
    """Train XGBoost with GridSearchCV."""
    from model.xgb.xgb_model import XGBoostPCSAFT

    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    xgb_model = XGBoostPCSAFT()
    xgb_model.fit(X_train, y_train, feature_names=feature_names, cv_folds=3)

    print(f"Best params: {xgb_model.best_params_}")

    # Save
    save_path = xgb_model.save()
    print(f"Model saved to {save_path}")
    return xgb_model


def train_chemprop(smiles_train, y_train, smiles_val, y_val, max_epochs=20):
    """Train chemprop D-MPNN."""
    from model.chemprop_model.chemprop_wrapper import ChempropPCSAFT

    print("\n" + "=" * 60)
    print("TRAINING CHEMPROP D-MPNN")
    print("=" * 60)

    cp_model = ChempropPCSAFT()
    cp_model.fit(
        smiles_train,
        y_train,
        smiles_val=smiles_val,
        y_val=y_val,
        max_epochs=max_epochs,
        batch_size=64,
        patience=5,
        hidden_dim=300,
        depth=3,
    )

    save_path = cp_model.save()
    print(f"Model saved to {save_path}")
    return cp_model


def evaluate_model(model_name, preds, y_test):
    """Evaluate predictions against test targets.

    Parameters
    ----------
    model_name : str
    preds : np.ndarray of shape (n, 3) or dict
    y_test : np.ndarray of shape (n, 3)

    Returns
    -------
    dict mapping target -> metrics dict
    """
    results = {}
    for i, target in enumerate(TARGETS):
        if isinstance(preds, dict):
            y_pred = preds[target]
        else:
            y_pred = preds[:, i]
        metrics = compute_metrics(y_test[:, i], y_pred)
        results[target] = metrics
    return results


def load_baseline_metrics():
    """Load RF and GNN baseline metrics from saved files if available."""
    baselines = {}

    # RF metrics from saved comparison
    comparison_path = SAVED_DIR / "comparison_metrics.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        for model_name in ["rf", "gc_pcsaft", "nn", "chemberta", "gnn"]:
            sub = df[df["model"] == model_name]
            if len(sub) == 0:
                continue
            baselines[model_name] = {}
            for _, row in sub.iterrows():
                baselines[model_name][row["target"]] = {
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "r2": row["r2"],
                }

    return baselines


def plot_r2_heatmap(all_metrics, save_dir):
    """Create R2 heatmap (models x targets)."""
    model_names = list(all_metrics.keys())
    targets = TARGETS

    data = np.zeros((len(model_names), len(targets)))
    for i, mname in enumerate(model_names):
        for j, target in enumerate(targets):
            data[i, j] = all_metrics[mname].get(target, {}).get("r2", np.nan)

    fig, ax = plt.subplots(figsize=(8, max(4, len(model_names) * 0.6 + 1)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        xticklabels=[TARGET_LABELS[t] for t in targets],
        yticklabels=model_names,
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("R\u00b2 by Model and Target", fontsize=14)
    fig.tight_layout()
    path = save_dir / "r2_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_parity_best_model(y_true, y_pred, model_name, save_dir):
    """Parity plots for the best model on all three targets."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, target in enumerate(TARGETS):
        ax = axes[i]
        mask = np.isfinite(y_true[:, i]) & np.isfinite(y_pred[:, i])
        if not mask.any():
            continue
        yt = y_true[mask, i]
        yp = y_pred[mask, i]

        ax.scatter(yt, yp, alpha=0.5, s=25, edgecolors="k", linewidths=0.3, c="#1f77b4")

        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", lw=1, alpha=0.5)

        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        label = TARGET_LABELS[target]
        ax.set_xlabel(f"True {label}", fontsize=12)
        ax.set_ylabel(f"Predicted {label}", fontsize=12)
        ax.set_title(f"{model_name} | {label}\nR\u00b2={r2:.3f}, MAE={mae:.3f}", fontsize=12)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(f"Parity Plots: {model_name} (Best Model)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_dir / "parity_best_model.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_xgb_feature_importance(xgb_model, save_dir, top_n=20):
    """Top N feature importances (averaged across targets)."""
    importances = xgb_model.feature_importances()
    names = xgb_model.feature_names_ or [f"f{i}" for i in range(len(importances))]

    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_names)), top_vals[::-1], color="#2ca02c", edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Mean Feature Importance (across targets)", fontsize=12)
    ax.set_title(f"Top {top_n} XGBoost Feature Importances", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = save_dir / "xgb_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_radar_ccc(all_metrics, save_dir):
    """Radar chart showing CCC (or R2 as proxy) for each model-target combo."""
    model_names = list(all_metrics.keys())
    targets = TARGETS
    labels = [f"{TARGET_LABELS[t]}" for t in targets]

    angles = np.linspace(0, 2 * np.pi, len(targets), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(model_names)))

    for mi, mname in enumerate(model_names):
        values = [all_metrics[mname].get(t, {}).get("r2", 0.0) for t in targets]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=mname, color=colors[mi])
        ax.fill(angles, values, alpha=0.1, color=colors[mi])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(-0.2, 1.0)
    ax.set_title("Model Comparison: R\u00b2 by Target", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    path = save_dir / "radar_ccc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Step 15: Train XGBoost and chemprop")
    parser.add_argument("--source", default="esper", help="Data source")
    parser.add_argument("--chemprop-epochs", type=int, default=20, help="chemprop max epochs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # ------------------------------------------------------------------
    # 1. Load and split data
    # ------------------------------------------------------------------
    print("Loading data...")
    df = load_data(source=args.source)
    train_df, test_df = split_data(df)
    print(f"Dataset: {len(df)} total, {len(train_df)} train, {len(test_df)} test")

    smiles_train = train_df["smiles"].tolist()
    smiles_test = test_df["smiles"].tolist()
    y_train = train_df[TARGETS].values.astype(np.float32)
    y_test = test_df[TARGETS].values.astype(np.float32)

    # Build features for XGBoost (same features as RF)
    print("Building features...")
    X_train, feature_names = build_features_with_names(smiles_train)
    X_test, _ = build_features_with_names(
        smiles_test,
        rdkit_names=[n for n in feature_names if not n.startswith("morgan_")],
    )

    # Filter NaN rows
    train_valid = np.isfinite(X_train).all(axis=1)
    test_valid = np.isfinite(X_test).all(axis=1)
    print(f"Valid features: {train_valid.sum()}/{len(X_train)} train, "
          f"{test_valid.sum()}/{len(X_test)} test")

    X_train_clean = X_train[train_valid]
    y_train_clean = y_train[train_valid]
    X_test_clean = X_test[test_valid]
    y_test_clean = y_test[test_valid]

    # ------------------------------------------------------------------
    # 2. Train XGBoost
    # ------------------------------------------------------------------
    xgb_model = train_xgboost(X_train_clean, y_train_clean, feature_names)

    # Evaluate XGBoost
    xgb_preds = xgb_model.predict(X_test_clean)
    xgb_metrics = evaluate_model("xgboost", xgb_preds, y_test_clean)

    print("\nXGBoost Test Results:")
    for target, m in xgb_metrics.items():
        print(f"  {target:12s}: R2={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

    # ------------------------------------------------------------------
    # 3. Train chemprop
    # ------------------------------------------------------------------
    # Use a validation split for early stopping
    n_val = max(10, int(len(smiles_train) * 0.1))
    smiles_val = smiles_train[-n_val:]
    y_val = y_train[-n_val:]
    smiles_train_cp = smiles_train[:-n_val]
    y_train_cp = y_train[:-n_val]

    cp_model = train_chemprop(
        smiles_train_cp, y_train_cp,
        smiles_val, y_val,
        max_epochs=args.chemprop_epochs,
    )

    # Evaluate chemprop
    cp_preds = cp_model.predict(smiles_test)
    cp_metrics = evaluate_model("chemprop", cp_preds, y_test)

    print("\nchemprop D-MPNN Test Results:")
    for target, m in cp_metrics.items():
        print(f"  {target:12s}: R2={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

    # ------------------------------------------------------------------
    # 4. Collect all metrics (including baselines)
    # ------------------------------------------------------------------
    all_metrics = load_baseline_metrics()
    all_metrics["xgboost"] = xgb_metrics
    all_metrics["chemprop"] = cp_metrics

    # Print full comparison
    print("\n" + "=" * 80)
    print("FULL MODEL COMPARISON")
    print("=" * 80)
    for model_name, metrics in all_metrics.items():
        print(f"\n  {model_name}:")
        for target in TARGETS:
            m = metrics.get(target, {})
            print(f"    {target:12s}: R2={m.get('r2', float('nan')):.4f}  "
                  f"MAE={m.get('mae', float('nan')):.4f}  "
                  f"RMSE={m.get('rmse', float('nan')):.4f}")

    # ------------------------------------------------------------------
    # 5. Promotion decision
    # ------------------------------------------------------------------
    rf_ek_r2 = all_metrics.get("rf", {}).get("epsilon_k", {}).get("r2", 0.0)
    cp_ek_r2 = cp_metrics.get("epsilon_k", {}).get("r2", 0.0)
    xgb_ek_r2 = xgb_metrics.get("epsilon_k", {}).get("r2", 0.0)

    print("\n" + "=" * 60)
    print("PROMOTION DECISION")
    print("=" * 60)
    print(f"  RF  epsilon_k R2:       {rf_ek_r2:.4f}")
    print(f"  XGBoost epsilon_k R2:   {xgb_ek_r2:.4f} (delta: {xgb_ek_r2 - rf_ek_r2:+.4f})")
    print(f"  chemprop epsilon_k R2:  {cp_ek_r2:.4f} (delta: {cp_ek_r2 - rf_ek_r2:+.4f})")

    # Find the best model
    all_ek_r2 = {name: metrics.get("epsilon_k", {}).get("r2", -999)
                 for name, metrics in all_metrics.items()}
    best_model = max(all_ek_r2, key=all_ek_r2.get)
    best_r2 = all_ek_r2[best_model]

    promoted = False
    if cp_ek_r2 - rf_ek_r2 >= 0.05:
        print(f"\n  >> chemprop beats RF by {cp_ek_r2 - rf_ek_r2:.4f} >= 0.05")
        print("  >> PROMOTE chemprop as new default")
        promoted = True
    else:
        print(f"\n  >> chemprop does NOT beat RF by >= 0.05 (delta={cp_ek_r2 - rf_ek_r2:.4f})")
        print(f"  >> KEEP RF as default. Best overall model: {best_model} (R2={best_r2:.4f})")

    # ------------------------------------------------------------------
    # 6. Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_r2_heatmap(all_metrics, FIGURES_DIR)

    # Find best model overall for parity plot
    # Best = highest average R2 across targets among newly trained models
    new_models = {"xgboost": (xgb_preds, y_test_clean), "chemprop": (cp_preds, y_test)}
    avg_r2 = {}
    for name, metrics in [("xgboost", xgb_metrics), ("chemprop", cp_metrics)]:
        r2_vals = [metrics[t]["r2"] for t in TARGETS if not np.isnan(metrics[t]["r2"])]
        avg_r2[name] = np.mean(r2_vals)

    best_new = max(avg_r2, key=avg_r2.get)
    preds_best, yt_best = new_models[best_new]
    if isinstance(preds_best, dict):
        preds_arr = np.column_stack([preds_best[t] for t in TARGETS])
    else:
        preds_arr = preds_best
    plot_parity_best_model(yt_best, preds_arr, best_new, FIGURES_DIR)

    plot_xgb_feature_importance(xgb_model, FIGURES_DIR)
    plot_radar_ccc(all_metrics, FIGURES_DIR)

    # ------------------------------------------------------------------
    # 7. Save metrics summary
    # ------------------------------------------------------------------
    summary = {
        "all_metrics": {
            name: {t: m for t, m in metrics.items()} for name, metrics in all_metrics.items()
        },
        "xgb_best_params": xgb_model.best_params_,
        "promotion_decision": "chemprop" if promoted else "rf (no change)",
        "best_overall_model": best_model,
    }
    summary_path = SAVED_DIR / "step15_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved to {summary_path}")

    print("\nStep 15 complete!")


if __name__ == "__main__":
    main()
