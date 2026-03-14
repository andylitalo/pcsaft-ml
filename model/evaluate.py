"""Unified evaluation harness for PC-SAFT prediction models.

Evaluates any combination of registered models (gc_pcsaft, rf, nn) on the
held-out test set and produces:
  - A printed comparison table
  - model/saved/comparison_metrics.csv
  - Parity comparison plot (grid: models x targets)
  - Residual distribution plot (grid: models x targets)
  - NN learning curves (if nn_history.json exists)
  - Uncertainty calibration plot (for models with UQ)

Usage:
    python -m model.evaluate --models gc_pcsaft rf nn
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.data.descriptors import build_features
from model.data.load import TARGETS
from model.registry import get_model, list_models

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent / "saved"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "03_evaluation_harness"

UNITS = {"m": "segments", "sigma": "\u00c5", "epsilon_k": "K"}
TARGET_LABELS = {"m": "m", "sigma": "\u03c3", "epsilon_k": "\u03b5/k"}

# Color scheme for models
MODEL_COLORS = {
    "gc_pcsaft": "#1b9e77",
    "rf": "#d95f02",
    "nn": "#7570b3",
}


# ---------------------------------------------------------------------------
# Feature helpers (for AD analysis)
# ---------------------------------------------------------------------------

def _load_feature_config() -> dict:
    config_path = SAVED_DIR / "feature_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {
        "features": "rdkit",
        "use_morgan": False,
        "use_rdkit": True,
        "morgan_radius": 2,
        "morgan_bits": 2048,
    }


def _get_rdkit_names_from_saved() -> list[str] | None:
    fnames_path = SAVED_DIR / "feature_names.joblib"
    if not fnames_path.exists():
        return None
    feature_names = joblib.load(fnames_path)
    rdkit_names = [n for n in feature_names if not n.startswith("morgan_")]
    return rdkit_names if rdkit_names else None


def _compute_test_features(smiles_list: list[str]) -> np.ndarray:
    config = _load_feature_config()
    use_rdkit = config.get("use_rdkit", True)
    rdkit_names = _get_rdkit_names_from_saved() if use_rdkit else None
    return build_features(
        smiles_list,
        use_morgan=config.get("use_morgan", False),
        use_rdkit=use_rdkit,
        morgan_radius=config.get("morgan_radius", 2),
        morgan_bits=config.get("morgan_bits", 2048),
        rdkit_names=rdkit_names,
    )


# ---------------------------------------------------------------------------
# AD analysis
# ---------------------------------------------------------------------------

def _get_ad_labels(smiles_list: list[str]) -> np.ndarray | None:
    """Return boolean array (True=in-domain) or None if AD model unavailable."""
    ad_path = SAVED_DIR / "ad_model.joblib"
    if not ad_path.exists():
        return None
    try:
        ad_model = joblib.load(ad_path)
        X = _compute_test_features(smiles_list)
        valid = np.isfinite(X).all(axis=1)
        labels = np.full(len(smiles_list), False)
        if valid.any():
            preds = ad_model.predict(X[valid])
            labels[valid] = preds == 1
        return labels
    except Exception:
        logger.warning("AD model loading failed; skipping AD analysis")
        return None


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute MAE, RMSE, R2 on valid (non-NaN) pairs."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": int(mask.sum())}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "n": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _plot_parity(
    all_results: dict,
    test_df: pd.DataFrame,
    model_names: list[str],
    ad_labels: np.ndarray | None,
) -> None:
    """Grid: rows=models, cols=targets. Color by AD status."""
    n_models = len(model_names)
    fig, axes = plt.subplots(
        n_models, 3, figsize=(15, 4.5 * n_models), squeeze=False
    )

    for i, mname in enumerate(model_names):
        preds = all_results[mname]["predictions"]
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            y_true = test_df[target].values
            y_pred = preds[target]
            mask = np.isfinite(y_true) & np.isfinite(y_pred)

            if ad_labels is not None and mask.any():
                in_dom = ad_labels & mask
                ood = ~ad_labels & mask
                if in_dom.any():
                    ax.scatter(
                        y_true[in_dom], y_pred[in_dom],
                        alpha=0.6, s=30, c="#1f77b4", edgecolors="k",
                        linewidths=0.3, label="In-domain", zorder=3,
                    )
                if ood.any():
                    ax.scatter(
                        y_true[ood], y_pred[ood],
                        alpha=0.6, s=30, c="#e74c3c", edgecolors="k",
                        linewidths=0.3, label="OOD", marker="^", zorder=3,
                    )
                ax.legend(fontsize=10, loc="upper left")
            elif mask.any():
                color = MODEL_COLORS.get(mname, "#333")
                ax.scatter(
                    y_true[mask], y_pred[mask],
                    alpha=0.6, s=30, c=color, edgecolors="k",
                    linewidths=0.3, zorder=3,
                )

            # Parity line
            all_vals = np.concatenate([y_true[mask], y_pred[mask]]) if mask.any() else [0, 1]
            lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
            margin = (hi - lo) * 0.05 if hi > lo else 1
            ax.plot(
                [lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", lw=1, alpha=0.5, zorder=2,
            )

            metrics = _compute_metrics(y_true, y_pred)
            label = TARGET_LABELS[target]
            ax.set_xlabel(f"True {label} ({UNITS[target]})", fontsize=12)
            ax.set_ylabel(f"Predicted {label} ({UNITS[target]})", fontsize=12)
            ax.set_title(
                f"{mname} | {label}: R\u00b2={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}",
                fontsize=13,
            )
            ax.tick_params(labelsize=11)
            ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "parity_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_residuals(
    all_results: dict,
    test_df: pd.DataFrame,
    model_names: list[str],
) -> None:
    """Grid: rows=models, cols=targets. Histogram of (pred - true)."""
    n_models = len(model_names)
    fig, axes = plt.subplots(
        n_models, 3, figsize=(15, 4 * n_models), squeeze=False
    )

    for i, mname in enumerate(model_names):
        preds = all_results[mname]["predictions"]
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            y_true = test_df[target].values
            y_pred = preds[target]
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            residuals = y_pred[mask] - y_true[mask]

            color = MODEL_COLORS.get(mname, "#333")
            ax.hist(residuals, bins=30, edgecolor="k", linewidth=0.5,
                    alpha=0.7, color=color)
            ax.axvline(0, color="r", linestyle="--", lw=1.5)

            label = TARGET_LABELS[target]
            ax.set_xlabel(f"Residual ({UNITS[target]})", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(
                f"{mname} | {label} residuals (mean={residuals.mean():.3f})",
                fontsize=13,
            )
            ax.tick_params(labelsize=11)

    fig.tight_layout()
    path = FIGURES_DIR / "residual_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_learning_curves() -> None:
    """Plot NN train/val loss curves from nn_history.json."""
    history_path = SAVED_DIR / "nn_history.json"
    if not history_path.exists():
        print("  (nn_history.json not found; skipping learning curves)")
        return
    history = json.loads(history_path.read_text())

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = history["epochs"]
    ax.plot(epochs, history["train_loss"], label="Train loss", lw=2, color="#d95f02")
    ax.plot(epochs, history["val_loss"], label="Val loss", lw=2, color="#7570b3")

    best_epoch = history.get("best_epoch")
    if best_epoch is not None:
        idx = epochs.index(best_epoch) if best_epoch in epochs else None
        if idx is not None:
            ax.axvline(
                best_epoch, color="grey", linestyle=":", lw=1,
                label=f"Best epoch ({best_epoch})",
            )
            ax.scatter(
                [best_epoch], [history["val_loss"][idx]],
                marker="*", s=150, c="gold", edgecolors="k", zorder=5,
            )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss (MSE, normalized targets)", fontsize=13)
    ax.set_title("PCSAFTNet Learning Curves", fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "nn_learning_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_uncertainty_calibration(
    all_results: dict,
    test_df: pd.DataFrame,
    model_names: list[str],
) -> None:
    """For models with UQ: bin by predicted std, plot vs mean abs error."""
    uq_models = [
        m for m in model_names
        if f"{TARGETS[0]}_std" in all_results[m].get("uncertainty", {})
    ]
    if not uq_models:
        print("  (No models with uncertainty; skipping calibration plot)")
        return

    n_models = len(uq_models)
    fig, axes = plt.subplots(
        n_models, 3, figsize=(15, 4.5 * n_models), squeeze=False
    )

    for i, mname in enumerate(uq_models):
        uq = all_results[mname]["uncertainty"]
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            y_true = test_df[target].values
            y_pred = uq[target]
            y_std = uq[f"{target}_std"]
            mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_std)

            if mask.sum() < 10:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12)
                continue

            abs_err = np.abs(y_pred[mask] - y_true[mask])
            pred_std = y_std[mask]

            # Bin by predicted std (5 bins)
            n_bins = 5
            sorted_idx = np.argsort(pred_std)
            bin_size = len(sorted_idx) // n_bins
            bin_means_std = []
            bin_means_err = []
            for b in range(n_bins):
                start = b * bin_size
                end = (b + 1) * bin_size if b < n_bins - 1 else len(sorted_idx)
                idxs = sorted_idx[start:end]
                bin_means_std.append(pred_std[idxs].mean())
                bin_means_err.append(abs_err[idxs].mean())

            color = MODEL_COLORS.get(mname, "#333")
            ax.scatter(bin_means_std, bin_means_err, c=color, s=80, edgecolors="k",
                       linewidths=0.5, zorder=3)
            ax.plot(bin_means_std, bin_means_err, c=color, lw=1.5, alpha=0.7)

            # Diagonal reference
            all_vals = bin_means_std + bin_means_err
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.05 if hi > lo else 0.1
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "k--", lw=1, alpha=0.4, label="Perfect calibration")

            label = TARGET_LABELS[target]
            ax.set_xlabel(f"Mean predicted std ({UNITS[target]})", fontsize=12)
            ax.set_ylabel(f"Mean |error| ({UNITS[target]})", fontsize=12)
            ax.set_title(f"{mname} | {label} calibration", fontsize=13)
            ax.legend(fontsize=10)
            ax.tick_params(labelsize=11)

    fig.tight_layout()
    path = FIGURES_DIR / "uncertainty_calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate(model_names: list[str] | None = None) -> pd.DataFrame:
    """Run unified evaluation across requested models.

    Parameters
    ----------
    model_names : list[str] | None
        Names of models to evaluate. Defaults to all registered models.

    Returns
    -------
    pd.DataFrame
        Comparison metrics (also saved as CSV).
    """
    if model_names is None:
        model_names = list_models()

    # Load test set
    test_path = SAVED_DIR / "test_set.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test set not found at {test_path}. Train a model first."
        )
    test_df = pd.read_csv(test_path)
    smiles_list = test_df["smiles"].tolist()

    print(f"Evaluating {len(model_names)} model(s) on {len(test_df)} test molecules")
    print(f"Models: {', '.join(model_names)}\n")

    # AD labels
    ad_labels = _get_ad_labels(smiles_list)
    if ad_labels is not None:
        n_in = ad_labels.sum()
        n_ood = len(ad_labels) - n_in
        pct = n_ood / len(ad_labels) * 100
        print(f"AD analysis: {n_in} in-domain, {n_ood} OOD ({pct:.1f}% flagged)\n")

    # Collect results
    all_results: dict[str, dict] = {}
    rows: list[dict] = []

    for mname in model_names:
        print(f"--- {mname} ---")
        try:
            model = get_model(mname)
            model.load()
        except Exception as exc:
            print(f"  SKIP: {exc}")
            continue

        # Point predictions
        preds = model.predict(smiles_list)
        all_results[mname] = {"predictions": preds}

        # Uncertainty
        try:
            uq = model.predict_with_uncertainty(smiles_list)
            all_results[mname]["uncertainty"] = uq
        except Exception:
            uq = None

        # Metrics per target
        for target in TARGETS:
            y_true = test_df[target].values
            y_pred = preds[target]
            metrics = _compute_metrics(y_true, y_pred)

            row = {
                "model": mname,
                "target": target,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "n": metrics["n"],
            }

            # Uncertainty column
            if uq is not None and f"{target}_std" in uq:
                std_vals = uq[f"{target}_std"]
                valid = np.isfinite(std_vals)
                row["mean_std"] = float(np.nanmean(std_vals[valid])) if valid.any() else np.nan
            else:
                row["mean_std"] = np.nan

            # AD split metrics
            if ad_labels is not None:
                mask_valid = np.isfinite(y_true) & np.isfinite(y_pred)
                in_mask = ad_labels & mask_valid
                ood_mask = ~ad_labels & mask_valid

                if in_mask.sum() >= 2:
                    row["r2_in_domain"] = float(r2_score(y_true[in_mask], y_pred[in_mask]))
                else:
                    row["r2_in_domain"] = np.nan

                if ood_mask.sum() >= 2:
                    row["r2_ood"] = float(r2_score(y_true[ood_mask], y_pred[ood_mask]))
                else:
                    row["r2_ood"] = np.nan

                row["frac_ood"] = float((~ad_labels).sum() / len(ad_labels))
            else:
                row["r2_in_domain"] = np.nan
                row["r2_ood"] = np.nan
                row["frac_ood"] = np.nan

            rows.append(row)

            print(
                f"  {target:10s}  MAE={metrics['mae']:.4f}  "
                f"RMSE={metrics['rmse']:.4f}  R\u00b2={metrics['r2']:.4f}"
            )
        print()

    # Build comparison DataFrame
    metrics_df = pd.DataFrame(rows)
    csv_path = SAVED_DIR / "comparison_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}\n")

    # Print formatted table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    for mname in model_names:
        if mname not in all_results:
            continue
        sub = metrics_df[metrics_df["model"] == mname]
        print(f"\n  {mname}:")
        for _, r in sub.iterrows():
            t = r["target"]
            has_std = not np.isnan(r.get("mean_std", np.nan))
            std_str = f"  mean_std={r['mean_std']:.4f}" if has_std else ""
            ad_str = ""
            if not np.isnan(r.get("r2_in_domain", np.nan)):
                ad_str = f"  R\u00b2(in)={r['r2_in_domain']:.3f}  R\u00b2(ood)={r['r2_ood']:.3f}"
            print(
                f"    {t:10s}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
                f"R\u00b2={r['r2']:.4f}{std_str}{ad_str}"
            )
    print("\n" + "=" * 80)

    # Generate figures
    evaluated_models = [m for m in model_names if m in all_results]
    if evaluated_models:
        print("\nGenerating figures...")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        _plot_parity(all_results, test_df, evaluated_models, ad_labels)
        _plot_residuals(all_results, test_df, evaluated_models)
        _plot_learning_curves()
        _plot_uncertainty_calibration(all_results, test_df, evaluated_models)

    return metrics_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PC-SAFT prediction models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to evaluate (default: all registered). "
             f"Available: {list_models()}",
    )
    args = parser.parse_args()
    evaluate(model_names=args.models)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
