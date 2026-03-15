"""Training and evaluation script for the weighted ensemble.

Usage:
    python -m model.ensemble.train_ensemble [--models rf nn chemberta]
    python -m model.ensemble.train_ensemble --baseline rf --models rf nn chemberta

Loads the test set (same split as all other models), runs the inverse-variance
weighted ensemble, and compares to a single-model baseline.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from model.data.load import TARGETS, load_data, split_data  # noqa: E402
from model.ensemble.weighted import WeightedEnsemble  # noqa: E402
from model.registry import get_model  # noqa: E402

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved"
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "figures" / "02c_ensemble"

TARGET_LABELS = {"m": "m (segments)", "sigma": "\u03c3 (\u00c5)", "epsilon_k": "\u03b5/k (K)"}
TARGET_UNITS = {"m": "segments", "sigma": "\u00c5", "epsilon_k": "K"}


def _load_test_set() -> tuple[list[str], dict[str, np.ndarray]]:
    """Load the test set using the same split as all other models."""
    test_csv = SAVED_DIR / "test_set.csv"
    if test_csv.exists():
        import pandas as pd

        test_df = pd.read_csv(test_csv)
        smiles = test_df["smiles"].tolist()
        targets = {t: test_df[t].values for t in TARGETS}
        return smiles, targets

    # Fall back to loading and splitting
    logger.info("No saved test_set.csv; splitting from source data")
    df = load_data(source="auto")
    _, test_df = split_data(df)
    smiles = test_df["smiles"].tolist()
    targets = {t: test_df[t].values for t in TARGETS}
    return smiles, targets


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute R2, MAE, RMSE for a pair of arrays (NaN-safe)."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
    }


def _load_baseline(baseline_name: str, smiles: list[str]) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, float]],
    str,
]:
    """Load and evaluate a single-model baseline.

    Returns (predictions_with_uncertainty, metrics_per_target, actual_name).
    """
    try:
        model = get_model(baseline_name)
        model.load()
        preds = model.predict_with_uncertainty(smiles)
        return preds, None, baseline_name
    except Exception as exc:
        logger.warning("Baseline %r failed: %s", baseline_name, exc)
        raise


def run_evaluation(
    model_names: list[str],
    baseline_name: str = "rf",
) -> dict:
    """Run ensemble evaluation and return metrics dict."""
    print("Loading test set...")
    smiles, true_targets = _load_test_set()
    n_test = len(smiles)
    print(f"Test set: {n_test} molecules")

    # ---- Single-model baseline ----
    print(f"\nEvaluating {baseline_name} baseline...")
    try:
        baseline_preds, _, baseline_actual = _load_baseline(baseline_name, smiles)
    except Exception:
        # Try fallback to first model in model_names
        baseline_actual = model_names[0]
        print(f"  Falling back to {baseline_actual} as baseline")
        baseline_preds, _, baseline_actual = _load_baseline(baseline_actual, smiles)

    baseline_metrics: dict[str, dict[str, float]] = {}
    for t in TARGETS:
        baseline_metrics[t] = _compute_metrics(true_targets[t], baseline_preds[t])
        print(
            f"  {baseline_actual} {t}: R2={baseline_metrics[t]['r2']:.4f}  "
            f"MAE={baseline_metrics[t]['mae']:.3f}  "
            f"RMSE={baseline_metrics[t]['rmse']:.3f}"
        )

    # ---- Per-model evaluation ----
    individual_metrics: dict[str, dict[str, dict[str, float]]] = {}
    individual_preds: dict[str, dict[str, np.ndarray]] = {}
    for name in model_names:
        try:
            m = get_model(name)
            m.load()
            preds = m.predict_with_uncertainty(smiles)
            individual_preds[name] = preds
            individual_metrics[name] = {}
            for t in TARGETS:
                individual_metrics[name][t] = _compute_metrics(
                    true_targets[t], preds[t]
                )
        except Exception as exc:
            logger.warning("Could not evaluate %r individually: %s", name, exc)

    # ---- Ensemble ----
    print(f"\nBuilding ensemble with models: {model_names}")
    ensemble = WeightedEnsemble(model_names)
    active_names = list(ensemble.models.keys())
    print(f"Active models: {active_names}")

    print("Evaluating ensemble...")
    ens_preds = ensemble.predict_with_uncertainty(smiles)

    ens_metrics: dict[str, dict[str, float]] = {}
    for t in TARGETS:
        ens_metrics[t] = _compute_metrics(true_targets[t], ens_preds[t])
        print(
            f"  Ensemble {t}: R2={ens_metrics[t]['r2']:.4f}  "
            f"MAE={ens_metrics[t]['mae']:.3f}  "
            f"RMSE={ens_metrics[t]['rmse']:.3f}"
        )

    # ---- Comparison ----
    print(f"\n===== R2 Comparison (baseline={baseline_actual}) =====")
    print(f"{'Target':<12} {baseline_actual:>10} {'Ensemble':>10} {'Delta':>8}")
    for t in TARGETS:
        delta = ens_metrics[t]["r2"] - baseline_metrics[t]["r2"]
        print(
            f"{t:<12} {baseline_metrics[t]['r2']:>10.4f} "
            f"{ens_metrics[t]['r2']:>10.4f} {delta:>+8.4f}"
        )

    # ---- Model disagreement ----
    print("\nComputing model disagreement...")
    disagreement = ensemble.predict_disagreement(smiles)
    for t in TARGETS:
        d = disagreement[t]
        valid = np.isfinite(d)
        if valid.any():
            print(
                f"  {t}: mean={d[valid].mean():.3f}  "
                f"max={d[valid].max():.3f}  "
                f"median={np.median(d[valid]):.3f}"
            )

    # ---- Uncertainty comparison ----
    print("\nUncertainty comparison (mean std):")
    for t in TARGETS:
        bl_std = baseline_preds[f"{t}_std"]
        ens_std = ens_preds[f"{t}_std"]
        bl_valid = np.isfinite(bl_std)
        ens_valid = np.isfinite(ens_std)
        bl_mean_std = bl_std[bl_valid].mean() if bl_valid.any() else float("nan")
        ens_mean_std = ens_std[ens_valid].mean() if ens_valid.any() else float("nan")
        reduction = (1 - ens_mean_std / bl_mean_std) * 100 if bl_mean_std > 0 else 0
        print(
            f"  {t}: {baseline_actual}={bl_mean_std:.4f}  "
            f"Ensemble={ens_mean_std:.4f}  "
            f"reduction={reduction:.1f}%"
        )

    # ---- Weight distribution ----
    print("\nWeight distribution:")
    weights = ensemble.get_per_model_weights(smiles)
    for name in weights:
        for t in TARGETS:
            w = weights[name][t]
            valid = np.isfinite(w)
            if valid.any():
                print(
                    f"  {name}/{t}: mean={w[valid].mean():.3f}  "
                    f"median={np.median(w[valid]):.3f}  "
                    f"min={w[valid].min():.3f}  max={w[valid].max():.3f}"
                )

    return {
        "baseline_name": baseline_actual,
        "baseline_metrics": baseline_metrics,
        "baseline_preds": baseline_preds,
        "ensemble_metrics": ens_metrics,
        "individual_metrics": individual_metrics,
        "model_names": active_names,
        "n_test": n_test,
        "ens_preds": ens_preds,
        "true_targets": true_targets,
        "smiles": smiles,
        "ensemble": ensemble,
    }


def generate_figures(results: dict) -> None:
    """Generate all three required figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    smiles = results["smiles"]
    true_targets = results["true_targets"]
    baseline_preds = results["baseline_preds"]
    ens_preds = results["ens_preds"]
    baseline_metrics = results["baseline_metrics"]
    ens_metrics = results["ensemble_metrics"]
    baseline_name = results["baseline_name"]
    ensemble = results["ensemble"]

    # ---- Figure 1: Parity plots (baseline vs Ensemble) ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for j, t in enumerate(TARGETS):
        y_true = true_targets[t]

        for row, (preds, label, metrics) in enumerate([
            (baseline_preds, baseline_name.upper(), baseline_metrics),
            (ens_preds, "Ensemble", ens_metrics),
        ]):
            ax = axes[row, j]
            y_pred = preds[t]
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if not mask.any():
                ax.set_title(f"{label}: {TARGET_LABELS[t]}\nNo valid predictions")
                continue
            ax.scatter(
                y_true[mask], y_pred[mask], alpha=0.3, s=15, c="#1f77b4",
            )
            lo = min(y_true[mask].min(), y_pred[mask].min())
            hi = max(y_true[mask].max(), y_pred[mask].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
            ax.set_xlabel(f"True {TARGET_LABELS[t]}", fontsize=12)
            ax.set_ylabel(f"Predicted {TARGET_LABELS[t]}", fontsize=12)
            ax.set_title(
                f"{label}: {TARGET_LABELS[t]}\n"
                f"R\u00b2={metrics[t]['r2']:.3f}, MAE={metrics[t]['mae']:.3f}",
                fontsize=13,
            )
            ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        f"{baseline_name.upper()} vs Ensemble: Parity Plots",
        fontsize=16, y=1.02,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "ensemble_vs_gnn_parity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 2: Weight distribution ----
    weights = ensemble.get_per_model_weights(smiles)
    model_names_active = list(weights.keys())
    n_models = len(model_names_active)
    n_targets = len(TARGETS)

    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))
    for j, t in enumerate(TARGETS):
        ax = axes[j]
        for i, name in enumerate(model_names_active):
            w = weights[name][t]
            valid = np.isfinite(w)
            if valid.any():
                ax.hist(
                    w[valid],
                    bins=50,
                    alpha=0.6,
                    label=name,
                    color=colors[i],
                    edgecolor="white",
                    linewidth=0.5,
                )
        ax.set_xlabel("Inverse-Variance Weight", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Weight Distribution: {TARGET_LABELS[t]}", fontsize=13)
        ax.legend(fontsize=10)

    fig.suptitle(
        "Model Weight Distribution Across Test Molecules",
        fontsize=16, y=1.02,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "model_weight_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 3: R2 improvement bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [TARGET_LABELS[t] for t in TARGETS]
    bl_r2 = [baseline_metrics[t]["r2"] for t in TARGETS]
    ens_r2 = [ens_metrics[t]["r2"] for t in TARGETS]
    deltas = [e - g for e, g in zip(ens_r2, bl_r2)]

    x = np.arange(len(TARGETS))
    width = 0.35
    ax.bar(
        x - width / 2, bl_r2, width,
        label=baseline_name.upper(), color="#1f77b4", alpha=0.8,
    )
    ax.bar(
        x + width / 2, ens_r2, width,
        label="Ensemble", color="#2ca02c", alpha=0.8,
    )

    # Annotate deltas
    for i, d in enumerate(deltas):
        y_pos = max(bl_r2[i], ens_r2[i]) + 0.02
        sign = "+" if d >= 0 else ""
        ax.text(
            x[i], y_pos, f"{sign}{d:.4f}",
            ha="center", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("R\u00b2", fontsize=13)
    ax.set_title(
        f"R\u00b2: {baseline_name.upper()} vs Ensemble (per target)",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(bl_r2), max(ens_r2)) + 0.1)

    fig.tight_layout()
    path = FIGURES_DIR / "ensemble_improvement_by_target.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_config(results: dict) -> None:
    """Save ensemble configuration to model/saved/ensemble_config.json."""
    config = {
        "models": results["model_names"],
        "n_test": results["n_test"],
        "baseline_name": results["baseline_name"],
        "baseline_metrics": results["baseline_metrics"],
        "ensemble_metrics": results["ensemble_metrics"],
        "individual_metrics": results.get("individual_metrics", {}),
    }
    path = SAVED_DIR / "ensemble_config.json"
    path.write_text(json.dumps(config, indent=2))
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty-weighted ensemble for PC-SAFT prediction."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "nn", "chemberta"],
        help="Model names to include in the ensemble (default: rf nn chemberta)",
    )
    parser.add_argument(
        "--baseline",
        default="rf",
        help="Single model to use as baseline for comparison (default: rf)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    results = run_evaluation(args.models, baseline_name=args.baseline)
    generate_figures(results)
    save_config(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
