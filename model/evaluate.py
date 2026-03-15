"""Unified evaluation harness for PC-SAFT prediction models.

Evaluates any combination of registered models (gc_pcsaft, rf, nn) on the
held-out test set and produces:
  - A printed comparison table
  - model/saved/comparison_metrics.csv
  - Parity comparison plot (grid: models x targets)
  - Residual distribution plot (grid: models x targets)
  - NN learning curves (if nn_history.json exists)
  - Uncertainty calibration plot (for models with UQ)
  - Bootstrap confidence intervals on all metrics (with --bootstrap)

Usage:
    python -m model.evaluate --models gc_pcsaft rf nn
    python -m model.evaluate --models rf --bootstrap   # add 95% CIs
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from model.data.descriptors import build_features
from model.data.load import TARGETS, load_data, split_data
from model.registry import get_model, list_models

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent / "saved"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "03_evaluation_harness"
FIGURES_DIR_METRICS = Path(__file__).resolve().parent.parent / "figures" / "11_metrics"

UNITS = {"m": "segments", "sigma": "\u00c5", "epsilon_k": "K"}
TARGET_LABELS = {"m": "m", "sigma": "\u03c3", "epsilon_k": "\u03b5/k"}

# Color scheme for models
MODEL_COLORS = {
    "gc_pcsaft": "#1b9e77",
    "rf": "#d95f02",
    "nn": "#7570b3",
    "chemberta": "#e7298a",
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

def _get_ad_labels(smiles_list: list[str], model_name: str | None = None) -> np.ndarray | None:
    """Return boolean array (True=in-domain) or None if AD model unavailable.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.
    model_name : str | None
        Model name. If "chemberta", use ChemBERTa embedding-space AD.
        Otherwise, use descriptor-space AD.

    Returns
    -------
    np.ndarray | None
        Boolean array or None if AD model unavailable.
    """
    # ChemBERTa uses its own embedding-space AD
    if model_name == "chemberta":
        try:
            model = get_model("chemberta")
            model.load()
            return model.predict_in_domain(smiles_list)
        except Exception:
            logger.warning("ChemBERTa AD model unavailable; skipping AD analysis")
            return None

    # RF, NN, and GC-PCSAFT use descriptor-space AD
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

def _ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient.

    Combines Pearson correlation (precision) with mean-bias correction
    (accuracy) into one number. Penalises systematic over/under-prediction
    that R² ignores.  Range: -1 to 1, perfect agreement = 1.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return np.nan
    mu_t, mu_p = yt.mean(), yp.mean()
    # Population variance (ddof=0) to match the standard CCC formula
    var_t = np.var(yt, ddof=0)
    var_p = np.var(yp, ddof=0)
    cov = np.cov(yt, yp, ddof=0)[0, 1]
    denom = var_t + var_p + (mu_t - mu_p) ** 2
    return float(2.0 * cov / denom) if denom > 1e-12 else np.nan


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute MAE, RMSE, R², MARE, CCC, and coverage thresholds.

    MARE (Mean Absolute Relative Error) = mean(|pred-true|/|true|).
    Equivalent to MAPE/100 and to AARD used in PC-SAFT literature.
    CCC is Lin's Concordance Correlation Coefficient.
    coverage_5pct / coverage_10pct: fraction of predictions within
    ±5% / ±10% of the true value (relative, scale-free).
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {
            "mae": np.nan, "rmse": np.nan, "r2": np.nan,
            "mare": np.nan, "ccc": np.nan,
            "coverage_5pct": np.nan, "coverage_10pct": np.nan,
            "n": int(mask.sum()),
        }
    yt, yp = y_true[mask], y_pred[mask]

    nonzero = np.abs(yt) > 1e-9
    if nonzero.any():
        rel_errors = np.abs((yp[nonzero] - yt[nonzero]) / yt[nonzero])
        mare = float(np.mean(rel_errors))
        coverage_5pct = float(np.mean(rel_errors <= 0.05))
        coverage_10pct = float(np.mean(rel_errors <= 0.10))
    else:
        mare = coverage_5pct = coverage_10pct = np.nan

    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "mare": mare,
        "ccc": _ccc(yt, yp),
        "coverage_5pct": coverage_5pct,
        "coverage_10pct": coverage_10pct,
        "n": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _plot_parity(
    all_results: dict,
    test_df: pd.DataFrame,
    model_names: list[str],
    ad_labels_dict: dict[str, np.ndarray | None],
) -> None:
    """Grid: rows=models, cols=targets. Color by AD status."""
    n_models = len(model_names)
    fig, axes = plt.subplots(
        n_models, 3, figsize=(15, 4.5 * n_models), squeeze=False
    )

    for i, mname in enumerate(model_names):
        preds = all_results[mname]["predictions"]
        ad_labels = ad_labels_dict.get(mname)
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
# Phase 2 figures: coverage bar chart, CCC vs R² scatter, MARE by target
# ---------------------------------------------------------------------------

def _plot_coverage_barchart(metrics_df: pd.DataFrame) -> None:
    """Bar chart: coverage@5% and coverage@10% for each model×target."""
    FIGURES_DIR_METRICS.mkdir(parents=True, exist_ok=True)
    models = metrics_df["model"].unique().tolist()
    targets = TARGETS

    x = np.arange(len(targets))
    width = 0.8 / (len(models) * 2 + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))

    for ax_idx, (thresh_col, thresh_label) in enumerate(
        [("coverage_5pct", "Coverage @ ±5%"), ("coverage_10pct", "Coverage @ ±10%")]
    ):
        ax = axes[ax_idx]
        for mi, (mname, color) in enumerate(zip(models, colors)):
            sub = metrics_df[metrics_df["model"] == mname]
            vals = [
                sub.loc[sub["target"] == t, thresh_col].values[0]
                if t in sub["target"].values else np.nan
                for t in targets
            ]
            offset = (mi - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9, label=mname,
                          color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.0%}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([TARGET_LABELS[t] for t in targets])
        ax.set_ylabel("Fraction of predictions", fontsize=11)
        ax.set_title(thresh_label, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Prediction Coverage at Relative Error Thresholds", fontsize=13)
    fig.tight_layout()
    path = FIGURES_DIR_METRICS / "coverage_bar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_ccc_vs_r2(metrics_df: pd.DataFrame) -> None:
    """Scatter: CCC vs R² for all model×target combinations."""
    FIGURES_DIR_METRICS.mkdir(parents=True, exist_ok=True)
    models = metrics_df["model"].unique().tolist()
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(7, 6))
    for mi, (mname, color) in enumerate(zip(models, colors)):
        sub = metrics_df[metrics_df["model"] == mname]
        for ti, target in enumerate(TARGETS):
            row = sub[sub["target"] == target]
            if row.empty:
                continue
            r2 = float(row["r2"].values[0])
            ccc = float(row["ccc"].values[0])
            if np.isnan(r2) or np.isnan(ccc):
                continue
            label = mname if ti == 0 else None
            ax.scatter(r2, ccc, color=color,
                       marker=markers[ti % len(markers)], s=80,
                       edgecolors="k", linewidths=0.5, label=label, zorder=3)
            ax.annotate(
                f"{mname[:6]}/{TARGET_LABELS[target]}",
                (r2, ccc), textcoords="offset points",
                xytext=(4, 3), fontsize=7, alpha=0.75,
            )

    # Diagonal where CCC == R² (no bias)
    lims = [-0.2, 1.0]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="CCC = R²\n(no bias)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("R² (test)", fontsize=12)
    ax.set_ylabel("CCC (Concordance Correlation Coefficient)", fontsize=12)
    ax.set_title("CCC vs R²: Points below diagonal indicate systematic bias", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR_METRICS / "ccc_vs_r2_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_mare_by_target(metrics_df: pd.DataFrame) -> None:
    """Grouped bar chart: MARE for each model×target."""
    FIGURES_DIR_METRICS.mkdir(parents=True, exist_ok=True)
    models = metrics_df["model"].unique().tolist()
    targets = TARGETS
    x = np.arange(len(targets))
    width = 0.8 / (len(models) + 0.5)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for mi, (mname, color) in enumerate(zip(models, colors)):
        sub = metrics_df[metrics_df["model"] == mname]
        vals = [
            sub.loc[sub["target"] == t, "mare"].values[0]
            if t in sub["target"].values else np.nan
            for t in targets
        ]
        offset = (mi - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=mname,
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{TARGET_LABELS[t]} ({UNITS[t]})" for t in targets])
    ax.set_ylabel("MARE (Mean Absolute Relative Error)", fontsize=11)
    ax.set_title(
        "MARE by Model and Target\n"
        "(lower = better; scale-free comparison across targets)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.10, color="red", linestyle="--", alpha=0.5, linewidth=1,
               label="10% threshold")
    fig.tight_layout()
    path = FIGURES_DIR_METRICS / "mare_by_target.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Cross-validated Q² (5-fold CV R²)
# ---------------------------------------------------------------------------

def compute_rf_cv_q2(n_folds: int = 5) -> dict[str, dict[str, float]]:
    """Compute Q² (K-fold cross-validated R²) for the RF on its training data.

    Q² is the standard QSPR validation metric (Golbraikh & Tropsha 2002).
    Unlike single-holdout R², it averages over multiple splits, giving a
    robust estimate of generalisability that is insensitive to the luck of
    one particular train/test draw.

    Threshold conventions:
        Q² > 0.5  —  model has predictive power (minimum acceptable)
        Q² > 0.6  —  good; Q² > 0.7  —  excellent

    Returns
    -------
    dict mapping target name → {"q2_mean": float, "q2_std": float, "folds": list[float]}
    """
    print(f"Computing Q² ({n_folds}-fold CV) for RF on training set...")

    # Reconstruct the same training split used when the model was saved.
    df = load_data("auto")
    train_df, _ = split_data(df, stratify_bins=5, random_state=42)

    smiles_train = train_df["smiles"].tolist()

    config = _load_feature_config()
    rdkit_names = _get_rdkit_names_from_saved() if config.get("use_rdkit", True) else None
    X = build_features(
        smiles_train,
        use_morgan=config.get("use_morgan", False),
        use_rdkit=config.get("use_rdkit", True),
        morgan_radius=config.get("morgan_radius", 2),
        morgan_bits=config.get("morgan_bits", 2048),
        rdkit_names=rdkit_names,
    )

    valid_mask = np.isfinite(X).all(axis=1)
    X = X[valid_mask]
    train_df = train_df.iloc[valid_mask].reset_index(drop=True)

    print(f"  Training molecules: {len(X)}, features: {X.shape[1]}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rf_cfg = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}

    results: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        y = train_df[target].values
        rf = RandomForestRegressor(**rf_cfg)
        fold_scores = cross_val_score(rf, X, y, cv=kf, scoring="r2")
        q2_mean = float(fold_scores.mean())
        q2_std = float(fold_scores.std())
        results[target] = {
            "q2_mean": q2_mean,
            "q2_std": q2_std,
            "folds": fold_scores.tolist(),
        }
        if q2_mean >= 0.6:
            threshold = "✓ good"
        elif q2_mean >= 0.5:
            threshold = "✓ acceptable"
        else:
            threshold = "✗ below threshold"
        print(
            f"  {target:10s}  Q²={q2_mean:.4f} ± {q2_std:.4f}"
            f"  folds={[f'{s:.3f}' for s in fold_scores]}  [{threshold}]"
        )

    return results


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate(
    model_names: list[str] | None = None,
    bootstrap: bool = False,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Run unified evaluation across requested models.

    Parameters
    ----------
    model_names : list[str] | None
        Names of models to evaluate. Defaults to all registered models.
    bootstrap : bool
        If True, compute bootstrap 95% CIs for R², MAE, and RMSE.
    n_boot : int
        Number of bootstrap resamples (default 2000).

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

    # Collect results
    all_results: dict[str, dict] = {}
    ad_labels_dict: dict[str, np.ndarray | None] = {}
    rows: list[dict] = []

    for mname in model_names:
        print(f"--- {mname} ---")
        try:
            model = get_model(mname)
            model.load()
        except Exception as exc:
            print(f"  SKIP: {exc}")
            continue

        # Get model-specific AD labels
        ad_labels = _get_ad_labels(smiles_list, model_name=mname)
        ad_labels_dict[mname] = ad_labels
        if ad_labels is not None:
            n_in = ad_labels.sum()
            n_ood = len(ad_labels) - n_in
            pct = n_ood / len(ad_labels) * 100
            print(f"  AD analysis: {n_in} in-domain, {n_ood} OOD ({pct:.1f}% flagged)")

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
                "mare": metrics["mare"],
                "ccc": metrics["ccc"],
                "coverage_5pct": metrics["coverage_5pct"],
                "coverage_10pct": metrics["coverage_10pct"],
                "n": metrics["n"],
            }

            # Bootstrap CIs
            if bootstrap:
                from model.uncertainty import bootstrap_metric_ci

                for metric_name, metric_fn in [
                    ("r2", r2_score),
                    ("mae", mean_absolute_error),
                    ("rmse", lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))),
                ]:
                    ci = bootstrap_metric_ci(
                        y_true, y_pred, metric_fn,
                        n_boot=n_boot, ci=0.95,
                    )
                    row[f"{metric_name}_lo"] = ci["ci_lo"]
                    row[f"{metric_name}_hi"] = ci["ci_hi"]
                    row[f"{metric_name}_boot_std"] = ci["std"]

            # Uncertainty: mean std and approximate 95% CI coverage
            if uq is not None and f"{target}_std" in uq:
                std_vals = uq[f"{target}_std"]
                valid = np.isfinite(std_vals)
                row["mean_std"] = float(np.nanmean(std_vals[valid])) if valid.any() else np.nan

                # Approximate 95% CI: pred ± 1.96 * sigma_trees
                # Empirical coverage = fraction of true values inside the interval.
                # A perfectly calibrated model would score ~0.95 here.
                mask_ci = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(std_vals)
                if mask_ci.sum() >= 2:
                    ci_half = 1.96 * std_vals[mask_ci]
                    inside = np.abs(y_true[mask_ci] - y_pred[mask_ci]) <= ci_half
                    row["ci95_coverage"] = float(inside.mean())
                    row["mean_ci95_width"] = float((2 * ci_half).mean())
                else:
                    row["ci95_coverage"] = np.nan
                    row["mean_ci95_width"] = np.nan
            else:
                row["mean_std"] = np.nan
                row["ci95_coverage"] = np.nan
                row["mean_ci95_width"] = np.nan

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

            ci_str = ""
            if not np.isnan(row.get("ci95_coverage", np.nan)):
                ci_str = (
                    f"  CI95_cov={row['ci95_coverage']:.3f}"
                    f"  CI95_width={row['mean_ci95_width']:.3f}"
                )
            boot_str = ""
            if bootstrap and "r2_lo" in row:
                boot_str = (
                    f"\n{'':14s}  R²_95CI=[{row['r2_lo']:.4f}, {row['r2_hi']:.4f}]"
                    f"  MAE_95CI=[{row['mae_lo']:.4f}, {row['mae_hi']:.4f}]"
                )
            print(
                f"  {target:10s}  MAE={metrics['mae']:.4f}  "
                f"RMSE={metrics['rmse']:.4f}  R\u00b2={metrics['r2']:.4f}  "
                f"MARE={metrics['mare']:.3f}  CCC={metrics['ccc']:.4f}  "
                f"cov5%={metrics['coverage_5pct']:.3f}  cov10%={metrics['coverage_10pct']:.3f}"
                f"{ci_str}{boot_str}"
            )
        print()

    # Build comparison DataFrame
    metrics_df = pd.DataFrame(rows)
    csv_path = SAVED_DIR / "comparison_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}\n")

    # Print formatted table
    print("=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
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
            mare_str = f"  MARE={r['mare']:.3f}" if not np.isnan(r.get("mare", np.nan)) else ""
            ccc_str = f"  CCC={r['ccc']:.4f}" if not np.isnan(r.get("ccc", np.nan)) else ""
            cov_str = ""
            if not np.isnan(r.get("coverage_5pct", np.nan)):
                cov_str = f"  cov5%={r['coverage_5pct']:.3f}  cov10%={r['coverage_10pct']:.3f}"
            ci_str = ""
            if not np.isnan(r.get("ci95_coverage", np.nan)):
                ci_str = (
                    f"  CI95_cov={r['ci95_coverage']:.3f}"
                    f"  CI95_width={r['mean_ci95_width']:.3f}"
                )
            boot_str = ""
            if "r2_lo" in r and not np.isnan(r.get("r2_lo", np.nan)):
                boot_str = (
                    f"\n{'':14s}  R²_95CI=[{r['r2_lo']:.4f}, {r['r2_hi']:.4f}]"
                    f"  MAE_95CI=[{r['mae_lo']:.4f}, {r['mae_hi']:.4f}]"
                    f"  RMSE_95CI=[{r['rmse_lo']:.4f}, {r['rmse_hi']:.4f}]"
                )
            print(
                f"    {t:10s}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
                f"R\u00b2={r['r2']:.4f}{mare_str}{ccc_str}{cov_str}{std_str}{ci_str}{ad_str}"
                f"{boot_str}"
            )
    print("\n" + "=" * 100)

    # Generate figures
    evaluated_models = [m for m in model_names if m in all_results]
    if evaluated_models:
        print("\nGenerating figures...")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        _plot_parity(all_results, test_df, evaluated_models, ad_labels_dict)
        _plot_residuals(all_results, test_df, evaluated_models)
        _plot_learning_curves()
        _plot_uncertainty_calibration(all_results, test_df, evaluated_models)

        # Phase 2 enhanced metrics figures
        print("\nGenerating Phase 2 enhanced metrics figures...")
        _plot_coverage_barchart(metrics_df)
        _plot_ccc_vs_r2(metrics_df)
        _plot_mare_by_target(metrics_df)

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
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help=(
            "Compute bootstrap 95%% CIs for R², MAE, and RMSE. "
            "Adds _lo/_hi/_boot_std columns to comparison_metrics.csv."
        ),
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        metavar="N",
        help="Number of bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        default=False,
        help=(
            "Also compute Q² (5-fold cross-validated R²) for the RF on its "
            "training set. Requires re-fitting 5 RF models; adds ~30 s on CPU."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        metavar="K",
        help="Number of CV folds for Q² computation (default: 5).",
    )
    args = parser.parse_args()
    metrics_df = evaluate(
        model_names=args.models,
        bootstrap=args.bootstrap,
        n_boot=args.n_boot,
    )

    if args.cv:
        print()
        q2_results = compute_rf_cv_q2(n_folds=args.cv_folds)
        print()
        print("=" * 60)
        print(f"Q² SUMMARY ({args.cv_folds}-fold CV, RF on training set)")
        print("=" * 60)
        print("  Threshold: Q² > 0.6 (good), > 0.5 (acceptable)")
        print()
        for target, res in q2_results.items():
            q2 = res["q2_mean"]
            std = res["q2_std"]
            single_holdout = metrics_df.loc[
                (metrics_df["model"] == "rf") & (metrics_df["target"] == target), "r2"
            ]
            r2_val = float(single_holdout.values[0]) if len(single_holdout) else float("nan")
            gap = r2_val - q2
            gap_flag = "  ← large gap, suspect overfit" if abs(gap) > 0.1 else ""
            print(
                f"  {target:10s}  Q²={q2:.4f} ± {std:.4f}  "
                f"R²(test)={r2_val:.4f}  gap={gap:+.4f}{gap_flag}"
            )
        print("=" * 60)

        # Save Q² to CSV alongside other metrics
        q2_path = SAVED_DIR / "cv_q2.json"
        import json as _json
        q2_path.write_text(_json.dumps(q2_results, indent=2))
        print(f"\nQ² results saved to {q2_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
