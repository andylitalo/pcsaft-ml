"""Step 51: Unified Model Comparison Chart.

Fills in missing fluorinated evaluations (XGBoost, SVR) and produces
a single consolidated comparison table and figure covering all tested
models across both Esper internal and fluorinated external validation.

Usage:
    python scripts/step51_unified_model_comparison.py [--section SECTION]

Sections:
    eval        Evaluate missing models on fluorinated set
    table       Build unified comparison table
    figure      Generate consolidated comparison figure
    report      Update model_comparison.md and Step 48 report
    all         Run everything (default)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import model.xgb  # noqa: F401, E402, I001
import model.svm.svm_model  # noqa: F401, E402
from model.data.load import TARGETS  # noqa: E402
from model.registry import _compute_features, get_model  # noqa: E402
from screening.hfo_screening import compute_boiling_point  # noqa: E402

logger = logging.getLogger(__name__)

SAVED_DIR = PROJECT_ROOT / "model" / "saved"
FIGURES_DIR = PROJECT_ROOT / "figures" / "51_unified_model_comparison"
FLUORINATED_VAL_PATH = SAVED_DIR / "gnn_fluorinated_validation_set.csv"

ALL_MODELS_DISPLAY = {
    "rf": "RF",
    "xgboost": "XGBoost",
    "svm": "SVR",
    "chemprop": "chemprop (D-MPNN)",
    "gnn": "GNN (GINEConv)",
    "chemberta": "ChemBERTa",
    "nn": "NN (MLP)",
    "ridge": "Ridge",
    "gc_pcsaft": "GC-PC-SAFT",
}

_SINGLE = "Single split (seed=42)"
_REPEAT = "10 outer splits (mean)"

ESPER_RESULTS = {
    "gc_pcsaft": {
        "m_r2": 0.40, "sigma_r2": -1.16, "epsk_r2": -0.04,
        "m_mae": 1.00, "sigma_mae": 0.49, "epsk_mae": 44.6,
        "protocol": "Deterministic",
    },
    "rf": {
        "m_r2": 0.62, "sigma_r2": 0.35, "epsk_r2": 0.33,
        "m_mae": 0.59, "sigma_mae": 0.19, "epsk_mae": 26.8,
        "protocol": _SINGLE,
    },
    "nn": {
        "m_r2": 0.47, "sigma_r2": 0.11, "epsk_r2": 0.14,
        "m_mae": 0.77, "sigma_mae": 0.24, "epsk_mae": 32.9,
        "protocol": _SINGLE,
    },
    "chemberta": {
        "m_r2": 0.53, "sigma_r2": 0.25, "epsk_r2": 0.27,
        "m_mae": 0.77, "sigma_mae": 0.23, "epsk_mae": 31.2,
        "protocol": _SINGLE,
    },
    "xgboost": {
        "m_r2": 0.64, "sigma_r2": 0.36, "epsk_r2": 0.33,
        "m_mae": 0.61, "sigma_mae": 0.20, "epsk_mae": 27.0,
        "protocol": _SINGLE,
    },
    "chemprop": {
        "m_r2": 0.54, "sigma_r2": 0.33, "epsk_r2": 0.39,
        "m_mae": 0.69, "sigma_mae": 0.21, "epsk_mae": 26.8,
        "protocol": _SINGLE,
    },
    "gnn": {
        "m_r2": 0.69, "sigma_r2": 0.34, "epsk_r2": 0.41,
        "m_mae": 0.76, "sigma_mae": 0.23, "epsk_mae": 28.2,
        "protocol": _SINGLE,
    },
    "svm": {
        "m_r2": 0.59, "sigma_r2": 0.25, "epsk_r2": 0.33,
        "m_mae": 0.73, "sigma_mae": 0.22, "epsk_mae": 29.9,
        "protocol": _REPEAT,
    },
    "ridge": {
        "m_r2": 0.52, "sigma_r2": -0.00, "epsk_r2": -0.43,
        "m_mae": 0.73, "sigma_mae": 0.24, "epsk_mae": 32.6,
        "protocol": _REPEAT,
    },
}

EXISTING_FLUOR = {
    "rf": {
        "epsk_mae": 14.3, "epsk_r2": -0.47,
        "bp_mae": 8.2, "bp_rmse": 10.3, "convergence": "6/15",
    },
    "chemprop": {
        "epsk_mae": 19.6, "epsk_r2": -1.22,
        "bp_mae": 19.3, "bp_rmse": 22.8, "convergence": "6/15",
    },
    "gnn": {
        "epsk_mae": 17.4, "epsk_r2": -0.84,
        "bp_mae": 23.9, "bp_rmse": 26.5, "convergence": "4/15",
    },
}

# Display order for the unified table
MODEL_ORDER = [
    "gc_pcsaft", "ridge", "nn", "chemberta", "svm", "rf", "xgboost",
    "chemprop", "gnn",
]

MODEL_COLORS = {
    "gc_pcsaft": "#999999",
    "ridge": "#bcbd22",
    "nn": "#17becf",
    "chemberta": "#9467bd",
    "svm": "#8c564b",
    "rf": "#d95f02",
    "xgboost": "#1b9e77",
    "chemprop": "#7570b3",
    "gnn": "#e7298a",
}


def _compute_features_for_model(smiles_list: list[str], model_name: str) -> np.ndarray:
    """Compute features using the feature names stored with a specific model.

    XGBoost and SVR were trained on a different clean_descriptors output
    (168 RDKit cols) than RF (170 RDKit cols). Using the wrong feature set
    causes a dimension mismatch. This function loads the correct feature
    names from each model's artifact and builds features to match.
    """
    import joblib as _joblib

    from model.data.descriptors import build_features

    model_feature_paths = {
        "xgboost": SAVED_DIR / "xgb" / "xgb_model.joblib",
        "svm": SAVED_DIR / "svm" / "svm_model.joblib",
    }
    path = model_feature_paths.get(model_name)
    if path is None or not path.exists():
        return _compute_features(smiles_list)

    data = _joblib.load(path)
    feature_names = data.get("feature_names")
    if feature_names is None:
        return _compute_features(smiles_list)

    rdkit_names = [n for n in feature_names if not n.startswith("morgan_")]
    use_morgan = any(n.startswith("morgan_") for n in feature_names)

    return build_features(
        smiles_list,
        use_morgan=use_morgan,
        use_rdkit=bool(rdkit_names),
        rdkit_names=rdkit_names if rdkit_names else None,
    )


def _evaluate_tabular_model_on_fluorinated(
    model_name: str,
    fluor_df: pd.DataFrame,
) -> dict:
    """Evaluate a tabular model (XGBoost or SVR) on the fluorinated set."""
    logger.info("Evaluating %s on fluorinated validation set...", model_name)

    smiles_list = fluor_df["smiles"].tolist()

    m = get_model(model_name)
    m.load()

    X = _compute_features_for_model(smiles_list, model_name)
    logger.info(
        "  Feature matrix: %s, NaN count: %d",
        X.shape, np.isnan(X).sum(),
    )

    valid_mask = np.isfinite(X).all(axis=1)
    logger.info(
        "  Valid samples: %d / %d", valid_mask.sum(), len(smiles_list),
    )

    preds_arr = m.predict(X[valid_mask])  # (n_valid, 3) or (n_valid,) if 1 target
    if preds_arr.ndim == 1:
        preds_arr = preds_arr.reshape(-1, 1)

    full_preds = np.full((len(smiles_list), len(TARGETS)), np.nan)
    full_preds[valid_mask] = preds_arr

    results: dict = {"per_target": {}, "boiling_point": {}}

    for i, target in enumerate(TARGETS):
        y_true = fluor_df[f"{target}_lit"].values
        y_pred = full_preds[:, i]
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            results["per_target"][target] = {
                "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "n": int(mask.sum()),
            }
            continue
        yt, yp = y_true[mask], y_pred[mask]
        results["per_target"][target] = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "r2": float(r2_score(yt, yp)),
            "n": int(mask.sum()),
        }
        logger.info(
            "  %s %s: MAE=%.2f, R2=%.3f (n=%d)",
            model_name, target,
            results["per_target"][target]["mae"],
            results["per_target"][target]["r2"],
            results["per_target"][target]["n"],
        )

    bp_exp = fluor_df["T_b_experimental_K"].values
    bp_pred = np.full(len(smiles_list), np.nan)
    for i in range(len(smiles_list)):
        m_val = full_preds[i, 0]
        s_val = full_preds[i, 1]
        e_val = full_preds[i, 2]
        if np.isfinite(m_val) and np.isfinite(s_val) and np.isfinite(e_val):
            bp_pred[i] = compute_boiling_point(m_val, s_val, e_val)

    mask = np.isfinite(bp_pred) & np.isfinite(bp_exp)
    n_converged = int(mask.sum())
    if n_converged >= 2:
        bp_mae = float(mean_absolute_error(bp_exp[mask], bp_pred[mask]))
        bp_rmse = float(np.sqrt(mean_squared_error(bp_exp[mask], bp_pred[mask])))
    else:
        bp_mae = float("nan")
        bp_rmse = float("nan")

    results["boiling_point"] = {
        "mae": bp_mae,
        "rmse": bp_rmse,
        "n_converged": n_converged,
        "n_total": len(smiles_list),
        "convergence": f"{n_converged}/{len(smiles_list)}",
    }
    logger.info(
        "  %s BP: MAE=%.1f K, RMSE=%.1f K, convergence=%d/%d",
        model_name, bp_mae, bp_rmse, n_converged, len(smiles_list),
    )

    return results


def section_eval() -> dict:
    """Evaluate XGBoost and SVR on the fluorinated validation set."""
    logger.info("=" * 70)
    logger.info("EVALUATING MISSING MODELS ON FLUORINATED SET")
    logger.info("=" * 70)

    fluor_df = pd.read_csv(FLUORINATED_VAL_PATH)
    logger.info("Fluorinated set: %d compounds", len(fluor_df))

    new_fluor_results = {}
    for model_name in ["xgboost", "svm"]:
        try:
            result = _evaluate_tabular_model_on_fluorinated(model_name, fluor_df)
            m_res = result["per_target"]["m"]
            sigma_res = result["per_target"]["sigma"]
            epsk = result["per_target"]["epsilon_k"]
            bp = result["boiling_point"]
            new_fluor_results[model_name] = {
                "m_mae": m_res["mae"],
                "m_r2": m_res["r2"],
                "sigma_mae": sigma_res["mae"],
                "sigma_r2": sigma_res["r2"],
                "epsk_mae": epsk["mae"],
                "epsk_r2": epsk["r2"],
                "bp_mae": bp["mae"],
                "bp_rmse": bp["rmse"],
                "convergence": bp["convergence"],
            }
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", model_name, exc, exc_info=True)

    artifact_path = SAVED_DIR / "step51_new_fluorinated_evals.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w") as f:
        json.dump(new_fluor_results, f, indent=2)
    logger.info("Saved new fluorinated evaluations: %s", artifact_path)

    return new_fluor_results


def _build_unified_data(new_fluor: dict) -> pd.DataFrame:
    """Merge Esper + fluorinated results into a single DataFrame."""
    all_fluor = {**EXISTING_FLUOR, **new_fluor}

    rows = []
    for model_key in MODEL_ORDER:
        esper = ESPER_RESULTS.get(model_key, {})
        fluor = all_fluor.get(model_key, {})
        display = ALL_MODELS_DISPLAY.get(model_key, model_key)

        rows.append({
            "model_key": model_key,
            "Model": display,
            "Esper eps/k R2": esper.get("epsk_r2"),
            "Esper eps/k MAE (K)": esper.get("epsk_mae"),
            "Esper m R2": esper.get("m_r2"),
            "Esper sigma R2": esper.get("sigma_r2"),
            "Fluor eps/k R2": fluor.get("epsk_r2"),
            "Fluor eps/k MAE (K)": fluor.get("epsk_mae"),
            "Fluor BP MAE (K)": fluor.get("bp_mae"),
            "Fluor Convergence": fluor.get("convergence"),
            "Protocol": esper.get("protocol", "---"),
        })

    return pd.DataFrame(rows)


def section_table(new_fluor: dict) -> pd.DataFrame:
    """Build and print the unified comparison table."""
    logger.info("=" * 70)
    logger.info("UNIFIED COMPARISON TABLE")
    logger.info("=" * 70)

    df = _build_unified_data(new_fluor)

    header = (
        f"{'Model':<22} "
        f"{'Esper ε/k R²':>14} {'Esper ε/k MAE':>15} "
        f"{'Fluor ε/k R²':>14} {'Fluor ε/k MAE':>15} "
        f"{'Fluor BP MAE':>14} {'Conv':>6}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for _, row in df.iterrows():
        def _fmt(v, fmt=".2f"):
            return f"{v:{fmt}}" if v is not None and np.isfinite(v) else "---"

        logger.info(
            f"  {row['Model']:<20} "
            f"{_fmt(row['Esper eps/k R2']):>14} "
            f"{_fmt(row['Esper eps/k MAE (K)'], '.1f'):>15} "
            f"{_fmt(row['Fluor eps/k R2']):>14} "
            f"{_fmt(row['Fluor eps/k MAE (K)'], '.1f'):>15} "
            f"{_fmt(row['Fluor BP MAE (K)'], '.1f'):>14} "
            f"{row['Fluor Convergence'] or '---':>6}"
        )

    artifact_path = SAVED_DIR / "step51_unified_comparison.csv"
    df.to_csv(artifact_path, index=False)
    logger.info("Saved unified table: %s", artifact_path)

    return df


def _style_mae_panel(ax, y_pos, models, label):
    """Apply common styling to a fluorinated MAE panel."""
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(models))
    ax.set_xlabel("MAE (K)", fontsize=12)
    ax.set_title(
        f"Fluorinated External: {label}",
        fontsize=13, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)


def section_figure(new_fluor: dict) -> None:
    """Generate the consolidated multi-panel comparison figure."""
    logger.info("=" * 70)
    logger.info("GENERATING UNIFIED COMPARISON FIGURE")
    logger.info("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _build_unified_data(new_fluor)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    models = df["Model"].tolist()
    model_keys = df["model_key"].tolist()
    y_pos = np.arange(len(models))
    colors = [MODEL_COLORS.get(k, "#333") for k in model_keys]

    # --- Panel 1: Esper epsilon/k R² ---
    ax = axes[0]
    vals = df["Esper eps/k R2"].values.astype(float)
    valid = np.isfinite(vals)
    valid_colors = [c for c, v in zip(colors, valid) if v]
    ax.barh(
        y_pos[valid], vals[valid], color=valid_colors,
        edgecolor="black", linewidth=0.5, height=0.7,
    )
    for i, (v, is_valid) in enumerate(zip(vals, valid)):
        if is_valid:
            x_text = max(v, 0.02) + 0.01 if v >= 0 else v - 0.01
            ha = "left" if v >= 0 else "right"
            ax.text(
                x_text, i, f"{v:.2f}",
                va="center", ha=ha, fontsize=9, fontweight="bold",
            )
        else:
            ax.text(0.02, i, "N/A", va="center", ha="left", fontsize=9, color="gray")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("R²", fontsize=12)
    ax.set_title("Esper Test: ε/k R²", fontsize=13, fontweight="bold")
    ax.axvline(0, color="k", linewidth=0.8, alpha=0.3)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(-0.6, 0.85)

    # --- Panel 2: Fluorinated epsilon/k MAE ---
    ax = axes[1]
    vals = df["Fluor eps/k MAE (K)"].values.astype(float)
    valid = np.isfinite(vals)
    for i, (v, is_valid) in enumerate(zip(vals, valid)):
        if is_valid:
            ax.barh(
                i, v, color=colors[i],
                edgecolor="black", linewidth=0.5, height=0.7,
            )
            ax.text(
                v + 0.3, i, f"{v:.1f}", va="center",
                ha="left", fontsize=9, fontweight="bold",
            )
        else:
            ax.barh(
                i, 0, color="#eee",
                edgecolor="#ccc", linewidth=0.5, height=0.7,
            )
            ax.text(
                0.5, i, "not evaluated", va="center",
                ha="left", fontsize=8, color="gray",
                style="italic",
            )
    _style_mae_panel(ax, y_pos, models, "\u03b5/k MAE")

    # --- Panel 3: Fluorinated Boiling Point MAE ---
    ax = axes[2]
    vals = df["Fluor BP MAE (K)"].values.astype(float)
    valid = np.isfinite(vals)
    for i, (v, is_valid) in enumerate(zip(vals, valid)):
        if is_valid:
            ax.barh(
                i, v, color=colors[i],
                edgecolor="black", linewidth=0.5, height=0.7,
            )
            ax.text(
                v + 0.3, i, f"{v:.1f}", va="center",
                ha="left", fontsize=9, fontweight="bold",
            )
        else:
            ax.barh(
                i, 0, color="#eee",
                edgecolor="#ccc", linewidth=0.5, height=0.7,
            )
            ax.text(
                0.5, i, "not evaluated", va="center",
                ha="left", fontsize=8, color="gray",
                style="italic",
            )
    _style_mae_panel(ax, y_pos, models, "BP MAE")

    fig.suptitle(
        "Unified Model Comparison: Esper Internal + Fluorinated External Validation",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = FIGURES_DIR / "unified_model_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)

    _fig_detailed_heatmap(new_fluor)


def _fig_detailed_heatmap(new_fluor: dict) -> None:
    """Heatmap with all models x all metrics."""
    df = _build_unified_data(new_fluor)
    models = df["Model"].tolist()

    metrics = [
        ("Esper m R2", "Esper m\nR²"),
        ("Esper sigma R2", "Esper σ\nR²"),
        ("Esper eps/k R2", "Esper ε/k\nR²"),
        ("Esper eps/k MAE (K)", "Esper ε/k\nMAE (K)"),
        ("Fluor eps/k R2", "Fluor ε/k\nR²"),
        ("Fluor eps/k MAE (K)", "Fluor ε/k\nMAE (K)"),
        ("Fluor BP MAE (K)", "Fluor BP\nMAE (K)"),
    ]
    col_keys = [m[0] for m in metrics]
    col_labels = [m[1] for m in metrics]

    data = df[col_keys].values.astype(float)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.9 * len(models))))
    masked = np.ma.array(data, mask=np.isnan(data))
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, ha="center")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if np.isfinite(val):
                text = f"{val:.2f}" if abs(val) < 10 else f"{val:.1f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color="black")
            else:
                ax.text(j, i, "---", ha="center", va="center", fontsize=9, color="gray")

    ax.set_title(
        "All Models × All Metrics (Esper + Fluorinated)",
        fontsize=13, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, shrink=0.6, label="Metric Value")
    plt.tight_layout()
    path = FIGURES_DIR / "unified_heatmap_all_metrics.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def section_report(new_fluor: dict) -> None:
    """Update model_comparison.md with the unified table and update Step 48 report."""
    logger.info("=" * 70)
    logger.info("UPDATING REPORTS")
    logger.info("=" * 70)

    all_fluor = {**EXISTING_FLUOR, **new_fluor}
    _update_model_comparison(all_fluor)
    _update_step48_report(all_fluor)


def _update_model_comparison(all_fluor: dict) -> None:
    """Add a unified summary table to model_comparison.md."""
    mc_path = PROJECT_ROOT / "supplementary_information" / "model_comparison.md"
    content = mc_path.read_text()

    section_marker = "\n---\n\n## Unified Model Comparison (Step 51)"
    if section_marker.strip() in content:
        logger.info("Unified section already exists in model_comparison.md; replacing.")
        idx = content.index("## Unified Model Comparison (Step 51)")
        next_hr = content.find("\n---\n", idx + 10)
        if next_hr == -1:
            content = content[:idx].rstrip()
        else:
            content = content[:idx].rstrip() + content[next_hr:]

    lines = [
        "",
        "---",
        "",
        "## Unified Model Comparison (Step 51)",
        "",
        (
            "This table consolidates all model results across "
            "Esper internal and fluorinated external validation "
            "into a single reference."
        ),
        "",
        (
            "| Model | Esper \u03b5/k R\u00b2 "
            "| Esper \u03b5/k MAE (K) "
            "| Fluor \u03b5/k MAE (K) "
            "| Fluor \u03b5/k R\u00b2 "
            "| Fluor BP MAE (K) "
            "| EOS Conv. | Protocol |"
        ),
        (
            "| ----- | ------------ "
            "| ----------------- "
            "| ----------------- "
            "| ------------ "
            "| --------------- "
            "| --------- | -------- |"
        ),
    ]

    for model_key in MODEL_ORDER:
        esper = ESPER_RESULTS.get(model_key, {})
        fluor = all_fluor.get(model_key, {})
        display = ALL_MODELS_DISPLAY.get(model_key, model_key)

        def _f(v, fmt=".2f"):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "---"
            return f"{v:{fmt}}"

        lines.append(
            f"| {display} "
            f"| {_f(esper.get('epsk_r2'))} "
            f"| {_f(esper.get('epsk_mae'), '.1f')} "
            f"| {_f(fluor.get('epsk_mae'), '.1f')} "
            f"| {_f(fluor.get('epsk_r2'))} "
            f"| {_f(fluor.get('bp_mae'), '.1f')} "
            f"| {fluor.get('convergence', '---')} "
            f"| {esper.get('protocol', '---')} |"
        )

    lines.extend([
        "",
        "**Notes:**",
        (
            "- SVR and Ridge Esper metrics are means across "
            "10 repeated stratified outer splits; all others "
            "use a single 80/20 split (seed=42)."
        ),
        (
            "- GC-PC-SAFT, NN (MLP), ChemBERTa, and Ridge "
            "were not evaluated on the fluorinated set (not "
            "deployment candidates)."
        ),
        (
            "- Negative R\u00b2 on the fluorinated set is "
            "expected: 15 compounds with narrow \u03b5/k range."
        ),
        (
            "- Fluorinated BP MAE is the decisive Tier 1 "
            "metric for model selection (see Step 48)."
        ),
        "",
        "See `figures/51_unified_model_comparison/` for consolidated comparison figures.",
        "",
    ])

    content = content.rstrip() + "\n" + "\n".join(lines)
    mc_path.write_text(content)
    logger.info("Updated: %s", mc_path)


def _update_step48_report(all_fluor: dict) -> None:
    """Update Step 48 report to include XGBoost and SVR in the fluorinated tables."""
    report_path = PROJECT_ROOT / "docs" / "reports" / "48_model_selection_validation.md"
    content = report_path.read_text()

    xgb = all_fluor.get("xgboost", {})
    svm = all_fluor.get("svm", {})

    old_param_table_end = "| GNN      | 0.705 | 0.056     | 17.4              | -0.841       |"
    param_parts = content.split("### Parameter Accuracy")
    xgb_in_param = (
        len(param_parts) > 1
        and "XGBoost" in param_parts[1].split("###")[0]
    )
    if old_param_table_end in content and not xgb_in_param:
        xgb_epsk_mae = xgb.get("epsk_mae")
        xgb_epsk_r2 = xgb.get("epsk_r2")
        svm_epsk_mae = svm.get("epsk_mae")
        svm_epsk_r2 = svm.get("epsk_r2")

        if xgb_epsk_mae is not None and np.isfinite(xgb_epsk_mae):
            xgb_m_mae = xgb.get("m_mae")
            xgb_sigma_mae = xgb.get("sigma_mae")
            xgb_m_str = (
                f"{xgb_m_mae:.3f}"
                if xgb_m_mae is not None and np.isfinite(xgb_m_mae)
                else "---"
            )
            xgb_s_str = (
                f"{xgb_sigma_mae:.3f}"
                if xgb_sigma_mae is not None and np.isfinite(xgb_sigma_mae)
                else "---"
            )
            new_rows = old_param_table_end
            new_rows += (
                f"\n| XGBoost  | {xgb_m_str} | {xgb_s_str}     "
                f"| {xgb_epsk_mae:.1f}              "
                f"| {xgb_epsk_r2:.3f}       |"
            )
            if svm_epsk_mae is not None and np.isfinite(svm_epsk_mae):
                svm_m_mae = svm.get("m_mae")
                svm_sigma_mae = svm.get("sigma_mae")
                svm_m_str = (
                    f"{svm_m_mae:.3f}"
                    if svm_m_mae is not None and np.isfinite(svm_m_mae)
                    else "---"
                )
                svm_s_str = (
                    f"{svm_sigma_mae:.3f}"
                    if svm_sigma_mae is not None
                    and np.isfinite(svm_sigma_mae)
                    else "---"
                )
                new_rows += (
                    f"\n| SVR      | {svm_m_str} | {svm_s_str}     "
                    f"| {svm_epsk_mae:.1f}              "
                    f"| {svm_epsk_r2:.3f}       |"
                )
            content = content.replace(
                old_param_table_end, new_rows,
            )
            logger.info(
                "Added XGBoost/SVR to param table.",
            )

    old_bp_table_end = "| GNN      | 23.9       | 26.5        | 4/15            |"
    bp_parts = content.split("### Boiling Point Accuracy")
    xgb_in_bp = (
        len(bp_parts) > 1
        and "XGBoost" in bp_parts[1].split("###")[0]
    )
    if old_bp_table_end in content and not xgb_in_bp:
        xgb_bp = xgb.get("bp_mae")
        xgb_bp_rmse = xgb.get("bp_rmse")
        xgb_conv = xgb.get("convergence", "---")
        svm_bp = svm.get("bp_mae")
        svm_bp_rmse = svm.get("bp_rmse")
        svm_conv = svm.get("convergence", "---")

        if xgb_bp is not None and np.isfinite(xgb_bp):
            new_rows = old_bp_table_end
            new_rows += (
                f"\n| XGBoost  | {xgb_bp:.1f}       "
                f"| {xgb_bp_rmse:.1f}        "
                f"| {xgb_conv}            |"
            )
            if svm_bp is not None and np.isfinite(svm_bp):
                new_rows += (
                    f"\n| SVR      | {svm_bp:.1f}       "
                    f"| {svm_bp_rmse:.1f}        "
                    f"| {svm_conv}            |"
                )
            content = content.replace(
                old_bp_table_end, new_rows,
            )
            logger.info(
                "Added XGBoost/SVR to BP table.",
            )

    ref_line = "5. `tier_summary_heatmap.png` - key metrics across tiers"
    if "unified_model_comparison" not in content:
        content = content.replace(
            ref_line,
            ref_line
            + "\n\nSee also "
            "`figures/51_unified_model_comparison/` for the "
            "consolidated all-model comparison.",
        )

    report_path.write_text(content)
    logger.info("Updated: %s", report_path)


def main():
    parser = argparse.ArgumentParser(
        description="Step 51: Unified Model Comparison Chart"
    )
    parser.add_argument(
        "--section", type=str, default="all",
        choices=["eval", "table", "figure", "report", "all"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    new_fluor: dict = {}

    if args.section in ("eval", "all"):
        new_fluor = section_eval()

    if not new_fluor:
        artifact_path = SAVED_DIR / "step51_new_fluorinated_evals.json"
        if artifact_path.exists():
            new_fluor = json.loads(artifact_path.read_text())
            logger.info("Loaded cached fluorinated evals: %s", artifact_path)

    if args.section in ("table", "all"):
        section_table(new_fluor)

    if args.section in ("figure", "all"):
        section_figure(new_fluor)

    if args.section in ("report", "all"):
        section_report(new_fluor)

    logger.info("=" * 70)
    logger.info("STEP 51 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
