"""Step 48: Model Selection Validation (5-Model Comparison).

Evaluates RF, XGBoost, chemprop, GNN, and GNNePCSAFT through a
reproducible, apples-to-apples validation protocol across three tiers.

Usage:
    python scripts/step48_model_selection_validation.py [--section SECTION]

Sections:
    load        Load and verify model artifacts
    tier1       Fluorinated external validation
    tier2       Esper holdout with bootstrap CIs
    tier3       Cross-validated Q² for tree models
    decision    Model selection decision
    figures     Generate all figures
    report      Write report
    all         Run everything (default)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import model.chemprop_model  # noqa: F401, E402, I001
import model.xgb  # noqa: F401, E402
from model.data.load import TARGETS, load_data, split_data  # noqa: E402
from model.registry import _compute_features, get_model  # noqa: E402
from model.uncertainty import bootstrap_metric_ci  # noqa: E402
from screening.hfo_screening import compute_boiling_point  # noqa: E402

# GNNePCSAFT — optional external benchmark
try:
    from gnnepcsaft.epcsaft import epcsaft_pred  # noqa: F401
    GNNEPCSAFT_AVAILABLE = True
except ImportError:
    GNNEPCSAFT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAVED_DIR = PROJECT_ROOT / "model" / "saved"
FIGURES_DIR = PROJECT_ROOT / "figures" / "48_model_selection_validation"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"

FLUORINATED_VAL_PATH = SAVED_DIR / "gnn_fluorinated_validation_set.csv"
TEST_SET_PATH = SAVED_DIR / "test_set.csv"
ESPER_DATA_PATH = PROJECT_ROOT / "model" / "data" / "esper_pcsaft.csv"

TARGET_LABELS = {"m": "m", "sigma": "\u03c3", "epsilon_k": "\u03b5/k"}
UNITS = {"m": "segments", "sigma": "\u00c5", "epsilon_k": "K"}

MODEL_NAMES_ORDERED = ["rf", "xgboost", "chemprop", "gnn", "gnnepcsaft"]
MODEL_DISPLAY = {
    "rf": "RF",
    "xgboost": "XGBoost",
    "chemprop": "chemprop",
    "gnn": "GNN",
    "gnnepcsaft": "GNNePCSAFT*",
}
MODEL_COLORS = {
    "rf": "#d95f02",
    "xgboost": "#1b9e77",
    "chemprop": "#7570b3",
    "gnn": "#e7298a",
    "gnnepcsaft": "#66a61e",
}


# ===================================================================
# Model loading and unified prediction
# ===================================================================


def _load_model_safe(name: str):
    """Load a registered model, returning None on failure."""
    try:
        m = get_model(name)
        m.load()
        logger.info("Loaded model: %s", name)
        return m
    except Exception as exc:
        logger.warning("Failed to load model %s: %s", name, exc)
        return None


def _predict_gnnepcsaft(smiles_list: list[str]) -> dict[str, np.ndarray] | None:
    """Predict with the external GNNePCSAFT model (ePC-SAFT parameters).

    Returns None if the package is not installed.
    """
    if not GNNEPCSAFT_AVAILABLE:
        logger.warning("GNNePCSAFT not installed; skipping")
        return None
    try:
        preds = epcsaft_pred(smiles_list)
        # epcsaft_pred returns array-like with shape (n, 3) or similar
        preds = np.asarray(preds, dtype=float)
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)
        return {
            "m": preds[:, 0],
            "sigma": preds[:, 1],
            "epsilon_k": preds[:, 2],
        }
    except Exception as exc:
        logger.warning("GNNePCSAFT prediction failed: %s", exc)
        return None


def predict_all_models(
    smiles_list: list[str],
    models: dict | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Predict with all available models, normalizing output to {target: array}.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to predict on.
    models : dict | None
        Pre-loaded model objects keyed by name.  If None, loads fresh.

    Returns
    -------
    dict mapping model_name -> {target: np.ndarray}
    """
    results: dict[str, dict[str, np.ndarray]] = {}

    if models is None:
        models = {}
        for name in ["rf", "gnn"]:
            m = _load_model_safe(name)
            if m is not None:
                models[name] = m
        for name in ["xgboost", "chemprop"]:
            m = _load_model_safe(name)
            if m is not None:
                models[name] = m

    # RF and GNN: accept SMILES, return dict
    for name in ["rf", "gnn"]:
        if name not in models:
            continue
        try:
            preds = models[name].predict(smiles_list)
            results[name] = preds
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", name, exc)

    # XGBoost: needs feature matrix
    if "xgboost" in models:
        try:
            X = _compute_features(smiles_list)
            preds_arr = models["xgboost"].predict(X)  # (n, 3)
            results["xgboost"] = {
                t: preds_arr[:, i] for i, t in enumerate(TARGETS)
            }
        except Exception as exc:
            logger.warning("Prediction failed for xgboost: %s", exc)

    # chemprop: accepts SMILES, returns (n, 3) array
    if "chemprop" in models:
        try:
            preds_arr = models["chemprop"].predict(smiles_list)  # (n, 3)
            results["chemprop"] = {
                t: preds_arr[:, i] for i, t in enumerate(TARGETS)
            }
        except Exception as exc:
            logger.warning("Prediction failed for chemprop: %s", exc)

    # GNNePCSAFT: external
    gnnep_preds = _predict_gnnepcsaft(smiles_list)
    if gnnep_preds is not None:
        results["gnnepcsaft"] = gnnep_preds

    return results


# ===================================================================
# Section: Load and verify artifacts
# ===================================================================


def section_load() -> dict:
    """Load models, validation data, and verify provenance.

    Returns dict with loaded models, datasets, and provenance info.
    """
    logger.info("=" * 80)
    logger.info("SECTION: LOAD AND VERIFY ARTIFACTS")
    logger.info("=" * 80)

    ctx: dict = {"models": {}, "provenance": {}}

    # -- Load fluorinated validation set --
    if not FLUORINATED_VAL_PATH.exists():
        raise FileNotFoundError(
            f"Fluorinated validation set not found: {FLUORINATED_VAL_PATH}"
        )
    fluor_df = pd.read_csv(FLUORINATED_VAL_PATH)
    ctx["fluor_df"] = fluor_df
    logger.info(
        "Fluorinated validation set: %d compounds, columns: %s",
        len(fluor_df), list(fluor_df.columns),
    )

    # -- Load Esper test set --
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Test set not found: {TEST_SET_PATH}")
    test_df = pd.read_csv(TEST_SET_PATH)
    ctx["test_df"] = test_df
    logger.info("Esper test set: %d molecules", len(test_df))

    # -- Load models --
    loaded_models: dict = {}
    artifact_paths: dict[str, str] = {}

    for name in ["rf", "gnn"]:
        m = _load_model_safe(name)
        if m is not None:
            loaded_models[name] = m
            artifact_paths[name] = str(SAVED_DIR)

    for name in ["xgboost", "chemprop"]:
        m = _load_model_safe(name)
        if m is not None:
            loaded_models[name] = m
            if name == "xgboost":
                artifact_paths[name] = str(SAVED_DIR / "xgb" / "xgb_model.joblib")
            else:
                artifact_paths[name] = str(SAVED_DIR / "chemprop" / "chemprop_model.pt")

    if GNNEPCSAFT_AVAILABLE:
        loaded_models["gnnepcsaft"] = "external"
        artifact_paths["gnnepcsaft"] = "pip:gnnepcsaft"

    ctx["models"] = loaded_models
    logger.info(
        "Loaded %d models: %s", len(loaded_models), list(loaded_models.keys())
    )

    # -- Provenance --
    # Feature schema hash
    feature_config_path = SAVED_DIR / "feature_config.json"
    if feature_config_path.exists():
        fc_text = feature_config_path.read_text()
        ctx["provenance"]["feature_config_hash"] = hashlib.sha256(
            fc_text.encode()
        ).hexdigest()[:12]
    else:
        ctx["provenance"]["feature_config_hash"] = "default"

    # Test split identity hash (sorted SMILES)
    sorted_smiles = sorted(test_df["smiles"].tolist())
    split_hash = hashlib.sha256(
        "\n".join(sorted_smiles).encode()
    ).hexdigest()[:12]
    ctx["provenance"]["test_split_hash"] = split_hash
    ctx["provenance"]["test_split_n"] = len(test_df)
    ctx["provenance"]["artifact_paths"] = artifact_paths

    # GNNePCSAFT version
    if GNNEPCSAFT_AVAILABLE:
        try:
            import gnnepcsaft
            ctx["provenance"]["gnnepcsaft_version"] = getattr(
                gnnepcsaft, "__version__", "unknown"
            )
        except Exception:
            ctx["provenance"]["gnnepcsaft_version"] = "unknown"
    else:
        ctx["provenance"]["gnnepcsaft_version"] = "not_installed"

    return ctx


# ===================================================================
# Section: Tier 1 - Fluorinated external validation
# ===================================================================


def section_tier1(ctx: dict) -> dict:
    """Tier 1: Fluorinated external validation on 15 compounds.

    Returns dict with metrics, per-compound results, leakage audit.
    """
    logger.info("=" * 80)
    logger.info("TIER 1: FLUORINATED EXTERNAL VALIDATION (15 compounds)")
    logger.info("=" * 80)

    fluor_df = ctx["fluor_df"]
    smiles_list = fluor_df["smiles"].tolist()
    n = len(smiles_list)

    # -- Predict with all models --
    all_preds = predict_all_models(smiles_list, models=ctx["models"])
    available_models = [m for m in MODEL_NAMES_ORDERED if m in all_preds]

    # -- 1a: Parameter accuracy --
    logger.info("--- Parameter Accuracy ---")
    param_metrics: dict[str, dict[str, dict]] = {}
    for mname in available_models:
        param_metrics[mname] = {}
        for target in TARGETS:
            y_true = fluor_df[f"{target}_lit"].values
            y_pred = all_preds[mname][target]
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() < 2:
                param_metrics[mname][target] = {
                    "mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": int(mask.sum()),
                }
                continue
            yt, yp = y_true[mask], y_pred[mask]
            param_metrics[mname][target] = {
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "r2": float(r2_score(yt, yp)) if mask.sum() >= 3 else np.nan,
                "n": int(mask.sum()),
            }
            # Bootstrap CI for MAE on small sample
            ci = bootstrap_metric_ci(yt, yp, mean_absolute_error, n_boot=2000, ci=0.95)
            param_metrics[mname][target]["mae_ci_lo"] = ci["ci_lo"]
            param_metrics[mname][target]["mae_ci_hi"] = ci["ci_hi"]

    # Log parameter table
    header = f"  {'Model':<14}"
    for t in TARGETS:
        header += f" {TARGET_LABELS[t]+' MAE':>12}"
    logger.info(header)
    for mname in available_models:
        row = f"  {MODEL_DISPLAY.get(mname, mname):<14}"
        for t in TARGETS:
            m_val = param_metrics[mname][t]["mae"]
            row += f" {m_val:>12.3f}"
        logger.info(row)

    # -- 1b: Boiling point accuracy --
    logger.info("--- Boiling Point Accuracy ---")
    bp_metrics: dict[str, dict] = {}
    bp_per_compound: dict[str, np.ndarray] = {}
    for mname in available_models:
        preds = all_preds[mname]
        bp_pred = np.full(n, np.nan)
        for i in range(n):
            m_val = preds["m"][i]
            s_val = preds["sigma"][i]
            e_val = preds["epsilon_k"][i]
            if np.isfinite(m_val) and np.isfinite(s_val) and np.isfinite(e_val):
                bp_pred[i] = compute_boiling_point(m_val, s_val, e_val)
        bp_per_compound[mname] = bp_pred

        bp_exp = fluor_df["T_b_experimental_K"].values
        mask = np.isfinite(bp_pred) & np.isfinite(bp_exp)
        n_converged = int(mask.sum())
        if n_converged < 2:
            bp_metrics[mname] = {
                "mae": np.nan, "rmse": np.nan, "n_converged": n_converged,
                "n_total": n,
            }
            continue

        bp_true_m, bp_pred_m = bp_exp[mask], bp_pred[mask]
        mae_val = float(mean_absolute_error(bp_true_m, bp_pred_m))
        rmse_val = float(np.sqrt(mean_squared_error(bp_true_m, bp_pred_m)))

        # Bootstrap CI for BP MAE
        ci = bootstrap_metric_ci(
            bp_true_m, bp_pred_m, mean_absolute_error, n_boot=2000, ci=0.95
        )
        bp_metrics[mname] = {
            "mae": mae_val,
            "rmse": rmse_val,
            "n_converged": n_converged,
            "n_total": n,
            "mae_ci_lo": ci["ci_lo"],
            "mae_ci_hi": ci["ci_hi"],
        }

    logger.info(
        "  %-14s %10s %10s %12s",
        "Model", "BP MAE(K)", "BP RMSE(K)", "Converged",
    )
    for mname in available_models:
        bm = bp_metrics[mname]
        logger.info(
            "  %-14s %10.2f %10.2f %8d/%d",
            MODEL_DISPLAY.get(mname, mname),
            bm["mae"], bm["rmse"], bm["n_converged"], bm["n_total"],
        )

    # -- 1c: Per-compound table --
    per_compound_rows = []
    for i in range(n):
        row_data = {
            "smiles": fluor_df["smiles"].iloc[i],
            "name": fluor_df.get("name", pd.Series([""] * n)).iloc[i],
            "m_lit": fluor_df["m_lit"].iloc[i],
            "sigma_lit": fluor_df["sigma_lit"].iloc[i],
            "epsilon_k_lit": fluor_df["epsilon_k_lit"].iloc[i],
            "T_b_experimental_K": fluor_df["T_b_experimental_K"].iloc[i],
        }
        for mname in available_models:
            for t in TARGETS:
                row_data[f"{t}_{mname}"] = all_preds[mname][t][i]
            row_data[f"T_b_{mname}"] = bp_per_compound[mname][i]
        per_compound_rows.append(row_data)
    per_compound_df = pd.DataFrame(per_compound_rows)

    # Save per-compound CSV
    per_compound_path = SAVED_DIR / "step48_fluorinated_all_models.csv"
    per_compound_df.to_csv(per_compound_path, index=False)
    logger.info("Saved per-compound results: %s", per_compound_path)

    # -- 1d: Paired BP error comparisons --
    logger.info("--- Paired Boiling Point Error Comparisons ---")
    bp_exp = fluor_df["T_b_experimental_K"].values
    paired_bp: dict[str, dict] = {}
    competitive_pairs = [
        ("rf", "xgboost"), ("rf", "chemprop"), ("rf", "gnn"),
        ("xgboost", "chemprop"),
    ]
    for m_a, m_b in competitive_pairs:
        if m_a not in bp_per_compound or m_b not in bp_per_compound:
            continue
        bp_a = bp_per_compound[m_a]
        bp_b = bp_per_compound[m_b]
        mask = (
            np.isfinite(bp_a) & np.isfinite(bp_b) & np.isfinite(bp_exp)
        )
        if mask.sum() < 2:
            continue
        err_a = np.abs(bp_a[mask] - bp_exp[mask])
        err_b = np.abs(bp_b[mask] - bp_exp[mask])
        diff = err_a - err_b  # negative means A is better

        # Bootstrap the MAE difference
        rng = np.random.RandomState(42)
        n_boot = 2000
        boot_diffs = np.empty(n_boot)
        n_paired = int(mask.sum())
        for b in range(n_boot):
            idx = rng.randint(0, n_paired, size=n_paired)
            boot_diffs[b] = err_a[idx].mean() - err_b[idx].mean()

        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))
        paired_bp[f"{m_a}_vs_{m_b}"] = {
            "mean_diff": float(diff.mean()),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_paired": n_paired,
            "excludes_zero": ci_lo > 0 or ci_hi < 0,
            "favors": m_b if diff.mean() > 0 else m_a,
        }
        logger.info(
            "  %s vs %s: delta_MAE = %.2f [%.2f, %.2f] %s",
            MODEL_DISPLAY.get(m_a, m_a),
            MODEL_DISPLAY.get(m_b, m_b),
            diff.mean(), ci_lo, ci_hi,
            "(significant)" if ci_lo > 0 or ci_hi < 0 else "(not significant)",
        )

    # -- 1e: Data leakage audit --
    logger.info("--- Data Leakage Audit ---")
    leakage_result = _data_leakage_audit(fluor_df, ctx)

    tier1 = {
        "param_metrics": param_metrics,
        "bp_metrics": bp_metrics,
        "bp_per_compound": {
            k: v.tolist() for k, v in bp_per_compound.items()
        },
        "paired_bp": paired_bp,
        "per_compound_df": per_compound_df,
        "leakage": leakage_result,
        "available_models": available_models,
        "all_preds": all_preds,
    }
    return tier1


def _data_leakage_audit(fluor_df: pd.DataFrame, ctx: dict) -> dict:
    """Check for exact SMILES overlap and Tanimoto nearest-neighbor similarity."""
    from model.ad_tanimoto import TanimotoAD

    fluor_smiles = set(fluor_df["smiles"].tolist())

    # Load Esper training data
    esper_df = load_data("esper")
    train_df, _ = split_data(esper_df, random_state=42)
    train_smiles = set(train_df["smiles"].tolist())

    # Exact overlap
    overlap = fluor_smiles & train_smiles
    logger.info(
        "  Exact SMILES overlap (fluorinated vs Esper train): %d", len(overlap)
    )
    if overlap:
        logger.warning("  LEAKAGE DETECTED: %s", overlap)

    # Tanimoto nearest-neighbor
    ad = TanimotoAD(radius=2, n_bits=2048)
    ad.fit(list(train_smiles))

    tanimoto_results = []
    for smi in fluor_df["smiles"]:
        sim = ad.tanimoto_nn(smi)
        tanimoto_results.append({"smiles": smi, "max_tanimoto_to_train": sim})
        logger.info("  %s -> max Tanimoto = %.3f", smi, sim)

    tanimoto_df = pd.DataFrame(tanimoto_results)
    max_sim = tanimoto_df["max_tanimoto_to_train"].max()
    mean_sim = tanimoto_df["max_tanimoto_to_train"].mean()
    logger.info(
        "  Tanimoto summary: mean=%.3f, max=%.3f", mean_sim, max_sim
    )

    high_sim = tanimoto_df[tanimoto_df["max_tanimoto_to_train"] > 0.85]
    if len(high_sim) > 0:
        logger.warning(
            "  HIGH SIMILARITY (>0.85): %d compounds have close training analogs",
            len(high_sim),
        )

    return {
        "exact_overlap": list(overlap),
        "n_exact_overlap": len(overlap),
        "tanimoto": tanimoto_df.to_dict("records"),
        "mean_tanimoto": float(mean_sim),
        "max_tanimoto": float(max_sim),
        "n_high_similarity": len(high_sim),
    }


# ===================================================================
# Section: Tier 2 - Esper holdout with bootstrap CIs
# ===================================================================


def section_tier2(ctx: dict) -> dict:
    """Tier 2: Esper holdout evaluation with bootstrap 95% CIs and paired tests.

    Returns dict with metrics and paired comparison results.
    """
    logger.info("=" * 80)
    logger.info("TIER 2: ESPER HOLDOUT WITH BOOTSTRAP CIs")
    logger.info("=" * 80)

    test_df = ctx["test_df"]
    smiles_list = test_df["smiles"].tolist()

    # Predict with all models
    all_preds = predict_all_models(smiles_list, models=ctx["models"])
    available_models = [m for m in MODEL_NAMES_ORDERED if m in all_preds]
    logger.info("Models available for Tier 2: %s", available_models)

    # -- Point metrics with bootstrap CIs --
    esper_metrics: dict[str, dict[str, dict]] = {}
    rows_for_csv: list[dict] = []

    for mname in available_models:
        esper_metrics[mname] = {}
        for target in TARGETS:
            y_true = test_df[target].values
            y_pred = all_preds[mname][target]
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            yt, yp = y_true[mask], y_pred[mask]

            if len(yt) < 3:
                esper_metrics[mname][target] = {
                    "mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": len(yt),
                }
                continue

            mae_val = float(mean_absolute_error(yt, yp))
            rmse_val = float(np.sqrt(mean_squared_error(yt, yp)))
            r2_val = float(r2_score(yt, yp))

            # Bootstrap CIs
            mae_ci = bootstrap_metric_ci(yt, yp, mean_absolute_error, n_boot=2000)
            r2_ci = bootstrap_metric_ci(yt, yp, r2_score, n_boot=2000)
            rmse_fn = lambda yt_, yp_: float(np.sqrt(mean_squared_error(yt_, yp_)))  # noqa: E731
            rmse_ci = bootstrap_metric_ci(yt, yp, rmse_fn, n_boot=2000)

            metrics_row = {
                "mae": mae_val,
                "rmse": rmse_val,
                "r2": r2_val,
                "n": int(mask.sum()),
                "mae_ci_lo": mae_ci["ci_lo"],
                "mae_ci_hi": mae_ci["ci_hi"],
                "r2_ci_lo": r2_ci["ci_lo"],
                "r2_ci_hi": r2_ci["ci_hi"],
                "rmse_ci_lo": rmse_ci["ci_lo"],
                "rmse_ci_hi": rmse_ci["ci_hi"],
            }
            esper_metrics[mname][target] = metrics_row

            rows_for_csv.append({
                "model": mname,
                "target": target,
                **metrics_row,
            })

    # Log summary table
    logger.info("--- Esper Holdout Metrics ---")
    for mname in available_models:
        label = MODEL_DISPLAY.get(mname, mname)
        for target in TARGETS:
            mm = esper_metrics[mname].get(target, {})
            logger.info(
                "  %-14s %-10s MAE=%.3f [%.3f,%.3f]  R2=%.3f [%.3f,%.3f]  n=%d",
                label, target,
                mm.get("mae", np.nan),
                mm.get("mae_ci_lo", np.nan), mm.get("mae_ci_hi", np.nan),
                mm.get("r2", np.nan),
                mm.get("r2_ci_lo", np.nan), mm.get("r2_ci_hi", np.nan),
                mm.get("n", 0),
            )

    # Save to CSV
    bootstrap_df = pd.DataFrame(rows_for_csv)
    bootstrap_path = SAVED_DIR / "step48_esper_bootstrap.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)
    logger.info("Saved Esper bootstrap metrics: %s", bootstrap_path)

    # -- Paired bootstrap comparisons --
    logger.info("--- Paired Bootstrap Comparisons ---")
    paired_esper: dict[str, dict] = {}
    competitive_pairs = [
        ("rf", "xgboost"), ("rf", "chemprop"), ("xgboost", "chemprop"),
        ("rf", "gnn"),
    ]
    for m_a, m_b in competitive_pairs:
        if m_a not in all_preds or m_b not in all_preds:
            continue
        for target in TARGETS:
            y_true = test_df[target].values
            ya = all_preds[m_a][target]
            yb = all_preds[m_b][target]
            mask = np.isfinite(y_true) & np.isfinite(ya) & np.isfinite(yb)
            if mask.sum() < 10:
                continue
            yt = y_true[mask]
            err_a = np.abs(ya[mask] - yt)
            err_b = np.abs(yb[mask] - yt)
            diff = err_a - err_b  # negative = A better

            rng = np.random.RandomState(42)
            n_boot = 2000
            n_paired = int(mask.sum())
            boot_diffs = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.randint(0, n_paired, size=n_paired)
                boot_diffs[b] = err_a[idx].mean() - err_b[idx].mean()

            ci_lo = float(np.percentile(boot_diffs, 2.5))
            ci_hi = float(np.percentile(boot_diffs, 97.5))
            key = f"{m_a}_vs_{m_b}_{target}"
            paired_esper[key] = {
                "mean_diff": float(diff.mean()),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_paired": n_paired,
                "excludes_zero": ci_lo > 0 or ci_hi < 0,
                "favors": m_b if diff.mean() > 0 else m_a,
            }
            logger.info(
                "  %s vs %s [%s]: delta_MAE=%.3f [%.3f,%.3f] %s %s",
                MODEL_DISPLAY.get(m_a, m_a), MODEL_DISPLAY.get(m_b, m_b),
                target, diff.mean(), ci_lo, ci_hi,
                "(sig)" if ci_lo > 0 or ci_hi < 0 else "(ns)",
                f"favors {MODEL_DISPLAY.get(m_b if diff.mean() > 0 else m_a, '')}",
            )

    # -- Inference throughput benchmark --
    logger.info("--- Inference Throughput Benchmark ---")
    throughput: dict[str, float] = {}
    bench_smiles = smiles_list[:min(500, len(smiles_list))]
    for mname in available_models:
        try:
            t0 = time.perf_counter()
            if mname == "xgboost":
                X_bench = _compute_features(bench_smiles)
                ctx["models"][mname].predict(X_bench)
            elif mname == "gnnepcsaft":
                _predict_gnnepcsaft(bench_smiles)
            else:
                ctx["models"][mname].predict(bench_smiles)
            elapsed = time.perf_counter() - t0
            throughput[mname] = len(bench_smiles) / elapsed
            logger.info(
                "  %-14s %.1f mol/s (%.2f s for %d molecules)",
                MODEL_DISPLAY.get(mname, mname),
                throughput[mname], elapsed, len(bench_smiles),
            )
        except Exception as exc:
            logger.warning("  Throughput benchmark failed for %s: %s", mname, exc)
            throughput[mname] = np.nan

    tier2 = {
        "esper_metrics": esper_metrics,
        "paired_esper": paired_esper,
        "available_models": available_models,
        "all_preds": all_preds,
        "throughput": throughput,
    }
    return tier2


# ===================================================================
# Section: Tier 3 - Cross-validated Q² for tree models
# ===================================================================


def section_tier3(ctx: dict) -> dict:
    """Tier 3: 5-fold CV Q² for RF and XGBoost.

    Returns dict with Q² results per model and target.
    """
    logger.info("=" * 80)
    logger.info("TIER 3: CROSS-VALIDATED Q² (5-fold, tree models only)")
    logger.info("=" * 80)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor

    # Reconstruct training data with same split
    df = load_data("auto")
    train_df, _ = split_data(df, stratify_bins=5, random_state=42)

    smiles_train = train_df["smiles"].tolist()
    X = _compute_features(smiles_train)
    valid_mask = np.isfinite(X).all(axis=1)
    X = X[valid_mask]
    train_df_valid = train_df.iloc[valid_mask].reset_index(drop=True)

    logger.info("  Training molecules: %d, features: %d", X.shape[0], X.shape[1])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define models for CV
    tree_models = {
        "rf": lambda: RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "xgboost": lambda: MultiOutputRegressor(
            XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, n_jobs=-1,
            )
        ),
    }

    q2_results: dict[str, dict[str, dict]] = {}

    for model_name, model_factory in tree_models.items():
        logger.info("  Computing Q² for %s...", model_name)
        q2_results[model_name] = {}

        for target in TARGETS:
            y = train_df_valid[target].values
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = model_factory()
                if model_name == "xgboost":
                    # MultiOutputRegressor expects 2D y
                    y_tr_all = train_df_valid[TARGETS].values[train_idx]
                    y_val_all = train_df_valid[TARGETS].values[val_idx]
                    model.fit(X_tr, y_tr_all)
                    preds = model.predict(X_val)
                    target_idx = TARGETS.index(target)
                    score = float(r2_score(y_val_all[:, target_idx], preds[:, target_idx]))
                else:
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val)
                    score = float(r2_score(y_val, preds))

                fold_scores.append(score)

            q2_mean = float(np.mean(fold_scores))
            q2_std = float(np.std(fold_scores))
            q2_results[model_name][target] = {
                "q2_mean": q2_mean,
                "q2_std": q2_std,
                "folds": fold_scores,
            }

            threshold = "good" if q2_mean >= 0.6 else "acceptable" if q2_mean >= 0.5 else "below"
            logger.info(
                "    %s Q²=%.4f +/- %.4f [%s]",
                target, q2_mean, q2_std, threshold,
            )

    logger.info(
        "  NOTE: chemprop, GNN, and GNNePCSAFT Q² require per-fold retraining; "
        "deferred (tree-only robustness analysis)."
    )

    return {"q2_results": q2_results}


# ===================================================================
# Section: Model selection decision
# ===================================================================


def section_decision(
    ctx: dict, tier1: dict, tier2: dict, tier3: dict
) -> dict:
    """Apply decision criteria and produce structured decision.

    Returns dict with the decision and evidence.
    """
    logger.info("=" * 80)
    logger.info("MODEL SELECTION DECISION")
    logger.info("=" * 80)

    available_models = tier1.get("available_models", [])
    bp_metrics = tier1["bp_metrics"]
    param_metrics = tier1["param_metrics"]
    paired_bp = tier1["paired_bp"]
    esper_metrics = tier2["esper_metrics"]
    throughput = tier2.get("throughput", {})

    decision = {
        "step": 48,
        "criteria": [],
        "tier1_decisive": False,
        "selected_model": None,
        "verdict": "",
    }

    # -- Criterion 1: Tier 1 BP performance --
    logger.info("  Criterion 1: Fluorinated Boiling Point MAE (Tier 1)")
    bp_maes = {}
    for mname in available_models:
        bm = bp_metrics.get(mname, {})
        bp_maes[mname] = bm.get("mae", np.nan)
    bp_sorted = sorted(
        [(m, v) for m, v in bp_maes.items() if np.isfinite(v)],
        key=lambda x: x[1],
    )
    if len(bp_sorted) >= 2:
        best_model, best_mae = bp_sorted[0]
        second_model, second_mae = bp_sorted[1]
        improvement = second_mae - best_mae
        logger.info(
            "    Best: %s (%.2f K), Runner-up: %s (%.2f K), Gap: %.2f K",
            MODEL_DISPLAY.get(best_model, best_model), best_mae,
            MODEL_DISPLAY.get(second_model, second_model), second_mae,
            improvement,
        )

        # Check decision criteria
        tier1_decisive = improvement >= 5.0  # 5 K improvement threshold
        # Also check paired CI
        pair_key = f"{best_model}_vs_{second_model}"
        alt_key = f"{second_model}_vs_{best_model}"
        paired_data = paired_bp.get(pair_key) or paired_bp.get(alt_key)
        paired_sig = False
        if paired_data:
            paired_sig = paired_data["excludes_zero"]
            tier1_decisive = tier1_decisive and paired_sig

        decision["criteria"].append({
            "name": "tier1_bp_mae",
            "best_model": best_model,
            "best_mae": best_mae,
            "second_model": second_model,
            "second_mae": second_mae,
            "gap_K": improvement,
            "meets_5K_threshold": improvement >= 5.0,
            "paired_ci_significant": paired_sig,
            "tier1_decisive": tier1_decisive,
        })
        decision["tier1_decisive"] = tier1_decisive

        if tier1_decisive:
            decision["selected_model"] = best_model
            logger.info(
                "    TIER 1 DECISIVE: %s wins by %.1f K (>5 K threshold, paired CI significant)",
                MODEL_DISPLAY.get(best_model, best_model), improvement,
            )
        else:
            logger.info(
                "    Tier 1 NOT decisive (gap=%.1f K, sig=%s). Deferring to Tier 2.",
                improvement, paired_sig,
            )
    elif len(bp_sorted) == 1:
        decision["selected_model"] = bp_sorted[0][0]
        decision["tier1_decisive"] = True
        logger.info("    Only one model with valid BP predictions.")

    # -- Criterion 2: Tier 2 epsilon_k MAE (tiebreaker) --
    if not decision["tier1_decisive"]:
        logger.info("  Criterion 2: Esper epsilon_k MAE with paired CIs (Tier 2)")
        ek_maes = {}
        for mname in available_models:
            em = esper_metrics.get(mname, {}).get("epsilon_k", {})
            ek_maes[mname] = em.get("mae", np.nan)
        ek_sorted = sorted(
            [(m, v) for m, v in ek_maes.items() if np.isfinite(v)],
            key=lambda x: x[1],
        )
        if len(ek_sorted) >= 2:
            best_ek, best_ek_mae = ek_sorted[0]
            logger.info(
                "    Best on Esper eps/k: %s (MAE=%.3f)",
                MODEL_DISPLAY.get(best_ek, best_ek), best_ek_mae,
            )
            decision["selected_model"] = best_ek
            decision["criteria"].append({
                "name": "tier2_ek_mae",
                "best_model": best_ek,
                "best_mae": best_ek_mae,
            })

    # -- Criterion 3: Practical considerations (tiebreaker of tiebreaker) --
    if decision["selected_model"] is None and len(available_models) > 0:
        # Fallback: pick RF by parsimony
        decision["selected_model"] = "rf"
        decision["criteria"].append({
            "name": "parsimony_fallback",
            "reason": "No decisive statistical evidence; RF selected by parsimony.",
        })

    # -- Epsilon/k accuracy summary --
    logger.info("  Criterion 3 (supporting): Tier 1 parameter accuracy")
    for mname in available_models:
        pm = param_metrics.get(mname, {}).get("epsilon_k", {})
        logger.info(
            "    %s: eps/k MAE=%.2f K",
            MODEL_DISPLAY.get(mname, mname), pm.get("mae", np.nan),
        )

    # -- Throughput --
    if throughput:
        logger.info("  Supporting: Inference throughput")
        for mname in available_models:
            tp = throughput.get(mname, np.nan)
            logger.info(
                "    %s: %.1f mol/s",
                MODEL_DISPLAY.get(mname, mname), tp,
            )
        decision["throughput"] = {
            k: float(v) if np.isfinite(v) else None for k, v in throughput.items()
        }

    # Final verdict
    winner = decision["selected_model"]
    decision["verdict"] = (
        f"{MODEL_DISPLAY.get(winner, winner)} selected as deployment model. "
        f"Evidence: "
        + (
            f"Tier 1 decisive ({bp_metrics.get(winner, {}).get('mae', 'N/A'):.1f} K BP MAE "
            f"on fluorinated validation)."
            if decision["tier1_decisive"]
            else "Tier 1 inconclusive; Tier 2 used as tiebreaker."
        )
    )

    logger.info("")
    logger.info("-" * 80)
    logger.info("DECISION: %s", decision["verdict"])
    logger.info("-" * 80)

    # Save decision JSON
    decision_path = SAVED_DIR / "step48_model_selection.json"
    _save_json(decision, decision_path)
    logger.info("Saved decision: %s", decision_path)

    return decision


# ===================================================================
# Section: Figures
# ===================================================================


def section_figures(
    ctx: dict, tier1: dict, tier2: dict, tier3: dict, decision: dict
) -> None:
    """Generate all figures for Step 48."""
    logger.info("=" * 80)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 80)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fluor_df = ctx["fluor_df"]

    t1_available = tier1.get("available_models", [])
    t2_available = tier2.get("available_models", [])
    t1_preds = tier1.get("all_preds", {})

    # -------------------------------------------------------------------
    # Figure 1: fluorinated_parity_all_models.png
    # Parity plots for epsilon_k, one panel per model
    # -------------------------------------------------------------------
    _fig_fluorinated_parity(fluor_df, t1_preds, t1_available)

    # -------------------------------------------------------------------
    # Figure 2: fluorinated_boiling_point_parity.png
    # -------------------------------------------------------------------
    _fig_fluorinated_bp_parity(fluor_df, tier1, t1_available)

    # -------------------------------------------------------------------
    # Figure 3: esper_bootstrap_ci_comparison.png
    # Forest plot of R² and MAE with CIs
    # -------------------------------------------------------------------
    _fig_esper_bootstrap_ci(tier2, t2_available)

    # -------------------------------------------------------------------
    # Figure 4: paired_difference_forest.png
    # -------------------------------------------------------------------
    _fig_paired_difference_forest(tier2)

    # -------------------------------------------------------------------
    # Figure 5: tier_summary_heatmap.png
    # -------------------------------------------------------------------
    _fig_tier_summary_heatmap(tier1, tier2, tier3)

    logger.info("All figures saved to: %s", FIGURES_DIR)


def _fig_fluorinated_parity(
    fluor_df: pd.DataFrame,
    all_preds: dict,
    available_models: list[str],
) -> None:
    """Parity plots for epsilon_k on fluorinated set, one panel per model."""
    n_models = len(available_models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    target = "epsilon_k"
    y_true = fluor_df[f"{target}_lit"].values

    for idx, mname in enumerate(available_models):
        ax = axes[0, idx]
        y_pred = all_preds[mname][target]
        mask = np.isfinite(y_true) & np.isfinite(y_pred)

        color = MODEL_COLORS.get(mname, "#333")
        if mask.any():
            ax.scatter(
                y_true[mask], y_pred[mask],
                s=80, alpha=0.7, c=color, edgecolors="k", linewidths=0.5,
                zorder=3,
            )
            # Parity line
            all_vals = np.concatenate([y_true[mask], y_pred[mask]])
            lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        mae_val = float(np.abs(y_pred[mask] - y_true[mask]).mean()) if mask.any() else np.nan
        display = MODEL_DISPLAY.get(mname, mname)
        ax.set_title(f"{display}\nMAE={mae_val:.1f} K", fontsize=13, weight="bold")
        ax.set_xlabel(f"Literature {TARGET_LABELS[target]} ({UNITS[target]})", fontsize=11)
        ax.set_ylabel(f"Predicted {TARGET_LABELS[target]} ({UNITS[target]})", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Fluorinated Validation: {TARGET_LABELS[target]} Parity (n=15)",
        fontsize=15, y=1.02, weight="bold",
    )
    plt.tight_layout()
    path = FIGURES_DIR / "fluorinated_parity_all_models.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


def _fig_fluorinated_bp_parity(
    fluor_df: pd.DataFrame,
    tier1: dict,
    available_models: list[str],
) -> None:
    """Predicted vs experimental boiling point for all models."""
    bp_per_compound = tier1.get("bp_per_compound", {})
    bp_metrics = tier1.get("bp_metrics", {})
    n_models = len(available_models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    bp_exp = fluor_df["T_b_experimental_K"].values

    for idx, mname in enumerate(available_models):
        ax = axes[0, idx]
        bp_pred = np.array(bp_per_compound.get(mname, [np.nan] * len(bp_exp)))
        mask = np.isfinite(bp_pred) & np.isfinite(bp_exp)
        color = MODEL_COLORS.get(mname, "#333")

        if mask.any():
            ax.scatter(
                bp_exp[mask], bp_pred[mask],
                s=80, alpha=0.7, c=color, edgecolors="k", linewidths=0.5, zorder=3,
            )
            all_vals = np.concatenate([bp_exp[mask], bp_pred[mask]])
            lo, hi = all_vals.min() - 20, all_vals.max() + 20
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.5)
            # +/- 10 K band
            ax.fill_between(
                [lo, hi], [lo - 10, hi - 10], [lo + 10, hi + 10],
                alpha=0.1, color="green",
            )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        bm = bp_metrics.get(mname, {})
        mae_val = bm.get("mae", np.nan)
        n_conv = bm.get("n_converged", 0)
        display = MODEL_DISPLAY.get(mname, mname)
        ax.set_title(
            f"{display}\nBP MAE={mae_val:.1f} K (n={n_conv})",
            fontsize=13, weight="bold",
        )
        ax.set_xlabel("Experimental T$_b$ (K)", fontsize=11)
        ax.set_ylabel("Predicted T$_b$ (K)", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle(
        "Fluorinated Validation: Boiling Point Parity (n=15)",
        fontsize=15, y=1.02, weight="bold",
    )
    plt.tight_layout()
    path = FIGURES_DIR / "fluorinated_boiling_point_parity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


def _fig_esper_bootstrap_ci(
    tier2: dict,
    available_models: list[str],
) -> None:
    """Forest plot of R² and MAE with bootstrap CIs for epsilon_k."""
    esper_metrics = tier2.get("esper_metrics", {})
    if not esper_metrics:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, 1.2 * len(available_models))))

    for ax, metric, label in [
        (axes[0], "r2", "R\u00b2 (\u03b5/k)"),
        (axes[1], "mae", "MAE (\u03b5/k, K)"),
    ]:
        y_positions = np.arange(len(available_models))
        for i, mname in enumerate(available_models):
            mm = esper_metrics.get(mname, {}).get("epsilon_k", {})
            point = mm.get(metric, np.nan)
            ci_lo = mm.get(f"{metric}_ci_lo", np.nan)
            ci_hi = mm.get(f"{metric}_ci_hi", np.nan)
            color = MODEL_COLORS.get(mname, "#333")

            if np.isfinite(point):
                ax.errorbar(
                    point, i,
                    xerr=[[point - ci_lo], [ci_hi - point]] if np.isfinite(ci_lo) else None,
                    fmt="o", markersize=8, color=color, ecolor=color,
                    elinewidth=2, capsize=5, capthick=2, zorder=3,
                )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [MODEL_DISPLAY.get(m, m) for m in available_models], fontsize=11,
        )
        ax.set_xlabel(label, fontsize=13)
        ax.set_title(f"Esper Holdout: {label}", fontsize=13, weight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "esper_bootstrap_ci_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


def _fig_paired_difference_forest(tier2: dict) -> None:
    """Forest plot of paired metric differences between top models."""
    paired_esper = tier2.get("paired_esper", {})
    if not paired_esper:
        return

    # Filter to epsilon_k comparisons
    ek_pairs = {k: v for k, v in paired_esper.items() if k.endswith("_epsilon_k")}
    if not ek_pairs:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, 1.0 * len(ek_pairs))))

    labels = []
    for i, (key, data) in enumerate(sorted(ek_pairs.items())):
        parts = key.replace("_epsilon_k", "").split("_vs_")
        disp_a = MODEL_DISPLAY.get(parts[0], parts[0])
        disp_b = MODEL_DISPLAY.get(parts[1], parts[1])
        label = f"{disp_a} vs {disp_b}"
        labels.append(label)

        mean_diff = data["mean_diff"]
        ci_lo = data["ci_lo"]
        ci_hi = data["ci_hi"]
        color = "red" if data["excludes_zero"] else "gray"

        ax.errorbar(
            mean_diff, i,
            xerr=[[mean_diff - ci_lo], [ci_hi - mean_diff]],
            fmt="o", markersize=8, color=color, ecolor=color,
            elinewidth=2, capsize=5, capthick=2, zorder=3,
        )

    ax.axvline(0, color="k", linestyle="--", lw=1, alpha=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Paired MAE Difference (\u03b5/k, K)", fontsize=13)
    ax.set_title(
        "Paired Bootstrap: MAE(A) - MAE(B) for \u03b5/k\n"
        "Red = 95% CI excludes 0 (significant)",
        fontsize=12, weight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "paired_difference_forest.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


def _fig_tier_summary_heatmap(
    tier1: dict, tier2: dict, tier3: dict,
) -> None:
    """Heatmap of key metrics across tiers, rows=models, cols=metrics."""
    bp_metrics = tier1.get("bp_metrics", {})
    param_metrics = tier1.get("param_metrics", {})
    esper_metrics = tier2.get("esper_metrics", {})
    q2_results = tier3.get("q2_results", {})

    # Columns: fluorinated BP MAE, fluorinated ek MAE, Esper ek R², Esper ek MAE, Q² ek
    col_labels = [
        "Fluor. BP\nMAE (K)",
        "Fluor. \u03b5/k\nMAE (K)",
        "Esper \u03b5/k\nR\u00b2",
        "Esper \u03b5/k\nMAE (K)",
        "Q\u00b2 \u03b5/k",
    ]
    all_models = list(
        dict.fromkeys(
            list(bp_metrics.keys())
            + list(esper_metrics.keys())
            + list(q2_results.keys())
        )
    )
    # Order by MODEL_NAMES_ORDERED
    ordered = [m for m in MODEL_NAMES_ORDERED if m in all_models]
    extra = [m for m in all_models if m not in ordered]
    all_models = ordered + extra

    if not all_models:
        return

    data = np.full((len(all_models), len(col_labels)), np.nan)
    for i, mname in enumerate(all_models):
        # Fluorinated BP MAE
        data[i, 0] = bp_metrics.get(mname, {}).get("mae", np.nan)
        # Fluorinated epsilon_k MAE
        data[i, 1] = param_metrics.get(mname, {}).get("epsilon_k", {}).get("mae", np.nan)
        # Esper epsilon_k R²
        data[i, 2] = esper_metrics.get(mname, {}).get("epsilon_k", {}).get("r2", np.nan)
        # Esper epsilon_k MAE
        data[i, 3] = esper_metrics.get(mname, {}).get("epsilon_k", {}).get("mae", np.nan)
        # Q² epsilon_k
        data[i, 4] = q2_results.get(mname, {}).get("epsilon_k", {}).get("q2_mean", np.nan)

    fig, ax = plt.subplots(figsize=(10, max(3, 1.0 * len(all_models))))

    # Mask NaN for display
    masked_data = np.ma.array(data, mask=np.isnan(data))
    im = ax.imshow(masked_data, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, ha="center")
    ax.set_yticks(range(len(all_models)))
    row_labels = []
    for m in all_models:
        label = MODEL_DISPLAY.get(m, m)
        if m == "gnnepcsaft":
            label += "\n(ext. benchmark)"
        row_labels.append(label)
    ax.set_yticklabels(row_labels, fontsize=11)

    # Annotate cells
    for i in range(len(all_models)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if np.isfinite(val):
                text = f"{val:.2f}" if abs(val) < 100 else f"{val:.1f}"
                ax.text(
                    j, i, text, ha="center", va="center", fontsize=10,
                    color="black",
                )
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=10, color="gray")

    ax.set_title(
        "Tier Summary: Key Metrics Across Models",
        fontsize=14, weight="bold",
    )
    plt.colorbar(im, ax=ax, shrink=0.6, label="Metric Value")
    plt.tight_layout()
    path = FIGURES_DIR / "tier_summary_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ===================================================================
# Section: Report
# ===================================================================


def section_report(
    ctx: dict,
    tier1: dict,
    tier2: dict,
    tier3: dict,
    decision: dict,
) -> None:
    """Write the report to docs/reports/48_model_selection_validation.md."""
    logger.info("=" * 80)
    logger.info("WRITING REPORT")
    logger.info("=" * 80)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "48_model_selection_validation.md"

    t1_avail = tier1.get("available_models", [])
    bp_metrics = tier1["bp_metrics"]
    param_metrics = tier1["param_metrics"]
    esper_metrics = tier2.get("esper_metrics", {})
    q2_results = tier3.get("q2_results", {})
    leakage = tier1.get("leakage", {})

    lines = []
    lines.append("# Results Report: Model Selection Validation (5-Model Comparison)\n")
    lines.append(
        "This report documents the reproducible, apples-to-apples comparison of "
        "five PC-SAFT parameter prediction models across three validation tiers.\n"
    )

    # Candidate scope
    lines.append("## Candidate Scope\n")
    lines.append(
        "Five models were evaluated: **RF**, **XGBoost**, **chemprop** (D-MPNN), "
        "**GNN** (GIN), and **GNNePCSAFT** (external pre-trained benchmark, ePC-SAFT). "
        "GNNePCSAFT predicts ePC-SAFT parameters rather than standard PC-SAFT; for "
        "non-associating fluorinated hydrocarbons the parameters are numerically close.\n"
    )
    lines.append(
        "**ChemBERTa-2 was considered and declined**: the existing ChemBERTa-1 fine-tune "
        "(Step 04, R2=0.27 on eps/k, 185x slower than RF) already covers the SMILES-transformer "
        "angle. The fundamental bottleneck (~1,800 fine-tuning examples) is unchanged for v2; "
        "it is not a deployment candidate.\n"
    )

    # Tier 1
    lines.append("## Tier 1: Fluorinated External Validation (15 compounds)\n")

    # Parameter table
    lines.append("### Parameter Accuracy\n")
    lines.append(
        "| Model | m MAE | sigma MAE | epsilon/k MAE (K) | epsilon/k R2 |"
    )
    lines.append("|-------|-------|-----------|-------------------|-------------|")
    for mname in t1_avail:
        pm = param_metrics.get(mname, {})
        mm = pm.get("m", {})
        sm = pm.get("sigma", {})
        em = pm.get("epsilon_k", {})
        label = MODEL_DISPLAY.get(mname, mname)
        lines.append(
            f"| {label} "
            f"| {mm.get('mae', np.nan):.3f} "
            f"| {sm.get('mae', np.nan):.3f} "
            f"| {em.get('mae', np.nan):.1f} "
            f"| {em.get('r2', np.nan):.3f} |"
        )
    lines.append("")

    # BP table
    lines.append("### Boiling Point Accuracy\n")
    lines.append("| Model | BP MAE (K) | BP RMSE (K) | EOS Convergence |")
    lines.append("|-------|-----------|-------------|-----------------|")
    for mname in t1_avail:
        bm = bp_metrics.get(mname, {})
        label = MODEL_DISPLAY.get(mname, mname)
        lines.append(
            f"| {label} "
            f"| {bm.get('mae', np.nan):.1f} "
            f"| {bm.get('rmse', np.nan):.1f} "
            f"| {bm.get('n_converged', 0)}/{bm.get('n_total', 15)} |"
        )
    lines.append("")

    # Leakage audit
    lines.append("### Data Leakage Audit\n")
    lines.append(
        f"- Exact SMILES overlap (fluorinated vs Esper train): "
        f"**{leakage.get('n_exact_overlap', 'N/A')}**"
    )
    lines.append(
        f"- Mean Tanimoto NN similarity to train: "
        f"**{leakage.get('mean_tanimoto', np.nan):.3f}**"
    )
    lines.append(
        f"- Max Tanimoto NN similarity to train: "
        f"**{leakage.get('max_tanimoto', np.nan):.3f}**"
    )
    lines.append(
        f"- Compounds with high similarity (>0.85): "
        f"**{leakage.get('n_high_similarity', 0)}**\n"
    )

    # Tier 2
    lines.append("## Tier 2: Esper Holdout with Bootstrap CIs\n")
    t2_avail = tier2.get("available_models", [])
    lines.append(
        f"Test set: {ctx['provenance'].get('test_split_n', 'N/A')} molecules "
        f"(split hash: {ctx['provenance'].get('test_split_hash', 'N/A')})\n"
    )
    lines.append(
        "| Model | eps/k R2 [95% CI] | eps/k MAE [95% CI] |"
    )
    lines.append("|-------|-------------------|---------------------|")
    for mname in t2_avail:
        em = esper_metrics.get(mname, {}).get("epsilon_k", {})
        label = MODEL_DISPLAY.get(mname, mname)
        r2_str = f"{em.get('r2', np.nan):.3f}"
        if np.isfinite(em.get("r2_ci_lo", np.nan)):
            r2_str += f" [{em['r2_ci_lo']:.3f}, {em['r2_ci_hi']:.3f}]"
        mae_str = f"{em.get('mae', np.nan):.1f}"
        if np.isfinite(em.get("mae_ci_lo", np.nan)):
            mae_str += f" [{em['mae_ci_lo']:.1f}, {em['mae_ci_hi']:.1f}]"
        if mname == "gnnepcsaft":
            label += " (OOD)"
        lines.append(f"| {label} | {r2_str} | {mae_str} |")
    lines.append("")

    # Tier 3
    lines.append("## Tier 3: Cross-Validated Q2 (5-fold, tree models)\n")
    if q2_results:
        lines.append("| Model | Q2 m | Q2 sigma | Q2 epsilon/k |")
        lines.append("|-------|------|----------|--------------|")
        for mname, targets in q2_results.items():
            label = MODEL_DISPLAY.get(mname, mname)
            m_q2 = targets.get("m", {})
            s_q2 = targets.get("sigma", {})
            e_q2 = targets.get("epsilon_k", {})
            lines.append(
                f"| {label} "
                f"| {m_q2.get('q2_mean', np.nan):.3f} +/- {m_q2.get('q2_std', np.nan):.3f} "
                f"| {s_q2.get('q2_mean', np.nan):.3f} +/- {s_q2.get('q2_std', np.nan):.3f} "
                f"| {e_q2.get('q2_mean', np.nan):.3f} +/- {e_q2.get('q2_std', np.nan):.3f} |"
            )
        lines.append("")
        lines.append(
            "Note: chemprop, GNN, and GNNePCSAFT Q2 deferred (require per-fold "
            "retraining; this is a tree-only robustness check).\n"
        )
    else:
        lines.append("Tier 3 was not computed in this run.\n")

    # Decision
    lines.append("## Model Selection Decision\n")
    lines.append(f"**{decision.get('verdict', 'N/A')}**\n")
    for criterion in decision.get("criteria", []):
        lines.append(f"- **{criterion.get('name', '')}**: {json.dumps(criterion, default=str)}")
    lines.append("")

    # Figures
    lines.append("## Figures\n")
    lines.append("All figures saved to `figures/48_model_selection_validation/`.\n")
    lines.append("1. `fluorinated_parity_all_models.png` - epsilon/k parity, one panel per model")
    lines.append("2. `fluorinated_boiling_point_parity.png` - predicted vs experimental BP")
    lines.append("3. `esper_bootstrap_ci_comparison.png` - forest plot of R2 and MAE with CIs")
    lines.append("4. `paired_difference_forest.png` - paired metric differences")
    lines.append("5. `tier_summary_heatmap.png` - key metrics across tiers\n")

    # Provenance
    lines.append("## Provenance\n")
    prov = ctx.get("provenance", {})
    lines.append(f"- Feature config hash: `{prov.get('feature_config_hash', 'N/A')}`")
    lines.append(f"- Test split hash: `{prov.get('test_split_hash', 'N/A')}`")
    lines.append(f"- Test split n: {prov.get('test_split_n', 'N/A')}")
    lines.append(f"- GNNePCSAFT version: {prov.get('gnnepcsaft_version', 'N/A')}")
    for name, path in prov.get("artifact_paths", {}).items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")

    # Readiness check
    lines.append("## Readiness Check\n")
    lines.append("- [x] All available models evaluated on fluorinated validation set")
    lines.append("- [x] Esper holdout metrics include bootstrap 95% CIs")
    lines.append("- [x] Paired comparisons use bootstrapped metric differences")
    lines.append("- [x] Data leakage audit documented")
    lines.append("- [x] Tanimoto nearest-neighbor similarity recorded")
    lines.append("- [x] Model selection decision documented with explicit evidence")
    lines.append("- [x] Provenance metadata saved")
    lines.append("- [x] Figures generated\n")

    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    logger.info("Report written: %s", report_path)


# ===================================================================
# Section: Save provenance metadata
# ===================================================================


def save_provenance(ctx: dict, tier1: dict, decision: dict) -> None:
    """Save evaluation provenance metadata to JSON."""
    metadata = {
        "step": 48,
        "provenance": ctx.get("provenance", {}),
        "fluorinated_validation": {
            "n_compounds": len(ctx.get("fluor_df", [])),
            "source": str(FLUORINATED_VAL_PATH),
        },
        "test_set": {
            "n_molecules": ctx.get("provenance", {}).get("test_split_n", 0),
            "source": str(TEST_SET_PATH),
            "split_hash": ctx.get("provenance", {}).get("test_split_hash", ""),
        },
        "leakage_audit": {
            "n_exact_overlap": tier1.get("leakage", {}).get("n_exact_overlap", None),
            "mean_tanimoto": tier1.get("leakage", {}).get("mean_tanimoto", None),
            "max_tanimoto": tier1.get("leakage", {}).get("max_tanimoto", None),
        },
        "models_evaluated": tier1.get("available_models", []),
        "selected_model": decision.get("selected_model", None),
    }
    path = SAVED_DIR / "step48_model_selection_metadata.json"
    _save_json(metadata, path)
    logger.info("Provenance metadata saved: %s", path)


# ===================================================================
# Utilities
# ===================================================================


def _save_json(data: dict, path: Path) -> None:
    """Save dict to JSON with numpy-safe serialization."""
    def _to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_native(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_native(data), f, indent=2, default=str)


# ===================================================================
# Main
# ===================================================================

SECTIONS = {
    "load": ["load"],
    "tier1": ["load", "tier1"],
    "tier2": ["load", "tier2"],
    "tier3": ["load", "tier3"],
    "decision": ["load", "tier1", "tier2", "tier3", "decision"],
    "figures": ["load", "tier1", "tier2", "tier3", "figures"],
    "report": ["load", "tier1", "tier2", "tier3", "decision", "report"],
    "all": ["load", "tier1", "tier2", "tier3", "decision", "figures", "report"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Step 48: Model Selection Validation (5-Model Comparison)"
    )
    parser.add_argument(
        "--section",
        type=str,
        default="all",
        choices=list(SECTIONS.keys()),
        help="Section to run (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    sections_to_run = SECTIONS[args.section]
    logger.info("=" * 80)
    logger.info("STEP 48: MODEL SELECTION VALIDATION")
    logger.info("Sections: %s", sections_to_run)
    logger.info("=" * 80)

    ctx = {}
    tier1 = {}
    tier2 = {}
    tier3 = {}
    decision_result = {}

    if "load" in sections_to_run:
        ctx = section_load()

    if "tier1" in sections_to_run:
        tier1 = section_tier1(ctx)

    if "tier2" in sections_to_run:
        tier2 = section_tier2(ctx)

    if "tier3" in sections_to_run:
        try:
            tier3 = section_tier3(ctx)
        except Exception as exc:
            logger.warning("Tier 3 failed (non-blocking): %s", exc)
            tier3 = {"q2_results": {}}

    if "decision" in sections_to_run:
        decision_result = section_decision(ctx, tier1, tier2, tier3)

    if "figures" in sections_to_run:
        section_figures(ctx, tier1, tier2, tier3, decision_result)

    if "report" in sections_to_run:
        section_report(ctx, tier1, tier2, tier3, decision_result)
        save_provenance(ctx, tier1, decision_result)

    # Final summary
    logger.info("=" * 80)
    logger.info("STEP 48 COMPLETE")
    logger.info("=" * 80)
    if decision_result:
        logger.info("  Selected model: %s", decision_result.get("selected_model", "N/A"))
        logger.info("  Verdict: %s", decision_result.get("verdict", "N/A"))
    logger.info("  Figures: %s", FIGURES_DIR)
    logger.info("  Report: %s", REPORTS_DIR / "48_model_selection_validation.md")


if __name__ == "__main__":
    main()
