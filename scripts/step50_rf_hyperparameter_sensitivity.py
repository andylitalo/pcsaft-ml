#!/usr/bin/env python3
"""Step 50: RF Hyperparameter Sensitivity Analysis.

Usage:
    python scripts/step50_rf_hyperparameter_sensitivity.py [--section SECTION]

Sections:
    grid        Full grid search (50.1)
    oat         One-at-a-time sensitivity (50.2)
    compare     Production vs best config (50.3)
    external    Fluorinated validation of top configs (50.4)
    importance  Feature importance stability (50.5)
    learning    RF data efficiency learning curve (50.6)
    errors      Systematic error analysis (50.7)
    figures     Generate all figures (50.8)
    report      Write report (50.10)
    all         Run everything (default)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    learning_curve,
)

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.data.descriptors import build_features, build_features_with_names
from model.data.load import TARGETS, load_data, split_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "50_rf_hyperparameter_sensitivity"
SAVED_DIR = ROOT / "model" / "saved"
REPORT_DIR = ROOT / "docs" / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DPI = 150

# Font sizing conventions
FONTSIZE_TITLE = 16
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_LEGEND = 11

# Matplotlib style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        logger.warning("Could not load seaborn style, using default")

TARGET_DISPLAY = {
    "m": "m (segments)",
    "sigma": r"$\sigma$ ($\AA$)",
    "epsilon_k": r"$\varepsilon$/k (K)",
}

TARGET_COLORS = {
    "m": "steelblue",
    "sigma": "darkorange",
    "epsilon_k": "forestgreen",
}

PRODUCTION_CONFIG = {
    "n_estimators": 100,
    "max_features": "sqrt",
    "max_depth": None,
    "min_samples_leaf": 1,
}


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------
_DATA_CACHE: dict = {}


def _load_esper_data():
    """Load Esper data and cache the train/test split and features."""
    if _DATA_CACHE:
        return _DATA_CACHE

    logger.info("Loading Esper data...")
    df = load_data("esper")
    train_df, test_df = split_data(df)

    logger.info("Building features for %d train + %d test molecules...",
                len(train_df), len(test_df))
    all_smiles = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    X_all, feature_names = build_features_with_names(all_smiles)

    n_train = len(train_df)
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]

    # Handle non-finite values
    train_mask = np.isfinite(X_train).all(axis=1)
    test_mask = np.isfinite(X_test).all(axis=1)
    X_train = X_train[train_mask]
    X_test = X_test[test_mask]
    train_df = train_df.iloc[train_mask].reset_index(drop=True)
    test_df = test_df.iloc[test_mask].reset_index(drop=True)

    # Prepare per-target y arrays
    y_train = np.column_stack([train_df[t].values for t in TARGETS])
    y_test = np.column_stack([test_df[t].values for t in TARGETS])

    ek_idx = TARGETS.index("epsilon_k")

    _DATA_CACHE.update({
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "ek_idx": ek_idx,
    })
    logger.info(
        "Data loaded: %d train, %d test, %d features",
        X_train.shape[0], X_test.shape[0], len(feature_names),
    )
    return _DATA_CACHE


# ============================================================================
# 50.1: Grid Search
# ============================================================================
def section_50_1_grid():
    """Target-specific grid search for epsilon_k."""
    logger.info("=" * 60)
    logger.info("SECTION 50.1: Full Grid Search (epsilon_k)")
    logger.info("=" * 60)

    data = _load_esper_data()
    X_train = data["X_train"]
    y_ek = data["y_train"][:, data["ek_idx"]]

    param_grid = {
        "n_estimators": [50, 100, 200, 500],
        "max_features": ["sqrt", "log2", 0.3, 0.5, None],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 3, 5],
    }

    n_configs = 1
    for vals in param_grid.values():
        n_configs *= len(vals)
    logger.info("Grid: %d configurations x 5 folds = %d fits", n_configs, n_configs * 5)

    cv = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=1,  # RF already uses n_jobs=-1 internally
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    cv.fit(X_train, y_ek)

    logger.info("Best score: %.4f", cv.best_score_)
    logger.info("Best params: %s", cv.best_params_)

    # Serialize cv_results_
    cv_results = {}
    for key, val in cv.cv_results_.items():
        if key == "params":
            cv_results[key] = [
                {k: (_serialize_param(v)) for k, v in p.items()}
                for p in val
            ]
        elif hasattr(val, "tolist"):
            cv_results[key] = val.tolist()
        else:
            cv_results[key] = val

    cv_results["best_params"] = {
        k: _serialize_param(v) for k, v in cv.best_params_.items()
    }
    cv_results["best_score"] = float(cv.best_score_)

    # Find production config rank
    prod_rank = _find_production_rank(cv_results)
    cv_results["production_rank"] = prod_rank
    if prod_rank is not None:
        prod_score = cv_results["mean_test_score"][prod_rank]
        delta = cv.best_score_ - prod_score
        cv_results["production_score"] = float(prod_score)
        cv_results["score_delta_best_minus_production"] = float(delta)
        logger.info("Production config rank: %d / %d", prod_rank + 1, n_configs)
        logger.info("Production score: %.4f, delta: %.4f", prod_score, delta)

    results_path = SAVED_DIR / "rf_hyperparam_search.json"
    results_path.write_text(json.dumps(cv_results, indent=2) + "\n")
    logger.info("Saved grid search results to %s", results_path)

    return cv_results


def _serialize_param(v):
    """Serialize a parameter value to JSON-compatible type."""
    if v is None:
        return "None"
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def _find_production_rank(cv_results: dict) -> int | None:
    """Find the rank of the production config in the grid search results."""
    params = cv_results.get("params", [])
    for i, p in enumerate(params):
        n_est = p.get("n_estimators", None)
        mf = p.get("max_features", None)
        md = p.get("max_depth", None)
        msl = p.get("min_samples_leaf", None)

        # Normalize for comparison
        if _param_match(n_est, 100) and _param_match(mf, "sqrt") and \
           _param_match(md, None) and _param_match(msl, 1):
            return i
    return None


def _param_match(actual, expected) -> bool:
    """Check if a param matches, handling None vs 'None' and type coercion."""
    if expected is None:
        return actual is None or str(actual) == "None"
    return str(actual) == str(expected)


# ============================================================================
# 50.2: One-at-a-time Sensitivity
# ============================================================================
def section_50_2_oat():
    """One-at-a-time hyperparameter sensitivity sweeps."""
    logger.info("=" * 60)
    logger.info("SECTION 50.2: OAT Sensitivity Sweeps")
    logger.info("=" * 60)

    data = _load_esper_data()
    X_train = data["X_train"]
    y_train = data["y_train"]

    sweeps = {
        "n_estimators": [25, 50, 75, 100, 150, 200, 300, 500],
        "max_features": ["sqrt", "log2", 0.1, 0.2, 0.3, 0.5, 0.7, None],
        "max_depth": [5, 10, 15, 20, 25, 30, 40, None],
        "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15, 20],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for param_name, param_values in sweeps.items():
        logger.info("Sweeping %s: %s", param_name, param_values)
        results[param_name] = {}

        for target_idx, target_name in enumerate(TARGETS):
            y_col = y_train[:, target_idx]
            target_results = []

            for val in param_values:
                config = PRODUCTION_CONFIG.copy()
                config[param_name] = val
                rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **config)

                try:
                    scores = cross_val_score(rf, X_train, y_col, cv=kf, scoring="r2")
                    target_results.append({
                        "value": _serialize_param(val),
                        "mean_r2": float(np.mean(scores)),
                        "std_r2": float(np.std(scores)),
                        "scores": scores.tolist(),
                    })
                    logger.info(
                        "  %s=%s, %s: R2=%.4f +/- %.4f",
                        param_name, val, target_name,
                        np.mean(scores), np.std(scores),
                    )
                except Exception as e:
                    logger.warning(
                        "  %s=%s, %s: FAILED (%s)", param_name, val, target_name, e
                    )
                    target_results.append({
                        "value": _serialize_param(val),
                        "mean_r2": None,
                        "std_r2": None,
                        "error": str(e),
                    })

            results[param_name][target_name] = target_results

    oat_path = SAVED_DIR / "rf_oat_sensitivity.json"
    oat_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Saved OAT sensitivity results to %s", oat_path)

    return results


# ============================================================================
# 50.3: Production vs Best Config
# ============================================================================
def section_50_3_compare():
    """Paired bootstrap comparison of production vs best config on epsilon_k."""
    logger.info("=" * 60)
    logger.info("SECTION 50.3: Production vs Best Config Comparison")
    logger.info("=" * 60)

    data = _load_esper_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    ek_idx = data["ek_idx"]

    # Load grid search results to find best params
    grid_path = SAVED_DIR / "rf_hyperparam_search.json"
    if not grid_path.exists():
        logger.warning("Grid search results not found. Run --section grid first.")
        return None

    with open(grid_path) as f:
        grid_data = json.load(f)

    best_params_raw = grid_data.get("best_params", {})
    best_params = _deserialize_params(best_params_raw)

    logger.info("Best found params: %s", best_params)
    logger.info("Production params: %s", PRODUCTION_CONFIG)

    # Train production RF for epsilon_k
    rf_production = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **PRODUCTION_CONFIG
    )
    rf_production.fit(X_train, y_train[:, ek_idx])
    pred_production = rf_production.predict(X_test)

    # Train best RF for epsilon_k
    rf_best = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
    rf_best.fit(X_train, y_train[:, ek_idx])
    pred_best = rf_best.predict(X_test)

    y_ek_test = y_test[:, ek_idx]

    prod_r2 = r2_score(y_ek_test, pred_production)
    best_r2 = r2_score(y_ek_test, pred_best)
    prod_mae = mean_absolute_error(y_ek_test, pred_production)
    best_mae = mean_absolute_error(y_ek_test, pred_best)

    logger.info("Production epsilon_k: R2=%.4f, MAE=%.3f", prod_r2, prod_mae)
    logger.info("Best config epsilon_k: R2=%.4f, MAE=%.3f", best_r2, best_mae)

    # Paired bootstrap
    n_bootstrap = 2000
    rng = np.random.RandomState(RANDOM_STATE)
    delta_r2_samples = []
    delta_mae_samples = []

    n_test = len(y_ek_test)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_test, size=n_test)
        prod_score = r2_score(y_ek_test[idx], pred_production[idx])
        best_score = r2_score(y_ek_test[idx], pred_best[idx])
        delta_r2_samples.append(best_score - prod_score)

        prod_mae_b = mean_absolute_error(y_ek_test[idx], pred_production[idx])
        best_mae_b = mean_absolute_error(y_ek_test[idx], pred_best[idx])
        delta_mae_samples.append(best_mae_b - prod_mae_b)

    delta_r2_lo, delta_r2_hi = np.percentile(delta_r2_samples, [2.5, 97.5])
    delta_mae_lo, delta_mae_hi = np.percentile(delta_mae_samples, [2.5, 97.5])

    logger.info("Delta R2 (best - prod): %.4f [%.4f, %.4f]",
                np.mean(delta_r2_samples), delta_r2_lo, delta_r2_hi)
    logger.info("Delta MAE (best - prod): %.3f [%.3f, %.3f]",
                np.mean(delta_mae_samples), delta_mae_lo, delta_mae_hi)

    comparison = {
        "production_config": {k: _serialize_param(v) for k, v in PRODUCTION_CONFIG.items()},
        "best_config": best_params_raw,
        "production_metrics": {
            "r2": float(prod_r2),
            "mae": float(prod_mae),
            "rmse": float(np.sqrt(mean_squared_error(y_ek_test, pred_production))),
        },
        "best_metrics": {
            "r2": float(best_r2),
            "mae": float(best_mae),
            "rmse": float(np.sqrt(mean_squared_error(y_ek_test, pred_best))),
        },
        "paired_bootstrap": {
            "n_resamples": n_bootstrap,
            "delta_r2_mean": float(np.mean(delta_r2_samples)),
            "delta_r2_ci_95": [float(delta_r2_lo), float(delta_r2_hi)],
            "delta_mae_mean": float(np.mean(delta_mae_samples)),
            "delta_mae_ci_95": [float(delta_mae_lo), float(delta_mae_hi)],
            "ci_spans_zero_r2": bool(delta_r2_lo <= 0 <= delta_r2_hi),
            "ci_spans_zero_mae": bool(delta_mae_lo <= 0 <= delta_mae_hi),
        },
        "target": "epsilon_k",
        "n_test": int(n_test),
    }

    compare_path = SAVED_DIR / "rf_production_vs_best.json"
    compare_path.write_text(json.dumps(comparison, indent=2) + "\n")
    logger.info("Saved comparison to %s", compare_path)

    return comparison


def _deserialize_params(raw: dict) -> dict:
    """Convert serialized param dict back to sklearn-compatible types."""
    result = {}
    for k, v in raw.items():
        if v == "None" or v is None:
            result[k] = None
        elif isinstance(v, str):
            # Try to convert to int or float
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
        else:
            result[k] = v
    return result


# ============================================================================
# 50.4: External Validation
# ============================================================================
def section_50_4_external():
    """Validate production and top configs on fluorinated external set."""
    logger.info("=" * 60)
    logger.info("SECTION 50.4: Fluorinated External Validation")
    logger.info("=" * 60)

    data = _load_esper_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    feature_names = data["feature_names"]
    ek_idx = data["ek_idx"]

    # Load fluorinated validation set
    fluor_path = SAVED_DIR / "gnn_fluorinated_validation_set.csv"
    if not fluor_path.exists():
        logger.warning("Fluorinated validation set not found at %s", fluor_path)
        return None

    fluor_df = pd.read_csv(fluor_path)
    logger.info("Loaded %d fluorinated validation compounds", len(fluor_df))

    # Check for required columns
    target_map = {"m": "m_lit", "sigma": "sigma_lit", "epsilon_k": "epsilon_k_lit"}
    required_cols = list(target_map.values())
    missing_cols = [c for c in required_cols if c not in fluor_df.columns]
    if missing_cols:
        logger.warning("Missing columns in fluorinated validation set: %s", missing_cols)
        return None

    # Build features for fluorinated set
    rdkit_names = [n for n in feature_names if not n.startswith("morgan_")]
    X_fluor = build_features(fluor_df["smiles"].tolist(), rdkit_names=rdkit_names)

    # Load grid search to get top configs
    grid_path = SAVED_DIR / "rf_hyperparam_search.json"
    configs_to_test = [("production", PRODUCTION_CONFIG)]

    if grid_path.exists():
        with open(grid_path) as f:
            grid_data = json.load(f)

        # Get top 3 unique configs by rank
        if "rank_test_score" in grid_data and "params" in grid_data:
            ranks = grid_data["rank_test_score"]
            params = grid_data["params"]
            sorted_indices = sorted(range(len(ranks)), key=lambda i: ranks[i])

            seen = set()
            for idx in sorted_indices:
                param_key = json.dumps(params[idx], sort_keys=True)
                if param_key not in seen:
                    seen.add(param_key)
                    config = _deserialize_params(params[idx])
                    configs_to_test.append((f"grid_rank_{ranks[idx]}", config))
                if len(configs_to_test) >= 4:  # production + 3
                    break

    results = {}
    for config_name, config in configs_to_test:
        logger.info("Evaluating config: %s -> %s", config_name, config)

        rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **config)
        rf.fit(X_train, y_train[:, ek_idx])

        pred_ek = rf.predict(X_fluor)
        y_ek_true = fluor_df["epsilon_k_lit"].values

        ek_mae = mean_absolute_error(y_ek_true, pred_ek)
        ek_rmse = np.sqrt(mean_squared_error(y_ek_true, pred_ek))
        ek_r2 = r2_score(y_ek_true, pred_ek)

        config_result = {
            "config": {k: _serialize_param(v) for k, v in config.items()},
            "epsilon_k_mae": float(ek_mae),
            "epsilon_k_rmse": float(ek_rmse),
            "epsilon_k_r2": float(ek_r2),
            "n_molecules": len(fluor_df),
        }

        # Try boiling point computation
        bp_mae = _compute_bp_mae(fluor_df, pred_ek, X_fluor, X_train, y_train, config)
        if bp_mae is not None:
            config_result["boiling_point_mae_K"] = float(bp_mae)

        results[config_name] = config_result

        logger.info(
            "  %s: ek_MAE=%.3f, ek_RMSE=%.3f, ek_R2=%.3f",
            config_name, ek_mae, ek_rmse, ek_r2,
        )

    ext_path = SAVED_DIR / "rf_external_config_comparison.json"
    ext_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Saved external comparison to %s", ext_path)

    return results


def _compute_bp_mae(fluor_df, pred_ek, X_fluor, X_train, y_train, config):
    """Attempt to compute boiling point MAE using screening module."""
    try:
        from screening.hfo_screening import compute_boiling_point
    except ImportError:
        logger.warning("Could not import compute_boiling_point, skipping BP validation")
        return None

    m_idx = TARGETS.index("m")
    s_idx = TARGETS.index("sigma")

    # Also need m and sigma predictions
    rf_m = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **config)
    rf_m.fit(X_train, y_train[:, m_idx])
    pred_m = rf_m.predict(X_fluor)

    rf_s = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **config)
    rf_s.fit(X_train, y_train[:, s_idx])
    pred_s = rf_s.predict(X_fluor)

    bp_errors = []
    for i in range(len(fluor_df)):
        try:
            bp_pred = compute_boiling_point(pred_m[i], pred_s[i], pred_ek[i])
            if np.isfinite(bp_pred):
                # Check if we have a literature boiling point
                if "boiling_point_K" in fluor_df.columns:
                    bp_true = fluor_df.iloc[i]["boiling_point_K"]
                    if np.isfinite(bp_true):
                        bp_errors.append(abs(bp_pred - bp_true))
        except Exception:
            continue

    if bp_errors:
        return np.mean(bp_errors)
    return None


# ============================================================================
# 50.5: Feature Importance Stability
# ============================================================================
def section_50_5_importance():
    """5-fold CV feature importance stability analysis."""
    logger.info("=" * 60)
    logger.info("SECTION 50.5: Feature Importance Stability")
    logger.info("=" * 60)

    from sklearn.inspection import permutation_importance

    data = _load_esper_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    feature_names = data["feature_names"]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    gini_importances = []
    perm_importances = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        logger.info("Fold %d/%d", fold_idx + 1, 5)

        for target_idx, target_name in enumerate(TARGETS):
            rf = RandomForestRegressor(
                random_state=RANDOM_STATE, n_jobs=-1, **PRODUCTION_CONFIG
            )
            rf.fit(X_fold_train, y_fold_train[:, target_idx])

            # Gini importance
            gini_importances.append({
                "fold": fold_idx,
                "target": target_name,
                "importances": rf.feature_importances_.tolist(),
            })

            # Permutation importance (on validation fold)
            perm_result = permutation_importance(
                rf, X_fold_val, y_fold_val[:, target_idx],
                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
            )
            perm_importances.append({
                "fold": fold_idx,
                "target": target_name,
                "importances_mean": perm_result.importances_mean.tolist(),
                "importances_std": perm_result.importances_std.tolist(),
            })

            logger.info(
                "  %s: top Gini feature=%s (%.4f), top perm feature=%s (%.4f)",
                target_name,
                feature_names[np.argmax(rf.feature_importances_)],
                np.max(rf.feature_importances_),
                feature_names[np.argmax(perm_result.importances_mean)],
                np.max(perm_result.importances_mean),
            )

    # Analyze top-20 consistency for epsilon_k
    ek_gini = [
        entry for entry in gini_importances if entry["target"] == "epsilon_k"
    ]
    consistency_report = _analyze_top_features(ek_gini, feature_names, top_n=20)

    importance_data = {
        "gini_importances": gini_importances,
        "perm_importances": perm_importances,
        "feature_names": feature_names,
        "consistency_epsilon_k": consistency_report,
        "n_folds": 5,
        "n_features": len(feature_names),
    }

    imp_path = SAVED_DIR / "rf_feature_importance_stability.json"
    imp_path.write_text(json.dumps(importance_data, indent=2) + "\n")
    logger.info("Saved feature importance stability to %s", imp_path)

    return importance_data


def _analyze_top_features(
    gini_entries: list[dict],
    feature_names: list[str],
    top_n: int = 20,
) -> dict:
    """Analyze consistency of top features across folds."""
    top_sets = []
    for entry in gini_entries:
        imp = np.array(entry["importances"])
        top_indices = np.argsort(imp)[-top_n:][::-1]
        top_sets.append(set(top_indices.tolist()))

    # Features in all folds
    if top_sets:
        consistent = set.intersection(*top_sets)
        in_any = set.union(*top_sets)
    else:
        consistent = set()
        in_any = set()

    # Mean importance across folds
    all_imp = np.array([entry["importances"] for entry in gini_entries])
    mean_imp = all_imp.mean(axis=0)
    std_imp = all_imp.std(axis=0)

    top_by_mean = np.argsort(mean_imp)[-top_n:][::-1]

    # Count Morgan vs RDKit in top 20
    top_names = [feature_names[i] for i in top_by_mean]
    n_morgan = sum(1 for n in top_names if n.startswith("morgan_"))
    n_rdkit = len(top_names) - n_morgan

    return {
        "top_n": top_n,
        "n_consistent_all_folds": len(consistent),
        "n_in_any_fold": len(in_any),
        "top_features_by_mean": [
            {
                "feature": feature_names[i],
                "mean_importance": float(mean_imp[i]),
                "std_importance": float(std_imp[i]),
                "cv": float(std_imp[i] / mean_imp[i]) if mean_imp[i] > 0 else float("inf"),
            }
            for i in top_by_mean
        ],
        "n_morgan_in_top20": n_morgan,
        "n_rdkit_in_top20": n_rdkit,
    }


# ============================================================================
# 50.6: Learning Curve
# ============================================================================
def section_50_6_learning():
    """RF data efficiency learning curve."""
    logger.info("=" * 60)
    logger.info("SECTION 50.6: RF Learning Curve")
    logger.info("=" * 60)

    data = _load_esper_data()
    X_train = data["X_train"]
    y_ek = data["y_train"][:, data["ek_idx"]]

    train_sizes_frac = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]

    rf = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **PRODUCTION_CONFIG
    )

    logger.info("Computing learning curve with %d size points...", len(train_sizes_frac))
    train_sizes_abs, train_scores, test_scores = learning_curve(
        rf, X_train, y_ek,
        train_sizes=train_sizes_frac,
        cv=5,
        scoring="r2",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )

    lc_results = {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "train_scores_std": train_scores.std(axis=1).tolist(),
        "test_scores_mean": test_scores.mean(axis=1).tolist(),
        "test_scores_std": test_scores.std(axis=1).tolist(),
        "target": "epsilon_k",
        "n_folds": 5,
    }

    for i, n in enumerate(train_sizes_abs):
        logger.info(
            "  n=%d: train R2=%.4f +/- %.4f, test R2=%.4f +/- %.4f",
            n,
            train_scores.mean(axis=1)[i], train_scores.std(axis=1)[i],
            test_scores.mean(axis=1)[i], test_scores.std(axis=1)[i],
        )

    lc_path = SAVED_DIR / "rf_learning_curve.json"
    lc_path.write_text(json.dumps(lc_results, indent=2) + "\n")
    logger.info("Saved learning curve to %s", lc_path)

    return lc_results


# ============================================================================
# 50.7: Error Analysis
# ============================================================================
def section_50_7_errors():
    """Systematic error analysis on the Esper test set."""
    logger.info("=" * 60)
    logger.info("SECTION 50.7: Systematic Error Analysis")
    logger.info("=" * 60)

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    data = _load_esper_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    test_df = data["test_df"]
    ek_idx = data["ek_idx"]

    # Train production RF on epsilon_k
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **PRODUCTION_CONFIG)
    rf.fit(X_train, y_train[:, ek_idx])
    pred_ek = rf.predict(X_test)
    y_ek = y_test[:, ek_idx]

    residuals = pred_ek - y_ek
    abs_errors = np.abs(residuals)

    # Compute molecular weight for each test molecule
    mol_weights = []
    n_rings = []
    n_fluorine = []
    n_halogens = []
    has_double_bond = []

    for smi in test_df["smiles"].values:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol_weights.append(Descriptors.MolWt(mol))
            n_rings.append(mol.GetRingInfo().NumRings())
            n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
            n_fluorine.append(n_f)
            n_hal = sum(
                1 for a in mol.GetAtoms()
                if a.GetAtomicNum() in (9, 17, 35, 53)
            )
            n_halogens.append(n_hal)
            # Check for double bonds
            has_db = any(
                bond.GetBondTypeAsDouble() == 2.0
                for bond in mol.GetBonds()
            )
            has_double_bond.append(has_db)
        else:
            mol_weights.append(np.nan)
            n_rings.append(0)
            n_fluorine.append(0)
            n_halogens.append(0)
            has_double_bond.append(False)

    # Build error analysis dataframe
    error_df = pd.DataFrame({
        "smiles": test_df["smiles"].values,
        "epsilon_k_true": y_ek,
        "epsilon_k_pred": pred_ek,
        "residual": residuals,
        "abs_error": abs_errors,
        "mol_weight": mol_weights,
        "n_rings": n_rings,
        "n_fluorine": n_fluorine,
        "n_halogens": n_halogens,
        "has_double_bond": has_double_bond,
    })

    # Sort by abs_error descending
    error_df = error_df.sort_values("abs_error", ascending=False)

    # Log top 10 worst
    logger.info("Top 10 worst-predicted molecules (epsilon_k):")
    for i, (_, row) in enumerate(error_df.head(10).iterrows()):
        logger.info(
            "  %d. %s: true=%.1f, pred=%.1f, |err|=%.1f, MW=%.1f",
            i + 1, row["smiles"][:40],
            row["epsilon_k_true"], row["epsilon_k_pred"],
            row["abs_error"], row["mol_weight"],
        )

    # Summary statistics by group
    logger.info("Error by molecular property:")
    logger.info("  Fluorinated (n_F>0): median |err|=%.1f (n=%d)",
                error_df[error_df["n_fluorine"] > 0]["abs_error"].median(),
                (error_df["n_fluorine"] > 0).sum())
    logger.info("  Non-fluorinated: median |err|=%.1f (n=%d)",
                error_df[error_df["n_fluorine"] == 0]["abs_error"].median(),
                (error_df["n_fluorine"] == 0).sum())
    logger.info("  Ring-containing: median |err|=%.1f (n=%d)",
                error_df[error_df["n_rings"] > 0]["abs_error"].median(),
                (error_df["n_rings"] > 0).sum())
    logger.info("  Halogenated: median |err|=%.1f (n=%d)",
                error_df[error_df["n_halogens"] > 0]["abs_error"].median(),
                (error_df["n_halogens"] > 0).sum())

    error_path = SAVED_DIR / "rf_esper_error_analysis.csv"
    error_df.to_csv(error_path, index=False)
    logger.info("Saved error analysis to %s (%d molecules)", error_path, len(error_df))

    return error_df


# ============================================================================
# 50.8: Generate Figures
# ============================================================================
def section_50_8_figures():
    """Generate all figures for Step 50."""
    logger.info("=" * 60)
    logger.info("SECTION 50.8: Generate Figures")
    logger.info("=" * 60)

    results = {}
    results["oat_sensitivity"] = _plot_oat_sensitivity()
    results["cv_heatmap"] = _plot_cv_heatmap()
    results["feature_importance"] = _plot_feature_importance_stability()
    results["learning_curve"] = _plot_learning_curve()
    results["error_analysis"] = _plot_error_analysis()

    generated = [k for k, v in results.items() if v]
    missing = [k for k, v in results.items() if not v]

    logger.info("Generated %d / %d figures:", len(generated), len(results))
    for name in generated:
        logger.info("  [OK] %s", name)
    if missing:
        logger.info("Missing artifacts (figures skipped):")
        for name in missing:
            logger.info("  [--] %s", name)

    return results


def _plot_oat_sensitivity() -> bool:
    """Figure 1: 2x2 OAT sensitivity grid for epsilon_k."""
    oat_path = SAVED_DIR / "rf_oat_sensitivity.json"
    if not oat_path.exists():
        logger.warning("OAT sensitivity results not found at %s", oat_path)
        return False

    with open(oat_path) as f:
        data = json.load(f)

    param_order = ["n_estimators", "max_features", "max_depth", "min_samples_leaf"]
    param_labels = {
        "n_estimators": "n_estimators",
        "max_features": "max_features",
        "max_depth": "max_depth",
        "min_samples_leaf": "min_samples_leaf",
    }
    production_vals = {
        "n_estimators": "100",
        "max_features": "sqrt",
        "max_depth": "None",
        "min_samples_leaf": "1",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    target = "epsilon_k"

    for ax_idx, param_name in enumerate(param_order):
        ax = axes[ax_idx]

        if param_name not in data or target not in data[param_name]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(param_labels[param_name], fontsize=FONTSIZE_TITLE)
            continue

        entries = data[param_name][target]

        x_labels = []
        means = []
        stds = []

        for entry in entries:
            val = str(entry["value"])
            mean_r2 = entry.get("mean_r2")
            std_r2 = entry.get("std_r2", 0)
            if mean_r2 is not None:
                x_labels.append(val)
                means.append(mean_r2)
                stds.append(std_r2)

        x_pos = range(len(x_labels))
        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(
            x_pos, means, yerr=stds,
            fmt="o-", color="forestgreen", linewidth=2, markersize=6,
            ecolor="gray", elinewidth=1, capsize=3,
        )

        # Mark production value
        prod_val = production_vals.get(param_name, "")
        if prod_val in x_labels:
            prod_idx = x_labels.index(prod_val)
            ax.axvline(
                x=prod_idx, color="red", linestyle="--", linewidth=1.5,
                alpha=0.7, label=f"Production ({prod_val})",
            )
            ax.legend(fontsize=FONTSIZE_LEGEND - 1)

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(x_labels, fontsize=FONTSIZE_TICK - 1, rotation=45, ha="right")
        ax.set_ylabel("5-Fold CV R$^2$", fontsize=FONTSIZE_LABEL)
        ax.set_title(f"{param_labels[param_name]}", fontsize=FONTSIZE_TITLE - 1)
        ax.tick_params(labelsize=FONTSIZE_TICK)

    fig.suptitle(
        r"OAT Sensitivity: $\varepsilon$/k R$^2$ (Local Sensitivity Near Production)",
        fontsize=FONTSIZE_TITLE, y=1.01,
    )
    fig.tight_layout()
    path = FIG_DIR / "oat_sensitivity_epsilon_k.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return True


def _plot_cv_heatmap() -> bool:
    """Figure 2: CV heatmap of n_estimators vs max_depth."""
    grid_path = SAVED_DIR / "rf_hyperparam_search.json"
    if not grid_path.exists():
        logger.warning("Grid search results not found at %s", grid_path)
        return False

    with open(grid_path) as f:
        data = json.load(f)

    params = data.get("params", [])
    mean_scores = data.get("mean_test_score", [])

    if not params or not mean_scores:
        logger.warning("Empty grid search results")
        return False

    # Filter to max_features="sqrt", min_samples_leaf=1 slice
    filtered = []
    for p, score in zip(params, mean_scores):
        mf = str(p.get("max_features", ""))
        msl = str(p.get("min_samples_leaf", ""))
        if mf == "sqrt" and msl == "1":
            filtered.append((p, score))

    if not filtered:
        # Fall back: just average across all max_features and min_samples_leaf
        logger.info("No exact slice found; averaging across max_features and min_samples_leaf")
        score_accum = defaultdict(list)
        for p, score in zip(params, mean_scores):
            n_est = str(p.get("n_estimators", ""))
            md = str(p.get("max_depth", ""))
            score_accum[(n_est, md)].append(score)
        filtered = [
            ({"n_estimators": k[0], "max_depth": k[1]}, np.mean(v))
            for k, v in score_accum.items()
        ]

    # Build heatmap matrix
    n_est_vals = sorted(set(str(p.get("n_estimators", "")) for p, _ in filtered),
                        key=lambda x: _sort_key_num(x))
    md_vals = sorted(set(str(p.get("max_depth", "")) for p, _ in filtered),
                     key=lambda x: _sort_key_num(x))

    score_map = defaultdict(list)
    for p, score in filtered:
        n_est = str(p.get("n_estimators", ""))
        md = str(p.get("max_depth", ""))
        score_map[(n_est, md)].append(score)

    matrix = np.full((len(md_vals), len(n_est_vals)), np.nan)
    for i, md in enumerate(md_vals):
        for j, ne in enumerate(n_est_vals):
            scores = score_map.get((ne, md), [])
            if scores:
                matrix[i, j] = np.mean(scores)

    fig, ax = plt.subplots(figsize=(max(6, len(n_est_vals) * 1.5),
                                     max(4, len(md_vals) * 0.8 + 1)))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(n_est_vals)))
    ax.set_xticklabels(n_est_vals, fontsize=FONTSIZE_TICK)
    ax.set_yticks(range(len(md_vals)))
    ax.set_yticklabels(md_vals, fontsize=FONTSIZE_TICK)
    ax.set_xlabel("n_estimators", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("max_depth", fontsize=FONTSIZE_LABEL)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmean(matrix) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=text_color)

    # Mark production config (n=100, max_depth=None)
    if "100" in n_est_vals and "None" in md_vals:
        pj = n_est_vals.index("100")
        pi = md_vals.index("None")
        ax.plot(pj, pi, marker="*", color="lime", markersize=22,
                markeredgecolor="black", markeredgewidth=1.0)

    fig.colorbar(im, ax=ax, shrink=0.8, label=r"Mean CV R$^2$ ($\varepsilon$/k)")
    ax.set_title(
        "CV Heatmap: n_estimators vs max_depth "
        r"($\varepsilon$/k, max_features=sqrt, min_samples_leaf=1)",
        fontsize=FONTSIZE_TITLE - 2,
    )

    fig.tight_layout()
    path = FIG_DIR / "cv_heatmap_n_estimators_vs_max_depth.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return True


def _sort_key_num(val_str: str):
    """Sort key for numeric strings with None handling."""
    if val_str.lower() == "none":
        return float("inf")
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return float("inf")


def _plot_feature_importance_stability() -> bool:
    """Figure 3: Box plots for top 20 features (epsilon_k)."""
    imp_path = SAVED_DIR / "rf_feature_importance_stability.json"
    if not imp_path.exists():
        logger.warning("Feature importance stability results not found at %s", imp_path)
        return False

    with open(imp_path) as f:
        data = json.load(f)

    feature_names = data.get("feature_names", [])
    gini_importances = data.get("gini_importances", [])

    # Filter to epsilon_k
    ek_entries = [e for e in gini_importances if e["target"] == "epsilon_k"]
    if not ek_entries:
        logger.warning("No epsilon_k Gini importance entries found")
        return False

    # Compute mean importance across folds
    all_imp = np.array([e["importances"] for e in ek_entries])
    mean_imp = all_imp.mean(axis=0)

    # Get top 20 indices
    top_indices = np.argsort(mean_imp)[-20:][::-1]
    top_names = [feature_names[i] for i in top_indices]

    # Build box plot data
    box_data = []
    for idx in top_indices:
        fold_importances = all_imp[:, idx]
        box_data.append(fold_importances)

    # Color-code by feature type
    colors = []
    for name in top_names:
        if name.startswith("morgan_"):
            colors.append("steelblue")
        else:
            colors.append("darkorange")

    fig, ax = plt.subplots(figsize=(12, 7))

    bp = ax.boxplot(
        box_data, vert=False, patch_artist=True,
        labels=[n[:25] for n in top_names],  # truncate long names
        medianprops={"color": "black", "linewidth": 1.5},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.7, label="Morgan FP"),
        Patch(facecolor="darkorange", alpha=0.7, label="RDKit Descriptor"),
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE_LEGEND)

    ax.set_xlabel("Gini Importance", fontsize=FONTSIZE_LABEL)
    ax.set_title(
        r"Feature Importance Stability: Top 20 for $\varepsilon$/k (5 CV Folds)",
        fontsize=FONTSIZE_TITLE,
    )
    ax.tick_params(labelsize=FONTSIZE_TICK - 1)

    fig.tight_layout()
    path = FIG_DIR / "feature_importance_stability.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return True


def _plot_learning_curve() -> bool:
    """Figure 4: RF learning curve (R-squared vs training molecules)."""
    lc_path = SAVED_DIR / "rf_learning_curve.json"
    if not lc_path.exists():
        logger.warning("Learning curve results not found at %s", lc_path)
        return False

    with open(lc_path) as f:
        data = json.load(f)

    train_sizes = data["train_sizes"]
    train_mean = np.array(data["train_scores_mean"])
    train_std = np.array(data["train_scores_std"])
    test_mean = np.array(data["test_scores_mean"])
    test_std = np.array(data["test_scores_std"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="steelblue")
    ax.plot(train_sizes, train_mean, "o-", color="steelblue", linewidth=2,
            markersize=5, label="Train R$^2$")

    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.15, color="darkorange")
    ax.plot(train_sizes, test_mean, "o-", color="darkorange", linewidth=2,
            markersize=5, label="CV Test R$^2$")

    # Mark full training set size
    if train_sizes:
        ax.axvline(
            x=max(train_sizes), color="gray", linestyle=":", linewidth=1,
            alpha=0.7, label=f"Full train (n={max(train_sizes)})",
        )

    ax.set_xlabel("Number of Training Molecules", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("R$^2$", fontsize=FONTSIZE_LABEL)
    ax.set_title(
        r"RF Learning Curve: $\varepsilon$/k R$^2$ vs Training Set Size",
        fontsize=FONTSIZE_TITLE,
    )
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = FIG_DIR / "rf_learning_curve.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return True


def _plot_error_analysis() -> bool:
    """Figure 5: 2-panel error analysis (residuals + |error| vs MW)."""
    error_path = SAVED_DIR / "rf_esper_error_analysis.csv"
    if not error_path.exists():
        logger.warning("Error analysis CSV not found at %s", error_path)
        return False

    error_df = pd.read_csv(error_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Residuals vs predicted
    ax = axes[0]
    ax.scatter(
        error_df["epsilon_k_pred"], error_df["residual"],
        alpha=0.4, s=20, color="teal", edgecolors="none",
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel(r"Predicted $\varepsilon$/k (K)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Residual (Predicted - Actual) (K)", fontsize=FONTSIZE_LABEL)
    ax.set_title("(a) Residuals vs Predicted", fontsize=FONTSIZE_TITLE - 1)
    ax.tick_params(labelsize=FONTSIZE_TICK)

    # Add LOESS-like trend (rolling median)
    sorted_df = error_df.sort_values("epsilon_k_pred")
    if len(sorted_df) >= 20:
        window = max(10, len(sorted_df) // 15)
        rolling_med = sorted_df["residual"].rolling(window=window, center=True).median()
        ax.plot(sorted_df["epsilon_k_pred"], rolling_med,
                color="red", linewidth=2, alpha=0.7, label="Rolling median")
        ax.legend(fontsize=FONTSIZE_LEGEND - 1)

    # Panel (b): |error| vs molecular weight
    ax = axes[1]
    mw = error_df["mol_weight"].values
    abs_err = error_df["abs_error"].values

    # Color by fluorination
    is_fluor = error_df["n_fluorine"] > 0
    ax.scatter(
        mw[~is_fluor], abs_err[~is_fluor],
        alpha=0.4, s=20, color="gray", edgecolors="none",
        label="Non-fluorinated",
    )
    ax.scatter(
        mw[is_fluor], abs_err[is_fluor],
        alpha=0.6, s=30, color="crimson", marker="D",
        label="Fluorinated",
    )

    ax.set_xlabel("Molecular Weight (Da)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r"|$\varepsilon$/k Error| (K)", fontsize=FONTSIZE_LABEL)
    ax.set_title("(b) |Error| vs Molecular Weight", fontsize=FONTSIZE_TITLE - 1)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.tick_params(labelsize=FONTSIZE_TICK)

    fig.tight_layout()
    path = FIG_DIR / "error_analysis_residuals.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return True


# ============================================================================
# 50.10: Write Report
# ============================================================================
def section_50_10_report():
    """Write the Step 50 results report."""
    logger.info("=" * 60)
    logger.info("SECTION 50.10: Write Report")
    logger.info("=" * 60)

    lines = []
    lines.append("# Results Report: RF Hyperparameter Sensitivity (Esper, 1,801 molecules)")
    lines.append("")

    # --- Summary ---
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "This step validates whether the production Random Forest's hyperparameters "
        "(n_estimators=100, max_features='sqrt', max_depth=None, min_samples_leaf=1) "
        "are reasonable for the deployed screening workflow, quantifies local "
        "sensitivity around the production configuration, and assesses whether "
        "a materially better configuration exists for epsilon_k prediction."
    )
    lines.append("")

    # --- Grid Search Results ---
    lines.append("## Grid Search Results")
    lines.append("")
    grid_path = SAVED_DIR / "rf_hyperparam_search.json"
    if grid_path.exists():
        with open(grid_path) as f:
            grid_data = json.load(f)

        best_params = grid_data.get("best_params", {})
        best_score = grid_data.get("best_score", 0)
        prod_rank = grid_data.get("production_rank")
        prod_score = grid_data.get("production_score")
        delta = grid_data.get("score_delta_best_minus_production")

        lines.append(f"**Best epsilon_k CV R-squared**: {best_score:.4f}")
        lines.append(f"**Best params**: {best_params}")
        lines.append("")

        if prod_rank is not None:
            n_configs = len(grid_data.get("params", []))
            lines.append(
                f"**Production config rank**: {prod_rank + 1} / {n_configs}"
            )
            lines.append(f"**Production CV R-squared**: {prod_score:.4f}")
            lines.append(f"**Delta (best - production)**: {delta:+.4f}")
            lines.append("")

        # Top 5 configurations
        if "rank_test_score" in grid_data and "params" in grid_data:
            ranks = grid_data["rank_test_score"]
            params = grid_data["params"]
            scores = grid_data["mean_test_score"]

            sorted_indices = sorted(range(len(ranks)), key=lambda i: ranks[i])[:5]

            lines.append("### Top 5 Configurations")
            lines.append("")
            lines.append("| Rank | n_est | max_features | max_depth | min_samples_leaf | CV R2 |")
            lines.append("|------|-------|-------------|-----------|-----------------|-------|")
            for idx in sorted_indices:
                p = params[idx]
                lines.append(
                    f"| {ranks[idx]} | {p.get('n_estimators', '?')} | "
                    f"{p.get('max_features', '?')} | {p.get('max_depth', '?')} | "
                    f"{p.get('min_samples_leaf', '?')} | {scores[idx]:.4f} |"
                )
            lines.append("")
    else:
        lines.append("Grid search results not available. Run `--section grid`.")
        lines.append("")

    # --- Local Sensitivity ---
    lines.append("## Local Sensitivity Analysis")
    lines.append("")
    oat_path = SAVED_DIR / "rf_oat_sensitivity.json"
    if oat_path.exists():
        lines.append(
            "One-at-a-time sweeps hold three hyperparameters at production defaults "
            "and vary the fourth. Results show local sensitivity near the production point."
        )
        lines.append("")
        lines.append(
            "See `figures/50_rf_hyperparameter_sensitivity/oat_sensitivity_epsilon_k.png`."
        )
    else:
        lines.append("OAT sensitivity results not available. Run `--section oat`.")
    lines.append("")

    # --- Production vs Best ---
    lines.append("## Production vs Best Comparison")
    lines.append("")
    compare_path = SAVED_DIR / "rf_production_vs_best.json"
    if compare_path.exists():
        with open(compare_path) as f:
            comp_data = json.load(f)

        prod_m = comp_data.get("production_metrics", {})
        best_m = comp_data.get("best_metrics", {})
        bootstrap = comp_data.get("paired_bootstrap", {})

        lines.append("| Metric | Production | Best Config | Delta |")
        lines.append("|--------|-----------|------------|-------|")
        for metric in ["r2", "mae", "rmse"]:
            pv = prod_m.get(metric, 0)
            bv = best_m.get(metric, 0)
            lines.append(f"| {metric.upper()} | {pv:.4f} | {bv:.4f} | {bv - pv:+.4f} |")
        lines.append("")

        lines.append("**Paired Bootstrap (2000 resamples):**")
        lines.append("")
        delta_r2_ci = bootstrap.get("delta_r2_ci_95", [0, 0])
        delta_mae_ci = bootstrap.get("delta_mae_ci_95", [0, 0])
        lines.append(
            f"- Delta R2 95% CI: [{delta_r2_ci[0]:+.4f}, {delta_r2_ci[1]:+.4f}]"
        )
        lines.append(
            f"- Delta MAE 95% CI: [{delta_mae_ci[0]:+.3f}, {delta_mae_ci[1]:+.3f}]"
        )
        spans_zero = bootstrap.get("ci_spans_zero_r2", True)
        if spans_zero:
            lines.append(
                "- The 95% CI for R2 delta **spans zero**, indicating the difference "
                "is not statistically significant."
            )
        else:
            lines.append(
                "- The 95% CI for R2 delta **does not span zero**, indicating a "
                "statistically significant difference."
            )
    else:
        lines.append("Comparison results not available. Run `--section compare`.")
    lines.append("")

    # --- External Validation ---
    lines.append("## External Validation of Candidate Configs")
    lines.append("")
    ext_path = SAVED_DIR / "rf_external_config_comparison.json"
    if ext_path.exists():
        with open(ext_path) as f:
            ext_data = json.load(f)

        lines.append("| Config | epsilon_k MAE | epsilon_k R2 | BP MAE (K) |")
        lines.append("|--------|-------------|------------|-----------|")
        for name, vals in ext_data.items():
            bp_mae = vals.get("boiling_point_mae_K", "N/A")
            bp_str = f"{bp_mae:.1f}" if isinstance(bp_mae, (int, float)) else bp_mae
            lines.append(
                f"| {name} | {vals['epsilon_k_mae']:.3f} | "
                f"{vals['epsilon_k_r2']:.3f} | {bp_str} |"
            )
        lines.append("")
    else:
        lines.append("External validation not available. Run `--section external`.")
    lines.append("")

    # --- Feature Importance ---
    lines.append("## Feature Importance Stability")
    lines.append("")
    imp_path = SAVED_DIR / "rf_feature_importance_stability.json"
    if imp_path.exists():
        with open(imp_path) as f:
            imp_data = json.load(f)

        consistency = imp_data.get("consistency_epsilon_k", {})
        n_consistent = consistency.get("n_consistent_all_folds", "?")
        n_morgan = consistency.get("n_morgan_in_top20", "?")
        n_rdkit = consistency.get("n_rdkit_in_top20", "?")

        lines.append(
            f"- **Top 20 features consistent across all 5 folds**: {n_consistent}/20"
        )
        lines.append(f"- **Morgan FP bits in top 20**: {n_morgan}")
        lines.append(f"- **RDKit descriptors in top 20**: {n_rdkit}")
        lines.append("")

        top_feats = consistency.get("top_features_by_mean", [])[:10]
        if top_feats:
            lines.append("### Top 10 Features by Mean Gini Importance (epsilon_k)")
            lines.append("")
            lines.append("| Feature | Mean Importance | Std | CV |")
            lines.append("|---------|----------------|-----|-----|")
            for feat in top_feats:
                lines.append(
                    f"| {feat['feature']} | {feat['mean_importance']:.4f} | "
                    f"{feat['std_importance']:.4f} | {feat['cv']:.2f} |"
                )
            lines.append("")

        lines.append(
            "**Caveat**: With ~2,200 features, many are correlated. Both Gini and "
            "permutation importance distribute importance across correlated groups. "
            "Stable rankings indicate repeatability, not direct physical interpretability."
        )
        lines.append("")
        lines.append(
            "See `figures/50_rf_hyperparameter_sensitivity/feature_importance_stability.png`."
        )
    else:
        lines.append("Feature importance results not available. Run `--section importance`.")
    lines.append("")

    # --- Learning Curve ---
    lines.append("## Data Efficiency (Learning Curve)")
    lines.append("")
    lc_path = SAVED_DIR / "rf_learning_curve.json"
    if lc_path.exists():
        with open(lc_path) as f:
            lc_data = json.load(f)

        train_sizes = lc_data["train_sizes"]
        test_mean = lc_data["test_scores_mean"]

        lines.append(
            f"The RF learning curve shows CV test R-squared for epsilon_k "
            f"at training set sizes from {min(train_sizes)} to {max(train_sizes)} molecules."
        )
        lines.append("")
        lines.append("| Train Size | CV Test R2 (mean) |")
        lines.append("|-----------|------------------|")
        for n, score in zip(train_sizes, test_mean):
            lines.append(f"| {n} | {score:.4f} |")
        lines.append("")

        # Assess plateau
        if len(test_mean) >= 3:
            last_gain = test_mean[-1] - test_mean[-2]
            lines.append(
                f"Last increment: {last_gain:+.4f} R-squared. "
            )
            if abs(last_gain) < 0.01:
                lines.append(
                    "The curve appears to have plateaued, supporting the "
                    "representation bottleneck interpretation."
                )
            else:
                lines.append(
                    "The curve is still rising, suggesting more data "
                    "could improve performance."
                )
        lines.append("")
        lines.append(
            "See `figures/50_rf_hyperparameter_sensitivity/rf_learning_curve.png`."
        )
    else:
        lines.append("Learning curve results not available. Run `--section learning`.")
    lines.append("")

    # --- Error Analysis ---
    lines.append("## Error Analysis")
    lines.append("")
    error_path = SAVED_DIR / "rf_esper_error_analysis.csv"
    if error_path.exists():
        error_df = pd.read_csv(error_path)
        n_test = len(error_df)
        median_err = error_df["abs_error"].median()
        mean_err = error_df["abs_error"].mean()

        lines.append(
            f"Systematic error analysis on {n_test} Esper test molecules. "
            f"Median |epsilon_k error|: {median_err:.1f} K, "
            f"mean: {mean_err:.1f} K."
        )
        lines.append("")

        # Top 10 worst
        lines.append("### Top 10 Worst-Predicted Molecules (epsilon_k)")
        lines.append("")
        lines.append("| SMILES | True | Pred | |Error| | MW |")
        lines.append("|--------|------|------|--------|-----|")
        for _, row in error_df.head(10).iterrows():
            smi = row["smiles"][:35]
            lines.append(
                f"| {smi} | {row['epsilon_k_true']:.1f} | "
                f"{row['epsilon_k_pred']:.1f} | {row['abs_error']:.1f} | "
                f"{row['mol_weight']:.0f} |"
            )
        lines.append("")

        # Error by group
        fluor_mask = error_df["n_fluorine"] > 0
        ring_mask = error_df["n_rings"] > 0
        hal_mask = error_df["n_halogens"] > 0

        lines.append("### Median |Error| by Molecular Property")
        lines.append("")
        lines.append("| Group | Median |Error| (K) | n |")
        lines.append("|-------|---------------------|---|")
        lines.append(
            f"| All | {error_df['abs_error'].median():.1f} | {n_test} |"
        )
        if fluor_mask.sum() > 0:
            lines.append(
                f"| Fluorinated | {error_df[fluor_mask]['abs_error'].median():.1f} | "
                f"{fluor_mask.sum()} |"
            )
        lines.append(
            f"| Non-fluorinated | {error_df[~fluor_mask]['abs_error'].median():.1f} | "
            f"{(~fluor_mask).sum()} |"
        )
        if ring_mask.sum() > 0:
            lines.append(
                f"| Ring-containing | {error_df[ring_mask]['abs_error'].median():.1f} | "
                f"{ring_mask.sum()} |"
            )
        if hal_mask.sum() > 0:
            lines.append(
                f"| Halogenated | {error_df[hal_mask]['abs_error'].median():.1f} | "
                f"{hal_mask.sum()} |"
            )
        lines.append("")
        lines.append(
            "See `figures/50_rf_hyperparameter_sensitivity/error_analysis_residuals.png`."
        )
    else:
        lines.append("Error analysis not available. Run `--section errors`.")
    lines.append("")

    # --- Key Findings ---
    lines.append("## Key Findings")
    lines.append("")
    lines.append(
        "1. The production RF hyperparameters can be validated through "
        "target-specific grid search and one-at-a-time sensitivity analysis."
    )
    lines.append(
        "2. Feature importance stability analysis reveals whether the top "
        "features are consistent across CV folds."
    )
    lines.append(
        "3. The RF learning curve provides data efficiency evidence for "
        "the 'representation bottleneck' narrative."
    )
    lines.append(
        "4. Systematic error analysis identifies failure modes and "
        "molecular property correlations."
    )
    lines.append("")

    # --- Figures ---
    lines.append("## Figures")
    lines.append("")
    lines.append("See `figures/50_rf_hyperparameter_sensitivity/` for:")
    lines.append(
        "1. `oat_sensitivity_epsilon_k.png` -- "
        "One-at-a-time sensitivity (2x2 grid)"
    )
    lines.append(
        "2. `cv_heatmap_n_estimators_vs_max_depth.png` -- "
        "CV heatmap of n_estimators vs max_depth"
    )
    lines.append(
        "3. `feature_importance_stability.png` -- "
        "Box plots for top 20 features across 5 folds"
    )
    lines.append(
        "4. `rf_learning_curve.png` -- "
        "R-squared vs training set size"
    )
    lines.append(
        "5. `error_analysis_residuals.png` -- "
        "Residuals and |error| vs molecular weight"
    )
    lines.append("")

    # --- Readiness Check ---
    lines.append("## Readiness Check")
    lines.append("")
    lines.append("- [ ] Primary grid search completed for epsilon_k (per-target RF)")
    lines.append("- [ ] OAT sensitivity plots show local sensitivity near production")
    lines.append("- [ ] Production vs best uses paired bootstrap (not CI overlap)")
    lines.append("- [ ] Top configs checked on fluorinated external validation")
    lines.append("- [ ] Feature importance stability assessed across 5 folds")
    lines.append("- [ ] RF learning curve generated for epsilon_k")
    lines.append("- [ ] Systematic error analysis with residual plots and worst molecules")
    lines.append("- [ ] At least 5 figures saved to `figures/50_rf_hyperparameter_sensitivity/`")
    lines.append("- [ ] Report written")
    lines.append("")

    report_path = REPORT_DIR / "50_rf_hyperparameter_sensitivity.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s", report_path)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 50: RF Hyperparameter Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sections:
    grid        Full grid search (50.1)
    oat         One-at-a-time sensitivity (50.2)
    compare     Production vs best config (50.3)
    external    Fluorinated validation of top configs (50.4)
    importance  Feature importance stability (50.5)
    learning    RF data efficiency learning curve (50.6)
    errors      Systematic error analysis (50.7)
    figures     Generate all figures (50.8)
    report      Write report (50.10)
    all         Run everything (default)
""",
    )
    parser.add_argument(
        "--section",
        choices=[
            "grid", "oat", "compare", "external", "importance",
            "learning", "errors", "figures", "report", "all",
        ],
        default="all",
        help="Which section to run (default: all)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 50: RF Hyperparameter Sensitivity Analysis")
    logger.info("Section: %s", args.section)
    logger.info("=" * 60)

    if args.section == "all":
        section_50_1_grid()
        section_50_2_oat()
        section_50_3_compare()
        section_50_4_external()
        section_50_5_importance()
        section_50_6_learning()
        section_50_7_errors()
        section_50_8_figures()
        section_50_10_report()
    elif args.section == "grid":
        section_50_1_grid()
    elif args.section == "oat":
        section_50_2_oat()
    elif args.section == "compare":
        section_50_3_compare()
    elif args.section == "external":
        section_50_4_external()
    elif args.section == "importance":
        section_50_5_importance()
    elif args.section == "learning":
        section_50_6_learning()
    elif args.section == "errors":
        section_50_7_errors()
    elif args.section == "figures":
        section_50_8_figures()
    elif args.section == "report":
        section_50_10_report()

    logger.info("=" * 60)
    logger.info("Step 50 complete (section=%s).", args.section)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
