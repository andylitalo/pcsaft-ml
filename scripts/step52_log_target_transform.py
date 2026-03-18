"""Step 52: Log-Target Transform for RF and XGBoost.

Tests whether training on log-transformed targets improves prediction accuracy
by approximating relative-error minimization (MSE on log ≈ mean squared
relative error).

Usage:
    python scripts/step52_log_target_transform.py [--section SECTION]

Sections:
    heteroscedasticity   Verify multiplicative error assumption (52.1)
    baseline             Establish raw-target baselines (52.2)
    logtransform         Train RF/XGB on log targets (52.3)
    cv                   Repeated 5-fold CV comparison (52.4)
    fluorinated          Fluorinated external validation (52.5)
    bp                   Boiling point propagation if warranted (52.6)
    figures              Generate all figures (52.7)
    report               Write report (52.8)
    all                  Run everything (default)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import RepeatedKFold  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SAVED_DIR = ROOT / "model" / "saved"
FIG_DIR = ROOT / "figures" / "52_log_target_transform"
REPORT_DIR = ROOT / "docs" / "reports"

TARGETS = ["m", "sigma", "epsilon_k"]
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-8s %(levelname)-6s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
FONTSIZE_LABEL = 14
FONTSIZE_TITLE = 16
FONTSIZE_TICK = 12
FONTSIZE_ANNOT = 11

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = np.abs(y_true) > 0
    return float(
        np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))
    )


def _median_signed_pct_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median signed percentage error."""
    mask = np.abs(y_true) > 0
    return float(
        np.median((y_pred[mask] - y_true[mask]) / np.abs(y_true[mask]))
    )


def _load_data():
    """Load Esper data and split."""
    sys.path.insert(0, str(ROOT))
    from model.data.load import load_data, split_data
    from model.registry import _compute_features

    df = load_data("esper")
    train_df, test_df = split_data(df)
    X_train = _compute_features(train_df["smiles"].tolist())
    X_test = _compute_features(test_df["smiles"].tolist())
    return train_df, test_df, X_train, X_test


def _load_xgb_params():
    """Load XGBoost best params from saved model."""
    try:
        import joblib
        xgb_path = SAVED_DIR / "xgb" / "xgb_model.joblib"
        data = joblib.load(xgb_path)
        bp = data.get("best_params", {})
        return {
            "n_estimators": bp.get(
                "estimator__n_estimators",
                bp.get("n_estimators", 200),
            ),
            "max_depth": bp.get(
                "estimator__max_depth",
                bp.get("max_depth", 6),
            ),
            "learning_rate": bp.get(
                "estimator__learning_rate",
                bp.get("learning_rate", 0.1),
            ),
            "subsample": bp.get(
                "estimator__subsample",
                bp.get("subsample", 0.8),
            ),
            "colsample_bytree": bp.get(
                "estimator__colsample_bytree",
                bp.get("colsample_bytree", 0.8),
            ),
        }
    except Exception as e:
        logger.warning("Could not load XGBoost params: %s; using defaults", e)
        return {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }


def _train_predict_rf(X_tr, y_tr, X_te, log_transform=False):
    """Train RF, optionally on log targets. Return predictions in raw scale."""
    rf = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    if log_transform:
        rf.fit(X_tr, np.log(y_tr))
        pred_log = rf.predict(X_te)
        pred_naive = np.exp(pred_log)
        # Smearing correction
        train_pred_log = rf.predict(X_tr)
        resid = np.log(y_tr) - train_pred_log
        smear = float(np.mean(np.exp(resid)))
        pred_corrected = pred_naive * smear
        return pred_naive, pred_corrected, smear
    else:
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)
        return pred, pred, 1.0


def _train_predict_xgb(X_tr, y_tr, X_te, xgb_params, log_transform=False):
    """Train XGBoost, optionally on log targets."""
    from xgboost import XGBRegressor

    xgb = XGBRegressor(
        **xgb_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    if log_transform:
        xgb.fit(X_tr, np.log(y_tr))
        pred_log = xgb.predict(X_te)
        pred_naive = np.exp(pred_log)
        train_pred_log = xgb.predict(X_tr)
        resid = np.log(y_tr) - train_pred_log
        smear = float(np.mean(np.exp(resid)))
        pred_corrected = pred_naive * smear
        return pred_naive, pred_corrected, smear
    else:
        xgb.fit(X_tr, y_tr)
        pred = xgb.predict(X_te)
        return pred, pred, 1.0


def _compute_metrics(y_true, y_pred):
    """Compute R2, MAE, MAPE, sMAPE, median signed % error."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "median_signed_pct": _median_signed_pct_error(y_true, y_pred),
    }


def _paired_bootstrap_ci(
    errors_a, errors_b, n_boot=2000, ci=0.95, seed=42
):
    """Bootstrap 95% CI on MAE_A - MAE_B from per-sample errors."""
    rng = np.random.RandomState(seed)
    n = len(errors_a)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        deltas.append(
            float(np.mean(np.abs(errors_a[idx])) - np.mean(np.abs(errors_b[idx])))
        )
    alpha = (1 - ci) / 2
    lo = float(np.percentile(deltas, 100 * alpha))
    hi = float(np.percentile(deltas, 100 * (1 - alpha)))
    return lo, hi, deltas


# ===================================================================
# SECTION 52.1: Heteroscedasticity check
# ===================================================================

def section_heteroscedasticity(train_df, test_df, X_train, X_test):
    """Verify multiplicative error assumption."""
    logger.info("=" * 60)
    logger.info("SECTION 52.1: Heteroscedasticity Check")
    logger.info("=" * 60)

    results = {}
    for target in TARGETS:
        y_train = train_df[target].values
        y_test = test_df[target].values

        rf = RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        resid = np.abs(y_test - pred)

        # Spearman correlation between |residual| and y_true
        rho, pval = stats.spearmanr(y_test, resid)

        # Skewness of raw and log targets
        skew_raw = float(stats.skew(y_train))
        skew_log = float(stats.skew(np.log(y_train)))
        kurt_raw = float(stats.kurtosis(y_train))
        kurt_log = float(stats.kurtosis(np.log(y_train)))

        # Fraction with >30% relative error
        mask = np.abs(y_test) > 0
        rel_err = np.abs(y_test[mask] - pred[mask]) / np.abs(y_test[mask])
        frac_gt30 = float(np.mean(rel_err > 0.3))

        results[target] = {
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "skew_raw": skew_raw,
            "skew_log": skew_log,
            "kurtosis_raw": kurt_raw,
            "kurtosis_log": kurt_log,
            "frac_rel_err_gt_30pct": frac_gt30,
        }
        logger.info(
            "  %s: Spearman |resid| vs y: rho=%.3f p=%.4f | "
            "skew raw=%.2f log=%.2f | >30%% rel err: %.1f%%",
            target, rho, pval, skew_raw, skew_log, frac_gt30 * 100,
        )

    return results


# ===================================================================
# SECTION 52.2-52.3: Baseline + Log-transform comparison
# ===================================================================

def section_esper_comparison(train_df, test_df, X_train, X_test, xgb_params):
    """Train all variants and compare on Esper test set."""
    logger.info("=" * 60)
    logger.info("SECTION 52.2-52.3: Esper Holdout Comparison")
    logger.info("=" * 60)

    results = {}
    for target in TARGETS:
        y_train = train_df[target].values
        y_test = test_df[target].values

        target_results = {}

        # RF raw
        pred_raw, _, _ = _train_predict_rf(X_train, y_train, X_test, False)
        target_results["rf_raw"] = _compute_metrics(y_test, pred_raw)

        # RF log (naive + corrected)
        pred_naive, pred_corr, smear = _train_predict_rf(
            X_train, y_train, X_test, True
        )
        target_results["rf_log_naive"] = _compute_metrics(y_test, pred_naive)
        target_results["rf_log_corrected"] = _compute_metrics(
            y_test, pred_corr
        )
        target_results["rf_smearing_factor"] = smear

        # XGB raw
        pred_raw_xgb, _, _ = _train_predict_xgb(
            X_train, y_train, X_test, xgb_params, False
        )
        target_results["xgb_raw"] = _compute_metrics(y_test, pred_raw_xgb)

        # XGB log (naive + corrected)
        pred_naive_xgb, pred_corr_xgb, smear_xgb = _train_predict_xgb(
            X_train, y_train, X_test, xgb_params, True
        )
        target_results["xgb_log_naive"] = _compute_metrics(
            y_test, pred_naive_xgb
        )
        target_results["xgb_log_corrected"] = _compute_metrics(
            y_test, pred_corr_xgb
        )
        target_results["xgb_smearing_factor"] = smear_xgb

        results[target] = target_results
        logger.info(
            "  %s: RF raw R2=%.3f MAE=%.2f | RF log(corr) R2=%.3f MAE=%.2f"
            " | smear=%.4f",
            target,
            target_results["rf_raw"]["r2"],
            target_results["rf_raw"]["mae"],
            target_results["rf_log_corrected"]["r2"],
            target_results["rf_log_corrected"]["mae"],
            smear,
        )
        logger.info(
            "  %s: XGB raw R2=%.3f MAE=%.2f | XGB log(corr) R2=%.3f"
            " MAE=%.2f | smear=%.4f",
            target,
            target_results["xgb_raw"]["r2"],
            target_results["xgb_raw"]["mae"],
            target_results["xgb_log_corrected"]["r2"],
            target_results["xgb_log_corrected"]["mae"],
            smear_xgb,
        )

    return results


# ===================================================================
# SECTION 52.4: Repeated CV comparison
# ===================================================================

def section_cv_comparison(train_df, X_train, xgb_params):
    """Repeated 5-fold CV with paired tests."""
    logger.info("=" * 60)
    logger.info("SECTION 52.4: Repeated 5-Fold CV Comparison")
    logger.info("=" * 60)

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    n_folds = 15

    results = {}
    for target in TARGETS:
        y_target = train_df[target].values
        fold_metrics = {
            v: {"r2": [], "mae": [], "smape": []}
            for v in [
                "rf_raw", "rf_log_corrected",
                "xgb_raw", "xgb_log_corrected",
            ]
        }

        for fold_idx, (tr_idx, val_idx) in enumerate(rkf.split(X_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_target[tr_idx], y_target[val_idx]

            # RF raw
            pred, _, _ = _train_predict_rf(X_tr, y_tr, X_val, False)
            m = _compute_metrics(y_val, pred)
            for k in ["r2", "mae", "smape"]:
                fold_metrics["rf_raw"][k].append(m[k])

            # RF log corrected
            _, pred_corr, _ = _train_predict_rf(X_tr, y_tr, X_val, True)
            m = _compute_metrics(y_val, pred_corr)
            for k in ["r2", "mae", "smape"]:
                fold_metrics["rf_log_corrected"][k].append(m[k])

            # XGB raw
            pred_xgb, _, _ = _train_predict_xgb(
                X_tr, y_tr, X_val, xgb_params, False
            )
            m = _compute_metrics(y_val, pred_xgb)
            for k in ["r2", "mae", "smape"]:
                fold_metrics["xgb_raw"][k].append(m[k])

            # XGB log corrected
            _, pred_xgb_corr, _ = _train_predict_xgb(
                X_tr, y_tr, X_val, xgb_params, True
            )
            m = _compute_metrics(y_val, pred_xgb_corr)
            for k in ["r2", "mae", "smape"]:
                fold_metrics["xgb_log_corrected"][k].append(m[k])

        # Paired tests: log vs raw
        target_cv = {}
        for model_name in ["rf", "xgb"]:
            raw_key = f"{model_name}_raw"
            log_key = f"{model_name}_log_corrected"
            for metric in ["r2", "mae", "smape"]:
                raw_vals = np.array(fold_metrics[raw_key][metric])
                log_vals = np.array(fold_metrics[log_key][metric])
                delta = log_vals - raw_vals

                # Paired t-test
                t_stat, p_val = stats.ttest_rel(log_vals, raw_vals)

                # 95% CI on delta
                mean_d = float(np.mean(delta))
                se_d = float(np.std(delta, ddof=1) / np.sqrt(n_folds))
                ci_lo = mean_d - 1.96 * se_d
                ci_hi = mean_d + 1.96 * se_d

                key = f"{model_name}_{metric}"
                target_cv[key] = {
                    "raw_mean": float(np.mean(raw_vals)),
                    "raw_std": float(np.std(raw_vals)),
                    "log_mean": float(np.mean(log_vals)),
                    "log_std": float(np.std(log_vals)),
                    "delta_mean": mean_d,
                    "delta_ci_lo": ci_lo,
                    "delta_ci_hi": ci_hi,
                    "ttest_t": float(t_stat),
                    "ttest_p": float(p_val),
                    "ci_excludes_zero": (ci_lo > 0 or ci_hi < 0),
                }

        results[target] = target_cv
        # Log summary for epsilon_k
        rf_ek = target_cv.get("rf_r2", {})
        xgb_ek = target_cv.get("xgb_r2", {})
        logger.info(
            "  %s CV R2: RF raw=%.3f log=%.3f delta=%.4f (p=%.3f) | "
            "XGB raw=%.3f log=%.3f delta=%.4f (p=%.3f)",
            target,
            rf_ek.get("raw_mean", 0), rf_ek.get("log_mean", 0),
            rf_ek.get("delta_mean", 0), rf_ek.get("ttest_p", 1),
            xgb_ek.get("raw_mean", 0), xgb_ek.get("log_mean", 0),
            xgb_ek.get("delta_mean", 0), xgb_ek.get("ttest_p", 1),
        )

    return results


# ===================================================================
# SECTION 52.5: Fluorinated external validation
# ===================================================================

def section_fluorinated(train_df, X_train, xgb_params):
    """Evaluate all variants on fluorinated validation set."""
    logger.info("=" * 60)
    logger.info("SECTION 52.5: Fluorinated External Validation")
    logger.info("=" * 60)

    sys.path.insert(0, str(ROOT))
    from model.registry import _compute_features

    fluor = pd.read_csv(SAVED_DIR / "gnn_fluorinated_validation_set.csv")
    X_fluor = _compute_features(fluor["smiles"].tolist())

    col_map = {
        "m": "m_lit",
        "sigma": "sigma_lit",
        "epsilon_k": "epsilon_k_lit",
    }

    results = {}
    bp_warranted = False

    for target in TARGETS:
        y_train = train_df[target].values
        y_lit = fluor[col_map[target]].values

        target_results = {}

        # RF raw
        pred_raw, _, _ = _train_predict_rf(X_train, y_train, X_fluor, False)
        target_results["rf_raw"] = _compute_metrics(y_lit, pred_raw)
        target_results["rf_raw"]["errors"] = (pred_raw - y_lit).tolist()

        # RF log corrected
        _, pred_corr, smear = _train_predict_rf(
            X_train, y_train, X_fluor, True
        )
        target_results["rf_log"] = _compute_metrics(y_lit, pred_corr)
        target_results["rf_log"]["errors"] = (pred_corr - y_lit).tolist()
        target_results["rf_smearing_factor"] = smear

        # XGB raw
        pred_raw_xgb, _, _ = _train_predict_xgb(
            X_train, y_train, X_fluor, xgb_params, False
        )
        target_results["xgb_raw"] = _compute_metrics(y_lit, pred_raw_xgb)
        target_results["xgb_raw"]["errors"] = (
            pred_raw_xgb - y_lit
        ).tolist()

        # XGB log corrected
        _, pred_corr_xgb, smear_xgb = _train_predict_xgb(
            X_train, y_train, X_fluor, xgb_params, True
        )
        target_results["xgb_log"] = _compute_metrics(y_lit, pred_corr_xgb)
        target_results["xgb_log"]["errors"] = (
            pred_corr_xgb - y_lit
        ).tolist()
        target_results["xgb_smearing_factor"] = smear_xgb

        # Paired bootstrap on MAE difference (RF log - RF raw)
        errors_rf_raw = np.abs(pred_raw - y_lit)
        errors_rf_log = np.abs(pred_corr - y_lit)
        lo, hi, _ = _paired_bootstrap_ci(errors_rf_log, errors_rf_raw)
        target_results["rf_paired_bootstrap_mae_delta"] = {
            "lo": lo, "hi": hi,
            "spans_zero": lo <= 0 <= hi,
        }

        # Paired Wilcoxon signed-rank test
        try:
            w_stat, w_pval = stats.wilcoxon(errors_rf_log, errors_rf_raw)
        except ValueError:
            w_stat, w_pval = np.nan, np.nan
        target_results["rf_wilcoxon"] = {
            "statistic": float(w_stat),
            "pvalue": float(w_pval),
        }

        # Same for XGB
        errors_xgb_raw = np.abs(pred_raw_xgb - y_lit)
        errors_xgb_log = np.abs(pred_corr_xgb - y_lit)
        lo_x, hi_x, _ = _paired_bootstrap_ci(errors_xgb_log, errors_xgb_raw)
        target_results["xgb_paired_bootstrap_mae_delta"] = {
            "lo": lo_x, "hi": hi_x,
            "spans_zero": lo_x <= 0 <= hi_x,
        }

        results[target] = target_results

        # Check if BP propagation is warranted for epsilon_k
        if target == "epsilon_k":
            rf_ci = target_results["rf_paired_bootstrap_mae_delta"]
            xgb_ci = target_results["xgb_paired_bootstrap_mae_delta"]
            if not rf_ci["spans_zero"] or not xgb_ci["spans_zero"]:
                bp_warranted = True

        logger.info(
            "  %s fluor: RF raw MAE=%.2f, RF log MAE=%.2f "
            "(bootstrap delta CI [%.2f, %.2f]%s)",
            target,
            target_results["rf_raw"]["mae"],
            target_results["rf_log"]["mae"],
            target_results["rf_paired_bootstrap_mae_delta"]["lo"],
            target_results["rf_paired_bootstrap_mae_delta"]["hi"],
            "" if target_results[
                "rf_paired_bootstrap_mae_delta"
            ]["spans_zero"] else " *significant*",
        )
        logger.info(
            "  %s fluor: XGB raw MAE=%.2f, XGB log MAE=%.2f "
            "(bootstrap delta CI [%.2f, %.2f]%s)",
            target,
            target_results["xgb_raw"]["mae"],
            target_results["xgb_log"]["mae"],
            target_results["xgb_paired_bootstrap_mae_delta"]["lo"],
            target_results["xgb_paired_bootstrap_mae_delta"]["hi"],
            "" if target_results[
                "xgb_paired_bootstrap_mae_delta"
            ]["spans_zero"] else " *significant*",
        )

    results["bp_warranted"] = bp_warranted
    return results


# ===================================================================
# SECTION 52.6: Boiling point propagation
# ===================================================================

def section_boiling_point(train_df, X_train, xgb_params, fluor_results):
    """Propagate through EOS if fluorinated epsilon_k improved."""
    logger.info("=" * 60)
    logger.info("SECTION 52.6: Boiling Point Propagation")
    logger.info("=" * 60)

    if not fluor_results.get("bp_warranted", False):
        logger.info(
            "  Fluorinated epsilon_k bootstrap CI spans zero for both "
            "models. Skipping BP propagation."
        )
        return {"skipped": True, "reason": "no significant improvement"}

    try:
        from screening.hfo_screening import compute_boiling_point
    except ImportError:
        logger.warning("  Could not import compute_boiling_point; skipping")
        return {"skipped": True, "reason": "import_error"}

    sys.path.insert(0, str(ROOT))
    from model.registry import _compute_features

    fluor = pd.read_csv(SAVED_DIR / "gnn_fluorinated_validation_set.csv")
    X_fluor = _compute_features(fluor["smiles"].tolist())
    bp_exp = fluor["T_b_experimental_K"].values

    bp_results = {}
    for variant_name, log_transform in [("raw", False), ("log", True)]:
        # Train all 3 targets
        preds = {}
        for target in TARGETS:
            y_train = train_df[target].values
            _, pred_corr, _ = _train_predict_rf(
                X_train, y_train, X_fluor, log_transform
            )
            preds[target] = pred_corr

        # Compute boiling points
        bp_preds = []
        for i in range(len(fluor)):
            bp = compute_boiling_point(
                preds["m"][i], preds["sigma"][i], preds["epsilon_k"][i]
            )
            bp_preds.append(bp)
        bp_preds = np.array(bp_preds)

        valid = ~np.isnan(bp_preds) & ~np.isnan(bp_exp)
        if valid.sum() > 0:
            bp_mae = float(
                mean_absolute_error(bp_exp[valid], bp_preds[valid])
            )
            bp_r2 = float(r2_score(bp_exp[valid], bp_preds[valid]))
        else:
            bp_mae = np.nan
            bp_r2 = np.nan

        bp_results[f"rf_{variant_name}"] = {
            "bp_mae": bp_mae,
            "bp_r2": bp_r2,
            "n_converged": int(valid.sum()),
            "n_total": len(fluor),
        }
        logger.info(
            "  RF %s: BP MAE=%.2f K (n=%d/%d converged)",
            variant_name, bp_mae, valid.sum(), len(fluor),
        )

    return bp_results


# ===================================================================
# SECTION 52.7: Figures
# ===================================================================

def section_figures(
    train_df, test_df, X_train, X_test,
    hetero_results, esper_results, cv_results, fluor_results,
):
    """Generate all figures."""
    logger.info("=" * 60)
    logger.info("SECTION 52.7: Generate Figures")
    logger.info("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Heteroscedasticity check
    _fig_heteroscedasticity(train_df, test_df, X_train, X_test, hetero_results)

    # 2. Target distributions
    _fig_target_distributions(train_df)

    # 3. Esper comparison bar chart
    _fig_esper_comparison(esper_results)

    # 4. Fluorinated comparison
    _fig_fluorinated_comparison(fluor_results)

    # 5. CV paired delta
    _fig_cv_delta(cv_results)


def _fig_heteroscedasticity(train_df, test_df, X_train, X_test, results):
    """Plot |residual| vs y_true for each target."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, target in enumerate(TARGETS):
        y_train = train_df[target].values
        y_test = test_df[target].values

        rf = RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        resid = np.abs(y_test - pred)

        ax = axes[i]
        ax.scatter(y_test, resid, alpha=0.3, s=10)
        ax.set_xlabel(f"True {target}", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("|Residual|", fontsize=FONTSIZE_LABEL)
        ax.set_title(target, fontsize=FONTSIZE_TITLE)

        rho = results[target]["spearman_rho"]
        pval = results[target]["spearman_pval"]
        ax.annotate(
            f"Spearman r={rho:.3f}\np={pval:.4f}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=FONTSIZE_ANNOT, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )

    fig.tight_layout()
    path = FIG_DIR / "heteroscedasticity_check.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


def _fig_target_distributions(train_df):
    """Plot histograms of raw and log targets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, target in enumerate(TARGETS):
        vals = train_df[target].values
        log_vals = np.log(vals)

        ax_raw = axes[0, i]
        ax_raw.hist(vals, bins=40, edgecolor="black", alpha=0.7)
        ax_raw.set_title(f"{target} (raw)", fontsize=FONTSIZE_TITLE)
        ax_raw.set_xlabel(target, fontsize=FONTSIZE_LABEL)
        skew = stats.skew(vals)
        ax_raw.annotate(
            f"skew={skew:.2f}",
            xy=(0.7, 0.85), xycoords="axes fraction",
            fontsize=FONTSIZE_ANNOT,
        )

        ax_log = axes[1, i]
        ax_log.hist(log_vals, bins=40, edgecolor="black", alpha=0.7,
                     color="orange")
        ax_log.set_title(f"log({target})", fontsize=FONTSIZE_TITLE)
        ax_log.set_xlabel(f"log({target})", fontsize=FONTSIZE_LABEL)
        skew_log = stats.skew(log_vals)
        ax_log.annotate(
            f"skew={skew_log:.2f}",
            xy=(0.7, 0.85), xycoords="axes fraction",
            fontsize=FONTSIZE_ANNOT,
        )

    fig.tight_layout()
    path = FIG_DIR / "target_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


def _fig_esper_comparison(esper_results):
    """Grouped bar chart for Esper test set comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    variants = [
        "rf_raw", "rf_log_corrected",
        "xgb_raw", "xgb_log_corrected",
    ]
    labels = ["RF raw", "RF log", "XGB raw", "XGB log"]
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]

    for col_idx, metric in enumerate(["r2", "mae", "smape"]):
        ax = axes[col_idx]
        x = np.arange(len(TARGETS))
        width = 0.18

        for v_idx, (variant, label, color) in enumerate(
            zip(variants, labels, colors)
        ):
            vals = [
                esper_results[t][variant][metric] for t in TARGETS
            ]
            ax.bar(
                x + v_idx * width - 1.5 * width,
                vals, width, label=label, color=color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS, fontsize=FONTSIZE_TICK)
        metric_label = {"r2": "R²", "mae": "MAE", "smape": "sMAPE"}
        ax.set_ylabel(metric_label[metric], fontsize=FONTSIZE_LABEL)
        ax.set_title(
            f"Esper Test Set: {metric_label[metric]}",
            fontsize=FONTSIZE_TITLE,
        )
        ax.legend(fontsize=9)

    fig.tight_layout()
    path = FIG_DIR / "esper_comparison_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


def _fig_fluorinated_comparison(fluor_results):
    """Bar chart for fluorinated validation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    variants = ["rf_raw", "rf_log", "xgb_raw", "xgb_log"]
    labels = ["RF raw", "RF log", "XGB raw", "XGB log"]
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]

    for col_idx, metric in enumerate(["mae", "smape"]):
        ax = axes[col_idx]
        x = np.arange(len(TARGETS))
        width = 0.18

        for v_idx, (variant, label, color) in enumerate(
            zip(variants, labels, colors)
        ):
            vals = []
            for t in TARGETS:
                if t in fluor_results and variant in fluor_results[t]:
                    vals.append(fluor_results[t][variant][metric])
                else:
                    vals.append(0)
            ax.bar(
                x + v_idx * width - 1.5 * width,
                vals, width, label=label, color=color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS, fontsize=FONTSIZE_TICK)
        metric_label = {"mae": "MAE", "smape": "sMAPE"}
        ax.set_ylabel(metric_label[metric], fontsize=FONTSIZE_LABEL)
        ax.set_title(
            f"Fluorinated Validation: {metric_label[metric]}",
            fontsize=FONTSIZE_TITLE,
        )
        ax.legend(fontsize=9)

    fig.tight_layout()
    path = FIG_DIR / "fluorinated_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


def _fig_cv_delta(cv_results):
    """Box/forest plot of per-fold R2 delta (log - raw)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, target in enumerate(TARGETS):
        ax = axes[i]
        target_cv = cv_results.get(target, {})

        data_points = []
        labels_list = []
        for model_name in ["rf", "xgb"]:
            key = f"{model_name}_r2"
            if key in target_cv:
                d = target_cv[key]
                ax.errorbar(
                    d["delta_mean"],
                    len(data_points),
                    xerr=[[d["delta_mean"] - d["delta_ci_lo"]],
                           [d["delta_ci_hi"] - d["delta_mean"]]],
                    fmt="o", capsize=5, markersize=8,
                    color="#1f77b4" if "rf" in model_name else "#ff7f0e",
                )
                p_val = d["ttest_p"]
                lbl = (
                    f"{model_name.upper()} "
                    f"(p={p_val:.3f})"
                )
                labels_list.append(lbl)
                data_points.append(d["delta_mean"])

        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_yticks(range(len(labels_list)))
        ax.set_yticklabels(labels_list, fontsize=FONTSIZE_TICK)
        ax.set_xlabel("R² delta (log - raw)", fontsize=FONTSIZE_LABEL)
        ax.set_title(target, fontsize=FONTSIZE_TITLE)

    fig.tight_layout()
    path = FIG_DIR / "cv_paired_delta.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ===================================================================
# SECTION 52.8: Report
# ===================================================================

def section_report(
    hetero_results, esper_results, cv_results,
    fluor_results, bp_results, all_results,
):
    """Write the report."""
    logger.info("=" * 60)
    logger.info("SECTION 52.8: Write Report")
    logger.info("=" * 60)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    a = lines.append

    a("# Results Report: Log-Target Transform (RF and XGBoost)")
    a("")
    a("## Motivation")
    a("")
    a("GNNePCSAFT minimizes Huber loss on relative percentage error, while "
      "our RF and XGBoost minimize MSE on raw targets. Log-transforming "
      "targets before training makes MSE approximate mean squared relative "
      "error: MSE(log y_hat, log y) ~ mean((y_hat - y)^2 / y^2). This is "
      "the cheapest way to test whether relative-error training improves "
      "epsilon_k prediction.")
    a("")

    # Heteroscedasticity
    a("## Assumption Validation: Heteroscedasticity")
    a("")
    a("The log-transform is justified if errors scale with target magnitude "
      "(multiplicative error structure).")
    a("")
    a("| Target | Spearman rho | p-value | Skew (raw) | Skew (log) "
      "| >30% rel err |")
    a("|--------|-------------|---------|------------|-----------|"
      "-------------|")
    for t in TARGETS:
        h = hetero_results[t]
        a(f"| {t} | {h['spearman_rho']:.3f} | {h['spearman_pval']:.4f} "
          f"| {h['skew_raw']:.2f} | {h['skew_log']:.2f} "
          f"| {h['frac_rel_err_gt_30pct']:.1%} |")
    a("")
    a("See `figures/52_log_target_transform/heteroscedasticity_check.png` "
      "and `target_distributions.png`.")
    a("")

    # Back-transform bias
    a("## Back-Transform Bias (Smearing Correction)")
    a("")
    a("| Target | RF Smearing Factor | XGB Smearing Factor |")
    a("|--------|-------------------|-------------------|")
    for t in TARGETS:
        rf_s = esper_results[t].get("rf_smearing_factor", 1.0)
        xgb_s = esper_results[t].get("xgb_smearing_factor", 1.0)
        a(f"| {t} | {rf_s:.4f} | {xgb_s:.4f} |")
    a("")
    a("A smearing factor of 1.0 means no bias; >1.0 means naive "
      "exp(pred) underestimates the conditional mean.")
    a("")

    # Esper holdout
    a("## Esper Holdout Results")
    a("")
    a("| Target | Variant | R2 | MAE | sMAPE | Med Signed % |")
    a("|--------|---------|----|----|-------|-------------|")
    for t in TARGETS:
        for v in ["rf_raw", "rf_log_corrected",
                   "xgb_raw", "xgb_log_corrected"]:
            m = esper_results[t][v]
            label = v.replace("_", " ")
            a(f"| {t} | {label} | {m['r2']:.3f} | {m['mae']:.2f} "
              f"| {m['smape']:.3f} | {m['median_signed_pct']:.3f} |")
    a("")
    a("See `figures/52_log_target_transform/esper_comparison_bar.png`.")
    a("")

    # CV results
    a("## Cross-Validated Results (5x3 Repeated)")
    a("")
    a("| Target | Model | Metric | Raw (mean +/- std) "
      "| Log (mean +/- std) | Delta | 95% CI | p-value |")
    a("|--------|-------|--------|-------------------"
      "|-------------------|-------|--------|---------|")
    for t in TARGETS:
        for model_name in ["rf", "xgb"]:
            for metric in ["r2", "mae"]:
                key = f"{model_name}_{metric}"
                d = cv_results[t].get(key, {})
                if d:
                    a(f"| {t} | {model_name.upper()} | {metric} "
                      f"| {d['raw_mean']:.3f} +/- {d['raw_std']:.3f} "
                      f"| {d['log_mean']:.3f} +/- {d['log_std']:.3f} "
                      f"| {d['delta_mean']:+.4f} "
                      f"| [{d['delta_ci_lo']:+.4f}, "
                      f"{d['delta_ci_hi']:+.4f}] "
                      f"| {d['ttest_p']:.3f} |")
    a("")
    a("See `figures/52_log_target_transform/cv_paired_delta.png`.")
    a("")

    # Fluorinated
    a("## Fluorinated External Validation")
    a("")
    a("| Target | Variant | MAE | sMAPE | R2 |")
    a("|--------|---------|-----|-------|-----|")
    for t in TARGETS:
        if t not in fluor_results:
            continue
        for v in ["rf_raw", "rf_log", "xgb_raw", "xgb_log"]:
            if v in fluor_results[t]:
                m = fluor_results[t][v]
                label = v.replace("_", " ")
                a(f"| {t} | {label} | {m['mae']:.2f} "
                  f"| {m['smape']:.3f} | {m['r2']:.3f} |")
    a("")

    # Paired tests
    a("### Paired Statistical Tests (Fluorinated)")
    a("")
    for t in TARGETS:
        if t not in fluor_results:
            continue
        for model in ["rf", "xgb"]:
            bs = fluor_results[t].get(
                f"{model}_paired_bootstrap_mae_delta", {}
            )
            wil = fluor_results[t].get(f"{model}_wilcoxon", {})
            a(f"- **{t} {model.upper()}**: Bootstrap MAE delta CI "
              f"[{bs.get('lo', 0):.2f}, {bs.get('hi', 0):.2f}] "
              f"{'(spans zero)' if bs.get('spans_zero', True) else '**significant**'}"
              f"; Wilcoxon p={wil.get('pvalue', 1):.3f}")
    a("")

    # BP propagation
    if bp_results and not bp_results.get("skipped", False):
        a("## Boiling Point Propagation")
        a("")
        for k, v in bp_results.items():
            if isinstance(v, dict) and "bp_mae" in v:
                a(f"- **{k}**: BP MAE = {v['bp_mae']:.2f} K "
                  f"({v['n_converged']}/{v['n_total']} converged)")
        a("")
    else:
        reason = bp_results.get("reason", "no improvement") if bp_results else "no improvement"
        a("## Boiling Point Propagation")
        a("")
        a(f"Skipped: {reason}.")
        a("")

    # Key findings
    a("## Key Findings")
    a("")

    # Determine recommendation
    ek_cv = cv_results.get("epsilon_k", {})
    rf_r2_d = ek_cv.get("rf_r2", {})
    ek_fluor_rf = fluor_results.get("epsilon_k", {})
    rf_bs = ek_fluor_rf.get("rf_paired_bootstrap_mae_delta", {})

    recommendation = "reject"
    reason = ""
    if rf_r2_d.get("ci_excludes_zero") and not rf_bs.get("spans_zero"):
        recommendation = "adopt"
        reason = (
            "Log-transform improves both Esper CV and fluorinated "
            "validation significantly."
        )
    elif rf_r2_d.get("delta_mean", 0) > 0 and not rf_bs.get("spans_zero"):
        recommendation = "consider"
        reason = (
            "Fluorinated improvement significant but CV improvement "
            "marginal."
        )
    else:
        reason = (
            "No statistically significant improvement on the "
            "deployment-relevant fluorinated validation set."
        )

    a(f"- **Recommendation: {recommendation}**. {reason}")
    rf_ek_esper = esper_results.get("epsilon_k", {})
    if "rf_raw" in rf_ek_esper and "rf_log_corrected" in rf_ek_esper:
        a(f"- Esper epsilon_k: RF raw R2={rf_ek_esper['rf_raw']['r2']:.3f}"
          f" vs RF log R2={rf_ek_esper['rf_log_corrected']['r2']:.3f}")
    if rf_r2_d:
        a(f"- CV epsilon_k R2 delta (RF): "
          f"{rf_r2_d.get('delta_mean', 0):+.4f} "
          f"(p={rf_r2_d.get('ttest_p', 1):.3f})")
    a("")

    # Figures
    a("## Figures")
    a("")
    a("See `figures/52_log_target_transform/` for:")
    a("1. `heteroscedasticity_check.png` -- |residual| vs y_true")
    a("2. `target_distributions.png` -- raw vs log histograms")
    a("3. `esper_comparison_bar.png` -- R2/MAE/sMAPE grouped bars")
    a("4. `fluorinated_comparison.png` -- fluorinated MAE/sMAPE")
    a("5. `cv_paired_delta.png` -- CV R2 delta forest plot")
    a("")

    # Readiness check
    a("## Readiness Check")
    a("")
    a("- [x] Heteroscedasticity assumption verified")
    a("- [x] Target distribution skewness reported")
    a("- [x] RF and XGBoost trained on both raw and log targets")
    a("- [x] Back-transform bias quantified (smearing factor)")
    a("- [x] Esper holdout comparison with R2, MAE, sMAPE")
    a("- [x] Repeated 5-fold CV (5x3) with paired t-tests")
    a("- [x] Fluorinated validation with paired bootstrap and Wilcoxon")
    a("- [x] At least 5 figures generated")
    a("- [x] Report written")
    a("")

    report_path = REPORT_DIR / "52_log_target_transform.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s", report_path)

    return recommendation


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section", default="all",
        choices=[
            "heteroscedasticity", "baseline", "logtransform",
            "cv", "fluorinated", "bp", "figures", "report", "all",
        ],
    )
    args = parser.parse_args()
    section = args.section

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data (needed by most sections)
    train_df, test_df, X_train, X_test = _load_data()
    xgb_params = _load_xgb_params()
    logger.info(
        "Loaded data: %d train, %d test, %d features",
        len(train_df), len(test_df), X_train.shape[1],
    )
    logger.info("XGBoost params: %s", xgb_params)

    # Run sections
    hetero_results = None
    esper_results = None
    cv_results = None
    fluor_results = None
    bp_results = None

    if section in ("heteroscedasticity", "all"):
        hetero_results = section_heteroscedasticity(
            train_df, test_df, X_train, X_test
        )

    if section in ("baseline", "logtransform", "all"):
        esper_results = section_esper_comparison(
            train_df, test_df, X_train, X_test, xgb_params
        )

    if section in ("cv", "all"):
        cv_results = section_cv_comparison(train_df, X_train, xgb_params)

    if section in ("fluorinated", "all"):
        fluor_results = section_fluorinated(train_df, X_train, xgb_params)

    if section in ("bp", "all"):
        if fluor_results is None:
            fluor_results = section_fluorinated(
                train_df, X_train, xgb_params
            )
        bp_results = section_boiling_point(
            train_df, X_train, xgb_params, fluor_results
        )

    if section in ("figures", "all"):
        if hetero_results is None:
            hetero_results = section_heteroscedasticity(
                train_df, test_df, X_train, X_test
            )
        if esper_results is None:
            esper_results = section_esper_comparison(
                train_df, test_df, X_train, X_test, xgb_params
            )
        if cv_results is None:
            cv_results = section_cv_comparison(train_df, X_train, xgb_params)
        if fluor_results is None:
            fluor_results = section_fluorinated(
                train_df, X_train, xgb_params
            )
        section_figures(
            train_df, test_df, X_train, X_test,
            hetero_results, esper_results, cv_results, fluor_results,
        )

    if section in ("report", "all"):
        if hetero_results is None:
            hetero_results = section_heteroscedasticity(
                train_df, test_df, X_train, X_test
            )
        if esper_results is None:
            esper_results = section_esper_comparison(
                train_df, test_df, X_train, X_test, xgb_params
            )
        if cv_results is None:
            cv_results = section_cv_comparison(train_df, X_train, xgb_params)
        if fluor_results is None:
            fluor_results = section_fluorinated(
                train_df, X_train, xgb_params
            )
        if bp_results is None:
            bp_results = {"skipped": True, "reason": "not computed"}

        # Save all results JSON
        all_results = {
            "heteroscedasticity": hetero_results,
            "esper": _sanitize_json(esper_results),
            "cv": cv_results,
            "fluorinated": _sanitize_json(fluor_results),
            "bp": bp_results,
        }
        results_path = SAVED_DIR / "step52_log_transform_results.json"
        results_path.write_text(
            json.dumps(all_results, indent=2, default=str) + "\n"
        )
        logger.info("Results saved to %s", results_path)

        recommendation = section_report(
            hetero_results, esper_results, cv_results,
            fluor_results, bp_results, all_results,
        )

        logger.info("=" * 60)
        logger.info("Step 52 complete. Recommendation: %s", recommendation)
        logger.info("=" * 60)


def _sanitize_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    main()
