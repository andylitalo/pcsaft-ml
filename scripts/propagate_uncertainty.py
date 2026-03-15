#!/usr/bin/env python3
"""Propagate RF prediction uncertainty through thermodynamic calculations.

For each VP-passing candidate, extracts per-tree (m, sigma, epsilon_k) from
the RF model and propagates each through teqp to compute distributions of
VP_ratio and H_ratio.  k_ij sensitivity is swept simultaneously.

Outputs:
    model/saved/uncertainty_propagated.csv
        Per-molecule columns: H_ratio_mean, H_ratio_lo, H_ratio_hi,
        H_ratio_tree_lo, H_ratio_tree_hi, VP_ratio_mean, VP_ratio_lo,
        VP_ratio_hi, plus original screening columns.

    model/saved/aggregate_uncertainty.json
        Bootstrap CIs on aggregate statistics (mean eps/k, fraction passing,
        SPT overlap R², etc.).

Usage:
    python scripts/propagate_uncertainty.py
    python scripts/propagate_uncertainty.py --max-molecules 50  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from model.registry import _compute_features, get_model
from model.thermodynamic import CYCLOPENTANE, HEXANE, T_REF
from model.uncertainty import (
    binomial_ci,
    bootstrap_metric_ci,
    combined_uncertainty,
    get_rf_tree_predictions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
SAVED_DIR = REPO_ROOT / "model" / "saved"
DATA_DIR = REPO_ROOT / "model" / "data"


def load_rf_models() -> dict:
    """Load the per-target RF models via the registry."""
    rf = get_model("rf")
    rf.load()
    return rf._models


def run_propagation(max_molecules: int | None = None):
    """Main propagation pipeline."""
    t0 = time.time()

    # --- Load data ---
    logger.info("Loading VP-passing candidates...")
    henrys_df = pd.read_csv(SAVED_DIR / "henrys_constant_screening.csv")
    logger.info("  %d candidates with Henry's data", len(henrys_df))

    if max_molecules is not None and max_molecules < len(henrys_df):
        logger.info("  Limiting to %d molecules for testing", max_molecules)
        henrys_df = henrys_df.head(max_molecules).copy()

    # --- Load RF models ---
    logger.info("Loading RF models...")
    rf_models = load_rf_models()
    n_trees = min(len(m.estimators_) for m in rf_models.values())
    logger.info("  %d trees per target", n_trees)

    # --- Compute features ---
    logger.info("Computing features for %d candidates...", len(henrys_df))
    smiles_list = henrys_df["smiles"].tolist()
    X = _compute_features(smiles_list)

    valid_mask = np.isfinite(X).all(axis=1)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning("  %d molecules have invalid features (NaN); skipping", n_invalid)

    # --- Extract per-tree predictions ---
    logger.info("Extracting per-tree predictions...")
    tree_preds = get_rf_tree_predictions(rf_models, X)
    logger.info("  tree_preds shape: %s", tree_preds.shape)

    # --- Run combined propagation ---
    k_ij_values = (-0.05, -0.025, 0.0, 0.025, 0.05)
    logger.info(
        "Running combined uncertainty propagation (%d trees x %d molecules x %d k_ij)...",
        n_trees, len(henrys_df), len(k_ij_values),
    )

    result = combined_uncertainty(
        tree_preds,
        k_ij_values=k_ij_values,
        reference=CYCLOPENTANE,
        solvent=HEXANE,
        T=T_REF,
        ci=0.95,
    )

    # --- Merge with original data ---
    for col, arr in result.items():
        henrys_df[col] = arr

    # --- Save ---
    out_path = SAVED_DIR / "uncertainty_propagated.csv"
    henrys_df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    logger.info("Saved %d rows to %s (%.1f min)", len(henrys_df), out_path, elapsed / 60)

    # --- Print summary for key candidates ---
    _print_candidate_summary(henrys_df)

    # --- Aggregate statistics with CIs ---
    logger.info("\nComputing aggregate uncertainty statistics...")
    agg = _compute_aggregate_uncertainty(henrys_df, rf_models, X)

    agg_path = SAVED_DIR / "aggregate_uncertainty.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info("Saved aggregate uncertainty to %s", agg_path)

    return henrys_df


def _print_candidate_summary(df: pd.DataFrame):
    """Print uncertainty for notable candidates."""
    logger.info("\n" + "=" * 80)
    logger.info("KEY CANDIDATE UNCERTAINTY SUMMARY")
    logger.info("=" * 80)

    notable = {
        "FC1=CCCC1": "1-Fluorocyclopentene",
        "FC1=CCCCC1": "1-Fluorocyclohexene",
        "C=C1C(F)(F)C1(F)F": "Tetrafluoromethylenecyclopropane",
    }

    for smiles, name in notable.items():
        row = df[df["smiles"] == smiles]
        if row.empty:
            logger.info("  %s (%s): not found in dataset", name, smiles)
            continue

        r = row.iloc[0]
        h_mean = r.get("H_ratio_tree_mean", r.get("H_ratio", np.nan))
        h_lo = r.get("H_ratio_tree_lo", np.nan)
        h_hi = r.get("H_ratio_tree_hi", np.nan)
        h_comb_lo = r.get("H_ratio_lo", np.nan)
        h_comb_hi = r.get("H_ratio_hi", np.nan)

        logger.info(
            "  %s (%s):", name, smiles,
        )
        logger.info(
            "    H_ratio = %.3f  tree-only 95%%CI [%.3f, %.3f]",
            h_mean, h_lo, h_hi,
        )
        logger.info(
            "    H_ratio combined (tree+k_ij) 95%%CI [%.3f, %.3f]",
            h_comb_lo, h_comb_hi,
        )
        if "VP_ratio_mean" in r:
            logger.info(
                "    VP_ratio = %.3f  95%%CI [%.3f, %.3f]",
                r["VP_ratio_mean"], r["VP_ratio_lo"], r["VP_ratio_hi"],
            )

    logger.info("=" * 80)


def _compute_aggregate_uncertainty(
    df: pd.DataFrame,
    rf_models: dict,
    X: np.ndarray,
) -> dict:
    """Compute bootstrap CIs on aggregate statistics."""
    agg: dict = {}

    # 1. Mean epsilon_k across all candidates
    eps_k_vals = df["epsilon_k"].dropna().values
    if len(eps_k_vals) > 0:
        from scipy.stats import sem

        mean_eps = float(np.mean(eps_k_vals))
        se = float(sem(eps_k_vals))
        agg["mean_epsilon_k"] = {
            "point": mean_eps,
            "ci_lo": mean_eps - 1.96 * se,
            "ci_hi": mean_eps + 1.96 * se,
            "n": len(eps_k_vals),
        }

    # 2. Fraction exceeding cyclopentane epsilon_k
    n_above = int((eps_k_vals > CYCLOPENTANE["epsilon_k"]).sum())
    agg["frac_above_cyclopentane_eps_k"] = binomial_ci(n_above, len(eps_k_vals))

    # 3. Fraction passing Henry's screen [0.5, 2.0]
    h_vals = df["H_ratio"].dropna().values
    n_pass = int(((h_vals >= 0.5) & (h_vals <= 2.0)).sum())
    agg["frac_henry_passing"] = binomial_ci(n_pass, len(h_vals))

    # 4. SPT-PCSAFT overlap R² with bootstrap CI
    spt_path = SAVED_DIR / "spt_pcsaft_overlap_validation.csv"
    if spt_path.exists():
        spt_df = pd.read_csv(spt_path)
        for param in ["m", "sigma", "epsilon_k"]:
            pred_col = param
            true_col = f"{param}_spt"
            if pred_col in spt_df.columns and true_col in spt_df.columns:
                mask = (
                    spt_df[pred_col].notna() & spt_df[true_col].notna()
                )
                if mask.sum() >= 5:
                    ci = bootstrap_metric_ci(
                        spt_df.loc[mask, true_col].values,
                        spt_df.loc[mask, pred_col].values,
                        r2_score,
                        n_boot=2000,
                    )
                    agg[f"spt_overlap_r2_{param}"] = ci

    # 5. H_ratio summary stats with CIs
    valid_h = df["H_ratio_tree_mean"].dropna().values
    if len(valid_h) > 10:
        from scipy.stats import sem as _sem

        agg["H_ratio_tree_mean_summary"] = {
            "mean": float(np.mean(valid_h)),
            "median": float(np.median(valid_h)),
            "std": float(np.std(valid_h)),
            "se": float(_sem(valid_h)),
            "n": len(valid_h),
        }

    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Propagate RF uncertainty through thermodynamic calculations",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Limit number of molecules (for testing).",
    )
    args = parser.parse_args()
    run_propagation(max_molecules=args.max_molecules)


if __name__ == "__main__":
    main()
