#!/usr/bin/env python3
"""Step 43: Chlorobutene Thermodynamic Near-Hit Deep-Dive.

Retrospective audit of the three chlorobutene near-hits identified in
Steps 20-21 (the ONLY molecules passing all 5 thermodynamic criteria).

This script implements Sections 43.1-43.8:
  43.1  Literature / external-data check (boiling point comparison)
  43.2  Benchmark nearest chlorinated alkenes in Esper
  43.3  Compute Tanimoto applicability-domain status
  43.4  Compute parameter-level uncertainty (CIs)
  43.5  Propagate uncertainty through the EOS
  43.6  RF vs SPT-PCSAFT comparison
  43.7  Chlorinated-compound bias assessment (NIST context)
  43.8  Save results and generate figures

Outputs:
    model/saved/chlorobutene_deep_dive.csv
    figures/43_chlorobutene_deep_dive/*.png
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.ad_tanimoto import TanimotoAD
from model.predict import predict_pcsaft, predict_pcsaft_with_ci
from model.registry import _compute_features, get_model
from model.thermodynamic import (
    CYCLOPENTANE,
    HEXANE,
    T_REF,
    compute_henrys_constant,
)
from model.uncertainty import (
    combined_uncertainty,
    get_rf_tree_predictions,
)
from screening.hfo_screening import compute_boiling_point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "43_chlorobutene_deep_dive"
SAVED_DIR = ROOT / "model" / "saved"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Font sizing conventions
FONT_TITLE = 16
FONT_LABEL = 14
FONT_TICK = 12
FONT_LEGEND = 11
DPI = 150

# Target molecules
TARGETS = ["C=C(Cl)CC", "C/C=C(/C)Cl", r"C/C=C(\C)Cl"]
TARGET_NAMES = [
    "1-chlorobut-1-ene",
    "(Z)-2-chloro-2-butene",
    "(E)-2-chloro-2-butene",
]

# Chlorinated alkene neighbors in Esper (Section 43.2)
NEIGHBOR_SMILES = ["C=CCl", "C=CCCl", "C=C(C)Cl", "C=C(Cl)Cl", "ClC=CCl"]
NEIGHBOR_NAMES = [
    "Vinyl chloride",
    "Allyl chloride",
    "2-Chloropropene",
    "1,1-Dichloroethene",
    "1,2-Dichloroethene",
]

# Experimental boiling points (CRC Handbook)
EXP_BOILING_POINTS = {
    "C=C(Cl)CC": 337.0,       # 64 deg C
    "C/C=C(/C)Cl": 341.0,     # 68 deg C
    r"C/C=C(\C)Cl": 336.0,    # 63 deg C
}


# ---------------------------------------------------------------------------
# 43.1  Literature and external-data check
# ---------------------------------------------------------------------------


def section_43_1_literature_check() -> pd.DataFrame:
    """Compare RF- and SPT-predicted boiling points to experimental values."""
    logger.info("=== 43.1  Literature boiling-point comparison ===")

    # Load SPT data
    spt = pd.read_csv(ROOT / "model" / "data" / "spt_pcsaft.csv")

    # Get RF predictions
    rf_preds = predict_pcsaft(TARGETS)

    rows = []
    for i, smi in enumerate(TARGETS):
        name = TARGET_NAMES[i]
        T_b_exp = EXP_BOILING_POINTS[smi]

        # RF parameters and boiling point
        rf_m = rf_preds.loc[i, "m"]
        rf_sigma = rf_preds.loc[i, "sigma"]
        rf_eps = rf_preds.loc[i, "epsilon_k"]
        rf_Tb = compute_boiling_point(rf_m, rf_sigma, rf_eps)

        # SPT parameters and boiling point
        spt_row = spt[spt["smiles"] == smi].iloc[0]
        spt_m = spt_row["m"]
        spt_sigma = spt_row["sigma"]
        spt_eps = spt_row["epsilon_k"]
        spt_Tb = compute_boiling_point(spt_m, spt_sigma, spt_eps)

        row = {
            "smiles": smi,
            "name": name,
            "T_b_exp_K": T_b_exp,
            "rf_m": rf_m,
            "rf_sigma": rf_sigma,
            "rf_epsilon_k": rf_eps,
            "rf_T_b_K": rf_Tb,
            "rf_T_b_error_K": rf_Tb - T_b_exp if np.isfinite(rf_Tb) else np.nan,
            "spt_m": spt_m,
            "spt_sigma": spt_sigma,
            "spt_epsilon_k": spt_eps,
            "spt_T_b_K": spt_Tb,
            "spt_T_b_error_K": spt_Tb - T_b_exp if np.isfinite(spt_Tb) else np.nan,
        }
        rows.append(row)

        logger.info(
            "  %s: T_b(exp)=%.0f K, T_b(RF)=%.1f K (err=%.1f), "
            "T_b(SPT)=%.1f K (err=%.1f)",
            name,
            T_b_exp,
            rf_Tb if np.isfinite(rf_Tb) else float("nan"),
            row["rf_T_b_error_K"] if np.isfinite(row["rf_T_b_error_K"]) else float("nan"),
            spt_Tb if np.isfinite(spt_Tb) else float("nan"),
            row["spt_T_b_error_K"] if np.isfinite(row["spt_T_b_error_K"]) else float("nan"),
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 43.2  Benchmark nearest chlorinated alkenes in Esper
# ---------------------------------------------------------------------------


def section_43_2_neighbor_benchmark() -> pd.DataFrame:
    """Predict neighbors with RF and compare to Esper ground truth."""
    logger.info("=== 43.2  Nearest chlorinated-alkene Esper benchmark ===")

    esper = pd.read_csv(ROOT / "model" / "data" / "esper_pcsaft.csv")
    spt = pd.read_csv(ROOT / "model" / "data" / "spt_pcsaft.csv")

    # RF predictions for neighbors
    rf_preds = predict_pcsaft(NEIGHBOR_SMILES)

    rows = []
    for i, smi in enumerate(NEIGHBOR_SMILES):
        name = NEIGHBOR_NAMES[i]

        # Esper ground truth
        esper_row = esper[esper["smiles"] == smi].iloc[0]
        esper_m = esper_row["m"]
        esper_sigma = esper_row["sigma"]
        esper_eps = esper_row["epsilon_k"]

        # RF prediction
        rf_m = rf_preds.loc[i, "m"]
        rf_sigma = rf_preds.loc[i, "sigma"]
        rf_eps = rf_preds.loc[i, "epsilon_k"]

        # SPT values (where available)
        spt_match = spt[spt["smiles"] == smi]
        spt_m = spt_match.iloc[0]["m"] if len(spt_match) > 0 else np.nan
        spt_sigma = spt_match.iloc[0]["sigma"] if len(spt_match) > 0 else np.nan
        spt_eps = spt_match.iloc[0]["epsilon_k"] if len(spt_match) > 0 else np.nan

        row = {
            "smiles": smi,
            "name": name,
            "esper_m": esper_m,
            "esper_sigma": esper_sigma,
            "esper_epsilon_k": esper_eps,
            "rf_m": rf_m,
            "rf_sigma": rf_sigma,
            "rf_epsilon_k": rf_eps,
            "rf_err_m": rf_m - esper_m,
            "rf_err_sigma": rf_sigma - esper_sigma,
            "rf_err_epsilon_k": rf_eps - esper_eps,
            "spt_m": spt_m,
            "spt_sigma": spt_sigma,
            "spt_epsilon_k": spt_eps,
            "spt_err_m": spt_m - esper_m if np.isfinite(spt_m) else np.nan,
            "spt_err_sigma": spt_sigma - esper_sigma if np.isfinite(spt_sigma) else np.nan,
            "spt_err_epsilon_k": spt_eps - esper_eps if np.isfinite(spt_eps) else np.nan,
        }
        rows.append(row)

        logger.info(
            "  %s: Esper eps/k=%.1f, RF eps/k=%.1f (err=%+.1f), "
            "SPT eps/k=%.1f (err=%+.1f)",
            name,
            esper_eps,
            rf_eps,
            row["rf_err_epsilon_k"],
            spt_eps if np.isfinite(spt_eps) else float("nan"),
            row["spt_err_epsilon_k"] if np.isfinite(row["spt_err_epsilon_k"]) else float("nan"),
        )

    df = pd.DataFrame(rows)

    # Summary of bias direction
    mean_rf_eps_err = df["rf_err_epsilon_k"].mean()
    mean_spt_eps_err = df["spt_err_epsilon_k"].dropna().mean()
    logger.info(
        "  RF mean eps/k signed error: %+.1f K (%s)",
        mean_rf_eps_err,
        "over-predicts" if mean_rf_eps_err > 0 else "under-predicts",
    )
    logger.info(
        "  SPT mean eps/k signed error: %+.1f K (%s)",
        mean_spt_eps_err,
        "over-predicts" if mean_spt_eps_err > 0 else "under-predicts",
    )

    return df


# ---------------------------------------------------------------------------
# 43.3  Tanimoto applicability-domain status
# ---------------------------------------------------------------------------


def section_43_3_tanimoto_ad() -> pd.DataFrame:
    """Compute Tanimoto similarity and AD status for targets."""
    logger.info("=== 43.3  Tanimoto applicability-domain status ===")

    esper = pd.read_csv(ROOT / "model" / "data" / "esper_pcsaft.csv")
    esper_smiles = esper["smiles"].tolist()

    ad = TanimotoAD(radius=2, n_bits=2048)
    ad.fit(esper_smiles)

    # Also find the nearest neighbor SMILES for each target
    from rdkit import DataStructs
    from rdkit.Chem import AllChem, MolFromSmiles

    rows = []
    for i, smi in enumerate(TARGETS):
        sim = ad.tanimoto_nn(smi)
        if sim >= 0.4:
            status = "in_domain"
        elif sim >= 0.3:
            status = "warning"
        else:
            status = "ood"

        # Find the actual nearest-neighbor SMILES
        mol = MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, ad.train_fps_)
        nn_idx = int(np.argmax(sims))
        nn_smiles = esper_smiles[nn_idx]
        nn_sim = float(sims[nn_idx])

        rows.append({
            "smiles": smi,
            "name": TARGET_NAMES[i],
            "tanimoto_nn": nn_sim,
            "nn_smiles": nn_smiles,
            "ad_status": status,
        })

        logger.info(
            "  %s: Tanimoto=%.3f (%s), NN=%s",
            TARGET_NAMES[i],
            nn_sim,
            status,
            nn_smiles,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 43.4  Parameter-level uncertainty (CIs)
# ---------------------------------------------------------------------------


def section_43_4_parameter_ci() -> pd.DataFrame:
    """Compute RF CIs for target molecules."""
    logger.info("=== 43.4  Parameter-level uncertainty (RF tree CIs) ===")

    ci_df = predict_pcsaft_with_ci(TARGETS)

    for i, smi in enumerate(TARGETS):
        row = ci_df.iloc[i]
        logger.info(
            "  %s: m=%.2f +/- %.2f, sigma=%.3f +/- %.3f, "
            "eps/k=%.1f +/- %.1f",
            TARGET_NAMES[i],
            row["m"],
            1.96 * row["m_std"],
            row["sigma"],
            1.96 * row["sigma_std"],
            row["epsilon_k"],
            1.96 * row["epsilon_k_std"],
        )

    return ci_df


# ---------------------------------------------------------------------------
# 43.5  Propagate uncertainty through EOS
# ---------------------------------------------------------------------------


def section_43_5_eos_propagation() -> dict[str, np.ndarray]:
    """Propagate per-tree uncertainty through teqp for VP and Henry's."""
    logger.info("=== 43.5  EOS-propagated uncertainty ===")

    rf = get_model("rf")
    rf.load()
    rf_models = rf._models

    X = _compute_features(TARGETS)
    tree_preds = get_rf_tree_predictions(rf_models, X)
    logger.info(
        "  Tree predictions shape: %s (trees=%d, mols=%d)",
        tree_preds.shape,
        tree_preds.shape[0],
        tree_preds.shape[1],
    )

    t0 = time.time()
    result = combined_uncertainty(
        tree_preds,
        k_ij_values=(-0.05, -0.025, 0.0, 0.025, 0.05),
        reference=CYCLOPENTANE,
        solvent=HEXANE,
        T=T_REF,
        ci=0.95,
    )
    elapsed = time.time() - t0
    logger.info("  Propagation took %.1f seconds", elapsed)

    for i, name in enumerate(TARGET_NAMES):
        logger.info(
            "  %s: VP_ratio=%.3f [%.3f, %.3f], "
            "H/Href(tree)=%.3f [%.3f, %.3f], "
            "H/Href(comb)=%.3f [%.3f, %.3f]",
            name,
            result["VP_ratio_mean"][i],
            result["VP_ratio_lo"][i],
            result["VP_ratio_hi"][i],
            result["H_ratio_tree_mean"][i],
            result["H_ratio_tree_lo"][i],
            result["H_ratio_tree_hi"][i],
            result["H_ratio_mean"][i],
            result["H_ratio_lo"][i],
            result["H_ratio_hi"][i],
        )

    return result


# ---------------------------------------------------------------------------
# 43.6  RF vs SPT comparison
# ---------------------------------------------------------------------------


def section_43_6_rf_vs_spt() -> pd.DataFrame:
    """Tabulate RF vs SPT disagreement and compute H/H_ref from SPT params."""
    logger.info("=== 43.6  RF vs SPT-PCSAFT comparison ===")

    spt = pd.read_csv(ROOT / "model" / "data" / "spt_pcsaft.csv")
    rf_preds = predict_pcsaft(TARGETS)

    rows = []
    for i, smi in enumerate(TARGETS):
        spt_row = spt[spt["smiles"] == smi].iloc[0]

        rf_m = rf_preds.loc[i, "m"]
        rf_sigma = rf_preds.loc[i, "sigma"]
        rf_eps = rf_preds.loc[i, "epsilon_k"]

        spt_m = spt_row["m"]
        spt_sigma = spt_row["sigma"]
        spt_eps = spt_row["epsilon_k"]

        # Compute H/H_ref from each parameter set
        ref_H_result = compute_henrys_constant(CYCLOPENTANE, HEXANE, T_REF, k_ij=0.0)
        H_ref = ref_H_result["H_Pa"] if ref_H_result else np.nan

        rf_H_result = compute_henrys_constant(
            {"m": rf_m, "sigma": rf_sigma, "epsilon_k": rf_eps},
            HEXANE, T_REF, k_ij=0.0,
        )
        rf_H = rf_H_result["H_Pa"] if rf_H_result else np.nan
        rf_H_ratio = rf_H / H_ref if np.isfinite(rf_H) and np.isfinite(H_ref) else np.nan

        spt_H_result = compute_henrys_constant(
            {"m": spt_m, "sigma": spt_sigma, "epsilon_k": spt_eps},
            HEXANE, T_REF, k_ij=0.0,
        )
        spt_H = spt_H_result["H_Pa"] if spt_H_result else np.nan
        spt_H_ratio = spt_H / H_ref if np.isfinite(spt_H) and np.isfinite(H_ref) else np.nan

        rows.append({
            "smiles": smi,
            "name": TARGET_NAMES[i],
            "rf_m": rf_m,
            "rf_sigma": rf_sigma,
            "rf_epsilon_k": rf_eps,
            "spt_m": spt_m,
            "spt_sigma": spt_sigma,
            "spt_epsilon_k": spt_eps,
            "delta_m": rf_m - spt_m,
            "delta_sigma": rf_sigma - spt_sigma,
            "delta_epsilon_k": rf_eps - spt_eps,
            "rf_H_ratio": rf_H_ratio,
            "spt_H_ratio": spt_H_ratio,
        })

        logger.info(
            "  %s: RF eps/k=%.1f, SPT eps/k=%.1f, delta=%+.1f K "
            "| RF H/Href=%.2f, SPT H/Href=%.2f",
            TARGET_NAMES[i],
            rf_eps,
            spt_eps,
            rf_eps - spt_eps,
            rf_H_ratio if np.isfinite(rf_H_ratio) else float("nan"),
            spt_H_ratio if np.isfinite(spt_H_ratio) else float("nan"),
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def plot_neighbor_benchmark(neighbor_df: pd.DataFrame) -> None:
    """Bar chart of RF prediction error (RF - Esper) for chlorinated-alkene neighbors."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    params = [
        ("rf_err_m", "m (segments)", "m"),
        ("rf_err_sigma", r"$\sigma$ (\u00c5)", "sigma"),
        ("rf_err_epsilon_k", r"$\varepsilon/k$ (K)", "epsilon_k"),
    ]

    x = np.arange(len(neighbor_df))
    names = neighbor_df["name"].tolist()

    for ax, (col, label, _param_key) in zip(axes, params):
        vals = neighbor_df[col].values
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=FONT_TICK - 2)
        ax.set_ylabel(f"RF error ({label})", fontsize=FONT_LABEL)
        ax.tick_params(axis="y", labelsize=FONT_TICK)

        # Add mean error annotation
        mean_err = np.mean(vals)
        ax.axhline(mean_err, color="blue", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            len(x) - 0.5,
            mean_err,
            f"mean={mean_err:+.2f}",
            color="blue",
            fontsize=FONT_LEGEND,
            va="bottom",
            ha="right",
        )

    fig.suptitle(
        "RF Prediction Error vs Esper Ground Truth\n(Chlorinated Alkene Neighbors)",
        fontsize=FONT_TITLE,
    )
    fig.tight_layout()
    out = FIG_DIR / "neighbor_benchmark.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out)


def plot_parameter_ci(
    ci_df: pd.DataFrame,
    rf_vs_spt_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
) -> None:
    """Dot plot: RF point estimate + CI, SPT overlay, Esper neighbor range."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    params = [
        ("m", "m_lo95", "m_hi95", "m (segments)"),
        ("sigma", "sigma_lo95", "sigma_hi95", r"$\sigma$ (\u00c5)"),
        ("epsilon_k", "eps_lo95", "eps_hi95", r"$\varepsilon/k$ (K)"),
    ]

    y_pos = np.arange(len(TARGETS))

    for ax, (param, lo_col, hi_col, label) in zip(axes, params):
        # RF point + CI
        rf_vals = ci_df[param].values
        lo = ci_df[lo_col].values
        hi = ci_df[hi_col].values
        xerr = np.array([rf_vals - lo, hi - rf_vals])

        ax.errorbar(
            rf_vals,
            y_pos,
            xerr=xerr,
            fmt="o",
            color="#1f77b4",
            capsize=5,
            capthick=1.5,
            markersize=8,
            label="RF (95% CI)",
            zorder=3,
        )

        # SPT overlay
        spt_col = f"spt_{param}" if param != "epsilon_k" else "spt_epsilon_k"
        spt_vals = rf_vs_spt_df[spt_col].values
        ax.scatter(
            spt_vals,
            y_pos + 0.15,
            marker="D",
            color="#d62728",
            s=60,
            label="SPT-PCSAFT",
            zorder=3,
        )

        # Esper neighbor range (shaded band)
        esper_col = f"esper_{param}" if param != "epsilon_k" else "esper_epsilon_k"
        esper_vals = neighbor_df[esper_col].values
        e_min, e_max = esper_vals.min(), esper_vals.max()
        ax.axvspan(e_min, e_max, alpha=0.15, color="green", label="Esper neighbor range")

        ax.set_xlabel(label, fontsize=FONT_LABEL)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(TARGET_NAMES, fontsize=FONT_TICK)
        ax.tick_params(axis="x", labelsize=FONT_TICK)
        ax.legend(fontsize=FONT_LEGEND - 1, loc="best")

    fig.suptitle(
        "Parameter Estimates with 95% CI\n(RF vs SPT, Esper Neighbor Range)",
        fontsize=FONT_TITLE,
    )
    fig.tight_layout()
    out = FIG_DIR / "parameter_ci_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out)


def plot_henry_ratio_ci(
    eos_result: dict[str, np.ndarray],
    rf_vs_spt_df: pd.DataFrame,
) -> None:
    """Forest plot of H/H_ref: point estimate, tree-only CI, combined CI."""
    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(TARGETS))
    offset = 0.15

    # Acceptance band [0.5, 2.0]
    ax.axvspan(0.5, 2.0, alpha=0.12, color="green", label="Acceptance band [0.5, 2.0]")

    # Combined CI (wider, outer)
    for i in range(len(TARGETS)):
        ax.plot(
            [eos_result["H_ratio_lo"][i], eos_result["H_ratio_hi"][i]],
            [y_pos[i] - offset, y_pos[i] - offset],
            color="#ff7f0e",
            linewidth=2.5,
            solid_capstyle="round",
        )
    ax.scatter(
        eos_result["H_ratio_mean"],
        y_pos - offset,
        marker="s",
        color="#ff7f0e",
        s=50,
        zorder=3,
        label="H/H_ref combined (tree + k_ij)",
    )

    # Tree-only CI (narrower, inner)
    for i in range(len(TARGETS)):
        ax.plot(
            [eos_result["H_ratio_tree_lo"][i], eos_result["H_ratio_tree_hi"][i]],
            [y_pos[i] + offset, y_pos[i] + offset],
            color="#1f77b4",
            linewidth=2.5,
            solid_capstyle="round",
        )
    ax.scatter(
        eos_result["H_ratio_tree_mean"],
        y_pos + offset,
        marker="o",
        color="#1f77b4",
        s=50,
        zorder=3,
        label="H/H_ref tree-only (k_ij=0)",
    )

    # SPT-implied H/H_ref
    spt_H = rf_vs_spt_df["spt_H_ratio"].values
    valid_spt = np.isfinite(spt_H)
    if valid_spt.any():
        ax.scatter(
            spt_H[valid_spt],
            y_pos[valid_spt],
            marker="D",
            color="#d62728",
            s=70,
            zorder=4,
            label="SPT H/H_ref",
        )

    # Reference line at H/H_ref = 1
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(TARGET_NAMES, fontsize=FONT_TICK)
    ax.set_xlabel("H / H(cyclopentane)", fontsize=FONT_LABEL)
    ax.set_title(
        "Henry's Constant Ratio: Uncertainty Propagation",
        fontsize=FONT_TITLE,
    )
    ax.legend(fontsize=FONT_LEGEND, loc="best")
    ax.tick_params(axis="x", labelsize=FONT_TICK)

    # Set x-axis to log scale if the range is large
    all_vals = np.concatenate([
        eos_result["H_ratio_lo"],
        eos_result["H_ratio_hi"],
        spt_H[valid_spt] if valid_spt.any() else [],
    ])
    if np.isfinite(all_vals).any():
        vmin = np.nanmin(all_vals)
        vmax = np.nanmax(all_vals)
        if vmax / max(vmin, 0.01) > 10:
            ax.set_xscale("log")

    fig.tight_layout()
    out = FIG_DIR / "henry_ratio_ci.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out)


def plot_ad_similarity_heatmap(
    ad_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
) -> None:
    """Optional Tanimoto similarity heatmap between targets and neighbors."""
    from rdkit import DataStructs
    from rdkit.Chem import AllChem, MolFromSmiles

    all_smiles = TARGETS + NEIGHBOR_SMILES
    n = len(all_smiles)

    # Compute fingerprints
    fps = []
    for smi in all_smiles:
        mol = MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)

    # Compute pairwise similarity
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])

    # Plot only target-vs-neighbor block
    n_targets = len(TARGETS)
    n_neighbors = len(NEIGHBOR_SMILES)
    sub_matrix = sim_matrix[:n_targets, n_targets:]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(sub_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(n_neighbors))
    ax.set_xticklabels(NEIGHBOR_NAMES, rotation=45, ha="right", fontsize=FONT_TICK)
    ax.set_yticks(np.arange(n_targets))
    ax.set_yticklabels(TARGET_NAMES, fontsize=FONT_TICK)

    # Annotate cells
    for i in range(n_targets):
        for j in range(n_neighbors):
            val = sub_matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=FONT_LEGEND, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Tanimoto Similarity", fontsize=FONT_LABEL)

    ax.set_title("Tanimoto Similarity: Targets vs Esper Neighbors", fontsize=FONT_TITLE)
    fig.tight_layout()
    out = FIG_DIR / "ad_similarity_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out)


# ---------------------------------------------------------------------------
# 43.8  Save results
# ---------------------------------------------------------------------------


def save_results(
    lit_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    ad_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    eos_result: dict[str, np.ndarray],
    rf_vs_spt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all target-level results into a single CSV."""
    logger.info("=== 43.8  Saving results ===")

    # Build a comprehensive per-target summary
    rows = []
    for i, smi in enumerate(TARGETS):
        row = {
            "smiles": smi,
            "name": TARGET_NAMES[i],
        }

        # Literature (43.1)
        lr = lit_df[lit_df["smiles"] == smi].iloc[0]
        row["T_b_exp_K"] = lr["T_b_exp_K"]
        row["rf_T_b_K"] = lr["rf_T_b_K"]
        row["rf_T_b_error_K"] = lr["rf_T_b_error_K"]
        row["spt_T_b_K"] = lr["spt_T_b_K"]
        row["spt_T_b_error_K"] = lr["spt_T_b_error_K"]

        # AD (43.3)
        ar = ad_df[ad_df["smiles"] == smi].iloc[0]
        row["tanimoto_nn"] = ar["tanimoto_nn"]
        row["nn_smiles"] = ar["nn_smiles"]
        row["ad_status"] = ar["ad_status"]

        # CI (43.4)
        cr = ci_df.iloc[i]
        row["rf_m"] = cr["m"]
        row["rf_sigma"] = cr["sigma"]
        row["rf_epsilon_k"] = cr["epsilon_k"]
        row["rf_m_std"] = cr["m_std"]
        row["rf_sigma_std"] = cr["sigma_std"]
        row["rf_epsilon_k_std"] = cr["epsilon_k_std"]
        row["rf_eps_lo95"] = cr["eps_lo95"]
        row["rf_eps_hi95"] = cr["eps_hi95"]

        # EOS propagation (43.5)
        row["VP_ratio_mean"] = eos_result["VP_ratio_mean"][i]
        row["VP_ratio_lo"] = eos_result["VP_ratio_lo"][i]
        row["VP_ratio_hi"] = eos_result["VP_ratio_hi"][i]
        row["H_ratio_tree_mean"] = eos_result["H_ratio_tree_mean"][i]
        row["H_ratio_tree_lo"] = eos_result["H_ratio_tree_lo"][i]
        row["H_ratio_tree_hi"] = eos_result["H_ratio_tree_hi"][i]
        row["H_ratio_combined_mean"] = eos_result["H_ratio_mean"][i]
        row["H_ratio_combined_lo"] = eos_result["H_ratio_lo"][i]
        row["H_ratio_combined_hi"] = eos_result["H_ratio_hi"][i]

        # RF vs SPT (43.6)
        sr = rf_vs_spt_df[rf_vs_spt_df["smiles"] == smi].iloc[0]
        row["spt_m"] = sr["spt_m"]
        row["spt_sigma"] = sr["spt_sigma"]
        row["spt_epsilon_k"] = sr["spt_epsilon_k"]
        row["delta_epsilon_k"] = sr["delta_epsilon_k"]
        row["rf_H_ratio"] = sr["rf_H_ratio"]
        row["spt_H_ratio"] = sr["spt_H_ratio"]

        rows.append(row)

    summary = pd.DataFrame(rows)
    out_path = SAVED_DIR / "chlorobutene_deep_dive.csv"
    summary.to_csv(out_path, index=False)
    logger.info("  Saved %s (%d rows)", out_path, len(summary))

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("Step 43: Chlorobutene Thermodynamic Near-Hit Deep-Dive")
    logger.info("=" * 60)

    # 43.1 Literature comparison
    lit_df = section_43_1_literature_check()

    # 43.2 Nearest-neighbor Esper benchmark
    neighbor_df = section_43_2_neighbor_benchmark()

    # 43.3 Tanimoto AD
    ad_df = section_43_3_tanimoto_ad()

    # 43.4 Parameter-level CIs
    ci_df = section_43_4_parameter_ci()

    # 43.5 EOS-propagated uncertainty
    eos_result = section_43_5_eos_propagation()

    # 43.6 RF vs SPT comparison
    rf_vs_spt_df = section_43_6_rf_vs_spt()

    # Generate figures
    logger.info("=== Generating figures ===")
    plot_neighbor_benchmark(neighbor_df)
    plot_parameter_ci(ci_df, rf_vs_spt_df, neighbor_df)
    plot_henry_ratio_ci(eos_result, rf_vs_spt_df)
    plot_ad_similarity_heatmap(ad_df, neighbor_df)

    # Save results
    summary = save_results(lit_df, neighbor_df, ad_df, ci_df, eos_result, rf_vs_spt_df)

    logger.info("=" * 60)
    logger.info("Step 43 complete. Summary:")
    logger.info(summary.to_string())

    return summary


if __name__ == "__main__":
    main()
