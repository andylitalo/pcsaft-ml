"""Step 42: Final Candidate Dossier -- Capstone Screening Synthesis.

Compiles all screening data (Steps 38b-41) into a comprehensive final dossier
with multi-criteria scoring, tiered ranking, per-candidate dossier cards,
four publication-quality figures, and a capstone report.

Key finding: No HFO/HCFO candidate is a suitable drop-in replacement for
cyclopentane. Parameter proximity does NOT predict property proximity for
fluorinated compounds (r = -0.709). However, the 50 candidates ARE the best
fluorinated alternatives if different operating conditions are acceptable.

Inputs:
  - screening/results/hfo_rf_shortlist_credibility.csv  (Step 39: 50 candidates)
  - screening/results/hfo_rf_thermo_validated.csv       (Step 40: EOS validation)
  - screening/results/hfo_rf_safety_assessed.csv        (Step 41: safety gate)

Outputs:
  - screening/results/hfo_rf_final_dossier.csv          (full 50-candidate table)
  - screening/results/hfo_rf_top10_dossier_cards.csv    (top 10 dossier cards)
  - figures/42_final_candidate_dossier/                  (4 figures)
  - docs/reports/42_final_candidate_dossier.md           (capstone report)

Usage:
    python scripts/step42_final_candidate_dossier.py
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

CREDIBILITY_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_shortlist_credibility.csv"
THERMO_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_thermo_validated.csv"
SAFETY_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_safety_assessed.csv"

DOSSIER_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_final_dossier.csv"
TOP10_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_top10_dossier_cards.csv"
FIGURES_DIR = PROJECT_ROOT / "figures" / "42_final_candidate_dossier"
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "42_final_candidate_dossier.md"

# Filter funnel CSV from Step 38b
FUNNEL_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_filter_funnel.csv"


# ===================================================================
# 42.1 -- Load and Merge All Screening Data
# ===================================================================


def load_and_merge() -> pd.DataFrame:
    """Load all three screening CSVs and merge on SMILES."""
    logger.info("[42.1] Loading and merging screening data...")

    df_cred = pd.read_csv(CREDIBILITY_CSV)
    logger.info("  Credibility (Step 39): %d rows, %d cols", *df_cred.shape)

    df_thermo = pd.read_csv(THERMO_CSV)
    logger.info("  Thermo validated (Step 40): %d rows, %d cols", *df_thermo.shape)

    df_safety = pd.read_csv(SAFETY_CSV)
    logger.info("  Safety assessed (Step 41): %d rows, %d cols", *df_safety.shape)

    # Start with credibility (has full uncertainty columns)
    df = df_cred.copy()

    # Merge thermo columns (unique to Step 40)
    thermo_cols = [
        "smiles", "eos_convergence_count", "vp_ratio_298", "rho_ratio_298",
        "property_distance_298", "H_ratio", "passes_all_thermo_gates",
    ]
    df = df.merge(df_thermo[thermo_cols], on="smiles", how="left")

    # Merge safety columns (unique to Step 41)
    safety_cols = [
        "smiles", "f_mass_fraction", "has_cf3", "has_reactive_f",
        "reactive_patterns", "is_associating", "mol_weight", "heavy_atom_count",
        "n_cc_double", "n_fluorine", "n_chlorine", "n_hydrogen", "n_carbon",
        "lifetime_class", "gwp_class", "flammability_class",
        "has_toxicity_flags", "toxicity_flags", "safety_score", "safety_gate",
    ]
    df = df.merge(df_safety[safety_cols], on="smiles", how="left")

    logger.info("  Merged DataFrame: %d rows, %d cols", *df.shape)
    return df


# ===================================================================
# 42.2 -- Multi-Criteria Dossier Scoring
# ===================================================================


def compute_dossier_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite dossier score (0-100) from four components."""
    logger.info("[42.2] Computing multi-criteria dossier scores...")

    # --- Credibility (0-30 points) ---
    cred_map = {
        "screening_ready": 30,
        "screening_with_warning": 15,
        "high_extrapolation_risk": 0,
    }
    df["score_credibility"] = df["credibility"].map(cred_map).fillna(0).astype(float)

    # --- Thermodynamic viability (0-30 points) ---
    df["score_thermo"] = 0.0

    # EOS convergence at all 3 temps: +10
    df.loc[df["eos_convergence_count"] == 3, "score_thermo"] += 10.0

    # VP ratio at 298K scoring
    vp = df["vp_ratio_298"].values
    vp_score = np.where(
        (vp >= 0.5) & (vp <= 2.0), 10.0,
        np.where((vp >= 0.3) & (vp <= 3.0), 5.0, 0.0),
    )
    df["score_thermo"] += vp_score

    # Henry's ratio scoring
    h_ratio = df["H_ratio"].values
    h_score = np.where(
        (h_ratio >= 0.5) & (h_ratio <= 2.0), 10.0,
        np.where((h_ratio >= 0.1) & (h_ratio <= 10.0), 5.0, 0.0),
    )
    df["score_thermo"] += h_score

    # --- Safety (0-20 points) ---
    df["score_safety"] = (df["safety_score"].clip(upper=5) * 4.0)

    # --- Parameter proximity (0-20 points) ---
    max_hfo_dist = df["hfo_distance"].max()
    df["score_proximity"] = (
        20.0 * np.maximum(0.0, 1.0 - df["hfo_distance"] / max_hfo_dist)
    )

    # --- Composite ---
    df["dossier_score"] = (
        df["score_credibility"]
        + df["score_thermo"]
        + df["score_safety"]
        + df["score_proximity"]
    )

    # Round for display
    for col in ["score_credibility", "score_thermo", "score_safety",
                "score_proximity", "dossier_score"]:
        df[col] = df[col].round(1)

    logger.info("  Score ranges:")
    logger.info("    Credibility: %.1f - %.1f", df["score_credibility"].min(),
                df["score_credibility"].max())
    logger.info("    Thermo:      %.1f - %.1f", df["score_thermo"].min(),
                df["score_thermo"].max())
    logger.info("    Safety:      %.1f - %.1f", df["score_safety"].min(),
                df["score_safety"].max())
    logger.info("    Proximity:   %.1f - %.1f", df["score_proximity"].min(),
                df["score_proximity"].max())
    logger.info("    Dossier:     %.1f - %.1f", df["dossier_score"].min(),
                df["dossier_score"].max())

    return df


# ===================================================================
# 42.3 -- Final Ranking and Tier Assignment
# ===================================================================


def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by dossier_score and assign tiers."""
    logger.info("[42.3] Ranking candidates and assigning tiers...")

    df = df.sort_values("dossier_score", ascending=False).reset_index(drop=True)
    df["dossier_rank"] = range(1, len(df) + 1)

    df["tier"] = pd.cut(
        df["dossier_score"],
        bins=[-np.inf, 50, 70, np.inf],
        labels=["Tier 3 (Not recommended)", "Tier 2 (Conditional)", "Tier 1 (Recommended)"],
    )

    tier_counts = df["tier"].value_counts()
    for tier_name in ["Tier 1 (Recommended)", "Tier 2 (Conditional)", "Tier 3 (Not recommended)"]:
        cnt = tier_counts.get(tier_name, 0)
        logger.info("  %s: %d candidates", tier_name, cnt)

    return df


# ===================================================================
# 42.4 -- Per-Candidate Dossier Cards (Top 10)
# ===================================================================


def _get_common_name(smiles: str) -> str:
    """Attempt to derive a common name or molecular formula from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Unknown"
    return Chem.rdMolDescriptors.CalcMolFormula(mol)


def _get_key_caveat(row: pd.Series) -> str:
    """Determine the single most important limitation for this candidate."""
    caveats = []

    # VP ratio caveat (most common issue)
    vp = row.get("vp_ratio_298", 0)
    if vp > 3.0:
        caveats.append(f"VP {vp:.1f}x cyclopentane (too volatile for drop-in use)")
    elif vp > 2.0:
        caveats.append(f"VP {vp:.1f}x cyclopentane (marginal volatility)")

    # Credibility caveat
    cred = row.get("credibility", "")
    if cred == "high_extrapolation_risk":
        caveats.append("High extrapolation risk (out of AD)")
    elif cred == "screening_with_warning":
        caveats.append("Moderate AD uncertainty")

    # Henry's ratio
    h_ratio = row.get("H_ratio", 0)
    if h_ratio > 10:
        caveats.append(f"High Henry's ratio ({h_ratio:.1f}x) -- poor foam retention")

    # Property distance
    prop_dist = row.get("property_distance_298", 0)
    if prop_dist > 5.0:
        caveats.append(f"Property distance {prop_dist:.1f} (far from cyclopentane)")

    if not caveats:
        return "No major caveats identified"
    return caveats[0]


def build_dossier_cards(df: pd.DataFrame) -> pd.DataFrame:
    """Build dossier cards for top 10 candidates."""
    logger.info("[42.4] Building dossier cards for top 10 candidates...")

    top10 = df.head(10).copy()
    cards = []

    for _, row in top10.iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else "N/A"
        mw = Descriptors.ExactMolWt(mol) if mol else 0

        card = {
            "dossier_rank": int(row["dossier_rank"]),
            "smiles": smi,
            "molecular_formula": formula,
            "molecular_weight": round(mw, 2),
            "m": round(row["m"], 3),
            "sigma": round(row["sigma"], 3),
            "epsilon_k": round(row["epsilon_k"], 1),
            "m_std": round(row["m_std"], 3),
            "sigma_std": round(row["sigma_std"], 3),
            "epsilon_k_std": round(row["epsilon_k_std"], 1),
            "credibility": row["credibility"],
            "ad_label": row["ad_label"],
            "boiling_point_K": round(row["boiling_point_K"], 1),
            "vp_ratio_298": round(row["vp_ratio_298"], 2),
            "H_ratio": round(row["H_ratio"], 2),
            "safety_score": int(row["safety_score"]),
            "gwp_class": row["gwp_class"],
            "flammability_class": row["flammability_class"],
            "dossier_score": row["dossier_score"],
            "tier": str(row["tier"]),
            "key_caveat": _get_key_caveat(row),
        }
        cards.append(card)

    df_cards = pd.DataFrame(cards)
    logger.info("  Built %d dossier cards", len(df_cards))
    return df_cards


# ===================================================================
# 42.5 -- Save Outputs
# ===================================================================


def save_outputs(df: pd.DataFrame, df_cards: pd.DataFrame):
    """Save full dossier CSV and top-10 cards CSV."""
    logger.info("[42.5] Saving outputs...")

    DOSSIER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOSSIER_CSV, index=False)
    logger.info("  Full dossier: %s (%d rows, %d cols)", DOSSIER_CSV, *df.shape)

    df_cards.to_csv(TOP10_CSV, index=False)
    logger.info("  Top-10 cards: %s (%d rows, %d cols)", TOP10_CSV, *df_cards.shape)


# ===================================================================
# 42.6 -- Figures
# ===================================================================


def generate_figures(df: pd.DataFrame):
    """Generate 4 publication-quality figures for Step 42."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("[42.6] Generating 4 figures to %s", FIGURES_DIR)

    _fig_dossier_score_breakdown(df)
    _fig_tier_distribution(df)
    _fig_radar_top5(df)
    _fig_screening_funnel(df)

    logger.info("  All 4 figures saved.")


def _fig_dossier_score_breakdown(df: pd.DataFrame):
    """Figure 1: Stacked horizontal bar chart of score components for top 20."""
    logger.info("  [1/4] dossier_score_breakdown.png")

    top20 = df.head(20).copy()
    top20 = top20.iloc[::-1]  # Reverse so rank 1 is at top

    fig, ax = plt.subplots(figsize=(12, 9))

    y_pos = np.arange(len(top20))
    bar_height = 0.7

    components = [
        ("score_credibility", "Credibility (0-30)", "#1f77b4"),
        ("score_thermo", "Thermo viability (0-30)", "#ff7f0e"),
        ("score_safety", "Safety (0-20)", "#2ca02c"),
        ("score_proximity", "Param. proximity (0-20)", "#d62728"),
    ]

    left = np.zeros(len(top20))
    for col, label, color in components:
        vals = top20[col].values
        ax.barh(y_pos, vals, left=left, height=bar_height,
                color=color, edgecolor="white", linewidth=0.5, label=label)
        left += vals

    # Labels
    labels = []
    for _, row in top20.iterrows():
        rank = int(row["dossier_rank"])
        smi = row["smiles"]
        smi_short = smi[:20] + "..." if len(smi) > 20 else smi
        labels.append(f"#{rank} {smi_short}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, family="monospace")
    ax.set_xlabel("Dossier Score", fontsize=14)
    ax.set_title("Dossier Score Breakdown: Top 20 Candidates", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 105)
    ax.axvline(70, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(50, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(71, len(top20) - 0.5, "Tier 1", fontsize=10, color="green", alpha=0.7)
    ax.text(51, len(top20) - 0.5, "Tier 2", fontsize=10, color="orange", alpha=0.7)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dossier_score_breakdown.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def _fig_tier_distribution(df: pd.DataFrame):
    """Figure 2: Bar chart of tier distribution."""
    logger.info("  [2/4] tier_distribution.png")

    tier_order = [
        "Tier 1 (Recommended)",
        "Tier 2 (Conditional)",
        "Tier 3 (Not recommended)",
    ]
    tier_colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    counts = []
    for t in tier_order:
        counts.append((df["tier"] == t).sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(tier_order)), counts, color=tier_colors,
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    # Annotate counts on bars
    for bar_obj, count in zip(bars, counts):
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xticks(range(len(tier_order)))
    ax.set_xticklabels(["Tier 1\n(Recommended)", "Tier 2\n(Conditional)",
                         "Tier 3\n(Not recommended)"], fontsize=12)
    ax.set_ylabel("Number of Candidates", fontsize=14)
    ax.set_title("Final Tier Distribution (n=50 HFO Candidates)", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, max(counts) + 5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tier_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def _fig_radar_top5(df: pd.DataFrame):
    """Figure 3: Radar/spider chart comparing top 5 candidates."""
    logger.info("  [3/4] radar_top5.png")

    top5 = df.head(5).copy()

    # Axes: credibility, VP similarity, Henry's similarity, safety, proximity
    # Normalize each to 0-1
    categories = [
        "Credibility",
        "VP Similarity",
        "Henry's Similarity",
        "Safety",
        "Param. Proximity",
    ]
    n_cats = len(categories)

    # Compute normalized values for each candidate
    candidate_data = []
    for _, row in top5.iterrows():
        cred_norm = row["score_credibility"] / 30.0
        # VP similarity: 1/VP_ratio, capped at 1.0
        vp_sim = min(1.0, 1.0 / max(row["vp_ratio_298"], 0.01))
        # Henry's similarity: 1/H_ratio, capped at 1.0
        h_sim = min(1.0, 1.0 / max(row["H_ratio"], 0.01))
        safety_norm = row["score_safety"] / 20.0
        prox_norm = row["score_proximity"] / 20.0
        candidate_data.append([cred_norm, vp_sim, h_sim, safety_norm, prox_norm])

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (_, row) in enumerate(top5.iterrows()):
        vals = candidate_data[i] + candidate_data[i][:1]  # Close polygon
        smi = row["smiles"]
        smi_short = smi[:18] + "..." if len(smi) > 18 else smi
        rank = int(row["dossier_rank"])
        ax.plot(angles, vals, "o-", color=colors[i], linewidth=2, markersize=5,
                label=f"#{rank} {smi_short}")
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=10)
    ax.set_title("Top 5 Candidates: Multi-Criteria Comparison", fontsize=16,
                 pad=20)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.35, 1.10),
              framealpha=0.9)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "radar_top5.png", dpi=150, bbox_inches="tight")
    plt.close()


def _fig_screening_funnel(df: pd.DataFrame):
    """Figure 4: Complete screening funnel from enumeration to final tiers."""
    logger.info("  [4/4] screening_funnel_complete.png")

    # Build funnel stages
    stages = []

    # Load filter funnel if available
    if FUNNEL_CSV.exists():
        df_funnel = pd.read_csv(FUNNEL_CSV)
        # Get initial and final counts
        initial = df_funnel.loc[df_funnel["stage"] == "Initial candidates", "count"]
        initial_count = int(initial.values[0]) if len(initial) > 0 else 10700
    else:
        initial_count = 10700

    stages.append(("Enumerated HFO/HCFOs", initial_count))
    stages.append(("7-filter cascade\n(bp, C=C, no Cl, F>=2, SA, F%, no reactive)", 50))

    # Credibility tiers from Step 39
    n_ready = (df["credibility"] == "screening_ready").sum()
    n_warning = (df["credibility"] == "screening_with_warning").sum()
    n_high_risk = (df["credibility"] == "high_extrapolation_risk").sum()
    stages.append((f"Credibility screened\n(ready={n_ready}, warn={n_warning}, "
                   f"risk={n_high_risk})", 50))

    # EOS convergence from Step 40
    n_eos_ok = (df["eos_convergence_count"] == 3).sum()
    stages.append(("EOS convergence (3/3 temps)", n_eos_ok))

    # Safety gate from Step 41
    n_pass_safety = (df["safety_gate"] == "PASS").sum()
    stages.append(("Safety gate PASS", n_pass_safety))

    # Final tiers
    n_t1 = (df["tier"] == "Tier 1 (Recommended)").sum()
    n_t2 = (df["tier"] == "Tier 2 (Conditional)").sum()
    n_t3 = (df["tier"] == "Tier 3 (Not recommended)").sum()
    stages.append((f"Tier 1: {n_t1} | Tier 2: {n_t2} | Tier 3: {n_t3}", 50))

    fig, ax = plt.subplots(figsize=(10, 8))

    n_stages = len(stages)
    max_width = 0.9
    y_positions = np.linspace(0.88, 0.12, n_stages)

    cmap = plt.cm.YlOrRd_r
    for i, ((label, count), y) in enumerate(zip(stages, y_positions)):
        # Width proportional to log of count (for visual effect)
        if count > 0:
            width_frac = max(0.15, max_width * np.log10(count + 1) /
                             np.log10(initial_count + 1))
        else:
            width_frac = 0.15

        color = cmap(i / (n_stages - 1) * 0.7 + 0.1)
        rect_left = 0.5 - width_frac / 2
        rect = plt.Rectangle((rect_left, y - 0.04), width_frac, 0.07,
                              facecolor=color, edgecolor="black",
                              linewidth=1.2, alpha=0.85,
                              transform=ax.transAxes)
        ax.add_patch(rect)

        # Count text
        ax.text(0.5, y, f"n = {count:,}", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, fontweight="bold")

        # Label to the right
        ax.text(0.5 + width_frac / 2 + 0.02, y, label,
                transform=ax.transAxes, ha="left", va="center", fontsize=11)

        # Arrow between stages
        if i < n_stages - 1:
            ax.annotate("", xy=(0.5, y_positions[i + 1] + 0.04),
                        xytext=(0.5, y - 0.04),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops={"arrowstyle": "->", "color": "gray",
                                    "lw": 1.5})

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Complete Screening Funnel: HFO/HCFO Blowing Agent Pipeline",
                 fontsize=16, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "screening_funnel_complete.png", dpi=150,
                bbox_inches="tight")
    plt.close()


# ===================================================================
# 42.7 -- Capstone Report
# ===================================================================


def generate_report(df: pd.DataFrame, df_cards: pd.DataFrame):
    """Generate the capstone report for Step 42."""
    logger.info("[42.7] Generating capstone report...")

    # Compute statistics
    n_total = len(df)
    n_t1 = (df["tier"] == "Tier 1 (Recommended)").sum()
    n_t2 = (df["tier"] == "Tier 2 (Conditional)").sum()
    n_t3 = (df["tier"] == "Tier 3 (Not recommended)").sum()

    top1 = df.iloc[0]
    top1_smi = top1["smiles"]
    top1_score = top1["dossier_score"]
    top1_tier = top1["tier"]

    # Median VP ratio
    median_vp = df["vp_ratio_298"].median()
    min_vp = df["vp_ratio_298"].min()
    max_vp = df["vp_ratio_298"].max()

    lines = []

    # ---- Title ----
    lines.append(
        "# Final Candidate Dossier: HFO/HCFO Blowing Agent Screening "
        "via ML-Predicted PC-SAFT Parameters"
    )
    lines.append("")

    # ---- Executive Summary ----
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"This dossier presents the final outcome of an end-to-end ML screening "
        f"pipeline for fluorinated (HFO/HCFO) blowing agent alternatives to "
        f"cyclopentane. From 10,700 enumerated candidates, a 7-filter cascade "
        f"selected 50 compounds that were characterized for uncertainty (Step 39), "
        f"thermodynamic properties via PC-SAFT EOS (Step 40), and safety (Step 41). "
        f"**The central finding is that no fluorinated candidate is a suitable "
        f"drop-in replacement for cyclopentane**: all 50 candidates have vapor "
        f"pressures {min_vp:.1f}--{max_vp:.1f}x that of cyclopentane at 298 K "
        f"(median {median_vp:.1f}x), making them too volatile for direct "
        f"substitution. However, the pipeline identified {n_t1} Tier 1 "
        f"(recommended for further study) and {n_t2} Tier 2 (conditional) "
        f"candidates, with the top-ranked compound being `{top1_smi}` "
        f"(dossier score {top1_score}, {top1_tier}). All 50 candidates have "
        f"excellent safety profiles (all A1 non-flammable, ultra-low/low GWP, "
        f"no toxicity flags), and may serve as blowing agents in reformulated "
        f"foam systems that accommodate higher volatility."
    )
    lines.append("")

    # ---- Screening Pipeline Overview ----
    lines.append("## Screening Pipeline Overview")
    lines.append("")
    lines.append(
        "The pipeline proceeds through the following stages:"
    )
    lines.append("")
    lines.append(
        "1. **Enumeration** (Step 20): Systematic generation of 10,700 "
        "HFO/HCFO candidate structures via combinatorial fluorination of "
        "C3--C5 olefin scaffolds."
    )
    lines.append(
        "2. **PC-SAFT Prediction** (Steps 01--04, 15): Random Forest models "
        "trained on 1,801 Esper dataset molecules predict m, sigma, epsilon_k "
        "from RDKit descriptors + Morgan fingerprints."
    )
    lines.append(
        "3. **7-Filter Cascade** (Step 38b): Boiling point [288--323 K], "
        "C=C bond present, no Cl, F >= 2, SA score <= 4.5, F mass "
        "fraction >= 65%, no reactive fluorination sites. Yields 50 candidates."
    )
    lines.append(
        "4. **Uncertainty & AD Screening** (Step 39): Tanimoto-based "
        "applicability domain assessment + RF tree-variance uncertainty "
        "quantification. 21 screening_ready, 28 warning, 1 high_risk."
    )
    lines.append(
        "5. **EOS Property Validation** (Step 40): Multi-temperature "
        "PC-SAFT EOS evaluation (273, 298, 323 K). 49/50 converge at "
        "all 3 temperatures. VP ratios 2.4--9.0x cyclopentane."
    )
    lines.append(
        "6. **Safety Gate** (Step 41): Structural safety heuristics, GWP "
        "proxy, flammability classification, toxicity screening. 50/50 pass "
        "(all score 5/5)."
    )
    lines.append(
        "7. **Final Dossier** (Step 42, this report): Multi-criteria scoring, "
        "tiered ranking, per-candidate dossier cards."
    )
    lines.append("")

    # ---- Model Selection ----
    lines.append("## Model Selection")
    lines.append("")
    lines.append(
        "Random Forest was selected over Graph Neural Network (GNN) based on "
        "rigorous domain-specific validation (Steps 38, 38c, 38d). The GNN "
        "catastrophically fails on fluorinated compounds: boiling point MAE "
        "of 133.7 K vs RF's 8.2 K (a 16.4x performance gap). This failure "
        "is architectural -- GNN message-passing aggregation over fluorine-heavy "
        "neighborhoods produces degenerate node representations -- and persists "
        "across retraining strategies (original Esper, augmented with fluorinated "
        "data, experimental compounds). RF with RDKit descriptors + Morgan "
        "fingerprints provides reliable predictions across the chemical space "
        "relevant to this screening."
    )
    lines.append("")

    # ---- Top 10 Candidate Profiles ----
    lines.append("## Top 10 Candidate Profiles")
    lines.append("")
    lines.append(
        "| Rank | SMILES | Formula | MW | m | sigma | eps/k | "
        "Cred. | bp (K) | VP ratio | H ratio | Safety | GWP | Flamm. | "
        "Score | Tier |"
    )
    lines.append(
        "|------|--------|---------|-----|-----|-------|-------|"
        "-------|--------|----------|---------|--------|-----|--------|"
        "-------|------|"
    )
    for _, card in df_cards.iterrows():
        smi = card["smiles"]
        smi_short = smi[:25] if len(smi) > 25 else smi
        tier_short = str(card["tier"]).split("(")[0].strip()
        lines.append(
            f"| {int(card['dossier_rank'])} "
            f"| `{smi_short}` "
            f"| {card['molecular_formula']} "
            f"| {card['molecular_weight']:.1f} "
            f"| {card['m']:.3f} "
            f"| {card['sigma']:.3f} "
            f"| {card['epsilon_k']:.1f} "
            f"| {card['credibility'][:8]} "
            f"| {card['boiling_point_K']:.1f} "
            f"| {card['vp_ratio_298']:.2f} "
            f"| {card['H_ratio']:.1f} "
            f"| {int(card['safety_score'])}/5 "
            f"| {card['gwp_class']} "
            f"| {card['flammability_class']} "
            f"| {card['dossier_score']} "
            f"| {tier_short} |"
        )
    lines.append("")

    # Key caveats per candidate
    lines.append("### Key Caveats per Candidate")
    lines.append("")
    for _, card in df_cards.iterrows():
        rank = int(card["dossier_rank"])
        lines.append(f"- **#{rank}** (`{card['smiles']}`): {card['key_caveat']}")
    lines.append("")

    # ---- Tier Distribution ----
    lines.append("## Tier Distribution")
    lines.append("")
    lines.append("| Tier | Criteria | Count | Fraction |")
    lines.append("|------|----------|-------|----------|")
    lines.append(
        f"| Tier 1 (Recommended) | dossier_score >= 70 | {n_t1} | "
        f"{100 * n_t1 / n_total:.0f}% |"
    )
    lines.append(
        f"| Tier 2 (Conditional) | dossier_score 50-69 | {n_t2} | "
        f"{100 * n_t2 / n_total:.0f}% |"
    )
    lines.append(
        f"| Tier 3 (Not recommended) | dossier_score < 50 | {n_t3} | "
        f"{100 * n_t3 / n_total:.0f}% |"
    )
    lines.append("")
    lines.append(
        "**Operational meaning**: Tier 1 candidates have the highest composite "
        "score across credibility, thermodynamic viability, safety, and parameter "
        "proximity. They are recommended for experimental validation (synthesis, "
        "property measurement, foam formulation trials). Tier 2 candidates may "
        "be viable with additional data or modified operating conditions. "
        "Tier 3 candidates are not recommended for further study in this context."
    )
    lines.append("")

    # ---- The Fundamental Finding ----
    lines.append("## The Fundamental Finding: Parameter Proximity Does Not "
                 "Predict Property Proximity")
    lines.append("")
    lines.append(
        "The most scientifically important result of this screening is that "
        "**proximity in PC-SAFT parameter space does not predict proximity "
        "in thermodynamic property space** for HFO/HCFO compounds. The "
        "Pearson correlation between HFO parameter distance and property-space "
        "distance at 298 K is r = -0.709 (negative: closer in parameters often "
        "means *further* in properties)."
    )
    lines.append("")
    lines.append(
        "This occurs because:"
    )
    lines.append("")
    lines.append(
        "1. **Nonlinear parameter-property mapping**: Vapor pressure depends "
        "nonlinearly on all three PC-SAFT parameters (m, sigma, epsilon_k). "
        "Small changes in the parameter combination can produce large changes "
        "in VP, especially near phase boundaries."
    )
    lines.append(
        "2. **Fluorination compensation effect**: Fluorination systematically "
        "lowers epsilon_k (dispersion energy) while increasing m (chain length) "
        "and sigma (segment diameter). The 5:2:1 weighting on epsilon_k in the "
        "HFO distance metric brings candidates close in parameter space, but "
        "the resulting VP depends on the *product* of these parameters in "
        "complex ways that the weighted distance cannot capture."
    )
    lines.append(
        "3. **VP divergence**: All 50 candidates have VP ratios of "
        f"{min_vp:.1f}--{max_vp:.1f}x cyclopentane at 298 K (median "
        f"{median_vp:.1f}x). This means they are 2--9 times more volatile "
        "than cyclopentane, making them unsuitable as drop-in replacements "
        "in existing foam formulations."
    )
    lines.append("")

    # ---- Implications for Industry ----
    lines.append("## Implications for Industry")
    lines.append("")
    lines.append(
        "1. **Drop-in cyclopentane replacement from fluorinated compounds is "
        "unlikely**: The fundamental thermodynamic difference between fluorinated "
        "olefins and cyclopentane (a hydrocarbon) means that no amount of "
        "structural optimization within the HFO/HCFO class will produce a "
        "compound with matching vapor pressure and density. This is a structural "
        "limitation of fluorinated chemistry, not a modeling artifact."
    )
    lines.append("")
    lines.append(
        f"2. **Best fluorinated alternatives for modified systems**: The top "
        f"{n_t1 + n_t2} candidates (Tier 1 + 2) may work with reformulated "
        f"foam systems designed for higher blowing agent volatility. VP ratios "
        f"of {min_vp:.1f}--{max_vp:.1f}x could be accommodated by adjusting "
        f"polyol reactivity, catalyst loading, and mold temperatures."
    )
    lines.append("")
    lines.append(
        "3. **Methodology transfers to other targets**: The screening pipeline "
        "(SMILES -> PC-SAFT -> EOS properties -> multi-criteria ranking) is "
        "target-agnostic. Replacing cyclopentane with a different reference "
        "compound (e.g., HFO-1234ze(E), HFC-245fa) or different property "
        "windows immediately repurposes the entire pipeline."
    )
    lines.append("")
    lines.append(
        "4. **Excellent safety profile**: All 50 candidates score 5/5 on "
        "safety heuristics, are classified as A1 (non-flammable), and have "
        f"ultra-low ({(df['gwp_class'] == 'ultra_low').sum()}) or low "
        f"({(df['gwp_class'] == 'low').sum()}) GWP. No toxicity red flags "
        "were identified. This confirms that HFO/HCFO candidates are "
        "inherently safe from a structural perspective."
    )
    lines.append("")

    # ---- Methodology Contributions ----
    lines.append("## Methodology Contributions")
    lines.append("")
    lines.append(
        "This project demonstrates several methodological advances for "
        "ML-assisted chemical screening:"
    )
    lines.append("")
    lines.append(
        "1. **End-to-end pipeline from SMILES to thermodynamic properties**: "
        "The pipeline predicts PC-SAFT parameters from molecular structure "
        "(RDKit descriptors + Morgan FP -> RF), then evaluates macroscopic "
        "properties (VP, density, Henry's constant) via the PC-SAFT EOS. "
        "This bridges the gap between ML prediction and engineering-relevant "
        "property estimation."
    )
    lines.append("")
    lines.append(
        "2. **Importance of property-space validation**: Parameter-space "
        "proximity is necessary but not sufficient for screening. The "
        "negative correlation (r = -0.709) between parameter and property "
        "distances demonstrates that EOS-level validation is essential. "
        "Screening on predicted parameters alone would produce misleading "
        "rankings."
    )
    lines.append("")
    lines.append(
        "3. **Uncertainty-aware screening with credibility tiers**: "
        "RF tree-variance uncertainty quantification combined with "
        "Tanimoto-based applicability domain assessment enables "
        "risk-stratified candidate lists. The 80% 1-sigma calibration "
        "coverage confirms that uncertainty estimates are conservative."
    )
    lines.append("")
    lines.append(
        "4. **Model selection via domain-specific validation**: "
        "The GNN's catastrophic failure on fluorinated compounds "
        "(16.4x worse bp MAE than RF) was only discovered through "
        "domain-specific validation on the target chemical class. "
        "Standard test-set metrics on the Esper dataset did not "
        "reveal this failure mode. This underscores the importance "
        "of validating models on the specific chemical space of interest."
    )
    lines.append("")

    # ---- Figures ----
    lines.append("## Figures")
    lines.append("")
    lines.append("See `figures/42_final_candidate_dossier/` for:")
    lines.append("")
    lines.append(
        "- `dossier_score_breakdown.png` -- Stacked horizontal bar chart "
        "showing the 4 score components (credibility, thermo viability, "
        "safety, parameter proximity) for the top 20 candidates. Tier "
        "boundaries at scores 50 and 70 are marked."
    )
    lines.append(
        "- `tier_distribution.png` -- Bar chart of final tier distribution "
        "across all 50 candidates."
    )
    lines.append(
        "- `radar_top5.png` -- Radar chart comparing the top 5 candidates "
        "across 5 normalized axes: credibility, VP similarity, Henry's "
        "similarity, safety, and parameter proximity."
    )
    lines.append(
        "- `screening_funnel_complete.png` -- Complete screening funnel "
        "from 10,700 enumerated candidates through filter cascade, "
        "credibility screening, EOS validation, safety gate, to final "
        "tier distribution."
    )
    lines.append("")

    # ---- Readiness Check ----
    lines.append("## Readiness Check")
    lines.append("")
    lines.append("- [x] All screening data loaded and merged (Steps 39-41)")
    lines.append(
        "- [x] Multi-criteria dossier score computed (credibility + thermo + "
        "safety + proximity, 0-100 scale)"
    )
    lines.append("- [x] 50 candidates ranked and tiered (Tier 1/2/3)")
    lines.append("- [x] Top 10 dossier cards with per-candidate caveats")
    lines.append("- [x] Full dossier CSV saved (hfo_rf_final_dossier.csv)")
    lines.append("- [x] Top 10 cards CSV saved (hfo_rf_top10_dossier_cards.csv)")
    lines.append("- [x] 4 publication-quality figures generated")
    lines.append(
        "- [x] Fundamental finding documented: parameter proximity does not "
        "predict property proximity (r = -0.709)"
    )
    lines.append(
        "- [x] Industry implications and methodology contributions described"
    )
    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    logger.info("  Report saved to: %s", REPORT_PATH)


# ===================================================================
# Main
# ===================================================================


def main():
    logger.info("=" * 80)
    logger.info("Step 42: Final Candidate Dossier -- Capstone Screening Synthesis")
    logger.info("=" * 80)

    # Check inputs exist
    for csv_path, step in [(CREDIBILITY_CSV, 39), (THERMO_CSV, 40), (SAFETY_CSV, 41)]:
        if not csv_path.exists():
            logger.error("Input CSV not found: %s", csv_path)
            logger.error("Run Step %d first.", step)
            sys.exit(1)

    # 42.1: Load and merge
    df = load_and_merge()

    # 42.2: Multi-criteria scoring
    df = compute_dossier_score(df)

    # 42.3: Ranking and tiers
    df = assign_tiers(df)

    # 42.4: Dossier cards
    df_cards = build_dossier_cards(df)

    # 42.5: Save outputs
    save_outputs(df, df_cards)

    # 42.6: Figures
    generate_figures(df)

    # 42.7: Report
    generate_report(df, df_cards)

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Step 42 COMPLETE -- Final Candidate Dossier")
    logger.info("=" * 80)
    logger.info("  Dossier CSV: %s", DOSSIER_CSV)
    logger.info("  Top-10 CSV:  %s", TOP10_CSV)
    logger.info("  Figures:     %s", FIGURES_DIR)
    logger.info("  Report:      %s", REPORT_PATH)
    logger.info("")
    logger.info("TOP 10 FINAL RANKING:")
    logger.info("-" * 100)
    for _, row in df.head(10).iterrows():
        rank = int(row["dossier_rank"])
        smi = row["smiles"][:35]
        score = row["dossier_score"]
        tier = str(row["tier"])
        vp = row["vp_ratio_298"]
        cred = row["credibility"][:8]
        logger.info(
            "  #%2d | %-35s | score=%.1f | %s | VP=%.1fx | cred=%s",
            rank, smi, score, tier, vp, cred,
        )


if __name__ == "__main__":
    main()
