"""Step 40: EOS Robustness & Property-Space Validation.

Validates the 50 RF-ranked HFO candidates (Step 39 credibility table) with:
  1. Multi-temperature EOS convergence (273, 298, 323 K) (40.1)
  2. Property-space distance to cyclopentane (VP/density ratios) (40.2)
  3. Henry's constant in hexane at 298 K (40.3)
  4. Thermodynamic screening gates (40.4)
  5. Merge with Step 39 credibility table (40.5)
  6. Four publication-quality figures (40.6)
  7. Report generation (40.7)

Inputs:
  - screening/results/hfo_rf_shortlist_credibility.csv (50 candidates from Step 39)

Outputs:
  - screening/results/hfo_rf_thermo_validated.csv
  - figures/40_eos_property_validation/ (4 figures)
  - docs/reports/40_eos_property_validation.md

Usage:
    python scripts/step40_eos_property_validation.py
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.thermodynamic import (  # noqa: E402
    CYCLOPENTANE,
    HEXANE,
    compute_henrys_constant,
    compute_properties,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_shortlist_credibility.csv"
OUTPUT_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_thermo_validated.csv"
FIGURES_DIR = PROJECT_ROOT / "figures" / "40_eos_property_validation"
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "40_eos_property_validation.md"

# Temperatures for multi-T sweep
TEMPERATURES = [273.15, 298.15, 323.15]

# Thermodynamic screening gates
VP_RATIO_BOUNDS = (0.3, 3.0)
PROPERTY_DISTANCE_MAX = 1.0
HENRYS_RATIO_BOUNDS = (0.1, 10.0)

# Figure styling
FONT_SIZES = {"xlabel": 14, "ylabel": 14, "title": 16, "tick": 12, "legend": 11}
FIG_DPI = 150

# Credibility-to-color mapping
CRED_COLORS = {
    "screening_ready": "#2ca02c",
    "screening_with_warning": "#ff7f0e",
    "high_extrapolation_risk": "#d62728",
}


# ===================================================================
# 40.1 — Multi-Temperature EOS Validation
# ===================================================================


def run_multi_temperature_eos(df):
    """Run EOS at 273, 298, 323 K for all candidates and cyclopentane reference.

    Returns:
        tuple: (results_df with per-temperature columns, ref_props dict keyed by T)
    """
    logger.info("[40.1] Multi-Temperature EOS Validation")
    logger.info("=" * 60)

    # Compute cyclopentane reference at each temperature
    ref_props = {}
    for T in TEMPERATURES:
        props = compute_properties(
            CYCLOPENTANE["m"], CYCLOPENTANE["sigma"], CYCLOPENTANE["epsilon_k"], T
        )
        ref_props[T] = props
        logger.info(
            "  Cyclopentane at %.1f K: VP=%.1f Pa, rho_L=%.1f mol/m3",
            T,
            props["vapor_pressure_Pa"],
            props["liquid_density_mol_m3"],
        )

    # Compute properties for each candidate at each temperature
    for T in TEMPERATURES:
        T_label = f"{T:.0f}"
        vp_col = f"vp_{T_label}"
        rho_col = f"rho_{T_label}"
        converge_col = f"converges_{T_label}"

        vp_vals, rho_vals, conv_vals = [], [], []
        for _, row in df.iterrows():
            props = compute_properties(row["m"], row["sigma"], row["epsilon_k"], T)
            vp = props["vapor_pressure_Pa"]
            rho = props["liquid_density_mol_m3"]
            converged = not (np.isnan(vp) or np.isnan(rho))

            vp_vals.append(vp)
            rho_vals.append(rho)
            conv_vals.append(converged)

        df[vp_col] = vp_vals
        df[rho_col] = rho_vals
        df[converge_col] = conv_vals

    # Count how many temperatures each candidate converges at
    converge_cols = [f"converges_{T:.0f}" for T in TEMPERATURES]
    df["eos_convergence_count"] = df[converge_cols].sum(axis=1).astype(int)

    n_all = (df["eos_convergence_count"] == 3).sum()
    n_some = ((df["eos_convergence_count"] > 0) & (df["eos_convergence_count"] < 3)).sum()
    n_none = (df["eos_convergence_count"] == 0).sum()
    logger.info(
        "  Convergence: %d/50 at all 3T, %d at some, %d at none",
        n_all,
        n_some,
        n_none,
    )

    return df, ref_props


# ===================================================================
# 40.2 — Property-Space Distance to Cyclopentane
# ===================================================================


def compute_property_distances(df, ref_props):
    """Compute VP ratio, density ratio, and property-space distance at each T.

    Also computes composite property distance across all 3 temperatures.
    """
    logger.info("\n[40.2] Property-Space Distance to Cyclopentane")
    logger.info("=" * 60)

    for T in TEMPERATURES:
        T_label = f"{T:.0f}"
        ref_vp = ref_props[T]["vapor_pressure_Pa"]
        ref_rho = ref_props[T]["liquid_density_mol_m3"]

        vp_col = f"vp_{T_label}"
        rho_col = f"rho_{T_label}"

        # VP ratio and density ratio
        df[f"vp_ratio_{T_label}"] = df[vp_col] / ref_vp
        df[f"rho_ratio_{T_label}"] = df[rho_col] / ref_rho

        # Property-space distance at this T
        df[f"property_distance_{T_label}"] = np.sqrt(
            (df[f"vp_ratio_{T_label}"] - 1.0) ** 2
            + (df[f"rho_ratio_{T_label}"] - 1.0) ** 2
        )

    # Composite property distance: RMS across the 3 temperatures
    dist_cols = [f"property_distance_{T:.0f}" for T in TEMPERATURES]
    df["property_distance_composite"] = np.sqrt(
        (df[dist_cols] ** 2).mean(axis=1)
    )

    # Summary for 298 K
    d298 = df["property_distance_298"]
    logger.info("  Property distance at 298 K: median=%.3f, mean=%.3f", d298.median(), d298.mean())
    logger.info(
        "  VP ratio at 298 K: median=%.3f, mean=%.3f",
        df["vp_ratio_298"].median(),
        df["vp_ratio_298"].mean(),
    )

    return df


# ===================================================================
# 40.3 — Henry's Constant Validation
# ===================================================================


def compute_henrys_constants(df):
    """Compute Henry's constant in hexane at 298 K for converging candidates."""
    logger.info("\n[40.3] Henry's Constant Validation")
    logger.info("=" * 60)

    # Cyclopentane reference
    ref_result = compute_henrys_constant(CYCLOPENTANE, HEXANE, 298.15, k_ij=0.0)
    if ref_result is None:
        logger.error("  Failed to compute cyclopentane Henry's constant!")
        df["H_Pa"] = np.nan
        df["H_ratio"] = np.nan
        return df

    H_ref = ref_result["H_Pa"]
    logger.info("  Cyclopentane H (hexane, 298 K): %.1f Pa", H_ref)

    h_pa_vals = []
    h_ratio_vals = []

    for _, row in df.iterrows():
        # Only compute for candidates that converge EOS at 298 K
        if not row.get("converges_298", False):
            h_pa_vals.append(np.nan)
            h_ratio_vals.append(np.nan)
            continue

        solute_params = {
            "m": row["m"],
            "sigma": row["sigma"],
            "epsilon_k": row["epsilon_k"],
        }
        result = compute_henrys_constant(solute_params, HEXANE, 298.15, k_ij=0.0)

        if result is None:
            h_pa_vals.append(np.nan)
            h_ratio_vals.append(np.nan)
        else:
            h_pa_vals.append(result["H_Pa"])
            h_ratio_vals.append(result["H_Pa"] / H_ref)

    df["H_Pa"] = h_pa_vals
    df["H_ratio"] = h_ratio_vals

    n_computed = df["H_Pa"].notna().sum()
    logger.info("  Henry's constant computed for %d/%d candidates", n_computed, len(df))
    if n_computed > 0:
        valid_h = df["H_ratio"].dropna()
        logger.info(
            "  H_ratio: median=%.3f, mean=%.3f, range=[%.3f, %.3f]",
            valid_h.median(),
            valid_h.mean(),
            valid_h.min(),
            valid_h.max(),
        )

    return df


# ===================================================================
# 40.4 — Thermodynamic Screening Gates
# ===================================================================


def apply_thermo_gates(df):
    """Apply pass/fail thermodynamic screening gates."""
    logger.info("\n[40.4] Thermodynamic Screening Gates")
    logger.info("=" * 60)

    # Gate 1: EOS convergence at all 3 temperatures
    df["gate_eos_all3"] = df["eos_convergence_count"] == 3

    # Gate 2: VP ratio in [0.3, 3.0] at 298 K
    df["gate_vp_ratio"] = df["vp_ratio_298"].between(
        VP_RATIO_BOUNDS[0], VP_RATIO_BOUNDS[1]
    )

    # Gate 3: Property distance < 1.0 at 298 K
    df["gate_prop_dist"] = df["property_distance_298"] < PROPERTY_DISTANCE_MAX

    # Gate 4: Henry's ratio in [0.1, 10]
    df["gate_henrys"] = df["H_ratio"].between(
        HENRYS_RATIO_BOUNDS[0], HENRYS_RATIO_BOUNDS[1]
    )

    # Combined: passes all gates
    df["passes_all_thermo_gates"] = (
        df["gate_eos_all3"]
        & df["gate_vp_ratio"]
        & df["gate_prop_dist"]
        & df["gate_henrys"]
    )

    # Report
    gate_names = {
        "gate_eos_all3": "EOS convergence (all 3T)",
        "gate_vp_ratio": f"VP ratio in {VP_RATIO_BOUNDS}",
        "gate_prop_dist": f"Property distance < {PROPERTY_DISTANCE_MAX}",
        "gate_henrys": f"Henry's ratio in {HENRYS_RATIO_BOUNDS}",
        "passes_all_thermo_gates": "ALL gates",
    }
    for col, name in gate_names.items():
        n_pass = df[col].sum()
        logger.info("  %s: %d/50 pass (%.0f%%)", name, n_pass, 100.0 * n_pass / len(df))

    return df


# ===================================================================
# 40.5 — Merge with Step 39 Credibility
# ===================================================================


def build_enriched_table(df):
    """Build the final enriched table with selected columns."""
    logger.info("\n[40.5] Building Enriched Table")
    logger.info("=" * 60)

    output_cols = [
        "smiles",
        "m",
        "sigma",
        "epsilon_k",
        "hfo_distance",
        "credibility",
        "ad_label",
        "mean_cv",
        "eos_convergence_count",
        "vp_ratio_298",
        "rho_ratio_298",
        "property_distance_298",
        "H_ratio",
        "passes_all_thermo_gates",
    ]

    # Keep only columns that exist
    available = [c for c in output_cols if c in df.columns]
    out = df[available].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logger.info(
        "  Saved enriched table to %s (%d rows, %d cols)",
        OUTPUT_CSV, len(out), len(available),
    )

    return out


# ===================================================================
# 40.6 — Figures
# ===================================================================


def plot_eos_convergence_by_credibility(df):
    """Stacked bar chart: convergence counts by credibility tier."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cred_order = ["screening_ready", "screening_with_warning", "high_extrapolation_risk"]
    conv_values = [0, 1, 2, 3]
    conv_colors = ["#d62728", "#ff7f0e", "#ffdd57", "#2ca02c"]
    conv_labels = ["0 temps", "1 temp", "2 temps", "3 temps"]

    x = np.arange(len(cred_order))
    width = 0.6
    bottoms = np.zeros(len(cred_order))

    for cv, color, label in zip(conv_values, conv_colors, conv_labels):
        counts = []
        for cred in cred_order:
            mask = (df["credibility"] == cred) & (df["eos_convergence_count"] == cv)
            counts.append(mask.sum())
        counts = np.array(counts, dtype=float)
        ax.bar(x, counts, width, bottom=bottoms, color=color, label=label, edgecolor="white")
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["screening_ready", "screening_with\n_warning", "high_extrapolation\n_risk"],
        fontsize=FONT_SIZES["tick"],
    )
    ax.set_ylabel("Number of candidates", fontsize=FONT_SIZES["ylabel"])
    ax.set_title("EOS Convergence by Credibility Tier", fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"], loc="upper right")
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])

    plt.tight_layout()
    path = FIGURES_DIR / "eos_convergence_by_credibility.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_vp_ratio_temperature_sweep(df):
    """Line plot: VP ratio vs temperature for top-20 candidates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by hfo_distance and take top 20
    top20 = df.nsmallest(20, "hfo_distance")

    for _, row in top20.iterrows():
        temps = []
        ratios = []
        for T in TEMPERATURES:
            T_label = f"{T:.0f}"
            ratio = row.get(f"vp_ratio_{T_label}", np.nan)
            if not np.isnan(ratio):
                temps.append(T)
                ratios.append(ratio)

        if len(temps) > 0:
            cred = row.get("credibility", "screening_with_warning")
            color = CRED_COLORS.get(cred, "#999999")
            ax.plot(
                temps, ratios, "o-", color=color,
                alpha=0.6, markersize=4, linewidth=1.2,
            )

    # Cyclopentane reference line
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Cyclopentane (ratio=1)")

    # Custom legend for credibility tiers
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0], color="black", linestyle="--",
            linewidth=1.5, label="Cyclopentane (ratio=1)",
        ),
        Line2D(
            [0], [0], color=CRED_COLORS["screening_ready"],
            marker="o", linestyle="-", label="screening_ready",
        ),
        Line2D(
            [0], [0], color=CRED_COLORS["screening_with_warning"],
            marker="o", linestyle="-",
            label="screening_with_warning",
        ),
        Line2D(
            [0], [0], color=CRED_COLORS["high_extrapolation_risk"],
            marker="o", linestyle="-",
            label="high_extrapolation_risk",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZES["legend"], loc="best")

    ax.set_xlabel("Temperature (K)", fontsize=FONT_SIZES["xlabel"])
    ax.set_ylabel(
        "VP Ratio (candidate / cyclopentane)", fontsize=FONT_SIZES["ylabel"],
    )
    ax.set_title(
        "VP Ratio vs Temperature (Top 20 by HFO Distance)",
        fontsize=FONT_SIZES["title"],
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    ax.set_xticks(TEMPERATURES)

    plt.tight_layout()
    path = FIGURES_DIR / "vp_ratio_temperature_sweep.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_property_vs_parameter_distance(df):
    """Scatter: parameter-space distance (x) vs property-space distance (y)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Only plot candidates with valid property distance at 298 K
    valid = df.dropna(subset=["property_distance_298", "hfo_distance"])

    for cred, color in CRED_COLORS.items():
        mask = valid["credibility"] == cred
        if mask.sum() > 0:
            ax.scatter(
                valid.loc[mask, "hfo_distance"],
                valid.loc[mask, "property_distance_298"],
                c=color,
                label=cred,
                s=50,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )

    # Reference line y=x
    lims = [0, max(valid["hfo_distance"].max(), valid["property_distance_298"].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="y = x")

    ax.set_xlabel("Parameter-space distance (HFO distance)", fontsize=FONT_SIZES["xlabel"])
    ax.set_ylabel("Property-space distance (298 K)", fontsize=FONT_SIZES["ylabel"])
    ax.set_title(
        "Parameter Distance vs Property Distance",
        fontsize=FONT_SIZES["title"],
    )
    ax.legend(fontsize=FONT_SIZES["legend"])
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])

    plt.tight_layout()
    path = FIGURES_DIR / "property_distance_vs_hfo_distance.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_henrys_vs_vp_ratio(df):
    """Scatter: VP ratio (x) vs Henry's ratio (y) at 298 K."""
    fig, ax = plt.subplots(figsize=(8, 6))

    valid = df.dropna(subset=["vp_ratio_298", "H_ratio"])

    if len(valid) == 0:
        logger.warning("  No valid data for Henry's vs VP ratio plot")
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        path = FIGURES_DIR / "henrys_vs_vp_ratio.png"
        fig.savefig(path, dpi=FIG_DPI)
        plt.close(fig)
        return

    for cred, color in CRED_COLORS.items():
        mask = valid["credibility"] == cred
        if mask.sum() > 0:
            ax.scatter(
                valid.loc[mask, "vp_ratio_298"],
                valid.loc[mask, "H_ratio"],
                c=color,
                label=cred,
                s=50,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )

    # Highlight discordant candidates (high VP ratio but low H ratio or vice versa)
    # Discordant = VP ratio and H ratio disagree in magnitude (one > 2, other < 0.5)
    discordant_mask = (
        ((valid["vp_ratio_298"] > 2.0) & (valid["H_ratio"] < 0.5))
        | ((valid["vp_ratio_298"] < 0.5) & (valid["H_ratio"] > 2.0))
    )
    if discordant_mask.sum() > 0:
        ax.scatter(
            valid.loc[discordant_mask, "vp_ratio_298"],
            valid.loc[discordant_mask, "H_ratio"],
            facecolors="none",
            edgecolors="red",
            s=120,
            linewidth=2,
            label=f"Discordant ({discordant_mask.sum()})",
            zorder=5,
        )

    # Reference lines
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("VP Ratio at 298 K (candidate / cyclopentane)", fontsize=FONT_SIZES["xlabel"])
    ax.set_ylabel("Henry's Ratio (candidate / cyclopentane)", fontsize=FONT_SIZES["ylabel"])
    ax.set_title("Henry's Constant vs Vapor Pressure Ratio", fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"])
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])

    plt.tight_layout()
    path = FIGURES_DIR / "henrys_vs_vp_ratio.png"
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    logger.info("  Saved %s", path)


def generate_all_figures(df):
    """Generate all 4 figures for Step 40."""
    logger.info("\n[40.6] Generating Figures")
    logger.info("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_eos_convergence_by_credibility(df)
    plot_vp_ratio_temperature_sweep(df)
    plot_property_vs_parameter_distance(df)
    plot_henrys_vs_vp_ratio(df)


# ===================================================================
# 40.7 — Report Generation
# ===================================================================


def generate_report(df, ref_props):
    """Write the Step 40 report."""
    logger.info("\n[40.7] Writing Report")
    logger.info("=" * 60)

    n_total = len(df)

    # --- EOS convergence stats ---
    conv_counts = df["eos_convergence_count"].value_counts().sort_index()

    # --- Gate pass rates ---
    gate_stats = {
        "EOS convergence (all 3T)": df["gate_eos_all3"].sum(),
        f"VP ratio in {VP_RATIO_BOUNDS} at 298 K": df["gate_vp_ratio"].sum(),
        f"Property distance < {PROPERTY_DISTANCE_MAX} at 298 K": df["gate_prop_dist"].sum(),
        f"Henry's ratio in {HENRYS_RATIO_BOUNDS}": df["gate_henrys"].sum(),
        "All gates": df["passes_all_thermo_gates"].sum(),
    }

    # --- Convergence by credibility ---
    cred_conv_table = []
    for cred in ["screening_ready", "screening_with_warning", "high_extrapolation_risk"]:
        subset = df[df["credibility"] == cred]
        if len(subset) > 0:
            n_all3 = (subset["eos_convergence_count"] == 3).sum()
            cred_conv_table.append((cred, len(subset), n_all3))

    # --- Top 20 by property distance at 298 K ---
    valid_298 = df.dropna(subset=["property_distance_298"]).nsmallest(20, "property_distance_298")

    # --- Henry's stats ---
    h_valid = df.dropna(subset=["H_ratio"])

    # --- Correlation between parameter and property distances ---
    corr_valid = df.dropna(subset=["hfo_distance", "property_distance_298"])
    if len(corr_valid) > 2:
        corr = corr_valid["hfo_distance"].corr(corr_valid["property_distance_298"])
    else:
        corr = np.nan

    # Build report content
    lines = []
    lines.append(
        "# Results Report: EOS Robustness & Property-Space Validation "
        "(Step 40, 50 RF Candidates)"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "Step 40 validates the 50 RF-ranked HFO candidates (from Step 39's credibility table) "
        "using thermodynamic property calculations from PC-SAFT EOS via teqp. "
        "Multi-temperature EOS convergence, vapor pressure and density ratios relative to "
        "cyclopentane, and Henry's constants in hexane are computed. Four thermodynamic screening "
        "gates filter candidates based on physical plausibility and proximity to cyclopentane's "
        "property profile."
    )
    lines.append("")

    # EOS Convergence Table
    lines.append("## EOS Convergence Statistics")
    lines.append("")
    lines.append("| Convergence Count | Candidates | Fraction |")
    lines.append("|-------------------|-----------|----------|")
    for count in range(4):
        n = conv_counts.get(count, 0)
        lines.append(f"| {count} of 3 temperatures | {n} | {100.0 * n / n_total:.0f}% |")
    lines.append("")

    lines.append("### Convergence by Credibility Tier")
    lines.append("")
    lines.append("| Credibility | Total | Converge at all 3T | Fraction |")
    lines.append("|-------------|-------|--------------------|----------|")
    for cred, n, n_all3 in cred_conv_table:
        frac = 100.0 * n_all3 / n if n > 0 else 0
        lines.append(f"| {cred} | {n} | {n_all3} | {frac:.0f}% |")
    lines.append("")

    # Cyclopentane reference properties
    lines.append("### Cyclopentane Reference Properties")
    lines.append("")
    lines.append("| Temperature (K) | VP (Pa) | Liquid Density (mol/m3) |")
    lines.append("|-----------------|---------|------------------------|")
    for T in TEMPERATURES:
        rp = ref_props[T]
        lines.append(
            f"| {T:.2f} | {rp['vapor_pressure_Pa']:.1f} | {rp['liquid_density_mol_m3']:.1f} |"
        )
    lines.append("")

    # Property-space results table (top 20)
    lines.append("## Property-Space Results (Top 20 by Property Distance at 298 K)")
    lines.append("")
    lines.append(
        "| SMILES | HFO Dist | Credibility | VP Ratio 298 | Rho Ratio 298 | "
        "Prop Dist 298 | H Ratio | All Gates |"
    )
    lines.append(
        "|--------|----------|-------------|-------------|--------------|"
        "--------------|---------|-----------|"
    )
    for _, row in valid_298.iterrows():
        smiles_short = row["smiles"]
        hfo_d = f"{row['hfo_distance']:.3f}"
        cred = row.get("credibility", "N/A")
        vp_val = row.get("vp_ratio_298", np.nan)
        vp_r = f"{vp_val:.3f}" if not np.isnan(vp_val) else "N/A"
        rho_val = row.get("rho_ratio_298", np.nan)
        rho_r = f"{rho_val:.3f}" if not np.isnan(rho_val) else "N/A"
        pd_val = row.get("property_distance_298", np.nan)
        pd298 = f"{pd_val:.3f}" if not np.isnan(pd_val) else "N/A"
        h_val = row.get("H_ratio", np.nan)
        hr = f"{h_val:.3f}" if not np.isnan(h_val) else "N/A"
        gates = "PASS" if row.get("passes_all_thermo_gates", False) else "FAIL"
        lines.append(
            f"| {smiles_short} | {hfo_d} | {cred} | {vp_r} | {rho_r} | {pd298} | {hr} | {gates} |"
        )
    lines.append("")

    # Henry's constant results
    lines.append("## Henry's Constant Results")
    lines.append("")
    if len(h_valid) > 0:
        lines.append(
            f"- Computed for {len(h_valid)}/{n_total} candidates "
            f"(those converging EOS at 298 K)"
        )
        lines.append(f"- H ratio median: {h_valid['H_ratio'].median():.3f}")
        lines.append(f"- H ratio mean: {h_valid['H_ratio'].mean():.3f}")
        h_min = h_valid["H_ratio"].min()
        h_max = h_valid["H_ratio"].max()
        lines.append(f"- H ratio range: [{h_min:.3f}, {h_max:.3f}]")
        n_h_pass = h_valid["H_ratio"].between(*HENRYS_RATIO_BOUNDS).sum()
        lines.append(
            f"- {n_h_pass}/{len(h_valid)} candidates have H ratio in "
            f"{HENRYS_RATIO_BOUNDS} (within order of magnitude of cyclopentane)"
        )
    else:
        lines.append("No Henry's constant values could be computed.")
    lines.append("")

    # Gate pass rates
    lines.append("## Thermodynamic Screening Gates")
    lines.append("")
    lines.append("| Gate | Pass | Fail | Pass Rate |")
    lines.append("|------|------|------|-----------|")
    for gate_name, n_pass in gate_stats.items():
        n_fail = n_total - n_pass
        rate = 100.0 * n_pass / n_total
        lines.append(f"| {gate_name} | {n_pass} | {n_fail} | {rate:.0f}% |")
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    lines.append(
        f"1. **EOS convergence is high**: {conv_counts.get(3, 0)}/{n_total} candidates "
        f"({100.0 * conv_counts.get(3, 0) / n_total:.0f}%) converge at all 3 temperatures, "
        "indicating the RF-predicted PC-SAFT parameters are "
        "physically plausible for most candidates."
    )
    lines.append("")

    n_pass_all = gate_stats["All gates"]
    lines.append(
        f"2. **Thermodynamic gates**: {n_pass_all}/{n_total} candidates "
        f"({100.0 * n_pass_all / n_total:.0f}%) pass all four thermodynamic screening gates. "
        f"The most restrictive gate filters candidates whose vapor pressure or Henry's constant "
        f"deviates too far from cyclopentane."
    )
    lines.append("")

    if not np.isnan(corr):
        lines.append(
            "3. **Parameter-property correlation**: "
            f"Pearson r = {corr:.3f} between "
            "parameter-space distance (HFO distance) and "
            "property-space distance at 298 K. "
        )
        abs_corr = abs(corr)
        if corr < -0.5:
            lines.append(
                "   This strong negative correlation is counter-intuitive: "
                "candidates closer in parameter space actually have "
                "*larger* property-space distances. This arises because "
                "the HFO distance metric weights epsilon_k heavily, "
                "while VP depends nonlinearly on all three PC-SAFT "
                "parameters. Thermodynamic validation cannot be "
                "replaced by parameter distance alone."
            )
        elif abs_corr > 0.5:
            lines.append(
                "   This indicates a moderate-to-strong positive "
                "correlation: candidates closer in parameter space "
                "tend to have more similar thermodynamic properties, "
                "validating the parameter-distance screening heuristic."
            )
        elif abs_corr > 0.2:
            lines.append(
                "   This indicates a weak-to-moderate correlation. "
                "Parameter proximity provides some but not complete "
                "predictive power for property similarity."
            )
        else:
            lines.append(
                "   This indicates a weak correlation. Parameter "
                "proximity does not strongly predict property "
                "similarity for this fluorinated candidate set, "
                "suggesting thermodynamic validation is essential."
            )
    else:
        lines.append(
            "3. **Parameter-property correlation**: Could not be "
            "computed due to insufficient data."
        )
    lines.append("")

    # Discordant Henry's vs VP
    if len(h_valid) > 0:
        discordant = (
            ((h_valid["vp_ratio_298"] > 2.0) & (h_valid["H_ratio"] < 0.5))
            | ((h_valid["vp_ratio_298"] < 0.5) & (h_valid["H_ratio"] > 2.0))
        )
        n_disc = discordant.sum()
        lines.append(
            f"4. **Discordant VP/Henry's candidates**: {n_disc} candidates "
            "show discordant behavior "
            "(high VP ratio but low Henry's ratio or vice versa). "
        )
        if n_disc == 0:
            lines.append(
                "   The absence of discordant candidates suggests VP and solubility behavior "
                "are directionally consistent across this candidate set."
            )
        else:
            lines.append(
                "   These candidates warrant extra scrutiny as their pure-component and "
                "mixture behavior diverge from expectations."
            )
    lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append("")
    lines.append("See `figures/40_eos_property_validation/` for:")
    lines.append(
        "- `eos_convergence_by_credibility.png` -- "
        "Stacked bar: convergence counts by credibility tier"
    )
    lines.append(
        "- `vp_ratio_temperature_sweep.png` -- VP ratio vs temperature for top 20 candidates"
    )
    lines.append(
        "- `property_distance_vs_hfo_distance.png` -- "
        "Parameter distance vs property distance scatter"
    )
    lines.append(
        "- `henrys_vs_vp_ratio.png` -- "
        "Henry's ratio vs VP ratio, with discordant candidates highlighted"
    )
    lines.append("")

    # Readiness check
    lines.append("## Readiness Check")
    lines.append("")
    lines.append("- [x] Multi-temperature EOS (273, 298, 323 K) run for all 50 candidates")
    lines.append("- [x] VP ratio, density ratio, and property-space distance computed")
    lines.append("- [x] Henry's constant in hexane at 298 K computed")
    lines.append("- [x] Four thermodynamic screening gates applied")
    lines.append("- [x] Results merged with Step 39 credibility labels")
    lines.append("- [x] Enriched CSV saved to screening/results/hfo_rf_thermo_validated.csv")
    lines.append("- [x] All 4 figures generated")
    lines.append("")

    report_text = "\n".join(lines)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)
    logger.info("  Saved report to %s", REPORT_PATH)


# ===================================================================
# Main
# ===================================================================


def main():
    logger.info("Step 40: EOS Robustness & Property-Space Validation")
    logger.info("=" * 60)

    # Load Step 39 credibility table
    df = pd.read_csv(INPUT_CSV)
    logger.info("Loaded %d candidates from %s", len(df), INPUT_CSV)

    # 40.1: Multi-temperature EOS
    df, ref_props = run_multi_temperature_eos(df)

    # 40.2: Property-space distances
    df = compute_property_distances(df, ref_props)

    # 40.3: Henry's constants
    df = compute_henrys_constants(df)

    # 40.4: Thermodynamic gates
    df = apply_thermo_gates(df)

    # 40.5: Build enriched table
    build_enriched_table(df)

    # 40.6: Figures
    generate_all_figures(df)

    # 40.7: Report
    generate_report(df, ref_props)

    logger.info("\nStep 40 complete.")
    logger.info("  Output CSV: %s", OUTPUT_CSV)
    logger.info("  Figures: %s", FIGURES_DIR)
    logger.info("  Report: %s", REPORT_PATH)

    # Print summary
    n_pass = df["passes_all_thermo_gates"].sum()
    n_eos = (df["eos_convergence_count"] == 3).sum()
    logger.info(
        "\n  SUMMARY: %d/50 converge at all 3T, %d/50 pass all thermo gates",
        n_eos,
        n_pass,
    )


if __name__ == "__main__":
    main()
