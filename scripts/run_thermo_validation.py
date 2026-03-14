#!/usr/bin/env python
"""Run thermodynamic validation on screening results.

This script validates that parameter-space proximity implies thermodynamic
property proximity by computing vapor pressure and liquid density for all
screening candidates using PC-SAFT EOS via teqp.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.thermodynamic import CYCLOPENTANE, T_REF, validate_candidates


def plot_param_vs_property_distance(df, output_dir, spearman_rho):
    """Scatter plot of parameter distance vs property distance."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter out NaN values for plotting
    valid = df.dropna(subset=["property_distance"])

    # Color by top-20 in parameter ranking
    colors = ["#e74c3c" if i < 20 else "#3498db" for i in range(len(valid))]

    ax.scatter(
        valid["distance"],
        valid["property_distance"],
        c=colors,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidths=0.5,
    )

    # Add Spearman correlation annotation
    textstr = f"Spearman ρ = {spearman_rho:.3f}"
    ax.text(
        0.95,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="black", label="Top-20 by parameter"),
        Patch(facecolor="#3498db", edgecolor="black", label="Other candidates"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

    ax.set_xlabel("Parameter-Space Distance (weighted Euclidean)", fontsize=14)
    ax.set_ylabel("Property-Space Distance (normalized)", fontsize=14)
    ax.set_title(
        "Parameter Distance vs Property Distance Validation", fontsize=16, pad=20
    )
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "param_vs_property_distance.png", dpi=300)
    plt.close()


def plot_vapor_pressure_comparison(df, ref_vp, output_dir):
    """Bar chart showing vapor pressure for top-20 candidates."""
    top20 = df.dropna(subset=["vapor_pressure_Pa"]).head(20)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert to kPa for readability
    vp_kpa = top20["vapor_pressure_Pa"].values / 1000
    ref_vp_kpa = ref_vp / 1000

    x = np.arange(len(top20))
    ax.bar(x, vp_kpa, color="#3498db", alpha=0.7, edgecolor="black")

    # Highlight reference line
    ax.axhline(
        y=ref_vp_kpa, color="#e74c3c", linestyle="--", linewidth=2, label="Cyclopentane"
    )

    # Add SMILES labels (rotate for readability)
    ax.set_xticks(x)
    ax.set_xticklabels(top20["smiles"].values, rotation=45, ha="right", fontsize=10)

    ax.set_xlabel("SMILES (Top-20 by Parameter Distance)", fontsize=14)
    ax.set_ylabel("Vapor Pressure (kPa)", fontsize=14)
    ax.set_title("Vapor Pressure at 298.15 K: Top-20 Candidates", fontsize=16, pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "vapor_pressure_comparison.png", dpi=300)
    plt.close()


def plot_rank_comparison(df, output_dir):
    """Compare rankings by parameter distance vs property distance."""
    # Get top-20 by parameter distance that have valid property distance
    top20_param = df.head(20).copy()
    valid_top20 = top20_param.dropna(subset=["property_distance"])

    # Re-rank by property distance
    all_valid = df.dropna(subset=["property_distance"]).copy()
    all_valid["property_rank"] = all_valid["property_distance"].rank()

    # Merge ranks
    valid_top20 = valid_top20.merge(
        all_valid[["smiles", "property_rank"]], on="smiles", how="left"
    )
    valid_top20["param_rank"] = np.arange(1, len(valid_top20) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw lines connecting ranks
    for _, row in valid_top20.iterrows():
        color = "#2ecc71" if row["property_rank"] <= 20 else "#e74c3c"
        ax.plot(
            [0, 1],
            [row["param_rank"], row["property_rank"]],
            color=color,
            alpha=0.6,
            linewidth=2,
        )

    # Add dots at endpoints
    ax.scatter(
        [0] * len(valid_top20),
        valid_top20["param_rank"],
        color="#3498db",
        s=100,
        zorder=3,
        edgecolors="black",
    )
    ax.scatter(
        [1] * len(valid_top20),
        valid_top20["property_rank"],
        color=[
            "#2ecc71" if r <= 20 else "#e74c3c"
            for r in valid_top20["property_rank"]
        ],
        s=100,
        zorder=3,
        edgecolors="black",
    )

    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Ranked by\nParameter Distance", "Ranked by\nProperty Distance"], fontsize=14
    )
    ax.set_ylabel("Rank (1 = best)", fontsize=14)
    ax.set_title(
        "Ranking Comparison: Top-20 Candidates by Parameter Distance", fontsize=16, pad=20
    )
    ax.invert_yaxis()  # Lower rank at top
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=12)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Stays in top-20"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Drops out of top-20"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "rank_comparison.png", dpi=300)
    plt.close()


def plot_sensitivity_analysis(output_dir):
    """Show how each parameter affects vapor pressure."""
    from model.thermodynamic import compute_properties

    ref = CYCLOPENTANE

    # Vary each parameter ±20% while holding others constant
    variations = np.linspace(0.8, 1.2, 50)

    # m variation
    m_results = [
        compute_properties(ref["m"] * v, ref["sigma"], ref["epsilon_k"])
        for v in variations
    ]
    m_vp = [r["vapor_pressure_Pa"] / 1000 for r in m_results]  # Convert to kPa

    # sigma variation
    sigma_results = [
        compute_properties(ref["m"], ref["sigma"] * v, ref["epsilon_k"])
        for v in variations
    ]
    sigma_vp = [r["vapor_pressure_Pa"] / 1000 for r in sigma_results]

    # epsilon_k variation
    eps_results = [
        compute_properties(ref["m"], ref["sigma"], ref["epsilon_k"] * v)
        for v in variations
    ]
    eps_vp = [r["vapor_pressure_Pa"] / 1000 for r in eps_results]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Reference vapor pressure
    ref_props = compute_properties(ref["m"], ref["sigma"], ref["epsilon_k"])
    ref_vp_kpa = ref_props["vapor_pressure_Pa"] / 1000

    ax.plot(
        (variations - 1) * 100, m_vp, label="m variation", linewidth=2, marker="o", markersize=3
    )
    ax.plot(
        (variations - 1) * 100,
        sigma_vp,
        label="σ variation",
        linewidth=2,
        marker="s",
        markersize=3,
    )
    ax.plot(
        (variations - 1) * 100,
        eps_vp,
        label="ε/k variation",
        linewidth=2,
        marker="^",
        markersize=3,
    )

    # Reference line
    ax.axhline(y=ref_vp_kpa, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Parameter Variation (%)", fontsize=14)
    ax.set_ylabel("Vapor Pressure (kPa)", fontsize=14)
    ax.set_title(
        "Sensitivity Analysis: Parameter Impact on Vapor Pressure", fontsize=16, pad=20
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_analysis.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Validate screening results with thermodynamic properties"
    )
    parser.add_argument(
        "--screening-results",
        type=Path,
        default=Path("screening/results/ranked_candidates_broad.csv"),
        help="Path to screening results CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/saved/thermo_validation.csv"),
        help="Path for output CSV with thermodynamic properties",
    )

    args = parser.parse_args()

    # Load screening results
    print(f"Loading screening results from {args.screening_results}...")
    candidates = pd.read_csv(args.screening_results)
    print(f"Loaded {len(candidates)} candidates")

    # Compute reference properties
    print(f"\nComputing reference properties for cyclopentane at {T_REF} K...")
    from model.thermodynamic import compute_properties

    ref_props = compute_properties(
        CYCLOPENTANE["m"], CYCLOPENTANE["sigma"], CYCLOPENTANE["epsilon_k"]
    )
    print(f"  Vapor pressure: {ref_props['vapor_pressure_Pa']/1000:.2f} kPa")
    print(f"  Liquid density: {ref_props['liquid_density_mol_m3']:.1f} mol/m³")

    # Validate all candidates
    print("\nComputing thermodynamic properties for all candidates...")
    validated_df, _ = validate_candidates(candidates)

    # Count failures
    n_failed = validated_df["vapor_pressure_Pa"].isna().sum()
    fail_pct = n_failed / len(candidates) * 100
    print(
        f"\nEOS calculation failed for {n_failed}/{len(candidates)} "
        f"candidates ({fail_pct:.1f}%)"
    )

    # Compute Spearman correlation (exclude NaN values)
    valid_for_corr = validated_df.dropna(subset=["property_distance"])
    if len(valid_for_corr) > 1:
        spearman_rho, p_value = spearmanr(
            valid_for_corr["distance"], valid_for_corr["property_distance"]
        )
        print(f"\nSpearman rank correlation: ρ = {spearman_rho:.3f} (p = {p_value:.2e})")
    else:
        print("\nInsufficient valid data for correlation analysis")
        spearman_rho = np.nan

    # Save results
    print(f"\nSaving results to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(args.output, index=False)

    # Generate figures
    fig_dir = Path("figures/09_thermodynamic_validation")
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures in {fig_dir}...")

    plot_param_vs_property_distance(validated_df, fig_dir, spearman_rho)
    print("  ✓ param_vs_property_distance.png")

    plot_vapor_pressure_comparison(
        validated_df, ref_props["vapor_pressure_Pa"], fig_dir
    )
    print("  ✓ vapor_pressure_comparison.png")

    plot_rank_comparison(validated_df, fig_dir)
    print("  ✓ rank_comparison.png")

    plot_sensitivity_analysis(fig_dir)
    print("  ✓ sensitivity_analysis.png")

    print("\n" + "=" * 60)
    print("Thermodynamic validation complete!")
    print(f"  Results: {args.output}")
    print(f"  Figures: {fig_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
