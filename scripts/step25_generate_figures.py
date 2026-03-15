"""Step 25: Generate HFO-Centric Screening Figures.

Creates 5 analysis figures from screening results:
1. screening_funnel.png - Horizontal bar chart of filter stages
2. boiling_point_vs_hfo_distance.png - Scatter plot with reference markers
3. parameter_space_comparison.png - 3-panel histograms
4. henrys_soft_preference.png - Henry's constant vs HFO distance
5. structural_motif_analysis.png - Backbone distribution

Usage:
    python scripts/step25_generate_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from screening.hfo_screening import HFO_1336MZZ


def main():
    """Generate all figures for Step 25."""
    # Paths
    results_dir = Path(__file__).parent.parent / "screening" / "results"
    figures_dir = Path(__file__).parent.parent / "figures" / "25_hfo_centric_screening"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ranked_path = results_dir / "hfo_centric_ranked.csv"
    funnel_path = results_dir / "hfo_filter_funnel.csv"

    if not ranked_path.exists():
        print(f"ERROR: {ranked_path} not found. Run step25_hfo_centric_screening.py first.")
        sys.exit(1)

    if not funnel_path.exists():
        print(f"ERROR: {funnel_path} not found. Run step25_hfo_centric_screening.py first.")
        sys.exit(1)

    df_ranked = pd.read_csv(ranked_path)
    df_funnel = pd.read_csv(funnel_path)

    print(f"Loaded {len(df_ranked)} ranked candidates")
    print(f"Loaded {len(df_funnel)} funnel stages")

    # Figure 1: Screening funnel
    print("\n[1/5] Generating screening_funnel.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = df_funnel["stage"].tolist()
    counts = df_funnel["count"].tolist()

    # Horizontal bar chart
    y_pos = np.arange(len(stages))
    colors = ["#4C72B0"] * (len(stages) - 1) + ["#DD8452"]  # Highlight final stage

    ax.barh(y_pos, counts, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=14)
    ax.set_xlabel("Candidate Count", fontsize=16)
    ax.set_title("HFO Screening Funnel (5 Filters)", fontsize=18, weight="bold")
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Annotate with counts
    for i, (count, y) in enumerate(zip(counts, y_pos)):
        ax.text(count + 50, y, f"{count:,}", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(figures_dir / "screening_funnel.png", dpi=300)
    plt.close()
    print(f"Saved: {figures_dir / 'screening_funnel.png'}")

    # Figure 2: Boiling point vs HFO distance
    print("\n[2/5] Generating boiling_point_vs_hfo_distance.png...")
    fig, ax = plt.subplots(figsize=(10, 7))

    x = df_ranked["hfo_distance"]
    y = df_ranked["boiling_point_K"] - 273.15  # Convert to Celsius
    c = df_ranked["n_fluorine"]

    scatter = ax.scatter(x, y, c=c, cmap="viridis", alpha=0.6, s=50, edgecolors="k", linewidth=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Number of Fluorines", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Mark HFO-1336mzz(Z)
    hfo_T_b = HFO_1336MZZ["boiling_point_K"] - 273.15
    ax.scatter([0], [hfo_T_b], marker="*", s=400, color="red", edgecolors="black",
               linewidth=1.5, label="HFO-1336mzz(Z)", zorder=10)

    # Horizontal lines at 15°C and 50°C
    ax.axhline(
        15, color="blue", linestyle="--", linewidth=1.5, alpha=0.7,
        label="15°C (lower bound)"
    )
    ax.axhline(
        50, color="orange", linestyle="--", linewidth=1.5, alpha=0.7,
        label="50°C (upper bound)"
    )

    ax.set_xlabel("HFO Parameter Distance", fontsize=16)
    ax.set_ylabel("Boiling Point (°C)", fontsize=16)
    ax.set_title(
        "Boiling Point vs HFO Distance (Filter-Passing Candidates)",
        fontsize=18, weight="bold"
    )
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "boiling_point_vs_hfo_distance.png", dpi=300)
    plt.close()
    print(f"Saved: {figures_dir / 'boiling_point_vs_hfo_distance.png'}")

    # Figure 3: Parameter space comparison (3-panel histograms)
    print("\n[3/5] Generating parameter_space_comparison.png...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    params = [("m", "Number of Segments (m)"),
              ("sigma", "Segment Diameter σ (Å)"),
              ("epsilon_k", "Dispersion Energy ε/k (K)")]

    for ax, (param, label) in zip(axes, params):
        # All candidates (from funnel — need to reload full dataset)
        # We'll use the ranked_df as proxy for "filter-passing"
        # For "all", we need to re-load or compute from scratch
        # For simplicity, we'll just show filter-passing vs HFO reference

        ax.hist(df_ranked[param], bins=30, alpha=0.6, color="steelblue",
                edgecolor="black", linewidth=0.8, label="Filter-passing")

        # Mark HFO reference
        ref_val = HFO_1336MZZ[param]
        ax.axvline(ref_val, color="red", linestyle="--", linewidth=2,
                   label="HFO-1336mzz(Z)")

        ax.set_xlabel(label, fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(alpha=0.3)

    fig.suptitle("PC-SAFT Parameter Space: Filter-Passing vs HFO Reference",
                 fontsize=18, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "parameter_space_comparison.png", dpi=300)
    plt.close()
    print(f"Saved: {figures_dir / 'parameter_space_comparison.png'}")

    # Figure 4: Henry's constant soft preference
    print("\n[4/5] Generating henrys_soft_preference.png...")

    # Filter to rows with valid H_Pa
    df_h = df_ranked[~df_ranked["H_Pa"].isna()].copy()

    if len(df_h) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))

        x = df_h["hfo_distance"]
        y = df_h["H_ratio"]
        c = df_h["boiling_point_K"] - 273.15

        scatter = ax.scatter(x, y, c=c, cmap="coolwarm", alpha=0.7, s=60,
                             edgecolors="k", linewidth=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Boiling Point (°C)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # Mark top 5 lowest H_ratio
        df_h_sorted = df_h.sort_values("H_ratio")
        top5 = df_h_sorted.head(5)
        ax.scatter(top5["hfo_distance"], top5["H_ratio"], marker="D", s=150,
                   color="gold", edgecolors="black", linewidth=1.5,
                   label="Top 5 lowest H", zorder=10)

        ax.set_xlabel("HFO Parameter Distance", fontsize=16)
        ax.set_ylabel("Henry's Constant Ratio (H / H_HFO)", fontsize=16)
        ax.set_yscale("log")
        ax.set_title("Henry's Constant as Soft Preference Metric", fontsize=18, weight="bold")
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(figures_dir / "henrys_soft_preference.png", dpi=300)
        plt.close()
        print(f"Saved: {figures_dir / 'henrys_soft_preference.png'}")
    else:
        print("No Henry's constant data available; skipping henrys_soft_preference.png")

    # Figure 5: Structural motif analysis
    print("\n[5/5] Generating structural_motif_analysis.png...")
    backbone_counts = df_ranked["backbone"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 7))

    backbones = backbone_counts.index.tolist()[:15]  # Top 15
    counts = backbone_counts.values.tolist()[:15]

    y_pos = np.arange(len(backbones))
    ax.barh(y_pos, counts, color="mediumseagreen", edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(backbones, fontsize=14)
    ax.set_xlabel("Count", fontsize=16)
    ax.set_title("Structural Motif Distribution (Top 15 Backbones)", fontsize=18, weight="bold")
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Annotate with counts
    for count, y in zip(counts, y_pos):
        ax.text(count + 0.5, y, f"{count}", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(figures_dir / "structural_motif_analysis.png", dpi=300)
    plt.close()
    print(f"Saved: {figures_dir / 'structural_motif_analysis.png'}")

    print("\n" + "=" * 80)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"Output directory: {figures_dir}")


if __name__ == "__main__":
    main()
