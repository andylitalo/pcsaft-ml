#!/usr/bin/env python
"""Generate figures for Step 12: Candidate Verification.

Creates three figures:
1. multi_temp_vp_ratio.png: VP ratio vs temperature for candidates passing criteria 1-4
2. criteria_breakdown.png: Funnel chart showing how many candidates fail at each criterion
3. parity_verified.png: Predicted vs reference for verified candidates only
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.thermodynamic import CYCLOPENTANE

TEMPERATURES = [273.15, 298.15, 323.15]
VP_RATIO_BOUNDS = (0.5, 1.5)


def plot_multi_temp_vp_ratio(df: pd.DataFrame, output_dir: Path):
    """VP ratio vs temperature for candidates passing criteria 1-4."""
    # Filter to candidates passing first 4 criteria (may fail SA)
    candidates = df[
        df["criterion_ad"]
        & df["criterion_param"]
        & df["criterion_eos"]
        & df["criterion_vp"]
    ].copy()

    # Separate verified (all 5) from non-verified (fail SA only)
    verified = candidates[candidates["all_criteria"]]
    non_verified = candidates[~candidates["all_criteria"]]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot shaded acceptable region
    ax.axhspan(
        VP_RATIO_BOUNDS[0],
        VP_RATIO_BOUNDS[1],
        alpha=0.15,
        color="green",
        label="Acceptable range [0.5, 1.5]",
    )

    # Plot non-verified candidates (gray, thin lines)
    for _, row in non_verified.iterrows():
        ratios = [row[f"vp_ratio_{int(T)}K"] for T in TEMPERATURES]
        ax.plot(
            TEMPERATURES,
            ratios,
            color="gray",
            alpha=0.3,
            linewidth=1,
            marker="o",
            markersize=3,
        )

    # Plot verified candidates (bold, colored)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for idx, (_, row) in enumerate(verified.iterrows()):
        ratios = [row[f"vp_ratio_{int(T)}K"] for T in TEMPERATURES]
        color = colors[idx % len(colors)]
        ax.plot(
            TEMPERATURES,
            ratios,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=8,
            label=row["smiles"],
        )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Temperature (K)", fontsize=16)
    ax.set_ylabel("VP Ratio to Cyclopentane", fontsize=16)
    ax.set_title(
        "Vapor Pressure Ratio Across Operating Temperatures\n"
        f"({len(candidates)} candidates passing criteria 1-4; "
        f"{len(verified)} verified)",
        fontsize=18,
        pad=20,
    )
    ax.legend(fontsize=11, loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.set_xticks(TEMPERATURES)
    ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))

    plt.tight_layout()
    plt.savefig(output_dir / "multi_temp_vp_ratio.png", dpi=300)
    plt.close()
    print("  ✓ multi_temp_vp_ratio.png")


def plot_criteria_breakdown(df: pd.DataFrame, output_dir: Path):
    """Funnel chart showing candidate attrition at each criterion."""
    # Count candidates passing each criterion
    criterion_names = [
        "All candidates",
        "1. In-domain (AD)",
        "2. Parameter proximity",
        "3. EOS convergence",
        "4. VP ratio [0.5, 1.5]",
        "5. SA score ≤ 4.5",
    ]

    counts = [
        len(df),
        df["criterion_ad"].sum(),
        (df["criterion_ad"] & df["criterion_param"]).sum(),
        (df["criterion_ad"] & df["criterion_param"] & df["criterion_eos"]).sum(),
        (
            df["criterion_ad"]
            & df["criterion_param"]
            & df["criterion_eos"]
            & df["criterion_vp"]
        ).sum(),
        df["all_criteria"].sum(),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create horizontal bar chart (funnel-like)
    y_pos = np.arange(len(criterion_names))
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

    bars = ax.barh(y_pos, counts, color=colors, edgecolor="black", linewidth=1.5)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            count + 5,
            i,
            f"{count}",
            va="center",
            ha="left",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(criterion_names, fontsize=13)
    ax.set_xlabel("Number of Candidates", fontsize=16)
    ax.set_title(
        "Screening Funnel: Candidate Attrition by Criterion",
        fontsize=18,
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="x", labelsize=14)
    ax.invert_yaxis()  # Top to bottom funnel

    plt.tight_layout()
    plt.savefig(output_dir / "criteria_breakdown.png", dpi=300)
    plt.close()
    print("  ✓ criteria_breakdown.png")


def plot_parity_verified(df: pd.DataFrame, output_dir: Path):
    """Parity plots for verified candidates only (m, σ, ε/k vs cyclopentane)."""
    verified = df[df["all_criteria"]].copy()

    if len(verified) == 0:
        print("  ⚠ No verified candidates; skipping parity_verified.png")
        return

    ref = CYCLOPENTANE
    targets = ["m", "sigma", "epsilon_k"]
    labels = {"m": "m (segments)", "sigma": "σ (Å)", "epsilon_k": "ε/k (K)"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, target in zip(axes, targets):
        pred = verified[target].values
        true_val = ref[target]

        # Plot reference line
        lim = [min(pred.min(), true_val) * 0.95, max(pred.max(), true_val) * 1.05]
        ax.plot(lim, lim, "k--", linewidth=2, alpha=0.5, label="Reference")

        # Plot horizontal line at cyclopentane value
        ax.axhline(y=true_val, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.7)
        ax.axvline(x=true_val, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.7)

        # Scatter predicted vs reference
        ax.scatter(
            [true_val] * len(pred),
            pred,
            s=150,
            alpha=0.7,
            edgecolors="black",
            linewidths=2,
            c=range(len(pred)),
            cmap="viridis",
        )

        # Annotate SMILES
        for i, (idx, row) in enumerate(verified.iterrows()):
            ax.annotate(
                row["smiles"],
                (true_val, row[target]),
                xytext=(10, 10 * (-1) ** i),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
            )

        ax.set_xlabel(f"Cyclopentane {labels[target]}", fontsize=14)
        ax.set_ylabel(f"Verified Candidates {labels[target]}", fontsize=14)
        ax.set_title(labels[target], fontsize=16, pad=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    plt.suptitle(
        f"Verified Candidates vs Cyclopentane (n={len(verified)})",
        fontsize=18,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "parity_verified.png", dpi=300)
    plt.close()
    print("  ✓ parity_verified.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate verification figures for Step 12"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("model/saved/verified_candidates.csv"),
        help="Path to verified candidates CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/12_candidate_verification"),
        help="Directory to save figures",
    )

    args = parser.parse_args()

    print(f"Loading verified candidates from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} candidates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures in {args.output_dir}...")

    plot_multi_temp_vp_ratio(df, args.output_dir)
    plot_criteria_breakdown(df, args.output_dir)
    plot_parity_verified(df, args.output_dir)

    print("\nFigure generation complete!")


if __name__ == "__main__":
    main()
