#!/usr/bin/env python3
"""Compare VP-only vs Henry-screened candidate sets (Step 22.3)."""

from pathlib import Path

import pandas as pd

# Paths
DATA_DIR = Path(__file__).parent.parent / "model" / "saved"
INPUT_CSV = DATA_DIR / "henrys_constant_screening.csv"
OUTPUT_CSV = DATA_DIR / "vp_vs_henrys_comparison.csv"


def main():
    print("Loading Henry's constant screening results...")
    df = pd.read_csv(INPUT_CSV)

    print(f"Total VP-passing candidates: {len(df)}")

    # VP-only shortlist: all candidates with VP ratio [0.5, 1.5]
    vp_only = df.copy()

    # Henry-screened: subset with H/H(ref) in [0.5, 2.0]
    henry_screened = df[(df["H_ratio"] >= 0.5) & (df["H_ratio"] <= 2.0)].copy()

    print(f"Henry-screened candidates (0.5 <= H/H(ref) <= 2.0): {len(henry_screened)}")

    # Compute how many VP-only candidates have H ratios > 10, > 100, > 1000
    n_gt_10 = (df["H_ratio"] > 10).sum()
    n_gt_100 = (df["H_ratio"] > 100).sum()
    n_gt_1000 = (df["H_ratio"] > 1000).sum()

    print("\n=== VP-only Counterfactual ===")
    print(f"VP-only candidates with H/H(ref) > 10: {n_gt_10}")
    print(f"VP-only candidates with H/H(ref) > 100: {n_gt_100}")
    print(f"VP-only candidates with H/H(ref) > 1000: {n_gt_1000}")

    # Percentage rejected by Henry's screening
    n_rejected = len(vp_only) - len(henry_screened)
    pct_rejected = 100 * n_rejected / len(vp_only) if len(vp_only) > 0 else 0

    print(f"\nRejected by Henry's screening: {n_rejected} ({pct_rejected:.1f}%)")

    # Create comparison table
    comparison = pd.DataFrame(
        {
            "screening_method": ["VP-only", "Henry-screened"],
            "n_candidates": [len(vp_only), len(henry_screened)],
            "H_ratio_min": [vp_only["H_ratio"].min(), henry_screened["H_ratio"].min()],
            "H_ratio_median": [
                vp_only["H_ratio"].median(),
                henry_screened["H_ratio"].median(),
            ],
            "H_ratio_max": [vp_only["H_ratio"].max(), henry_screened["H_ratio"].max()],
            "n_H_gt_10": [n_gt_10, 0],
            "n_H_gt_100": [n_gt_100, 0],
            "n_H_gt_1000": [n_gt_1000, 0],
        }
    )

    comparison.to_csv(OUTPUT_CSV, index=False)
    print(f"\nComparison table saved to {OUTPUT_CSV}")
    print("\n=== Comparison Table ===")
    print(comparison.to_string(index=False))

    # Distribution analysis
    print("\n=== H_ratio Distribution (VP-only) ===")
    h_ratio_bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0, 10, 100, 1000, float("inf")]
    h_ratio_labels = [
        "< 0.5",
        "0.5-1.0",
        "1.0-1.5",
        "1.5-2.0",
        "2.0-5.0",
        "5.0-10",
        "10-100",
        "100-1000",
        ">= 1000",
    ]
    h_ratio_dist = pd.cut(
        vp_only["H_ratio"], bins=h_ratio_bins, labels=h_ratio_labels
    ).value_counts(sort=False)
    print(h_ratio_dist)

    # Check for specific molecules
    print("\n=== Sample High-H Molecules (if any) ===")
    high_h = vp_only[vp_only["H_ratio"] > 2.0].copy()
    if len(high_h) > 0:
        high_h_sorted = high_h.sort_values("H_ratio", ascending=False).head(10)
        print(
            high_h_sorted[
                ["smiles", "m", "sigma", "epsilon_k", "vp_ratio_to_ref", "H_ratio"]
            ].to_string(index=False)
        )
    else:
        print("No molecules with H/H(ref) > 2.0 found in VP-passing set.")


if __name__ == "__main__":
    main()
