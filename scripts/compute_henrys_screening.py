#!/usr/bin/env python3
"""Compute Henry's constants for VP-passing candidates (Step 22.2)."""

from pathlib import Path

import pandas as pd

from model.thermodynamic import CYCLOPENTANE, HEXANE, batch_henrys_constant

# Paths
DATA_DIR = Path(__file__).parent.parent / "model" / "saved"
INPUT_CSV = DATA_DIR / "thermo_validation_systematic.csv"
OUTPUT_CSV = DATA_DIR / "henrys_constant_screening.csv"

def main():
    print("Loading systematic validation data...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total candidates: {len(df)}")

    # Filter for VP-passing candidates (0.5 <= vp_ratio <= 1.5)
    vp_passing = df[
        (df["vp_ratio_to_ref"] >= 0.5) & (df["vp_ratio_to_ref"] <= 1.5)
    ].copy()
    print(f"VP-passing candidates (0.5 <= VP ratio <= 1.5): {len(vp_passing)}")

    # Compute Henry's constants
    print("\nComputing Henry's constants in n-hexane at 298.15 K...")
    print("(This will take ~15-20 minutes for ~950 molecules)")

    result_df = batch_henrys_constant(
        vp_passing,
        solvent_params=HEXANE,
        T=298.15,
        k_ij=0.0,
        reference=CYCLOPENTANE
    )

    # Save results
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total computed: {len(result_df)}")
    print(f"Failed computations (NaN): {result_df['H_Pa'].isna().sum()}")
    print(f"Successful computations: {result_df['H_Pa'].notna().sum()}")

    # H_ratio statistics
    h_ratio_valid = result_df["H_ratio"].dropna()
    if len(h_ratio_valid) > 0:
        print(f"\nH_ratio statistics (n={len(h_ratio_valid)}):")
        print(f"  Min: {h_ratio_valid.min():.3f}")
        print(f"  25%: {h_ratio_valid.quantile(0.25):.3f}")
        print(f"  Median: {h_ratio_valid.median():.3f}")
        print(f"  75%: {h_ratio_valid.quantile(0.75):.3f}")
        print(f"  Max: {h_ratio_valid.max():.3f}")
        print(f"  Mean: {h_ratio_valid.mean():.3f}")

        # Count by H_ratio range
        print("\nH_ratio distribution:")
        print(f"  H/H(ref) < 0.5: {(h_ratio_valid < 0.5).sum()}")
        print(f"  0.5 <= H/H(ref) < 2.0: {((h_ratio_valid >= 0.5) & (h_ratio_valid < 2.0)).sum()}")
        print(f"  2.0 <= H/H(ref) < 10: {((h_ratio_valid >= 2.0) & (h_ratio_valid < 10)).sum()}")
        print(f"  10 <= H/H(ref) < 100: {((h_ratio_valid >= 10) & (h_ratio_valid < 100)).sum()}")
        count_100_1000 = ((h_ratio_valid >= 100) & (h_ratio_valid < 1000)).sum()
        print(f"  100 <= H/H(ref) < 1000: {count_100_1000}")
        print(f"  H/H(ref) >= 1000: {(h_ratio_valid >= 1000).sum()}")

if __name__ == "__main__":
    main()
