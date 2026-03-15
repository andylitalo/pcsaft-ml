#!/usr/bin/env python3
"""Polyol solvent sensitivity analysis (Step 22.4)."""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from model.thermodynamic import CYCLOPENTANE, batch_henrys_constant

# Paths
DATA_DIR = Path(__file__).parent.parent / "model" / "saved"
INPUT_CSV = DATA_DIR / "henrys_constant_screening.csv"
OUTPUT_CSV = DATA_DIR / "henrys_polyol_sensitivity.csv"


def main():
    print("Loading Henry's constant screening results (hexane solvent)...")
    df_hexane = pd.read_csv(INPUT_CSV)

    # Extract candidate parameters
    candidates_df = df_hexane[["smiles", "m", "sigma", "epsilon_k"]].copy()

    # Try alternative solvents as polyol proxies
    # Real PPG/PEG has m ~ 100-200 and epsilon_k ~ 228 K, but no VLE at 298 K
    # Use n-decane (C10) as a higher-MW hydrocarbon proxy
    # Literature PC-SAFT: m=4.66, sigma=3.84 Å, epsilon/k=243.9 K
    polyol_params_list = [
        {"m": 4.66, "sigma": 3.84, "epsilon_k": 243.9, "name": "n-decane"},
        {"m": 5.0, "sigma": 3.9, "epsilon_k": 250.0, "name": "heavy_alkane"},
    ]

    df_polyol = None
    working_params = None

    for polyol_params in polyol_params_list:
        print(f"\n=== Trying polyol-like solvent: {polyol_params['name']} ===")
        try:
            df_polyol = batch_henrys_constant(
                candidates_df,
                solvent_params=polyol_params,
                T=298.15,
                k_ij=0.0,
                reference=CYCLOPENTANE,
            )

            # Check if we got valid results
            n_valid = df_polyol["H_Pa"].notna().sum()
            print(f"Valid computations: {n_valid}/{len(df_polyol)}")

            if n_valid > 0.9 * len(df_polyol):  # At least 90% success rate
                working_params = polyol_params
                print(f"✓ Success with {polyol_params['name']}")
                break
            else:
                print("✗ Too many failures, trying smaller m...")
        except Exception as e:
            print(f"✗ Failed with error: {e}")

    if df_polyol is None or working_params is None:
        print("\nERROR: Could not compute Henry's constants for any polyol-like solvent")
        return

    print(f"\n=== Using {working_params['name']} for analysis ===")

    # Merge hexane and polyol results
    result_df = df_hexane[["smiles", "m", "sigma", "epsilon_k", "H_ratio"]].copy()
    result_df = result_df.rename(columns={"H_ratio": "H_ratio_hexane"})
    result_df["H_ratio_polyol"] = df_polyol["H_ratio"]
    result_df["polyol_solvent"] = working_params["name"]

    # Compute Spearman correlation
    valid_mask = (
        result_df["H_ratio_hexane"].notna() & result_df["H_ratio_polyol"].notna()
    )
    if valid_mask.sum() > 0:
        rho, p_value = spearmanr(
            result_df.loc[valid_mask, "H_ratio_hexane"],
            result_df.loc[valid_mask, "H_ratio_polyol"],
        )
        print(f"\nSpearman ρ between hexane and polyol H ratios: {rho:.3f} (p={p_value:.3e})")
    else:
        rho = float("nan")
        p_value = float("nan")
        print("\nERROR: No valid pairs for correlation")

    # Save results
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Summary statistics
    print("\n=== H_ratio Comparison ===")
    print(f"Hexane H_ratio: min={result_df['H_ratio_hexane'].min():.3f}, "
          f"median={result_df['H_ratio_hexane'].median():.3f}, "
          f"max={result_df['H_ratio_hexane'].max():.3f}")
    print(f"Polyol H_ratio: min={result_df['H_ratio_polyol'].min():.3f}, "
          f"median={result_df['H_ratio_polyol'].median():.3f}, "
          f"max={result_df['H_ratio_polyol'].max():.3f}")

    # Check for rank reversals
    result_df["hexane_rank"] = result_df["H_ratio_hexane"].rank()
    result_df["polyol_rank"] = result_df["H_ratio_polyol"].rank()
    result_df["rank_delta"] = abs(
        result_df["hexane_rank"] - result_df["polyol_rank"]
    )

    print(f"\nMean absolute rank delta: {result_df['rank_delta'].mean():.1f}")
    print(f"Max absolute rank delta: {result_df['rank_delta'].max():.0f}")

    # Show top candidates by hexane vs polyol
    print("\n=== Top 10 by Hexane H_ratio ===")
    top_hexane = result_df.nsmallest(10, "H_ratio_hexane")
    print(
        top_hexane[
            ["smiles", "H_ratio_hexane", "H_ratio_polyol", "hexane_rank", "polyol_rank"]
        ].to_string(index=False)
    )

    print("\n=== Top 10 by Polyol H_ratio ===")
    top_polyol = result_df.nsmallest(10, "H_ratio_polyol")
    print(
        top_polyol[
            ["smiles", "H_ratio_hexane", "H_ratio_polyol", "hexane_rank", "polyol_rank"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
