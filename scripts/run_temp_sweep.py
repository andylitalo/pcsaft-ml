"""Temperature sweep validation for top-50 candidates.

Computes vapor pressure and liquid density at T = [230, 250, 270, 290, 310, 330] K
for the top-50 candidates from Step 09 thermodynamic validation.

Outputs:
    - model/saved/temp_sweep_results.csv: Full results with VP/density at all T
    - figures/19_temperature_sweep/vp_vs_T.png: Clausius-Clapeyron plot for top-5
    - figures/19_temperature_sweep/rank_stability.png: Heatmap of rank correlations
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model.thermodynamic import CYCLOPENTANE, compute_properties  # noqa: E402

# Temperature sweep range: 230-330 K in 20 K steps
TEMPERATURES = [230, 250, 270, 290, 310, 330]
T_REF = 298.15  # Reference temperature from Step 09


def load_top50_candidates():
    """Load top-50 candidates by parameter-space distance from thermo_validation.csv."""
    thermo_path = project_root / "model" / "saved" / "thermo_validation.csv"
    df = pd.read_csv(thermo_path)

    # Sort by distance (parameter-space distance) and take top 50
    df_sorted = df.sort_values("distance").head(50).copy()
    df_sorted.reset_index(drop=True, inplace=True)

    dist_min = df_sorted['distance'].min()
    dist_max = df_sorted['distance'].max()
    print(f"Loaded top-50 candidates (distance range: {dist_min:.4f} to {dist_max:.4f})")
    return df_sorted


def compute_temp_sweep(candidates_df):
    """Compute VP and density at all temperatures for each candidate.

    Args:
        candidates_df: DataFrame with m, sigma, epsilon_k columns

    Returns:
        DataFrame with columns:
            - Original columns (smiles, m, sigma, epsilon_k, distance, etc.)
            - For each T: vp_{T}K, rho_{T}K, prop_dist_{T}K
            - rank_298K, rank_230K, rank_250K, ..., rank_330K
    """
    # First, compute cyclopentane reference at all temperatures
    ref_props = {}
    for T in TEMPERATURES:
        ref_props[T] = compute_properties(
            CYCLOPENTANE["m"], CYCLOPENTANE["sigma"], CYCLOPENTANE["epsilon_k"], T
        )

    # Compute properties for each candidate at each temperature
    results = []
    for idx, row in candidates_df.iterrows():
        result = {
            "smiles": row["smiles"],
            "m": row["m"],
            "sigma": row["sigma"],
            "epsilon_k": row["epsilon_k"],
            "distance": row["distance"],
        }

        for T in TEMPERATURES:
            props = compute_properties(row["m"], row["sigma"], row["epsilon_k"], T)
            result[f"vp_{T}K"] = props["vapor_pressure_Pa"]
            result[f"rho_{T}K"] = props["liquid_density_mol_m3"]

            # Compute property-space distance at this temperature
            ref_vp = ref_props[T]["vapor_pressure_Pa"]
            ref_rho = ref_props[T]["liquid_density_mol_m3"]

            if np.isnan(props["vapor_pressure_Pa"]) or np.isnan(props["liquid_density_mol_m3"]):
                result[f"prop_dist_{T}K"] = np.nan
            else:
                # Normalized Euclidean distance in (VP, density) space
                result[f"prop_dist_{T}K"] = np.sqrt(
                    ((props["vapor_pressure_Pa"] - ref_vp) / ref_vp) ** 2
                    + ((props["liquid_density_mol_m3"] - ref_rho) / ref_rho) ** 2
                )

        results.append(result)

    df_results = pd.DataFrame(results)

    # Add rankings by property distance at each temperature
    for T in TEMPERATURES:
        col = f"prop_dist_{T}K"
        rank_col = f"rank_{T}K"
        # rank(ascending=True) so rank 1 = smallest distance = best
        df_results[rank_col] = df_results[col].rank(method="min", ascending=True)

    return df_results, ref_props


def analyze_rank_stability(df_results):
    """Compute Spearman correlation between rankings at different temperatures.

    Strategy: Use pairwise-complete correlation (drop NaN per pair) to handle
    candidates that fail at extreme temperatures but succeed at moderate temps.

    Args:
        df_results: DataFrame with rank_{T}K columns

    Returns:
        DataFrame: Pairwise Spearman correlations between temperature rankings
    """
    rank_cols = [f"rank_{T}K" for T in TEMPERATURES]
    rank_data = df_results[rank_cols].copy()

    # Check validity per temperature
    print("\nCandidates with valid EOS at each temperature:")
    for T in TEMPERATURES:
        n_valid = rank_data[f"rank_{T}K"].notna().sum()
        print(f"  {T}K: {n_valid}/{len(df_results)}")

    # Compute pairwise Spearman correlations (handles missing data)
    corr_matrix = pd.DataFrame(index=rank_cols, columns=rank_cols, dtype=float)

    for col1 in rank_cols:
        for col2 in rank_cols:
            # For each pair, use only rows where both are non-NaN
            valid_mask = rank_data[[col1, col2]].notna().all(axis=1)
            if valid_mask.sum() >= 3:  # Need at least 3 points for correlation
                rho, _ = spearmanr(rank_data.loc[valid_mask, col1],
                                   rank_data.loc[valid_mask, col2])
                corr_matrix.loc[col1, col2] = rho
            else:
                corr_matrix.loc[col1, col2] = np.nan

    # Pretty print correlations vs 290K (reference)
    print("\nSpearman ρ between rankings at different temperatures:")
    print("(Reference: 290K is closest to 298.15K, pairwise-complete correlation)")
    for T in TEMPERATURES:
        if T == 290:
            continue  # Skip self-comparison
        rho = corr_matrix.loc["rank_290K", f"rank_{T}K"]
        if pd.notna(rho):
            print(f"  290K vs {T}K: ρ = {rho:.3f}")
        else:
            print(f"  290K vs {T}K: ρ = N/A (insufficient data)")

    return corr_matrix


def plot_vp_vs_temperature(df_results, ref_props, output_path):
    """Plot VP vs T (Clausius-Clapeyron) for top-5 candidates + cyclopentane.

    Args:
        df_results: DataFrame with vp_{T}K columns
        ref_props: Dict of reference properties at each temperature
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot cyclopentane reference
    ref_vp = [ref_props[T]["vapor_pressure_Pa"] / 1e3 for T in TEMPERATURES]  # kPa
    ax.plot(
        TEMPERATURES, ref_vp, "k-", linewidth=2.5,
        label="Cyclopentane (reference)", marker="o"
    )

    # Plot top-5 candidates
    colors = plt.cm.tab10(range(5))
    for idx in range(5):
        row = df_results.iloc[idx]
        vp_values = [row[f"vp_{T}K"] / 1e3 for T in TEMPERATURES]  # kPa

        # Only plot if at least 4 temperatures have valid data
        valid_count = sum(1 for vp in vp_values if not np.isnan(vp))
        if valid_count >= 4:
            label = f"Rank {idx+1}: {row['smiles']}"
            ax.plot(TEMPERATURES, vp_values, "-", color=colors[idx],
                   linewidth=2, alpha=0.8, label=label, marker="s")

    ax.set_xlabel("Temperature (K)", fontsize=14)
    ax.set_ylabel("Vapor Pressure (kPa)", fontsize=14)
    ax.set_title("Vapor Pressure vs Temperature\nTop-5 Candidates vs Cyclopentane", fontsize=15)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved VP vs T plot to {output_path}")


def plot_rank_stability_heatmap(corr_matrix, output_path):
    """Plot heatmap of rank correlation matrix.

    Args:
        corr_matrix: DataFrame of pairwise Spearman correlations
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Rename columns/rows for cleaner labels
    labels = [f"{T}K" for T in TEMPERATURES]
    corr_matrix.index = labels
    corr_matrix.columns = labels

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        center=0.8,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax
    )

    ax.set_title("Rank Stability Across Temperature Range\n(Spearman Correlation)", fontsize=15)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved rank stability heatmap to {output_path}")


def identify_rank_shifters(df_results, threshold=10):
    """Identify candidates whose rank changes by >threshold positions.

    Uses temperatures where data is available (requires at least 3 valid temps).

    Args:
        df_results: DataFrame with rank_{T}K columns
        threshold: Rank change threshold (default 10)

    Returns:
        DataFrame of candidates with large rank changes
    """
    rank_cols = [f"rank_{T}K" for T in TEMPERATURES]

    # For each candidate, compute max rank change among valid temperatures
    results = []
    for idx, row in df_results.iterrows():
        valid_ranks = {col: row[col] for col in rank_cols if pd.notna(row[col])}

        if len(valid_ranks) >= 3:  # Need at least 3 temps to assess stability
            rank_values = list(valid_ranks.values())
            max_change = max(rank_values) - min(rank_values)
            ref_rank = valid_ranks.get("rank_290K", np.nan)

            results.append({
                "smiles": row["smiles"],
                "max_rank_change": max_change,
                "rank_290K": ref_rank,
                "n_valid_temps": len(valid_ranks)
            })

    if not results:
        print("\n0 candidates with sufficient data (need ≥3 valid temperatures)")
        return pd.DataFrame()

    df_shifters = pd.DataFrame(results)
    shifters = df_shifters[df_shifters["max_rank_change"] > threshold].copy()
    shifters_sorted = shifters.sort_values("max_rank_change", ascending=False)

    print(
        f"\n{len(shifters)} candidates with rank change > {threshold} "
        f"positions (≥3 valid temps):"
    )
    for _, row in shifters_sorted.head(10).iterrows():
        print(f"  {row['smiles']:20s} - max Δrank = {row['max_rank_change']:.0f} "
              f"(290K rank: {row['rank_290K']:.0f}, {row['n_valid_temps']} valid temps)")

    return shifters_sorted


def main():
    """Run temperature sweep validation."""
    print("=" * 70)
    print("Step 19: Temperature Sweep Validation")
    print("=" * 70)

    # Load top-50 candidates
    df_candidates = load_top50_candidates()

    # Compute temperature sweep
    print(
        f"\nComputing VP and density at {len(TEMPERATURES)} temperatures "
        f"for {len(df_candidates)} candidates..."
    )
    df_results, ref_props = compute_temp_sweep(df_candidates)

    # Save results
    output_path = project_root / "model" / "saved" / "temp_sweep_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    # Analyze rank stability
    corr_matrix = analyze_rank_stability(df_results)

    # Identify rank shifters
    shifters = identify_rank_shifters(df_results, threshold=10)

    # Generate figures
    figures_dir = project_root / "figures" / "19_temperature_sweep"

    plot_vp_vs_temperature(
        df_results,
        ref_props,
        figures_dir / "vp_vs_T.png"
    )

    plot_rank_stability_heatmap(
        corr_matrix,
        figures_dir / "rank_stability.png"
    )

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total candidates analyzed: {len(df_results)}")
    rank_cols = [f'rank_{T}K' for T in TEMPERATURES]
    n_valid_all = len(df_results.dropna(subset=rank_cols))
    print(f"Candidates valid across all temperatures: {n_valid_all}")
    print(f"Candidates with rank change > 10 positions: {len(shifters)}")

    if n_valid_all > 0:
        # corr_matrix has been relabeled to just "290K", etc.
        print(f"\nMean Spearman ρ vs 290K: {corr_matrix.loc['290K'].drop('290K').mean():.3f}")
        print(f"Min Spearman ρ vs 290K: {corr_matrix.loc['290K'].drop('290K').min():.3f}")
    else:
        print(
            "\nWARNING: No candidates valid across all temperatures "
            "- cannot compute rank stability."
        )

    print("\nOutputs:")
    print(f"  - {output_path}")
    print(f"  - {figures_dir / 'vp_vs_T.png'}")
    print(f"  - {figures_dir / 'rank_stability.png'}")


if __name__ == "__main__":
    main()
