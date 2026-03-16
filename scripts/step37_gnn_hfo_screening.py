"""Step 37: GNN HFO-Centric Screening — Full Pipeline Rerun.

Re-run complete HFO screening using the GNN model from Step 31:
1. Generate candidates via systematic enumeration (same as Step 25)
2. Predict PC-SAFT parameters via GNN (model.registry.get_model("gnn"))
3. Compute boiling points via teqp EOS
4. Compute SA scores
5. Apply all 7 HFO-centric filters (including Step 32 fluorination filters)
6. Rank by HFO parameter distance with 5:2:1 weighting (epsilon_k:sigma:m)
7. Compute Henry's constant for top 50 filter-passing candidates
8. Save ranked results to new GNN output paths

Usage:
    python scripts/step37_gnn_hfo_screening.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.registry import get_model
from model.thermodynamic import HEXANE, compute_henrys_constant
from screening.generate import generate_systematic_candidates
from screening.hfo_screening import (
    HFO_1336MZZ,
    apply_hfo_filters,
    batch_boiling_points,
    hfo_parameter_distance,
)


def print_warning_banner():
    """Print OOD warning banner."""
    print("\n" + "=" * 80)
    print("WARNING: OUT-OF-DOMAIN PREDICTION WORKFLOW (GNN-BASED)")
    print("=" * 80)
    print("This workflow ranks OOD hypotheses using GNN-predicted PC-SAFT parameters")
    print("and EOS-derived properties. It is suitable for prioritization and analysis,")
    print("not for final material selection without experimental validation.")
    print()
    print("Model Basis:")
    print("  - GNN model from Step 31 (unified Esper + fluorinated data)")
    print("  - Fluorinated test set: R²(m)=0.86, R²(σ)=0.73, R²(ε/k)=0.91")
    print("  - 0% candidate overlap with training set (pure extrapolation)")
    print("  - Boiling point MAE: 8.2 K (from Step 24 validation)")
    print("  - PC-SAFT: 3-parameter model, no association, no dipoles")
    print()
    print("Parameter Weighting:")
    print("  - Using 5:2:1 weighting (ε/k : σ : m) per Step 9 recommendation")
    print("  - Weighting reflects thermodynamic importance for vapor pressure")
    print()
    print("NOTE: This is a model-promotion rerun, not a final candidate shortlist.")
    print("Further validation (uncertainty, EOS re-ranking, safety) is required.")
    print("=" * 80 + "\n")


def main():
    """Run full GNN HFO-centric screening pipeline."""
    print_warning_banner()

    output_dir = Path(__file__).parent.parent / "screening" / "results"
    output_dir.mkdir(exist_ok=True)

    # Stage 1: Generate candidates
    print("\n[1/8] Generating systematic candidates (max_cl=1, max_mw=200)...")
    start = time.time()
    candidates = generate_systematic_candidates(max_cl=1, max_mw=200.0)
    print(f"Generated {len(candidates)} candidates in {time.time() - start:.1f}s")

    # Convert to DataFrame
    smiles_list = [smi for smi, _ in candidates]
    df = pd.DataFrame({"smiles": smiles_list})

    # Stage 2: Predict PC-SAFT parameters via GNN
    print("\n[2/8] Predicting PC-SAFT parameters via GNN...")
    start = time.time()
    gnn = get_model("gnn")
    gnn.load()
    predictions = gnn.predict(smiles_list)
    df["m"] = predictions["m"]
    df["sigma"] = predictions["sigma"]
    df["epsilon_k"] = predictions["epsilon_k"]
    print(f"Predicted {len(df)} PC-SAFT parameter sets in {time.time() - start:.1f}s")

    # Stage 3: Compute boiling points
    print("\n[3/8] Computing boiling points via teqp vapor pressure root finding...")
    start = time.time()
    df = batch_boiling_points(df)
    n_valid_bp = (~df["boiling_point_K"].isna()).sum()
    print(f"Computed {n_valid_bp}/{len(df)} valid boiling points in {time.time() - start:.1f}s")

    # Stage 4: Compute SA scores
    print("\n[4/8] Computing synthetic accessibility scores...")
    start = time.time()
    sa_scores = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            sa_scores.append(10.0)
        else:
            try:
                sa_scores.append(sascorer.calculateScore(mol))
            except Exception:
                sa_scores.append(10.0)
    df["sa_score"] = sa_scores
    print(f"Computed SA scores in {time.time() - start:.1f}s")

    # Stage 5: Apply HFO-centric filters (all 7 filters)
    print("\n[5/8] Applying HFO-centric screening filters (7 filters)...")
    print("Filters:")
    print("  1. Boiling point [288-323K] (15-50°C)")
    print("  2. Has C=C bond (olefin)")
    print("  3. No chlorine (Cl-free)")
    print("  4. At least 2 fluorine atoms")
    print("  5. SA score <= 4.5")
    print("  6. Fluorine mass fraction >= 65%")
    print("  7. No reactive fluorination sites")
    print()

    filtered_df, filter_stats = apply_hfo_filters(df)

    # Save filter stats (expanded to include all 7 filters)
    filter_funnel = pd.DataFrame([{
        "stage": "Initial candidates",
        "count": len(df),
    }, {
        "stage": "Boiling point [288-323K]",
        "count": filter_stats["boiling_point"],
    }, {
        "stage": "Has C=C bond",
        "count": filter_stats["has_double_bond"],
    }, {
        "stage": "No chlorine",
        "count": filter_stats["no_chlorine"],
    }, {
        "stage": "F count >= 2",
        "count": filter_stats["fluorine_count"],
    }, {
        "stage": "SA score <= 4.5",
        "count": filter_stats["sa_score"],
    }, {
        "stage": "Fluorine mass fraction >= 65%",
        "count": filter_stats["fluorine_mass_fraction"],
    }, {
        "stage": "No reactive fluorination sites",
        "count": filter_stats["no_reactive_sites"],
    }, {
        "stage": "Final (all filters)",
        "count": filter_stats["total"],
    }])

    filter_funnel_path = output_dir / "hfo_gnn_filter_funnel.csv"
    filter_funnel.to_csv(filter_funnel_path, index=False)
    print(f"\nFilter funnel saved to: {filter_funnel_path}")

    # Stage 6: Rank by HFO parameter distance (5:2:1 weighting)
    print(f"\n[6/8] Ranking {len(filtered_df)} candidates by HFO-1336mzz(Z) parameter distance...")
    print("Using 5:2:1 weighting (ε/k : σ : m) per Step 9 recommendation")
    start = time.time()

    distances = []
    for _, row in filtered_df.iterrows():
        dist = hfo_parameter_distance(
            row["m"], row["sigma"], row["epsilon_k"],
            reference=HFO_1336MZZ,
            weights={"epsilon_k": 5.0, "sigma": 2.0, "m": 1.0},
        )
        distances.append(dist)

    filtered_df = filtered_df.copy()
    filtered_df["hfo_distance"] = distances
    filtered_df = filtered_df.sort_values("hfo_distance").reset_index(drop=True)
    print(f"Ranked candidates in {time.time() - start:.1f}s")

    # Add fluorine count for analysis
    def count_fluorines(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F")

    filtered_df["n_fluorine"] = filtered_df["smiles"].apply(count_fluorines)

    # Add backbone annotation for structural analysis
    def get_backbone(smiles):
        """Classify backbone type."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "unknown"

        # Count carbons
        n_c = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")

        # Check if cyclic
        if mol.GetRingInfo().NumRings() > 0:
            return f"cyclic_C{n_c}"
        else:
            return f"acyclic_C{n_c}"

    filtered_df["backbone"] = filtered_df["smiles"].apply(get_backbone)

    # Stage 7: Compute Henry's constant for top 50
    print("\n[7/8] Computing Henry's constant for top 50 filter-passing candidates...")
    if len(filtered_df) > 0:
        top50 = filtered_df.head(50).copy()
        start = time.time()

        H_values = []
        H_ratios = []

        # Compute reference Henry's constant
        ref_result = compute_henrys_constant(HFO_1336MZZ, HEXANE, T=298.15)
        if ref_result is None:
            print("WARNING: Failed to compute reference Henry's constant")
            H_ref = np.nan
        else:
            H_ref = ref_result["H_Pa"]
            print(f"Reference H (HFO-1336mzz(Z) in hexane): {H_ref:.2e} Pa")

        for idx, row in top50.iterrows():
            solute_params = {
                "m": row["m"],
                "sigma": row["sigma"],
                "epsilon_k": row["epsilon_k"],
            }
            result = compute_henrys_constant(solute_params, HEXANE, T=298.15)

            if result is None:
                H_values.append(np.nan)
                H_ratios.append(np.nan)
            else:
                H_values.append(result["H_Pa"])
                H_ratios.append(result["H_Pa"] / H_ref if not np.isnan(H_ref) else np.nan)

        top50["H_Pa"] = H_values
        top50["H_ratio"] = H_ratios

        # Merge back into filtered_df
        filtered_df = filtered_df.merge(
            top50[["smiles", "H_Pa", "H_ratio"]],
            on="smiles",
            how="left",
        )

        print(f"Computed Henry's constants in {time.time() - start:.1f}s")
    else:
        print("No candidates passed filters; skipping Henry's constant computation")
        filtered_df["H_Pa"] = np.nan
        filtered_df["H_ratio"] = np.nan

    # Stage 8: Save results
    print("\n[8/8] Saving results...")
    output_path = output_dir / "hfo_gnn_ranked.csv"
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved {len(filtered_df)} ranked candidates to: {output_path}")

    # Print top 20
    print("\n" + "=" * 80)
    print("TOP 20 GNN HFO-CENTRIC CANDIDATES")
    print("=" * 80)
    print()

    if len(filtered_df) > 0:
        top20 = filtered_df.head(20)
        for rank, row in enumerate(top20.itertuples(), start=1):
            T_c = row.boiling_point_K - 273.15
            print(f"Rank {rank:2d}: {row.smiles}")
            print(f"  PC-SAFT: m={row.m:.3f}, σ={row.sigma:.3f} Å, ε/k={row.epsilon_k:.1f} K")
            print(
                f"  T_b={T_c:.1f}°C, SA={row.sa_score:.2f}, F={row.n_fluorine}, "
                f"HFO dist={row.hfo_distance:.4f}"
            )
            if not np.isnan(row.H_Pa):
                print(f"  Henry's constant: {row.H_Pa:.2e} Pa (ratio to HFO: {row.H_ratio:.3f})")
            print()
    else:
        print("No candidates passed all filters.")
        print()

    print("=" * 80)
    print("GNN SCREENING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
