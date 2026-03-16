"""Step 38b: RF HFO Screening Re-Run.

Step 38 found that the GNN model catastrophically fails on fluorinated
compounds (boiling point MAE = 133.7 K vs RF's 8.2 K). This step re-runs
the full HFO screening pipeline using the RF model instead of GNN.

Pipeline stages:
1. Generate candidates via systematic enumeration (same as Step 25/37)
2. Predict PC-SAFT parameters via RF (model.registry.get_model("rf"))
3. Compute boiling points via teqp EOS
4. Compute SA scores
5. Apply all 7 HFO-centric filters (including Step 32 fluorination filters)
6. Rank by HFO parameter distance with 5:2:1 weighting (epsilon_k:sigma:m)
7. Add RF uncertainty estimates (tree-level std)
8. Save ranked results to new RF-specific output paths

Usage:
    python scripts/step38b_rf_hfo_screening.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Contrib.SA_Score import sascorer  # noqa: E402

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.registry import get_model  # noqa: E402
from model.thermodynamic import HEXANE, compute_henrys_constant  # noqa: E402
from screening.filters import count_cf3_groups  # noqa: E402
from screening.generate import generate_systematic_candidates  # noqa: E402
from screening.hfo_screening import (  # noqa: E402
    HFO_1336MZZ,
    apply_hfo_filters,
    batch_boiling_points,
    compute_boiling_point,
    hfo_parameter_distance,
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "screening" / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "38b_rf_hfo_screening"
VALIDATION_SET_PATH = PROJECT_ROOT / "model" / "saved" / "gnn_fluorinated_validation_set.csv"


def print_warning_banner():
    """Print OOD warning banner for RF-based screening."""
    print("\n" + "=" * 80)
    print("WARNING: OUT-OF-DOMAIN PREDICTION WORKFLOW (RF-BASED)")
    print("=" * 80)
    print("This workflow ranks OOD hypotheses using RF-predicted PC-SAFT parameters")
    print("and EOS-derived properties. It is suitable for prioritization and analysis,")
    print("not for final material selection without experimental validation.")
    print()
    print("Model Basis:")
    print("  - Random Forest model trained on Esper dataset (1,801 molecules)")
    print("  - Esper test set: R2(m)=0.61, R2(sigma)=0.32, R2(eps/k)=0.27")
    print("  - Fluorinated validation MAE: 8.2 K boiling point (Step 38)")
    print("  - RF is demonstrably more reliable for fluorinated compounds than GNN")
    print("  - 0% candidate overlap with training set (pure extrapolation)")
    print("  - PC-SAFT: 3-parameter model, no association, no dipoles")
    print()
    print("Parameter Weighting:")
    print("  - Using 5:2:1 weighting (eps/k : sigma : m) per Step 9 recommendation")
    print("  - Weighting reflects thermodynamic importance for vapor pressure")
    print()
    print("Context:")
    print("  - Step 38 found GNN boiling point MAE = 133.7 K on fluorinated compounds")
    print("  - RF boiling point MAE = 8.2 K on the same set")
    print("  - This step re-runs the Step 37 screening with RF as the prediction model")
    print()
    print("NOTE: This is a model-correction rerun, not a final candidate shortlist.")
    print("Further validation (uncertainty, EOS re-ranking, safety) is required.")
    print("=" * 80 + "\n")


def run_validation_sanity_check():
    """Run RF through the 15-compound fluorinated validation set from Step 38.

    Returns:
        tuple: (validation_df, bp_mae)
    """
    print("\n" + "=" * 80)
    print("RF VALIDATION SANITY CHECK (15-compound fluorinated set)")
    print("=" * 80)

    # Load validation set
    val_df = pd.read_csv(VALIDATION_SET_PATH)
    smiles_list = val_df["smiles"].tolist()
    print(f"Loaded {len(smiles_list)} molecules from validation set")

    # Predict with RF
    rf = get_model("rf")
    rf.load()
    rf_preds = rf.predict_with_uncertainty(smiles_list)

    val_df["m_rf"] = rf_preds["m"]
    val_df["sigma_rf"] = rf_preds["sigma"]
    val_df["epsilon_k_rf"] = rf_preds["epsilon_k"]
    val_df["m_std"] = rf_preds["m_std"]
    val_df["sigma_std"] = rf_preds["sigma_std"]
    val_df["epsilon_k_std"] = rf_preds["epsilon_k_std"]

    # Compute boiling points from RF predictions
    bp_values = []
    for _, row in val_df.iterrows():
        T_b = compute_boiling_point(row["m_rf"], row["sigma_rf"], row["epsilon_k_rf"])
        bp_values.append(T_b)
    val_df["T_b_predicted_K"] = bp_values

    # Compute errors
    valid = ~val_df["T_b_predicted_K"].isna()
    val_df["T_b_error_K"] = val_df["T_b_predicted_K"] - val_df["T_b_experimental_K"]

    if valid.sum() > 0:
        errors = val_df.loc[valid, "T_b_error_K"].values
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        median_ae = np.median(np.abs(errors))
        max_ae = np.abs(errors).max()
        mean_err = errors.mean()

        print("\nRF Boiling Point Validation:")
        print(f"  Converged:    {valid.sum()}/{len(val_df)}")
        print(f"  MAE:          {mae:.2f} K")
        print(f"  RMSE:         {rmse:.2f} K")
        print(f"  Median AE:    {median_ae:.2f} K")
        print(f"  Max AE:       {max_ae:.2f} K")
        print(f"  Mean error:   {mean_err:.2f} K (bias)")

        if mae < 15.0:
            print(f"\n  RF boiling point MAE = {mae:.1f} K: PASS (< 15 K threshold)")
        else:
            print(f"\n  RF boiling point MAE = {mae:.1f} K: WARNING (>= 15 K threshold)")
    else:
        mae = np.nan
        print("\n  WARNING: No valid boiling point predictions converged.")

    # Per-molecule detail
    print("\nPer-molecule results:")
    print(f"  {'SMILES':<35s} {'T_b_exp':>8s} {'T_b_pred':>9s} {'Error':>7s}")
    print("  " + "-" * 65)
    for _, row in val_df.iterrows():
        if not np.isnan(row["T_b_predicted_K"]):
            pred_str = f"{row['T_b_predicted_K']:.1f}"
        else:
            pred_str = "FAIL"
        if not np.isnan(row["T_b_error_K"]):
            err_str = f"{row['T_b_error_K']:.1f}"
        else:
            err_str = "N/A"
        smi = row["smiles"]
        t_exp = row["T_b_experimental_K"]
        print(f"  {smi:<35s} {t_exp:8.1f} {pred_str:>9s} {err_str:>7s}")

    return val_df, mae


def run_screening_pipeline():
    """Run the full RF HFO screening pipeline.

    Returns:
        tuple: (filtered_df, filter_funnel, df_full)
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Stage 1: Generate candidates
    print("\n[1/8] Generating systematic candidates (max_cl=1, max_mw=200)...")
    start = time.time()
    candidates = generate_systematic_candidates(max_cl=1, max_mw=200.0)
    print(f"Generated {len(candidates)} candidates in {time.time() - start:.1f}s")

    smiles_list = [smi for smi, _ in candidates]
    df = pd.DataFrame({"smiles": smiles_list})

    # Stage 2: Predict PC-SAFT parameters via RF with uncertainty
    print("\n[2/8] Predicting PC-SAFT parameters via RF (with uncertainty)...")
    start = time.time()
    rf = get_model("rf")
    rf.load()
    predictions = rf.predict_with_uncertainty(smiles_list)
    df["m"] = predictions["m"]
    df["sigma"] = predictions["sigma"]
    df["epsilon_k"] = predictions["epsilon_k"]
    df["m_std"] = predictions["m_std"]
    df["sigma_std"] = predictions["sigma_std"]
    df["epsilon_k_std"] = predictions["epsilon_k_std"]
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
    print("  1. Boiling point [288-323K] (15-50 C)")
    print("  2. Has C=C bond (olefin)")
    print("  3. No chlorine (Cl-free)")
    print("  4. At least 2 fluorine atoms")
    print("  5. SA score <= 4.5")
    print("  6. Fluorine mass fraction >= 65%")
    print("  7. No reactive fluorination sites")
    print()

    filtered_df, filter_stats = apply_hfo_filters(df)

    # Build filter funnel DataFrame
    filter_funnel = pd.DataFrame([
        {"stage": "Initial candidates", "count": len(df)},
        {"stage": "Boiling point [288-323K]", "count": filter_stats["boiling_point"]},
        {"stage": "Has C=C bond", "count": filter_stats["has_double_bond"]},
        {"stage": "No chlorine", "count": filter_stats["no_chlorine"]},
        {"stage": "F count >= 2", "count": filter_stats["fluorine_count"]},
        {"stage": "SA score <= 4.5", "count": filter_stats["sa_score"]},
        {
            "stage": "Fluorine mass fraction >= 65%",
            "count": filter_stats["fluorine_mass_fraction"],
        },
        {"stage": "No reactive fluorination sites", "count": filter_stats["no_reactive_sites"]},
        {"stage": "Final (all filters)", "count": filter_stats["total"]},
    ])

    funnel_path = OUTPUT_DIR / "hfo_rf_filter_funnel.csv"
    filter_funnel.to_csv(funnel_path, index=False)
    print(f"\nFilter funnel saved to: {funnel_path}")

    # Stage 6: Rank by HFO parameter distance (5:2:1 weighting)
    print(f"\n[6/8] Ranking {len(filtered_df)} candidates by HFO-1336mzz(Z) parameter distance...")
    print("Using 5:2:1 weighting (eps/k : sigma : m) per Step 9 recommendation")
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

    # Add fluorine count
    def count_fluorines(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F")

    filtered_df["n_fluorine"] = filtered_df["smiles"].apply(count_fluorines)

    # Add backbone annotation
    def get_backbone(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "unknown"
        n_c = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
        if mol.GetRingInfo().NumRings() > 0:
            return f"cyclic_C{n_c}"
        else:
            return f"acyclic_C{n_c}"

    filtered_df["backbone"] = filtered_df["smiles"].apply(get_backbone)

    # Add model_type column
    filtered_df["model_type"] = "rf"

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

        for _, row in top50.iterrows():
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
                H_ratios.append(
                    result["H_Pa"] / H_ref if not np.isnan(H_ref) else np.nan
                )

        top50["H_Pa"] = H_values
        top50["H_ratio"] = H_ratios

        # Merge back
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
    output_path = OUTPUT_DIR / "hfo_rf_ranked.csv"
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved {len(filtered_df)} ranked candidates to: {output_path}")

    # Print top 20
    print("\n" + "=" * 80)
    print("TOP 20 RF HFO-CENTRIC CANDIDATES")
    print("=" * 80)
    print()

    if len(filtered_df) > 0:
        top20 = filtered_df.head(20)
        for rank, row in enumerate(top20.itertuples(), start=1):
            T_c = row.boiling_point_K - 273.15
            print(f"Rank {rank:2d}: {row.smiles}")
            print(
                f"  PC-SAFT: m={row.m:.3f}, sigma={row.sigma:.3f} A, "
                f"eps/k={row.epsilon_k:.1f} K"
            )
            print(
                f"  Uncertainty: m_std={row.m_std:.3f}, sigma_std={row.sigma_std:.3f}, "
                f"eps_k_std={row.epsilon_k_std:.1f}"
            )
            print(
                f"  T_b={T_c:.1f} C, SA={row.sa_score:.2f}, F={row.n_fluorine}, "
                f"HFO dist={row.hfo_distance:.4f}"
            )
            if hasattr(row, "H_Pa") and not np.isnan(row.H_Pa):
                print(f"  Henry's constant: {row.H_Pa:.2e} Pa (ratio to HFO: {row.H_ratio:.3f})")
            print()
    else:
        print("No candidates passed all filters.")
        print()

    return filtered_df, filter_funnel, df


def generate_figures(filtered_df, filter_funnel, df_full, validation_df):
    """Generate all figures for Step 38b.

    Args:
        filtered_df: RF ranked candidates (filter-passing)
        filter_funnel: RF filter funnel DataFrame
        df_full: Full candidate set (pre-filter) with RF predictions
        validation_df: Validation sanity check results
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: RF screening funnel
    print("\n[Fig 1/4] Generating RF screening funnel...")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(filter_funnel))
    ax.barh(y_pos, filter_funnel["count"], color="darkorange", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(filter_funnel["stage"], fontsize=10)
    ax.set_xlabel("Candidate Count", fontsize=12)
    ax.set_title(
        "RF HFO Screening Filter Funnel (7 Filters, Step 38b)",
        fontsize=14, weight="bold",
    )
    ax.invert_yaxis()
    for i, count in enumerate(filter_funnel["count"]):
        ax.text(count + 20, i, f"{count:,}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "screening_funnel_rf.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'screening_funnel_rf.png'}")

    # Figure 2: Boiling point vs HFO distance (RF)
    print("\n[Fig 2/4] Generating boiling point vs HFO distance scatter...")
    if len(filtered_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        cf3_counts = filtered_df["smiles"].apply(count_cf3_groups)
        scatter = ax.scatter(
            filtered_df["hfo_distance"],
            filtered_df["boiling_point_K"] - 273.15,
            c=cf3_counts,
            cmap="viridis",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            s=50,
        )
        ax.set_xlabel("HFO Distance (5:2:1 weighting)", fontsize=12)
        ax.set_ylabel("Boiling Point (C)", fontsize=12)
        ax.set_title(
            "RF Candidates: Boiling Point vs HFO Distance (Step 38b)",
            fontsize=14, weight="bold",
        )
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("CF3 Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "boiling_point_vs_hfo_distance_rf.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'boiling_point_vs_hfo_distance_rf.png'}")
    else:
        print("  Skipped: no filter-passing candidates")

    # Figure 3: RF vs GNN funnel comparison
    print("\n[Fig 3/4] Generating RF vs GNN funnel comparison...")
    gnn_funnel_path = OUTPUT_DIR / "hfo_gnn_filter_funnel.csv"
    if gnn_funnel_path.exists():
        gnn_funnel = pd.read_csv(gnn_funnel_path)
        fig, ax = plt.subplots(figsize=(12, 7))
        y_pos = np.arange(len(filter_funnel))
        bar_height = 0.35

        bars_rf = ax.barh(
            y_pos - bar_height / 2, filter_funnel["count"],
            bar_height, label="RF (Step 38b)", color="darkorange", alpha=0.7,
        )
        bars_gnn = ax.barh(
            y_pos + bar_height / 2, gnn_funnel["count"],
            bar_height, label="GNN (Step 37)", color="steelblue", alpha=0.7,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(filter_funnel["stage"], fontsize=10)
        ax.set_xlabel("Candidate Count", fontsize=12)
        ax.set_title(
            "RF vs GNN: Filter Funnel Comparison",
            fontsize=14, weight="bold",
        )
        ax.legend(fontsize=11, loc="upper right")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        # Add count labels
        for bar, count in zip(bars_rf, filter_funnel["count"]):
            ax.text(
                bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9, color="darkorange",
            )
        for bar, count in zip(bars_gnn, gnn_funnel["count"]):
            ax.text(
                bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9, color="steelblue",
            )

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "rf_vs_gnn_funnel_comparison.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'rf_vs_gnn_funnel_comparison.png'}")
    else:
        print("  Skipped: GNN funnel data not found")

    # Figure 4: RF validation sanity check
    print("\n[Fig 4/4] Generating RF validation sanity check plot...")
    if validation_df is not None and len(validation_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 7))

        valid = ~validation_df["T_b_predicted_K"].isna()
        if valid.sum() > 0:
            exp_vals = validation_df.loc[valid, "T_b_experimental_K"].values
            pred_vals = validation_df.loc[valid, "T_b_predicted_K"].values

            ax.scatter(
                exp_vals, pred_vals,
                s=100, alpha=0.7, color="darkorange", edgecolor="black", zorder=5,
            )

            # Label each point
            for _, row in validation_df[valid].iterrows():
                name = row.get("name", row["smiles"])
                if isinstance(name, str) and len(name) > 20:
                    name = name[:20] + "..."
                ax.annotate(
                    name, (row["T_b_experimental_K"], row["T_b_predicted_K"]),
                    fontsize=7, alpha=0.7, textcoords="offset points", xytext=(5, 5),
                )

            # Parity line
            all_vals = np.concatenate([exp_vals, pred_vals])
            vmin, vmax = all_vals.min() - 10, all_vals.max() + 10
            ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, lw=2, label="Parity")

            # +/- 10K bands
            ax.fill_between(
                [vmin, vmax], [vmin - 10, vmax - 10], [vmin + 10, vmax + 10],
                alpha=0.1, color="green", label="+/- 10 K",
            )

            errors = pred_vals - exp_vals
            mae = np.abs(errors).mean()

            ax.set_xlabel("Experimental T_b (K)", fontsize=14)
            ax.set_ylabel("RF Predicted T_b (K)", fontsize=14)
            ax.set_title(
                f"RF Validation: Boiling Points (MAE = {mae:.1f} K, n={valid.sum()})",
                fontsize=14, weight="bold",
            )
            ax.legend(fontsize=11, loc="upper left")
            ax.grid(alpha=0.3)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "rf_validation_sanity.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'rf_validation_sanity.png'}")
    else:
        print("  Skipped: no validation data")

    print(f"\nAll figures saved to: {FIGURES_DIR}")


def main():
    """Run full RF HFO screening pipeline with validation and figures."""
    print_warning_banner()

    # Step 0: Validation sanity check
    validation_df, bp_mae = run_validation_sanity_check()

    # Step 1-8: Full screening pipeline
    print("\n" + "=" * 80)
    print("RF HFO SCREENING PIPELINE")
    print("=" * 80)
    filtered_df, filter_funnel, df_full = run_screening_pipeline()

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    generate_figures(filtered_df, filter_funnel, df_full, validation_df)

    # Compare with GNN results
    print("\n" + "=" * 80)
    print("RF vs GNN COMPARISON")
    print("=" * 80)

    gnn_ranked_path = OUTPUT_DIR / "hfo_gnn_ranked.csv"
    if gnn_ranked_path.exists():
        df_gnn = pd.read_csv(gnn_ranked_path)
        print(f"\nGNN candidates (Step 37): {len(df_gnn)}")
        print(f"RF candidates (Step 38b): {len(filtered_df)}")

        gnn_smiles = set(df_gnn["smiles"])
        rf_smiles = set(filtered_df["smiles"])
        overlap = gnn_smiles & rf_smiles
        print(f"Overlap in filter-passing sets: {len(overlap)}")

        if len(filtered_df) > 0 and len(df_gnn) > 0:
            rf_top20 = set(filtered_df.head(20)["smiles"])
            gnn_top20 = set(df_gnn.head(20)["smiles"])
            print(f"Overlap in top 20: {len(rf_top20 & gnn_top20)}")

            rf_top50 = set(filtered_df.head(50)["smiles"])
            gnn_top50 = set(df_gnn.head(50)["smiles"])
            print(f"Overlap in top 50: {len(rf_top50 & gnn_top50)}")
    else:
        print("GNN ranked results not found; comparison skipped")

    # Final summary
    print("\n" + "=" * 80)
    print("RF HFO SCREENING COMPLETE (Step 38b)")
    print("=" * 80)
    print(f"\nValidation MAE: {bp_mae:.1f} K")
    print(f"Candidates passing all 7 filters: {len(filtered_df)}")
    print(f"Results saved to: {OUTPUT_DIR / 'hfo_rf_ranked.csv'}")
    print(f"Funnel saved to: {OUTPUT_DIR / 'hfo_rf_filter_funnel.csv'}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
