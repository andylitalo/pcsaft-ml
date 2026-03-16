"""Step 37: Filter Sensitivity Analysis and Comparison Figures.

This script performs:
1. Filter threshold sensitivity analysis for GNN screening
2. Comparison figures between RF (Step 25) and GNN (Step 37) results

Usage:
    python scripts/step37_sensitivity_and_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from screening.filters import fluorine_mass_fraction, has_reactive_fluorine


def filter_sensitivity_analysis(df_full):
    """Perform threshold sensitivity analysis on GNN candidates.

    Args:
        df_full: Full candidate DataFrame with PC-SAFT params and boiling points

    Returns:
        dict: Sensitivity results for each threshold variation
    """
    results = {
        "boiling_point": [],
        "sa_score": [],
        "fluorine_mass_fraction": [],
    }

    # Baseline filters (for reference)
    baseline_bp = df_full["boiling_point_K"].between(288, 323)
    baseline_sa = df_full["sa_score"] <= 4.5
    baseline_f_mass = df_full["smiles"].apply(
        lambda s: fluorine_mass_fraction(s) >= 0.65
    )

    # Common filters (applied to all sensitivity tests)
    pattern_cc = Chem.MolFromSmarts("C=C")
    has_cc = df_full["smiles"].apply(
        lambda s: Chem.MolFromSmiles(s).HasSubstructMatch(pattern_cc)
        if Chem.MolFromSmiles(s) is not None else False
    )

    pattern_cl = Chem.MolFromSmarts("[Cl]")
    no_cl = df_full["smiles"].apply(
        lambda s: not Chem.MolFromSmiles(s).HasSubstructMatch(pattern_cl)
        if Chem.MolFromSmiles(s) is not None else False
    )

    def count_fluorines(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F")

    f_count = df_full["smiles"].apply(lambda s: count_fluorines(s) >= 2)

    def is_safe(smiles):
        has_reactive, _ = has_reactive_fluorine(smiles)
        return not has_reactive

    no_reactive = df_full["smiles"].apply(is_safe)

    # 1. Boiling point sensitivity: ±5K
    print("\n=== Boiling Point Sensitivity ===")
    for lower, upper, label in [
        (283, 328, "wider (+5K)"),
        (288, 323, "baseline"),
        (293, 318, "narrower (-5K)"),
    ]:
        bp_mask = df_full["boiling_point_K"].between(lower, upper)
        combined = bp_mask & has_cc & no_cl & f_count & baseline_sa & baseline_f_mass & no_reactive
        count = combined.sum()
        results["boiling_point"].append({
            "range": f"[{lower}, {upper}]K",
            "label": label,
            "count": count,
        })
        print(f"  {label:20s} [{lower}-{upper}]K: {count:4d} candidates")

    # 2. SA score sensitivity: 4.0 to 5.0 in 0.5 steps
    print("\n=== SA Score Sensitivity ===")
    for threshold in [4.0, 4.5, 5.0]:
        sa_mask = df_full["sa_score"] <= threshold
        combined = baseline_bp & has_cc & no_cl & f_count & sa_mask & baseline_f_mass & no_reactive
        count = combined.sum()
        label = "baseline" if threshold == 4.5 else f"threshold={threshold}"
        results["sa_score"].append({
            "threshold": threshold,
            "label": label,
            "count": count,
        })
        print(f"  SA <= {threshold:3.1f} ({label:20s}): {count:4d} candidates")

    # 3. Fluorine mass fraction sensitivity: 0.55 to 0.75 in 0.05 steps
    print("\n=== Fluorine Mass Fraction Sensitivity ===")
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        f_mass_mask = df_full["smiles"].apply(
            lambda s: fluorine_mass_fraction(s) >= threshold
        )
        combined = baseline_bp & has_cc & no_cl & f_count & baseline_sa & f_mass_mask & no_reactive
        count = combined.sum()
        label = "baseline" if threshold == 0.65 else f"threshold={threshold:.2f}"
        results["fluorine_mass_fraction"].append({
            "threshold": threshold,
            "label": label,
            "count": count,
        })
        print(f"  F_mass >= {threshold:.2f} ({label:20s}): {count:4d} candidates")

    return results


def generate_figures(df_gnn, df_rf, df_gnn_full, sensitivity_results, figures_dir):
    """Generate all comparison and sensitivity figures.

    Args:
        df_gnn: GNN ranked candidates (passing all filters)
        df_rf: RF ranked candidates (passing all filters)
        df_gnn_full: All GNN candidates (pre-filter)
        sensitivity_results: Output from filter_sensitivity_analysis
        figures_dir: Directory to save figures
    """
    figures_dir.mkdir(exist_ok=True)

    # Figure 1: GNN filter funnel
    print("\n[1/6] Generating GNN filter funnel...")
    funnel_path = (
        Path(__file__).parent.parent / "screening" / "results" / "hfo_gnn_filter_funnel.csv"
    )
    funnel = pd.read_csv(funnel_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(funnel))
    ax.barh(y_pos, funnel["count"], color="steelblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(funnel["stage"], fontsize=10)
    ax.set_xlabel("Candidate Count", fontsize=12)
    ax.set_title("GNN HFO Screening Filter Funnel (7 Filters)", fontsize=14, weight="bold")
    ax.invert_yaxis()

    # Add count labels
    for i, count in enumerate(funnel["count"]):
        ax.text(count + 50, i, f"{count:,}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "screening_funnel_gnn.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {figures_dir / 'screening_funnel_gnn.png'}")

    # Figure 2: RF vs GNN parameter histograms
    print("\n[2/6] Generating parameter comparison histograms...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (param, label) in enumerate([
        ("m", "m (segments)"),
        ("sigma", "σ (Å)"),
        ("epsilon_k", "ε/k (K)"),
    ]):
        ax = axes[i]
        ax.hist(
            df_rf[param], bins=30, alpha=0.5, label="RF (Step 25)",
            color="orange", edgecolor="black"
        )
        ax.hist(
            df_gnn[param], bins=30, alpha=0.5, label="GNN (Step 37)",
            color="steelblue", edgecolor="black"
        )
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "RF vs GNN: Predicted PC-SAFT Parameters (Filter-Passing Candidates)",
        fontsize=14, weight="bold"
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "rf_vs_gnn_parameter_histograms.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {figures_dir / 'rf_vs_gnn_parameter_histograms.png'}")

    # Figure 3: RF vs GNN rank shift
    print("\n[3/6] Generating rank shift comparison...")
    # Find overlapping candidates in top 50
    rf_top50 = set(df_rf.head(50)["smiles"])
    gnn_top50 = set(df_gnn.head(50)["smiles"])
    overlap = rf_top50 & gnn_top50

    print(f"  Overlap in top 50: {len(overlap)} candidates")

    if len(overlap) > 0:
        # Get ranks for overlapping candidates
        rf_ranks = {smi: i + 1 for i, smi in enumerate(df_rf["smiles"])}
        gnn_ranks = {smi: i + 1 for i, smi in enumerate(df_gnn["smiles"])}

        overlap_data = []
        for smi in overlap:
            overlap_data.append({
                "smiles": smi,
                "rf_rank": rf_ranks.get(smi, 999),
                "gnn_rank": gnn_ranks.get(smi, 999),
                "shift": rf_ranks.get(smi, 999) - gnn_ranks.get(smi, 999),
            })

        overlap_df = pd.DataFrame(overlap_data).sort_values("gnn_rank")

        fig, ax = plt.subplots(figsize=(10, 8))
        for _, row in overlap_df.iterrows():
            color = "green" if row["shift"] > 0 else "red" if row["shift"] < 0 else "gray"
            ax.plot(
                [1, 2], [row["rf_rank"], row["gnn_rank"]],
                'o-', color=color, alpha=0.6, linewidth=1.5
            )

        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["RF (Step 25)", "GNN (Step 37)"], fontsize=12)
        ax.set_ylabel("Rank (lower is better)", fontsize=12)
        ax.set_title(
            f"Rank Shift for Top-50 Overlapping Candidates (n={len(overlap)})",
            fontsize=14, weight="bold"
        )
        ax.invert_yaxis()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(figures_dir / "rf_vs_gnn_rank_shift.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {figures_dir / 'rf_vs_gnn_rank_shift.png'}")
    else:
        print("  No overlap in top 50; skipping rank shift plot")

    # Figure 4: Boiling point vs HFO distance (GNN)
    print("\n[4/6] Generating boiling point vs HFO distance scatter...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute cf3_count for color coding
    from screening.filters import count_cf3_groups
    cf3_counts = df_gnn["smiles"].apply(count_cf3_groups)

    scatter = ax.scatter(
        df_gnn["hfo_distance"],
        df_gnn["boiling_point_K"] - 273.15,
        c=cf3_counts,
        cmap="viridis",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        s=50,
    )

    ax.set_xlabel("HFO Distance (5:2:1 weighting)", fontsize=12)
    ax.set_ylabel("Boiling Point (°C)", fontsize=12)
    ax.set_title("GNN Candidates: Boiling Point vs HFO Distance", fontsize=14, weight="bold")
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("CF3 Count", fontsize=12)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "boiling_point_vs_hfo_distance_gnn.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: {figures_dir / 'boiling_point_vs_hfo_distance_gnn.png'}")

    # Figure 5: RF vs GNN top 20 comparison
    print("\n[5/6] Generating top 20 comparison table...")
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    # Get top 20 from each
    rf_top20 = df_rf.head(20)
    gnn_top20 = df_gnn.head(20)

    # Create comparison table
    table_data = []
    for i in range(20):
        rf_smi = rf_top20.iloc[i]["smiles"] if i < len(rf_top20) else ""
        gnn_smi = gnn_top20.iloc[i]["smiles"] if i < len(gnn_top20) else ""

        # Truncate long SMILES
        rf_smi_short = rf_smi[:30] + "..." if len(rf_smi) > 30 else rf_smi
        gnn_smi_short = gnn_smi[:30] + "..." if len(gnn_smi) > 30 else gnn_smi

        match = "✓" if rf_smi == gnn_smi and rf_smi != "" else ""

        table_data.append([
            f"{i+1}",
            rf_smi_short,
            gnn_smi_short,
            match,
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Rank", "RF (Step 25)", "GNN (Step 37)", "Match"],
        cellLoc="left",
        loc="center",
        colWidths=[0.08, 0.42, 0.42, 0.08],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight matching rows
    for i, row in enumerate(table_data, start=1):
        if row[3] == "✓":
            for j in range(4):
                table[(i, j)].set_facecolor("#E8F5E9")

    ax.set_title("Top 20 Candidates: RF vs GNN Comparison", fontsize=14, weight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(figures_dir / "rf_vs_gnn_top20_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {figures_dir / 'rf_vs_gnn_top20_comparison.png'}")

    # Figure 6: Filter sensitivity
    print("\n[6/6] Generating filter sensitivity plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Boiling point sensitivity
    ax = axes[0]
    bp_data = sensitivity_results["boiling_point"]
    labels = [d["label"] for d in bp_data]
    counts = [d["count"] for d in bp_data]
    colors = ["lightblue", "steelblue", "lightblue"]
    ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Candidate Count", fontsize=12)
    ax.set_title("Boiling Point Window", fontsize=12, weight="bold")
    ax.tick_params(axis="x", labelrotation=15)
    for i, count in enumerate(counts):
        ax.text(i, count + 1, str(count), ha="center", fontsize=10, weight="bold")

    # SA score sensitivity
    ax = axes[1]
    sa_data = sensitivity_results["sa_score"]
    labels = [f"≤{d['threshold']}" for d in sa_data]
    counts = [d["count"] for d in sa_data]
    colors = ["lightblue", "steelblue", "lightblue"]
    ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Candidate Count", fontsize=12)
    ax.set_title("SA Score Threshold", fontsize=12, weight="bold")
    for i, count in enumerate(counts):
        ax.text(i, count + 2, str(count), ha="center", fontsize=10, weight="bold")

    # Fluorine mass fraction sensitivity
    ax = axes[2]
    f_data = sensitivity_results["fluorine_mass_fraction"]
    labels = [f"≥{d['threshold']:.0%}" for d in f_data]
    counts = [d["count"] for d in f_data]
    colors = ["lightblue"] * len(labels)
    colors[2] = "steelblue"  # Baseline is 0.65 (index 2)
    ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Candidate Count", fontsize=12)
    ax.set_title("Fluorine Mass Fraction", fontsize=12, weight="bold")
    ax.tick_params(axis="x", labelrotation=15)
    for i, count in enumerate(counts):
        ax.text(i, count + 5, str(count), ha="center", fontsize=10, weight="bold")

    fig.suptitle("GNN Filter Sensitivity Analysis", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "filter_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {figures_dir / 'filter_sensitivity.png'}")


def main():
    """Run sensitivity analysis and generate all figures."""
    print("=" * 80)
    print("Step 37: Filter Sensitivity Analysis and Comparison Figures")
    print("=" * 80)

    # Load data
    results_dir = Path(__file__).parent.parent / "screening" / "results"
    df_gnn = pd.read_csv(results_dir / "hfo_gnn_ranked.csv")
    df_rf = pd.read_csv(results_dir / "hfo_centric_ranked.csv")

    print(f"\nLoaded GNN results: {len(df_gnn)} candidates")
    print(f"Loaded RF results: {len(df_rf)} candidates")

    # Load full GNN candidate set for sensitivity analysis
    # We need to regenerate this quickly
    print("\nLoading full candidate set for sensitivity analysis...")
    from rdkit.Contrib.SA_Score import sascorer

    from model.registry import get_model
    from screening.generate import generate_systematic_candidates
    from screening.hfo_screening import batch_boiling_points

    # Generate candidates
    candidates = generate_systematic_candidates(max_cl=1, max_mw=200.0)
    smiles_list = [smi for smi, _ in candidates]
    df_full = pd.DataFrame({"smiles": smiles_list})

    # Predict with GNN
    print("Predicting PC-SAFT parameters with GNN...")
    gnn = get_model("gnn")
    gnn.load()
    predictions = gnn.predict(smiles_list)
    df_full["m"] = predictions["m"]
    df_full["sigma"] = predictions["sigma"]
    df_full["epsilon_k"] = predictions["epsilon_k"]

    # Compute boiling points
    print("Computing boiling points...")
    df_full = batch_boiling_points(df_full)

    # Compute SA scores
    print("Computing SA scores...")
    sa_scores = []
    for smi in df_full["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            sa_scores.append(10.0)
        else:
            try:
                sa_scores.append(sascorer.calculateScore(mol))
            except Exception:
                sa_scores.append(10.0)
    df_full["sa_score"] = sa_scores

    # Run sensitivity analysis
    print("\n" + "=" * 80)
    print("Filter Sensitivity Analysis")
    print("=" * 80)
    sensitivity_results = filter_sensitivity_analysis(df_full)

    # Generate figures
    print("\n" + "=" * 80)
    print("Generating Comparison Figures")
    print("=" * 80)
    figures_dir = Path(__file__).parent.parent / "figures" / "37_gnn_hfo_screening"
    generate_figures(df_gnn, df_rf, df_full, sensitivity_results, figures_dir)

    print("\n" + "=" * 80)
    print("Analysis and Figure Generation Complete")
    print("=" * 80)
    print(f"\nFigures saved to: {figures_dir}")
    print("\nSummary:")
    print(f"  GNN candidates passing all filters: {len(df_gnn)}")
    print(f"  RF candidates passing all filters: {len(df_rf)}")
    gnn_top20 = set(df_gnn.head(20)['smiles'])
    rf_top20 = set(df_rf.head(20)['smiles'])
    print(f"  Overlap in top 20: {len(gnn_top20 & rf_top20)}")
    gnn_top50 = set(df_gnn.head(50)['smiles'])
    rf_top50 = set(df_rf.head(50)['smiles'])
    print(f"  Overlap in top 50: {len(gnn_top50 & rf_top50)}")


if __name__ == "__main__":
    main()
