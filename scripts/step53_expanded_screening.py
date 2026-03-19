#!/usr/bin/env python
"""Step 53: Expanded Candidate Screening — All Non-Associating C2–C6 Halogenated Hydrocarbons.

Enumerates the full space of H/F/Cl/Br/I-substituted C2–C6 hydrocarbons
(both olefin and saturated backbones), predicts PC-SAFT parameters with
the RF model, applies cyclopentane-centric screening filters, and annotates
with applicability domain and regulatory status.
"""

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.ad_tanimoto import TanimotoAD
from model.predict import predict_pcsaft
from screening.filters import CYCLOPENTANE_EPS_K, CYCLOPENTANE_M, CYCLOPENTANE_SIGMA
from screening.generate import (
    ALKENE_BACKBONES,
    ALL_BACKBONES,
    generate_systematic_candidates,
)
from screening.hfo_screening import (
    apply_cyclopentane_filters,
    batch_boiling_points,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path("figures/53_expanded_screening")
RESULTS_DIR = Path("screening/results")

# Cyclopentane reference PC-SAFT parameters
CYCLOPENTANE = {"m": CYCLOPENTANE_M, "sigma": CYCLOPENTANE_SIGMA, "epsilon_k": CYCLOPENTANE_EPS_K}

# Weights for cyclopentane parameter distance (from step guide)
DISTANCE_WEIGHTS = {"m": 1.0, "sigma": 2.0, "epsilon_k": 5.0}

# Known commercial blowing agents (canonical SMILES → name)
KNOWN_AGENTS = {
    "C1CCCC1": "cyclopentane",
    "CCCCC": "n-pentane",
    "CC(C)CC": "isopentane",
    "FC(F)CC(F)(F)F": "HFC-245fa",
    "CC(F)(F)Cl": "HCFC-141b",
    "FC(F)Cl": "HCFC-22",
    "CC(F)F": "HFC-152a",
    "FCC(F)(F)F": "HFC-134a",
    "FC(F)C(F)(F)F": "HFC-125",
    "CC(F)(F)F": "HFC-143a",
    "C=C(F)C(F)(F)F": "HFO-1234yf",
    "ClC=CC(F)(F)F": "HCFO-1233zd",
}

# Canonicalize known agent SMILES
_canonical_known = {}
for smi, name in KNOWN_AGENTS.items():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        _canonical_known[Chem.MolToSmiles(mol)] = name
KNOWN_AGENTS_CANONICAL = _canonical_known


def compute_sa_scores(smiles_list):
    """Compute SA scores for a list of SMILES."""
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores.append(10.0)
            continue
        try:
            scores.append(sascorer.calculateScore(mol))
        except Exception:
            scores.append(10.0)
    return scores


def compute_cyclopentane_distance(m, sigma, epsilon_k, weights=None):
    """Compute weighted normalized distance to cyclopentane."""
    if weights is None:
        weights = DISTANCE_WEIGHTS
    wm = weights["m"]
    ws = weights["sigma"]
    we = weights["epsilon_k"]
    dm = wm * ((m - CYCLOPENTANE["m"]) / CYCLOPENTANE["m"]) ** 2
    ds = ws * ((sigma - CYCLOPENTANE["sigma"]) / CYCLOPENTANE["sigma"]) ** 2
    de = we * ((epsilon_k - CYCLOPENTANE["epsilon_k"]) / CYCLOPENTANE["epsilon_k"]) ** 2
    return float(np.sqrt(dm + ds + de))


def annotate_candidates(df):
    """Add structural, regulatory, and known-agent annotations."""
    pattern_cc = Chem.MolFromSmarts("C=C")
    pattern_cl = Chem.MolFromSmarts("[Cl]")
    pattern_br = Chem.MolFromSmarts("[Br]")
    pattern_i = Chem.MolFromSmarts("[#53]")

    has_double_bond = []
    contains_cl = []
    contains_br = []
    contains_i = []
    n_f_list = []
    n_halogens_list = []
    reg_flags = []
    known_names = []

    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            has_double_bond.append(False)
            contains_cl.append(False)
            contains_br.append(False)
            contains_i.append(False)
            n_f_list.append(0)
            n_halogens_list.append(0)
            reg_flags.append("unknown")
            known_names.append("")
            continue

        has_cc = mol.HasSubstructMatch(pattern_cc)
        has_cl = mol.HasSubstructMatch(pattern_cl)
        has_br = mol.HasSubstructMatch(pattern_br)
        has_i = mol.HasSubstructMatch(pattern_i)
        n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
        n_cl = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17)
        n_br = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 35)
        n_i = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 53)
        n_hal = n_f + n_cl + n_br + n_i

        # Regulatory classification
        if has_i:
            reg = "I-unstable"
        elif has_br:
            reg = "Br-ODP-nonstarter"
        elif has_cl and not has_cc:
            reg = "HCFC-Montreal-phaseout"
        elif has_cl and has_cc:
            reg = "HCFO-transitional"
        elif n_hal == 0:
            reg = "flammable-hydrocarbon"
        elif not has_cc and n_f > 0:
            reg = "HFC-Kigali-phasedown"
        elif has_cc and n_f > 0:
            reg = "HFO-compliant"
        else:
            reg = "other"

        has_double_bond.append(has_cc)
        contains_cl.append(has_cl)
        contains_br.append(has_br)
        contains_i.append(has_i)
        n_f_list.append(n_f)
        n_halogens_list.append(n_hal)
        reg_flags.append(reg)

        # Known agent matching
        can_smi = Chem.MolToSmiles(mol)
        known_names.append(KNOWN_AGENTS_CANONICAL.get(can_smi, ""))

    out = df.copy()
    out["has_double_bond"] = has_double_bond
    out["contains_cl"] = contains_cl
    out["contains_br"] = contains_br
    out["contains_i"] = contains_i
    out["n_fluorine"] = n_f_list
    out["n_halogens"] = n_halogens_list
    out["regulatory_flag"] = reg_flags
    out["known_agent"] = known_names
    return out


def halogen_class(row):
    """Classify molecule by halogen composition."""
    has_f = row.get("n_fluorine", 0) > 0
    has_cl = row.get("contains_cl", False)
    has_br = row.get("contains_br", False)
    has_i = row.get("contains_i", False)
    n_hal = row.get("n_halogens", 0)

    if n_hal == 0:
        return "unsubstituted"
    parts = []
    if has_f:
        parts.append("F")
    if has_cl:
        parts.append("Cl")
    if has_br:
        parts.append("Br")
    if has_i:
        parts.append("I")
    return "+".join(parts) if parts else "other"


def generate_figures(all_df, filtered_df, step20_count=10700):
    """Generate all figures for Step 53."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Parameter space by regulatory class
    fig, ax = plt.subplots(figsize=(10, 7))
    bp_mask = (all_df["boiling_point_K"] >= 288) & (all_df["boiling_point_K"] <= 323)
    bp_df = all_df[bp_mask].copy()

    reg_colors = {
        "HFO-compliant": "#2ecc71",
        "HCFO-transitional": "#f39c12",
        "HFC-Kigali-phasedown": "#e74c3c",
        "HCFC-Montreal-phaseout": "#9b59b6",
        "Br-ODP-nonstarter": "#e67e22",
        "I-unstable": "#95a5a6",
        "flammable-hydrocarbon": "#3498db",
        "other": "#bdc3c7",
    }

    for reg, color in reg_colors.items():
        mask = bp_df["regulatory_flag"] == reg
        if mask.sum() == 0:
            continue
        subset = bp_df[mask]
        ax.scatter(
            subset["m"], subset["epsilon_k"],
            c=color, label=f"{reg} (n={mask.sum()})",
            alpha=0.5, s=15, edgecolors="none",
        )

    ax.scatter(
        CYCLOPENTANE["m"], CYCLOPENTANE["epsilon_k"],
        marker="*", s=300, c="black", zorder=10, label="Cyclopentane",
    )
    ax.set_xlabel("m (segments)", fontsize=14)
    ax.set_ylabel("ε/k (K)", fontsize=14)
    ax.set_title("Parameter Space by Regulatory Class (BP-passing candidates)", fontsize=16)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parameter_space_by_regulatory_class.png", dpi=150)
    plt.close(fig)

    # Figure 2: Filter funnel comparison
    total_expanded = len(all_df)
    bp_count = int(bp_mask.sum())
    final_count = len(filtered_df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Step 20 funnel (approximate from the supplementary info)
    step20_stages = ["Enumerated", "BP [288-323K]", "All filters"]
    step20_counts = [step20_count, int(step20_count * 0.08), 0]  # approx 8% pass BP
    axes[0].barh(step20_stages, step20_counts, color="#3498db", alpha=0.7)
    axes[0].set_title("Step 20: Olefin-Only (F/Cl)", fontsize=14)
    axes[0].set_xlabel("Candidate Count", fontsize=12)
    for i, v in enumerate(step20_counts):
        axes[0].text(v + max(step20_counts) * 0.02, i, str(v), va="center", fontsize=11)

    # Step 53 funnel
    step53_stages = ["Enumerated", "BP [288-323K]", "BP + SA ≤ 4.5"]
    step53_counts = [total_expanded, bp_count, final_count]
    axes[1].barh(step53_stages, step53_counts, color="#e74c3c", alpha=0.7)
    axes[1].set_title("Step 53: Expanded (All halogens)", fontsize=14)
    axes[1].set_xlabel("Candidate Count", fontsize=12)
    for i, v in enumerate(step53_counts):
        axes[1].text(v + max(step53_counts) * 0.02, i, str(v), va="center", fontsize=11)

    fig.suptitle("Filter Funnel Comparison", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "filter_funnel_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Halogen class distribution
    if "halogen_class" not in all_df.columns:
        all_df = all_df.copy()
        all_df["halogen_class"] = all_df.apply(halogen_class, axis=1)
    if "halogen_class" not in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df["halogen_class"] = filtered_df.apply(halogen_class, axis=1)

    classes = sorted(all_df["halogen_class"].unique())
    stages = ["Enumerated", "BP-passing", "Final (BP+SA)"]
    stage_dfs = [all_df, all_df[bp_mask], filtered_df]

    class_counts = {cls: [] for cls in classes}
    for sdf in stage_dfs:
        if "halogen_class" not in sdf.columns:
            sdf = sdf.copy()
            sdf["halogen_class"] = sdf.apply(halogen_class, axis=1)
        vc = sdf["halogen_class"].value_counts()
        for cls in classes:
            class_counts[cls].append(vc.get(cls, 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stages))
    width = 0.8 / len(classes)
    cmap = plt.cm.Set3
    for i, cls in enumerate(classes):
        offset = (i - len(classes) / 2 + 0.5) * width
        ax.bar(x + offset, class_counts[cls], width, label=cls, color=cmap(i / len(classes)))

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=12)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Halogen Class Distribution at Each Filter Stage", fontsize=16)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "halogen_class_distribution.png", dpi=150)
    plt.close(fig)

    # Figure 4: AD coverage histogram
    if "ad_tanimoto_max" in all_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Separate by halogen content
        mask_br = all_df["contains_br"] == True  # noqa: E712
        mask_i = all_df["contains_i"] == True  # noqa: E712
        mask_neither = ~mask_br & ~mask_i

        if mask_neither.sum() > 0:
            ax.hist(
                all_df.loc[mask_neither, "ad_tanimoto_max"].dropna(),
                bins=50, alpha=0.6, label=f"No Br/I (n={mask_neither.sum()})",
                color="#3498db",
            )
        if mask_br.sum() > 0:
            ax.hist(
                all_df.loc[mask_br, "ad_tanimoto_max"].dropna(),
                bins=50, alpha=0.6, label=f"Contains Br (n={mask_br.sum()})",
                color="#e74c3c",
            )
        if mask_i.sum() > 0:
            ax.hist(
                all_df.loc[mask_i, "ad_tanimoto_max"].dropna(),
                bins=50, alpha=0.6, label=f"Contains I (n={mask_i.sum()})",
                color="#f39c12",
            )

        ax.axvline(0.4, color="black", linestyle="--", linewidth=2, label="AD threshold (0.4)")
        ax.set_xlabel("Max Tanimoto Similarity to Training Set", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_title("Applicability Domain Coverage", fontsize=16)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "ad_coverage.png", dpi=150)
        plt.close(fig)

    print(f"Figures saved to {FIGURES_DIR}/")


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Enumerate candidates
    # =========================================================================
    print("=" * 70)
    print("STEP 53: Expanded Candidate Screening")
    print("=" * 70)
    print("\n--- Phase 1: Enumeration ---")

    candidates = generate_systematic_candidates(
        backbones=ALL_BACKBONES,
        max_cl=1,
        max_br=2,
        max_i=1,
        max_mw=200.0,
        include_parent=True,
        n_jobs=-1,
    )
    smiles_list = [smi for smi, _ in candidates]
    print(f"Total unique candidates (with stereoisomers): {len(smiles_list)}")

    # Breakdown by backbone type
    olefin_only = generate_systematic_candidates(
        backbones=ALKENE_BACKBONES,
        max_cl=1,
        max_br=2,
        max_i=1,
        max_mw=200.0,
        include_parent=True,
        n_jobs=-1,
    )
    olefin_smiles = {smi for smi, _ in olefin_only}
    saturated_smiles = {smi for smi in smiles_list if smi not in olefin_smiles}
    print(f"  Olefin backbone candidates: {len(olefin_smiles)}")
    print(f"  Saturated backbone candidates: {len(saturated_smiles)}")

    # =========================================================================
    # 2. Predict PC-SAFT parameters
    # =========================================================================
    print("\n--- Phase 2: RF Prediction ---")
    predictions = predict_pcsaft(smiles_list)
    print(f"Predictions computed for {len(predictions)} molecules")

    # =========================================================================
    # 3. Compute boiling points (parallelized)
    # =========================================================================
    print("\n--- Phase 3: Boiling Points ---")
    df = batch_boiling_points(predictions, n_jobs=-1)
    n_valid_bp = df["boiling_point_K"].notna().sum()
    print(f"Valid boiling points: {n_valid_bp}/{len(df)}")

    # =========================================================================
    # 4. SA scores
    # =========================================================================
    print("\n--- Phase 4: SA Scores ---")
    df["sa_score"] = compute_sa_scores(df["smiles"].tolist())
    print(f"SA scores computed for {len(df)} molecules")

    # =========================================================================
    # 5. Broad filter (boiling point + SA only)
    # =========================================================================
    print("\n--- Phase 5: Cyclopentane-Centric Filtering ---")
    filtered, filter_stats = apply_cyclopentane_filters(df)
    print(f"\nCandidates passing broad filter: {len(filtered)}")

    # =========================================================================
    # 6. Rank by cyclopentane parameter distance
    # =========================================================================
    print("\n--- Phase 6: Ranking ---")
    filtered["cyclopentane_distance"] = filtered.apply(
        lambda row: compute_cyclopentane_distance(row["m"], row["sigma"], row["epsilon_k"]),
        axis=1,
    )
    filtered = filtered.sort_values("cyclopentane_distance").reset_index(drop=True)
    print(f"Ranked {len(filtered)} candidates by cyclopentane distance")

    # =========================================================================
    # 7. Applicability domain check
    # =========================================================================
    print("\n--- Phase 7: Applicability Domain ---")
    ad_path = Path("model/saved/tanimoto_ad.joblib")
    if ad_path.exists():
        tanimoto_ad = joblib.load(ad_path)
        print("Loaded saved TanimotoAD model")
    else:
        # Fit from training data
        from model.data.load import load_esper

        train_df, _ = load_esper()
        tanimoto_ad = TanimotoAD()
        tanimoto_ad.fit(train_df["smiles"].tolist())
        print("Fitted TanimotoAD from Esper training data")

    # Compute AD for all candidates (not just filtered)
    all_ad_max = [tanimoto_ad.tanimoto_nn(smi) for smi in df["smiles"]]
    df["ad_tanimoto_max"] = all_ad_max
    df["ad_in_domain"] = [v >= 0.4 for v in all_ad_max]

    # Also add to filtered
    filtered_ad = [tanimoto_ad.tanimoto_nn(smi) for smi in filtered["smiles"]]
    filtered["ad_tanimoto_max"] = filtered_ad
    filtered["ad_in_domain"] = [v >= 0.4 for v in filtered_ad]

    n_in_domain = sum(filtered["ad_in_domain"])
    print(f"In-domain (Tanimoto >= 0.4): {n_in_domain}/{len(filtered)}")

    # =========================================================================
    # 8. Annotate candidates
    # =========================================================================
    print("\n--- Phase 8: Annotation ---")
    df = annotate_candidates(df)
    filtered = annotate_candidates(filtered)
    filtered["halogen_class"] = filtered.apply(halogen_class, axis=1)
    df["halogen_class"] = df.apply(halogen_class, axis=1)
    print("Annotations added (structural, regulatory, known-agent)")

    # =========================================================================
    # 9. Save results
    # =========================================================================
    print("\n--- Phase 9: Save Results ---")
    filtered.to_csv(RESULTS_DIR / "expanded_ranked.csv", index=False)
    print(f"Saved {len(filtered)} ranked candidates to {RESULTS_DIR}/expanded_ranked.csv")

    # =========================================================================
    # 10. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal enumerated:      {len(df)}")
    print(f"Valid boiling points:  {n_valid_bp}")
    print(f"BP in [288, 323K]:     {filter_stats['boiling_point']}")
    print(f"SA score <= 4.5:       {filter_stats['sa_score']}")
    print(f"Pass all filters:      {filter_stats['total']}")

    # Breakdown by regulatory class
    print("\nFiltered candidates by regulatory class:")
    if len(filtered) > 0:
        for reg, count in filtered["regulatory_flag"].value_counts().items():
            print(f"  {reg}: {count}")

    # Breakdown by halogen class
    print("\nFiltered candidates by halogen class:")
    if len(filtered) > 0:
        for hc, count in filtered["halogen_class"].value_counts().items():
            print(f"  {hc}: {count}")

    # AD breakdown
    print("\nAD coverage in filtered candidates:")
    if len(filtered) > 0:
        for hc in filtered["halogen_class"].unique():
            hc_mask = filtered["halogen_class"] == hc
            n_total = hc_mask.sum()
            n_ad = (filtered.loc[hc_mask, "ad_in_domain"]).sum()
            print(f"  {hc}: {n_ad}/{n_total} in-domain ({100*n_ad/n_total:.0f}%)")

    # Top 20 candidates
    print("\nTop 20 candidates by cyclopentane distance:")
    top20_cols = [
        "smiles", "m", "sigma", "epsilon_k", "boiling_point_K",
        "cyclopentane_distance", "regulatory_flag", "ad_in_domain", "known_agent",
    ]
    if len(filtered) > 0:
        top20 = filtered.head(20)[top20_cols]
        print(top20.to_string(index=False))

    # Known agents found
    print("\nKnown commercial agents found in expanded space:")
    known_found = filtered[filtered["known_agent"] != ""]
    if len(known_found) > 0:
        for _, row in known_found.iterrows():
            print(
                f"  {row['known_agent']}: rank={filtered.index.get_loc(row.name)+1}, "
                f"dist={row['cyclopentane_distance']:.4f}, "
                f"reg={row['regulatory_flag']}, "
                f"AD={'in' if row['ad_in_domain'] else 'OUT'}"
            )
    else:
        print("  (none found in filtered set — check full enumerated set)")
        # Check in full df
        df_annotated_known = df[df["known_agent"] != ""]
        if len(df_annotated_known) > 0:
            print("  Known agents in full enumerated set (before filtering):")
            for _, row in df_annotated_known.iterrows():
                bp = row.get("boiling_point_K", np.nan)
                print(f"    {row['known_agent']}: BP={bp:.1f} K, reg={row['regulatory_flag']}")

    # =========================================================================
    # 11. Comparison with Step 20/38b
    # =========================================================================
    print("\n--- Comparison with Step 20/38b ---")
    hfo_path = RESULTS_DIR / "hfo_rf_ranked.csv"
    if hfo_path.exists():
        hfo_df = pd.read_csv(hfo_path)
        hfo_smiles = set(hfo_df["smiles"])
        if len(filtered) > 0:
            top50 = set(filtered.head(50)["smiles"])
            overlap = top50 & hfo_smiles
            new = top50 - hfo_smiles
            print(f"Step 38b ranked candidates: {len(hfo_df)}")
            print(f"Expanded top-50 overlap with Step 38b: {len(overlap)}")
            print(f"New candidates in top-50: {len(new)}")
            if new:
                new_df = filtered[filtered["smiles"].isin(new)].head(10)
                print("\nNew candidates not in Step 38b (top 10):")
                print(new_df[top20_cols].to_string(index=False))
    else:
        print("Step 38b results not found — skipping comparison")

    # =========================================================================
    # 12. Generate figures
    # =========================================================================
    print("\n--- Phase 12: Figures ---")
    generate_figures(df, filtered)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save a summary JSON for the report
    summary = {
        "total_enumerated": len(df),
        "valid_boiling_points": int(n_valid_bp),
        "filter_stats": filter_stats,
        "n_filtered": len(filtered),
        "n_in_domain": int(n_in_domain),
        "olefin_candidates": len(olefin_smiles),
        "saturated_candidates": len(saturated_smiles),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(RESULTS_DIR / "expanded_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {RESULTS_DIR}/expanded_summary.json")


if __name__ == "__main__":
    main()
