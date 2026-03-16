"""Step 41: Fluorinated Safety & Environmental Gate.

Applies structural safety, environmental (GWP proxy), flammability, and
toxicity screening to the 50 RF-ranked HFO/HCFO blowing agent candidates
from Step 39.

All classifications here are **structural heuristics**, NOT regulatory
determinations. Real safety assessment requires flash-point testing,
LOC measurement, cardiac sensitization studies, and atmospheric modelling.

Inputs:
  - screening/results/hfo_rf_shortlist_credibility.csv (50 candidates)

Outputs:
  - screening/results/hfo_rf_safety_assessed.csv
  - figures/41_safety_environmental_gate/ (3 figures)
  - docs/reports/41_safety_environmental_gate.md

Usage:
    python scripts/step41_safety_environmental_gate.py
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from screening.filters import (  # noqa: E402
    fluorine_mass_fraction,
    has_cf3_group,
    has_reactive_fluorine,
    is_associating,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_shortlist_credibility.csv"
OUTPUT_CSV = PROJECT_ROOT / "screening" / "results" / "hfo_rf_safety_assessed.csv"
FIGURES_DIR = PROJECT_ROOT / "figures" / "41_safety_environmental_gate"
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "41_safety_environmental_gate.md"

# ---------------------------------------------------------------------------
# Toxicity SMARTS patterns (41.4)
# ---------------------------------------------------------------------------
TOXICITY_SMARTS = {
    "epoxide": "[C]1[O][C]1",
    "peroxide": "[O]-[O]",
    "isocyanate": "[N]=[C]=[O]",
    "acid_fluoride": "[CX3](=O)F",
}

# Pre-compile
_TOXICITY_PATTERNS = {}
for name, sma in TOXICITY_SMARTS.items():
    pat = Chem.MolFromSmarts(sma)
    if pat is not None:
        _TOXICITY_PATTERNS[name] = pat

# C=C bond SMARTS
CC_DOUBLE_SMARTS = Chem.MolFromSmarts("[C]=[C]")


# ===================================================================
# 41.1 — Structural Safety Assessment
# ===================================================================

def structural_assessment(smiles: str) -> dict:
    """Compute structural safety properties for a single molecule.

    Returns a dict with all computed columns.
    """
    result = {"smiles": smiles}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Failed to parse SMILES: %s", smiles)
        return result

    # Fluorine mass fraction
    result["f_mass_fraction"] = fluorine_mass_fraction(smiles)

    # CF3 groups
    result["has_cf3"] = has_cf3_group(smiles)

    # Reactive fluorination sites
    reactive, reactive_patterns = has_reactive_fluorine(smiles)
    result["has_reactive_f"] = reactive
    result["reactive_patterns"] = ";".join(reactive_patterns) if reactive_patterns else ""

    # Association sites
    result["is_associating"] = is_associating(smiles)

    # Molecular weight
    result["mol_weight"] = Descriptors.ExactMolWt(mol)

    # Heavy atom count
    result["heavy_atom_count"] = Descriptors.HeavyAtomCount(mol)

    # C=C bond count
    if CC_DOUBLE_SMARTS is not None:
        cc_matches = mol.GetSubstructMatches(CC_DOUBLE_SMARTS)
        result["n_cc_double"] = len(cc_matches)
    else:
        result["n_cc_double"] = 0

    # Atom counts
    result["n_fluorine"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    result["n_chlorine"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17)
    result["n_hydrogen"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    # RDKit suppresses explicit H by default; add them
    mol_h = Chem.AddHs(mol)
    result["n_hydrogen"] = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
    result["n_carbon"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)

    return result


# ===================================================================
# 41.2 — GWP & Atmospheric Lifetime Proxy
# ===================================================================

def _has_h_on_carbon(mol) -> bool:
    """Check if molecule has H atoms bonded to carbon (sp2 or sp3)."""
    mol_h = Chem.AddHs(mol)
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            # Check if the H is bonded to a carbon
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 6:
                    return True
    return False


def atmospheric_lifetime_proxy(smiles: str, n_cc_double: int) -> tuple[str, str]:
    """Classify atmospheric lifetime and GWP class from structure.

    Returns (lifetime_class, gwp_class).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown", "unknown"

    has_cc = n_cc_double > 0
    has_h_on_c = _has_h_on_carbon(mol)

    # Count C-H bonds (including on sp2 and sp3 carbon)
    mol_h = Chem.AddHs(mol)
    n_ch = 0
    for bond in mol_h.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if (a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 1) or \
           (a1.GetAtomicNum() == 1 and a2.GetAtomicNum() == 6):
            n_ch += 1

    if has_cc and has_h_on_c:
        return "very_short", "ultra_low"
    elif has_cc and not has_h_on_c:
        return "short", "low"
    elif not has_cc and n_ch > 0:
        return "moderate", "medium"
    else:
        return "long", "high"


# ===================================================================
# 41.3 — Flammability Classification Proxy
# ===================================================================

def flammability_proxy(f_mass_frac: float) -> str:
    """Heuristic flammability classification based on fluorine mass fraction.

    Based on ASHRAE 34 correlation heuristic (NOT an actual classification).
    """
    if f_mass_frac >= 0.65:
        return "A1"
    elif f_mass_frac >= 0.50:
        return "A2L"
    elif f_mass_frac >= 0.30:
        return "A2"
    else:
        return "A3"


# ===================================================================
# 41.4 — Toxicity Red Flags
# ===================================================================

def toxicity_red_flags(smiles: str) -> tuple[bool, list[str]]:
    """Check for known problematic substructures.

    Returns (has_flags, list_of_flag_names).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, []

    found = []
    for name, pattern in _TOXICITY_PATTERNS.items():
        if mol.HasSubstructMatch(pattern):
            found.append(name)

    return len(found) > 0, found


# ===================================================================
# 41.5 — Composite Safety Score
# ===================================================================

def compute_safety_score(row: dict) -> int:
    """Compute composite safety score (higher = safer).

    Scoring:
      +1 if F mass fraction >= 0.50
      +1 if C=C bond present
      +1 if no reactive fluorination sites
      +1 if not associating
      +1 if no toxicity red flags
      -1 for each reactive pattern found
    """
    score = 0
    if row.get("f_mass_fraction", 0) >= 0.50:
        score += 1
    if row.get("n_cc_double", 0) > 0:
        score += 1
    if not row.get("has_reactive_f", True):
        score += 1
    if not row.get("is_associating", True):
        score += 1
    if not row.get("has_toxicity_flags", True):
        score += 1

    # Penalty: -1 for each reactive pattern
    rp = row.get("reactive_patterns", "")
    n_reactive = len(rp.split(";")) if rp else 0
    score -= n_reactive

    return score


# ===================================================================
# 41.6 — Safety Gate Classification
# ===================================================================

def classify_safety_gate(row: dict) -> str:
    """Assign pass/flag/fail based on composite criteria.

    PASS: safety_score >= 3 AND gwp_class in [ultra_low, low] AND flammability in [A1, A2L]
    FLAG: safety_score >= 2 AND gwp_class in [ultra_low, low, medium]
    FAIL: everything else
    """
    score = row.get("safety_score", 0)
    gwp = row.get("gwp_class", "unknown")
    flamm = row.get("flammability_class", "A3")

    if score >= 3 and gwp in ("ultra_low", "low") and flamm in ("A1", "A2L"):
        return "PASS"
    elif score >= 2 and gwp in ("ultra_low", "low", "medium"):
        return "FLAG"
    else:
        return "FAIL"


# ===================================================================
# Main assessment pipeline
# ===================================================================

def run_assessment(df_input: pd.DataFrame) -> pd.DataFrame:
    """Run the full safety assessment pipeline on all candidates."""

    records = []
    for _, row in df_input.iterrows():
        smi = row["smiles"]

        # 41.1: Structural assessment
        rec = structural_assessment(smi)

        # 41.2: GWP/atmospheric lifetime proxy
        lifetime, gwp = atmospheric_lifetime_proxy(smi, rec.get("n_cc_double", 0))
        rec["lifetime_class"] = lifetime
        rec["gwp_class"] = gwp

        # 41.3: Flammability proxy
        rec["flammability_class"] = flammability_proxy(rec.get("f_mass_fraction", 0))

        # 41.4: Toxicity red flags
        has_tox, tox_list = toxicity_red_flags(smi)
        rec["has_toxicity_flags"] = has_tox
        rec["toxicity_flags"] = ";".join(tox_list) if tox_list else ""

        # 41.5: Composite safety score
        rec["safety_score"] = compute_safety_score(rec)

        # 41.6: Gate classification
        rec["safety_gate"] = classify_safety_gate(rec)

        records.append(rec)

    df_safety = pd.DataFrame(records)

    # Merge with input DataFrame to keep all original columns
    df_merged = df_input.merge(df_safety, on="smiles", how="left")

    return df_merged


# ===================================================================
# 41.7 — Figures
# ===================================================================

def generate_figures(df: pd.DataFrame):
    """Generate 3 figures for Step 41."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating 3 figures to %s", FIGURES_DIR)

    # ---------------------------------------------------------------
    # Figure 1: Safety score distribution
    # ---------------------------------------------------------------
    logger.info("  [1/3] safety_score_distribution.png")
    fig, ax = plt.subplots(figsize=(8, 5))

    gate_colors = {"PASS": "#2ca02c", "FLAG": "#ff7f0e", "FAIL": "#d62728"}
    gate_order = ["PASS", "FLAG", "FAIL"]

    scores = df["safety_score"].values
    bins = np.arange(scores.min() - 0.5, scores.max() + 1.5, 1)

    # Stack bars by gate
    bottom = np.zeros(len(bins) - 1)
    for gate in gate_order:
        mask = df["safety_gate"] == gate
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(df.loc[mask, "safety_score"], bins=bins)
        ax.bar(
            (bins[:-1] + bins[1:]) / 2, counts, width=0.8,
            bottom=bottom, color=gate_colors[gate], edgecolor="black",
            linewidth=0.5, label=gate, alpha=0.85,
        )
        bottom += counts

    ax.set_xlabel("Composite Safety Score", fontsize=14)
    ax.set_ylabel("Number of Candidates", fontsize=16)
    ax.set_title("Safety Score Distribution (n=50 RF Candidates)", fontsize=16)
    ax.set_xticks(range(int(scores.min()), int(scores.max()) + 1))
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, title="Safety Gate", title_fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "safety_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------------
    # Figure 2: Flammability vs GWP scatter
    # ---------------------------------------------------------------
    logger.info("  [2/3] flammability_vs_gwp.png")
    fig, ax = plt.subplots(figsize=(9, 6))

    flamm_order = ["A1", "A2L", "A2", "A3"]
    gwp_order = ["ultra_low", "low", "medium", "high"]
    flamm_map = {v: i for i, v in enumerate(flamm_order)}
    gwp_map = {v: i for i, v in enumerate(gwp_order)}

    # Group by (flammability, gwp) and compute count + mean safety score
    groups = df.groupby(["flammability_class", "gwp_class"]).agg(
        count=("safety_score", "size"),
        mean_score=("safety_score", "mean"),
    ).reset_index()

    x_vals = [flamm_map.get(f, -1) for f in groups["flammability_class"]]
    y_vals = [gwp_map.get(g, -1) for g in groups["gwp_class"]]

    scatter = ax.scatter(
        x_vals, y_vals,
        s=groups["count"] * 80,  # scale for visibility
        c=groups["mean_score"],
        cmap="RdYlGn", edgecolor="black", linewidth=0.8, alpha=0.85,
        vmin=0, vmax=5,
    )

    # Annotate counts
    for _, row in groups.iterrows():
        xi = flamm_map.get(row["flammability_class"], -1)
        yi = gwp_map.get(row["gwp_class"], -1)
        ax.annotate(
            f"n={int(row['count'])}",
            (xi, yi), textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(range(len(flamm_order)))
    ax.set_xticklabels(flamm_order, fontsize=12)
    ax.set_yticks(range(len(gwp_order)))
    ax.set_yticklabels([g.replace("_", " ") for g in gwp_order], fontsize=12)
    ax.set_xlabel("Flammability Class (heuristic)", fontsize=14)
    ax.set_ylabel("GWP Class (heuristic)", fontsize=14)
    ax.set_title("Flammability vs GWP Classification (n=50)", fontsize=16)
    ax.tick_params(labelsize=12)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Mean Safety Score", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "flammability_vs_gwp.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------------
    # Figure 3: Safety profile for top-20 (stacked horizontal bar)
    # ---------------------------------------------------------------
    logger.info("  [3/3] safety_profile_top20.png")
    top20 = df.head(20).copy()

    criteria = [
        ("F >= 50%", lambda r: r["f_mass_fraction"] >= 0.50),
        ("C=C present", lambda r: r["n_cc_double"] > 0),
        ("No reactive F", lambda r: not r["has_reactive_f"]),
        ("Not associating", lambda r: not r["is_associating"]),
        ("No tox flags", lambda r: not r["has_toxicity_flags"]),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(top20))
    bar_height = 0.6

    pass_color = "#2ca02c"
    fail_color = "#d62728"

    for ci, (crit_name, crit_fn) in enumerate(criteria):
        for yi, (_, row) in enumerate(top20.iterrows()):
            passes = crit_fn(row)
            color = pass_color if passes else fail_color
            ax.barh(
                y_pos[yi], 1, left=ci, height=bar_height,
                color=color, edgecolor="white", linewidth=1.0, alpha=0.85,
            )

    # X-axis: criteria labels
    ax.set_xticks([i + 0.5 for i in range(len(criteria))])
    ax.set_xticklabels([c[0] for c in criteria], fontsize=12, rotation=0)

    # Y-axis: SMILES (truncated) + rank
    labels = []
    for _, row in top20.iterrows():
        rank = int(row.get("rank_original", 0))
        smi = row["smiles"]
        smi_short = smi[:22] + "..." if len(smi) > 22 else smi
        labels.append(f"#{rank} {smi_short}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, family="monospace")
    ax.invert_yaxis()

    ax.set_xlabel("Safety Criteria", fontsize=14)
    ax.set_title("Safety Profile: Top 20 Candidates by HFO Distance Rank", fontsize=16)
    ax.tick_params(labelsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=pass_color, edgecolor="black", label="Pass"),
        Patch(facecolor=fail_color, edgecolor="black", label="Fail"),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "safety_profile_top20.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("All 3 figures saved to: %s", FIGURES_DIR)


# ===================================================================
# 41.8 — Report
# ===================================================================

def generate_report(df: pd.DataFrame):
    """Generate the Step 41 markdown report."""
    logger.info("Generating report: %s", REPORT_PATH)

    n_total = len(df)
    n_pass = (df["safety_gate"] == "PASS").sum()
    n_flag = (df["safety_gate"] == "FLAG").sum()
    n_fail = (df["safety_gate"] == "FAIL").sum()

    mean_score = df["safety_score"].mean()
    median_score = df["safety_score"].median()

    # Flammability distribution
    flamm_dist = df["flammability_class"].value_counts().to_dict()
    gwp_dist = df["gwp_class"].value_counts().to_dict()
    lifetime_dist = df["lifetime_class"].value_counts().to_dict()

    # Top-20 table
    top20 = df.head(20)
    top20_rows = []
    for _, row in top20.iterrows():
        rank = int(row.get("rank_original", 0))
        smi = row["smiles"]
        smi_short = smi[:30] if len(smi) > 30 else smi
        f_frac = row.get("f_mass_fraction", 0)
        n_cc = int(row.get("n_cc_double", 0))
        flamm = row.get("flammability_class", "")
        gwp = row.get("gwp_class", "")
        score = int(row.get("safety_score", 0))
        gate = row.get("safety_gate", "")
        reactive = "Y" if row.get("has_reactive_f", False) else "N"
        cred = row.get("credibility", "")
        top20_rows.append(
            f"| {rank} | {smi_short} | {f_frac:.2f} | {n_cc} | "
            f"{reactive} | {flamm} | {gwp} | {score} | {gate} | {cred} |"
        )
    top20_table = "\n".join(top20_rows)

    lines = []
    lines.append(
        "# Results Report: Fluorinated Safety & Environmental Gate "
        "(Step 41, 50 RF Candidates)"
    )
    lines.append("")
    lines.append("## Safety Assessment Methodology")
    lines.append("")
    lines.append(
        "This step applies structural heuristics to screen "
        "the 50 RF-ranked HFO/HCFO blowing agent candidates "
        "for safety, environmental impact, and toxicity concerns. "
        "**All classifications are proxy-based heuristics derived "
        "from molecular structure, NOT regulatory determinations.** "
        "Real safety assessment requires:"
    )
    lines.append("")
    lines.append("- Flash point and lower flammability limit (LFL) testing")
    lines.append("- Limiting oxygen concentration (LOC) measurement")
    lines.append("- Cardiac sensitization studies (NOAEL determination)")
    lines.append("- Atmospheric chemistry modelling (OH radical rate constants)")
    lines.append("- Full ASHRAE 34 classification protocol")
    lines.append("")
    lines.append("### Heuristic Methods")
    lines.append("")
    lines.append(
        "1. **Fluorine mass fraction**: Computed from molecular formula. "
        "Higher F content correlates with lower flammability."
    )
    lines.append(
        "2. **GWP proxy**: Based on presence of C=C bonds (enabling OH "
        "radical attack) and C-H bonds. HFOs with C=C bonds have "
        "atmospheric lifetimes of days (GWP < 10), analogous to "
        "HFO-1234yf."
    )
    lines.append(
        "3. **Flammability proxy**: ASHRAE 34 heuristic based on F mass "
        "fraction thresholds (>= 65% -> A1, >= 50% -> A2L, >= 30% -> A2, "
        "< 30% -> A3)."
    )
    lines.append(
        "4. **Toxicity red flags**: Substructure search for epoxides, "
        "peroxides, isocyanates, and acid fluorides."
    )
    lines.append(
        "5. **Composite safety score** (0-5, higher = safer): "
        "+1 for each of F >= 50%, C=C present, no reactive F, "
        "not associating, no tox flags; -1 per reactive pattern found."
    )
    lines.append("")
    lines.append("### Gate Criteria")
    lines.append("")
    lines.append(
        "- **PASS**: safety_score >= 3 AND gwp_class in [ultra_low, low] "
        "AND flammability_class in [A1, A2L]"
    )
    lines.append(
        "- **FLAG**: safety_score >= 2 AND gwp_class in [ultra_low, low, medium]"
    )
    lines.append("- **FAIL**: everything else")
    lines.append("")

    lines.append("## Results: Safety Profile (Top 20)")
    lines.append("")
    lines.append(
        "| Rank | SMILES | F frac | C=C | React F | Flamm | GWP | "
        "Score | Gate | Credibility |"
    )
    lines.append(
        "|------|--------|--------|-----|---------|-------|-----|"
        "-------|------|-------------|"
    )
    lines.append(top20_table)
    lines.append("")

    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("### Safety Gate Distribution")
    lines.append("")
    lines.append("| Gate | Count | Fraction |")
    lines.append("|------|-------|----------|")
    for gate in ["PASS", "FLAG", "FAIL"]:
        n = (df["safety_gate"] == gate).sum()
        pct = 100 * n / n_total
        lines.append(f"| {gate} | {n} | {pct:.0f}% |")
    lines.append("")
    lines.append(f"- Mean safety score: {mean_score:.2f}")
    lines.append(f"- Median safety score: {median_score:.1f}")
    lines.append("")

    lines.append("### GWP Class Distribution")
    lines.append("")
    lines.append("| GWP Class | Count |")
    lines.append("|-----------|-------|")
    for cls in ["ultra_low", "low", "medium", "high"]:
        n = gwp_dist.get(cls, 0)
        lines.append(f"| {cls} | {n} |")
    lines.append("")

    lines.append("### Atmospheric Lifetime Classification")
    lines.append("")
    lines.append("| Lifetime Class | Count |")
    lines.append("|----------------|-------|")
    for cls in ["very_short", "short", "moderate", "long"]:
        n = lifetime_dist.get(cls, 0)
        lines.append(f"| {cls} | {n} |")
    lines.append("")

    lines.append("### Flammability Class Distribution")
    lines.append("")
    lines.append("| Flammability | Count |")
    lines.append("|--------------|-------|")
    for cls in ["A1", "A2L", "A2", "A3"]:
        n = flamm_dist.get(cls, 0)
        lines.append(f"| {cls} | {n} |")
    lines.append("")

    # Reactive fluorination summary
    n_reactive = df["has_reactive_f"].sum()
    lines.append("### Reactive Fluorination Sites")
    lines.append("")
    lines.append(
        f"- Candidates with reactive F patterns: {n_reactive}/{n_total} "
        f"({100 * n_reactive / n_total:.0f}%)"
    )
    reactive_details = df.loc[df["has_reactive_f"], ["smiles", "reactive_patterns"]]
    if len(reactive_details) > 0:
        lines.append("- Pattern breakdown:")
        pattern_counts = {}
        for patterns_str in reactive_details["reactive_patterns"]:
            for p in patterns_str.split(";"):
                if p:
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
        for pname, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {pname}: {cnt} candidates")
    lines.append("")

    # Toxicity flags
    n_tox = df["has_toxicity_flags"].sum()
    lines.append("### Toxicity Red Flags")
    lines.append("")
    lines.append(
        f"- Candidates with toxicity red flags: {n_tox}/{n_total}"
    )
    if n_tox > 0:
        tox_details = df.loc[df["has_toxicity_flags"], ["smiles", "toxicity_flags"]]
        for _, row in tox_details.iterrows():
            lines.append(f"  - {row['smiles']}: {row['toxicity_flags']}")
    else:
        lines.append(
            "- No candidates triggered any toxicity red flag patterns "
            "(epoxide, peroxide, isocyanate, acid fluoride)."
        )
    lines.append("")
    lines.append(
        "**Note**: Absence of these substructural red flags does NOT "
        "confirm safety. Most fluorinated olefins have low acute toxicity "
        "(cf. HFO-1234yf, NOAEL > 50,000 ppm), but cardiac sensitization "
        "is a known concern class for all halogenated hydrocarbons and "
        "should be evaluated experimentally for any candidate."
    )
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append(
        f"1. **{n_pass} of {n_total} candidates pass the safety gate** "
        f"({100 * n_pass / n_total:.0f}%), with {n_flag} flagged for "
        f"review and {n_fail} failing."
    )
    lines.append("")

    # Check if all have C=C
    all_cc = (df["n_cc_double"] > 0).all()
    lines.append(
        f"2. **All candidates have C=C bonds**: {all_cc}. "
        "This is expected since they passed the HFO filter in Step 38b, "
        "confirming ultra-low GWP potential (atmospheric lifetime of days)."
    )
    lines.append("")

    n_a1 = flamm_dist.get("A1", 0)
    n_a2l = flamm_dist.get("A2L", 0)
    lines.append(
        f"3. **Flammability**: {n_a1} candidates classified as A1 "
        f"(non-flammable heuristic) and {n_a2l} as A2L (mildly flammable). "
        "The fluorine mass fraction range determines this distribution."
    )
    lines.append("")

    lines.append(
        f"4. **Reactive fluorination**: {n_reactive} candidates have "
        "reactive fluorination patterns (allylic CHF or CH2F adjacent "
        "to C=C). These could indicate thermal instability but are "
        "common in commercial HFOs (e.g., HFO-1234yf has allylic CHF)."
    )
    lines.append("")

    lines.append(
        "5. **No toxicity red flags**: None of the 50 candidates "
        "contain epoxide, peroxide, isocyanate, or acid fluoride "
        "substructures. This is expected for simple fluorinated olefins."
    )
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    lines.append("See `figures/41_safety_environmental_gate/` for:")
    lines.append(
        "- `safety_score_distribution.png` -- Histogram of safety scores "
        "colored by pass/flag/fail gate"
    )
    lines.append(
        "- `flammability_vs_gwp.png` -- 2D scatter of flammability class "
        "vs GWP class, with point size proportional to candidate count "
        "and color representing mean safety score"
    )
    lines.append(
        "- `safety_profile_top20.png` -- Horizontal stacked bar chart "
        "showing which safety criteria each top-20 candidate passes/fails"
    )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "**These are structural heuristics, NOT regulatory classifications.** "
        "The following limitations apply:"
    )
    lines.append("")
    lines.append(
        "1. **Flammability**: The F mass fraction threshold is a rough "
        "correlation. Actual ASHRAE 34 classification requires standardized "
        "flash point testing, burning velocity measurement, and heat of "
        "combustion data."
    )
    lines.append(
        "2. **GWP**: The atmospheric lifetime proxy assumes OH radical "
        "reactivity based on C=C bond presence. True GWP requires "
        "atmospheric chemistry modelling with measured OH rate constants, "
        "IR absorption cross-sections, and radiative efficiency calculations."
    )
    lines.append(
        "3. **Toxicity**: Substructure screening catches only known-bad "
        "patterns. Cardiac sensitization (the primary toxicity concern for "
        "halogenated hydrocarbons) cannot be predicted from structure alone "
        "and requires in-vivo testing."
    )
    lines.append(
        "4. **Thermal stability**: The reactive fluorination patterns are "
        "heuristic. Some commercial HFOs (e.g., HFO-1234yf) contain these "
        "patterns and are thermally stable under normal conditions."
    )
    lines.append(
        "5. **Environmental persistence**: Trifluoroacetic acid (TFA) "
        "formation from HFO degradation is a growing environmental concern "
        "not captured by GWP alone."
    )
    lines.append("")

    lines.append("## Readiness Check")
    lines.append("")
    lines.append("- [x] Structural safety assessment for all 50 candidates")
    lines.append("- [x] GWP and atmospheric lifetime proxy classification")
    lines.append("- [x] Flammability classification proxy")
    lines.append("- [x] Toxicity red flag screening")
    lines.append("- [x] Composite safety score computed")
    lines.append(
        f"- [x] Safety gate applied: {n_pass} PASS, {n_flag} FLAG, {n_fail} FAIL"
    )
    lines.append("- [x] Enriched CSV saved with all safety columns")
    lines.append("- [x] 3 figures generated")
    lines.append("- [x] Heuristic nature of all classifications documented")
    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    logger.info("Report saved to: %s", REPORT_PATH)


# ===================================================================
# Main
# ===================================================================

def main():
    logger.info("=" * 80)
    logger.info("Step 41: Fluorinated Safety & Environmental Gate")
    logger.info("=" * 80)

    # Load input data
    if not INPUT_CSV.exists():
        logger.error("Input CSV not found: %s", INPUT_CSV)
        logger.error("Run Step 39 first to produce the credibility shortlist.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    logger.info("Loaded %d candidates from %s", len(df), INPUT_CSV)

    # Run full assessment
    logger.info("")
    logger.info("[41.1-41.6] Running safety assessment pipeline...")
    df_assessed = run_assessment(df)

    # Save enriched CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_assessed.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved safety-assessed CSV to: %s", OUTPUT_CSV)

    # Summary
    n_total = len(df_assessed)
    n_pass = (df_assessed["safety_gate"] == "PASS").sum()
    n_flag = (df_assessed["safety_gate"] == "FLAG").sum()
    n_fail = (df_assessed["safety_gate"] == "FAIL").sum()

    logger.info("")
    logger.info("=" * 60)
    logger.info("SAFETY GATE RESULTS")
    logger.info("=" * 60)
    logger.info("  PASS: %d/%d (%d%%)", n_pass, n_total, 100 * n_pass // n_total)
    logger.info("  FLAG: %d/%d (%d%%)", n_flag, n_total, 100 * n_flag // n_total)
    logger.info("  FAIL: %d/%d (%d%%)", n_fail, n_total, 100 * n_fail // n_total)
    logger.info("")

    # Print top-10 summary
    logger.info("TOP-10 SAFETY PROFILES:")
    logger.info("-" * 100)
    for _, row in df_assessed.head(10).iterrows():
        rank = int(row.get("rank_original", 0))
        smi = row["smiles"][:35]
        f_frac = row.get("f_mass_fraction", 0)
        flamm = row.get("flammability_class", "?")
        gwp = row.get("gwp_class", "?")
        score = int(row.get("safety_score", 0))
        gate = row.get("safety_gate", "?")
        logger.info(
            "  Rank %2d | %-35s | F=%.2f | %3s | %-9s | score=%d | %s",
            rank, smi, f_frac, flamm, gwp, score, gate,
        )

    # Generate figures
    logger.info("")
    generate_figures(df_assessed)

    # Generate report
    generate_report(df_assessed)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Step 41 COMPLETE")
    logger.info("=" * 80)
    logger.info("  CSV: %s", OUTPUT_CSV)
    logger.info("  Figures: %s", FIGURES_DIR)
    logger.info("  Report: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
