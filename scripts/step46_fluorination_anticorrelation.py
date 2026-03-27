"""Step 46: Cross-class fluorination context in the Esper dataset.

Tests whether the fluorination-lowers-epsilon_k pattern observed in the
halogenated olefin screen also appears across a broader set of carbon-
containing Esper molecules. This is supporting context for the project
narrative, not a proof of a universal physical law.

Fluorination axis: fluorine mass fraction (consistent with Step 41 ASHRAE
heuristic). The F/(F+H) ratio is reported as a secondary descriptor only.

Produces:
  figures/46_universal_anticorrelation/epsk_vs_fluorination.png
  figures/46_universal_anticorrelation/class_comparison_bars.png
  model/saved/step46_esper_classified.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdAll")

DATA_PATH = Path("model/data/esper_pcsaft.csv")
FIG_DIR = Path("figures/46_universal_anticorrelation")
SAVE_CSV = Path("model/saved/step46_esper_classified.csv")

CYCLOPENTANE_EK = 288.8

# Step 41 ASHRAE heuristic boundaries (fluorine mass fraction)
A1_THRESHOLD = 0.65
A2L_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Filtering and classification
# ---------------------------------------------------------------------------

def has_carbon(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())


def fluorine_mass_fraction(smiles: str) -> float:
    """Mass of fluorine atoms / total molecular weight."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    mw = Descriptors.MolWt(mol)
    if mw == 0:
        return 0.0
    return (n_f * 18.998) / mw


def f_over_fh(smiles: str) -> float:
    """F/(F+H) ratio — secondary descriptor only."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    mol = Chem.AddHs(mol)
    n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    n_h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    if n_f + n_h == 0:
        return 0.0
    return n_f / (n_f + n_h)


def ashrae_class(f_mass_frac: float) -> str:
    if f_mass_frac >= A1_THRESHOLD:
        return "A1"
    elif f_mass_frac >= A2L_THRESHOLD:
        return "A2L"
    elif f_mass_frac >= 0.30:
        return "A2"
    else:
        return "A3"


def classify_molecule(smiles: str) -> str:
    """Coarse functional-class label (descriptive bin, not authoritative)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"

    atom_nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
    has_f = 9 in atom_nums
    has_cl = 17 in atom_nums
    has_br = 35 in atom_nums
    has_o = 8 in atom_nums
    has_n = 7 in atom_nums
    has_s = 16 in atom_nums
    has_si = 14 in atom_nums

    if has_si:
        return "organosilicon"
    if has_s:
        return "sulfur-containing"
    if has_n:
        return "nitrogen-containing"
    if has_f and has_o:
        return "fluoroether/fluoro-oxy"
    if has_f:
        return "fluorinated (no O)"
    if has_cl or has_br:
        return "chloro/bromo"
    if has_o:
        return "oxygenated (no halogen)"

    is_aromatic = any(mol.GetAtomWithIdx(i).GetIsAromatic()
                      for i in range(mol.GetNumAtoms()))
    if is_aromatic:
        return "aromatic HC"
    return "aliphatic HC"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "aliphatic HC": "#2c7bb6",
    "aromatic HC": "#abd9e9",
    "oxygenated (no halogen)": "#fdae61",
    "fluoroether/fluoro-oxy": "#d7191c",
    "fluorinated (no O)": "#d73027",
    "chloro/bromo": "#1a9641",
    "nitrogen-containing": "#b2abd2",
    "sulfur-containing": "#8073ac",
    "organosilicon": "#542788",
    "invalid": "#999999",
}


def plot_scatter(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for cls in df["mol_class"].unique():
        sub = df[df["mol_class"] == cls]
        color = CLASS_COLORS.get(cls, "#999999")
        ax.scatter(
            sub["f_mass_frac"], sub["epsilon_k"],
            c=color, label=f"{cls} (n={len(sub)})",
            alpha=0.55, s=22, edgecolors="none",
        )

    ax.axhline(CYCLOPENTANE_EK, color="black", ls="--", lw=1.2,
               label=f"cyclopentane ε/k = {CYCLOPENTANE_EK} K")
    ax.axvline(A2L_THRESHOLD, color="grey", ls=":", lw=1,
               label=f"A2L heuristic (F mass frac = {A2L_THRESHOLD})")
    ax.axvline(A1_THRESHOLD, color="grey", ls="-.", lw=1,
               label=f"A1 heuristic (F mass frac = {A1_THRESHOLD})")

    # Light shading for A2L+ region above cyclopentane ε/k
    ax.fill_between(
        [A2L_THRESHOLD, 1.02], CYCLOPENTANE_EK - 20,
        df["epsilon_k"].max() + 20,
        color="red", alpha=0.05,
    )
    ax.text(
        0.78, CYCLOPENTANE_EK + 12,
        "A2L+ & cyclopentane-like ε/k\n(sparsely populated — see triage)",
        ha="center", va="bottom", fontsize=8, color="#a00000", style="italic",
    )

    ax.set_xlabel("Fluorine mass fraction", fontsize=12)
    ax.set_ylabel("ε/k  (K)", fontsize=12)
    ax.set_title(
        "Dispersion energy vs fluorine mass fraction\n"
        f"Esper dataset — {len(df)} carbon-containing molecules, "
        "coarse functional-class labels",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(50, 570)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, ncol=2, loc="upper right",
              framealpha=0.85, borderpad=0.8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_class_bars(df: pd.DataFrame, outpath: Path) -> None:
    """Paired bar chart: base class mean ε/k vs fluorinated counterpart."""
    pairs = [
        ("aliphatic HC", "fluorinated (no O)",
         "Aliphatic HC", "Fluorinated HC"),
        ("aromatic HC", "fluorinated (no O)",
         "Aromatic HC", "Fluoroaromatic*"),
        ("oxygenated (no halogen)", "fluoroether/fluoro-oxy",
         "Oxygenated (no F)", "Fluoro-oxygenated"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = []
    labels = []
    colors = []
    values = []
    counts = []

    pos = 0
    for cls_base, cls_fluor, base_label, fluor_label in pairs:
        base_data = df[df["mol_class"] == cls_base]
        fluor_data = df[df["mol_class"] == cls_fluor]

        # For aromatic pair, filter fluorinated to aromatics only
        if "Aromatic" in base_label:
            fluor_data = fluor_data[fluor_data["smiles"].apply(_is_aromatic)]
            if len(fluor_data) < 5:
                fluor_data = df[df["mol_class"] == cls_fluor]

        for sub, color, label in [
            (base_data, "#2c7bb6", base_label),
            (fluor_data, "#d73027", fluor_label),
        ]:
            if len(sub) == 0:
                continue
            x_pos.append(pos)
            labels.append(label)
            colors.append(color)
            values.append(sub["epsilon_k"].mean())
            counts.append(len(sub))
            pos += 1
        pos += 0.5

    bars = ax.bar(x_pos, values, color=colors, width=0.8,
                  edgecolor="white", linewidth=0.5)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="#555")

    ax.axhline(CYCLOPENTANE_EK, color="black", ls="--", lw=1, alpha=0.7)
    ax.text(len(x_pos) - 0.5, CYCLOPENTANE_EK + 4,
            f"cyclopentane ({CYCLOPENTANE_EK} K)",
            fontsize=8, ha="right", va="bottom", color="#333")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Mean ε/k  (K)", fontsize=11)
    ax.set_title(
        "Mean ε/k by coarse class — base vs fluorinated counterpart\n"
        "(descriptive comparison; subgroup counts shown)",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(0, max(values) + 40)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def _is_aromatic(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(mol.GetAtomWithIdx(i).GetIsAromatic()
               for i in range(mol.GetNumAtoms()))


# ---------------------------------------------------------------------------
# Tabular analysis
# ---------------------------------------------------------------------------

def fluorination_gradient_table(df: pd.DataFrame) -> str:
    bins = [
        (0.0, 0.001, "No fluorine"),
        (0.001, 0.30, "Low F mass frac (<0.30)"),
        (0.30, 0.50, "Moderate (0.30–0.50, A2 heuristic)"),
        (0.50, 0.65, "Elevated (0.50–0.65, A2L heuristic)"),
        (0.65, 1.01, "High (≥0.65, A1 heuristic)"),
    ]
    lines = [
        "| Fluorine mass fraction | ASHRAE heuristic | N | Mean ε/k (K) | Median | Max |",
        "|---|---|---|---|---|---|",
    ]
    for lo, hi, label in bins:
        sub = df[(df["f_mass_frac"] >= lo) & (df["f_mass_frac"] < hi)]
        if len(sub) == 0:
            continue
        ashrae = "A3" if hi <= 0.30 else ("A2" if hi <= 0.50 else ("A2L" if hi <= 0.65 else "A1"))
        lines.append(
            f"| {label} | {ashrae} | {len(sub)} | "
            f"{sub['epsilon_k'].mean():.1f} | {sub['epsilon_k'].median():.1f} | "
            f"{sub['epsilon_k'].max():.1f} |"
        )
    return "\n".join(lines)


def class_pair_table(df: pd.DataFrame) -> str:
    lines = [
        "| Class pair | N (base) | Mean ε/k (base) | N (fluor.) | Mean ε/k (fluor.) | Δ ε/k |",
        "|---|---|---|---|---|---|",
    ]
    for base, fluor, label in [
        ("aliphatic HC", "fluorinated (no O)", "Aliphatic HC → Fluorinated"),
        ("oxygenated (no halogen)", "fluoroether/fluoro-oxy", "Oxygenated → Fluoro-oxygenated"),
    ]:
        b = df[df["mol_class"] == base]
        f = df[df["mol_class"] == fluor]
        if len(b) == 0 or len(f) == 0:
            continue
        delta = f["epsilon_k"].mean() - b["epsilon_k"].mean()
        lines.append(
            f"| {label} | {len(b)} | {b['epsilon_k'].mean():.1f} | "
            f"{len(f)} | {f['epsilon_k'].mean():.1f} | {delta:+.1f} |"
        )
    return "\n".join(lines)


def triage_hits(df: pd.DataFrame) -> str:
    """List and manually classify all molecules with high ε/k AND A2L+ fluorination."""
    high_ek = df["epsilon_k"] >= (CYCLOPENTANE_EK - 20)  # ≥269 K
    a2l_plus = df["f_mass_frac"] >= A2L_THRESHOLD

    hits = df[high_ek & a2l_plus].sort_values("epsilon_k", ascending=False)

    lines = [
        f"\n=== TRIAGE: molecules with ε/k ≥ 269 K AND F mass frac ≥ {A2L_THRESHOLD} ===",
        f"Total hits: {len(hits)}\n",
    ]
    if len(hits) == 0:
        lines.append("No molecules found in this region.")
        return "\n".join(lines)

    lines.append(
        f"{'ε/k':>7} {'F mass%':>8} {'ASHRAE':>7} {'MW':>7} "
        f"{'Class':<25} {'SMILES':<40} Relevance"
    )
    lines.append("-" * 130)
    for _, row in hits.iterrows():
        lines.append(
            f"{row['epsilon_k']:>7.1f} {row['f_mass_frac']:>8.2f} "
            f"{row['ashrae']:>7} {row['mw']:>7.1f} "
            f"{row['mol_class']:<25} {row['smiles']:<40} "
            f"(see manual triage)"
        )
    return "\n".join(lines)


def quadrant_counts(df: pd.DataFrame) -> str:
    high_ek = df["epsilon_k"] >= (CYCLOPENTANE_EK - 20)
    a2l_plus = df["f_mass_frac"] >= A2L_THRESHOLD
    a1_plus = df["f_mass_frac"] >= A1_THRESHOLD

    flam_hi = (high_ek & ~a2l_plus).sum()
    a2l_hi = (high_ek & a2l_plus).sum()
    a1_hi = (high_ek & a1_plus).sum()
    a2l_lo = (~high_ek & a2l_plus).sum()
    flam_lo = (~high_ek & ~a2l_plus).sum()
    lines = [
        "\n=== QUADRANT COUNTS ===\n",
        f"  ε/k ≥ 269 K, F < {A2L_THRESHOLD} (flammable, high ε/k): {flam_hi}",
        f"  ε/k ≥ 269 K, F ≥ {A2L_THRESHOLD} (A2L+, high ε/k):     {a2l_hi}",
        f"  ε/k ≥ 269 K, F ≥ {A1_THRESHOLD} (A1, high ε/k):       {a1_hi}",
        f"  ε/k < 269 K, F ≥ {A2L_THRESHOLD} (A2L+, low ε/k):      {a2l_lo}",
        f"  ε/k < 269 K, F < {A2L_THRESHOLD} (flammable, low ε/k):  {flam_lo}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(raw)} molecules from {DATA_PATH}")

    # Filter to carbon-containing species
    carbon_mask = raw["smiles"].apply(has_carbon)
    df = raw[carbon_mask].copy()
    n_excluded = len(raw) - len(df)
    print(f"Excluded {n_excluded} non-carbon species; {len(df)} molecules retained")

    # Compute descriptors
    df["mol_class"] = df["smiles"].apply(classify_molecule)
    df["f_mass_frac"] = df["smiles"].apply(fluorine_mass_fraction)
    df["f_over_fh"] = df["smiles"].apply(f_over_fh)
    df["mw"] = df["smiles"].apply(
        lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else np.nan
    )
    df["ashrae"] = df["f_mass_frac"].apply(ashrae_class)

    df.to_csv(SAVE_CSV, index=False)
    print(f"Saved classified data to {SAVE_CSV}")

    # Class distribution
    print(f"\n=== CLASS DISTRIBUTION ({len(df)} carbon-containing molecules) ===\n")
    for cls, grp in df.groupby("mol_class"):
        print(f"  {cls:<30} n={len(grp):>4}  mean ε/k={grp['epsilon_k'].mean():>6.1f}")

    # Fluorination gradient
    print("\n=== FLUORINATION GRADIENT ===\n")
    print(fluorination_gradient_table(df))

    # Class-pair comparison
    print("\n=== CLASS-PAIR COMPARISON ===\n")
    print(class_pair_table(df))

    # Quadrant counts
    print(quadrant_counts(df))

    # Triage hits
    print(triage_hits(df))

    # Bottom-line: candidate-like subset
    candidate_like = df[(df["mw"] >= 50) & (df["mw"] <= 200)]
    print("\n=== CANDIDATE-LIKE SUBSET (MW 50–200 Da, carbon-containing) ===")
    print(f"Total: {len(candidate_like)}")
    high_ek_cl = candidate_like[candidate_like["epsilon_k"] >= 270]
    print(f"With ε/k ≥ 270 K: {len(high_ek_cl)}")
    a2l_cl = high_ek_cl[high_ek_cl["f_mass_frac"] >= A2L_THRESHOLD]
    a1_cl = high_ek_cl[high_ek_cl["f_mass_frac"] >= A1_THRESHOLD]
    print(f"  of which A2L+ (F mass frac ≥ {A2L_THRESHOLD}): {len(a2l_cl)}")
    print(f"  of which A1   (F mass frac ≥ {A1_THRESHOLD}): {len(a1_cl)}")
    if len(a2l_cl) > 0:
        print("\n  These molecules:")
        for _, row in a2l_cl.sort_values("epsilon_k", ascending=False).iterrows():
            print(f"    ε/k={row['epsilon_k']:.1f}  F_mass={row['f_mass_frac']:.2f}  "
                  f"MW={row['mw']:.1f}  {row['ashrae']}  {row['mol_class']}  {row['smiles']}")

    # Generate figures
    plot_scatter(df, FIG_DIR / "epsk_vs_fluorination.png")
    plot_class_bars(df, FIG_DIR / "class_comparison_bars.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
