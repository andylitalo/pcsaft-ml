"""Generate 5 figures for the ML chemistry narrative report.

Figure 1: epsilon_k vs n_fluorine scatter (colored by molecule class)
Figure 2: VP vs Henry's screening funnel (two-panel)
Figure 3: Henry's sensitivity to epsilon_ij (Boltzmann curve overlay)
Figure 4: Uncertainty envelope for standout candidates
Figure 5: Pareto frontier (H ratio vs n_fluorine)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_molecule(smiles: str) -> str:
    """Classify as HFO (F, no Cl), HCFO (F and Cl), or Cl-olefin (Cl, no F)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"

    n_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    n_cl = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)

    if n_f > 0 and n_cl == 0:
        return "HFO"
    elif n_f > 0 and n_cl > 0:
        return "HCFO"
    elif n_cl > 0 and n_f == 0:
        return "Cl-olefin"
    return "other"

REPO_ROOT = Path(__file__).parent.parent
SAVED_DIR = REPO_ROOT / "model" / "saved"
FIG_DIR = REPO_ROOT / "figures" / "22_ml_chemistry_narrative"

# Reference values (cyclopentane)
CYCLOPENTANE = {"m": 2.3655, "sigma": 3.7114, "epsilon_k": 288.84}

# Acceptance band for Henry's ratio
H_RATIO_MIN = 0.5
H_RATIO_MAX = 2.0


def load_data():
    """Load all required datasets."""
    logger.info("Loading datasets...")

    # Full candidate set with all metadata
    full_df = pd.read_csv(SAVED_DIR / "novel_pcsaft_predictions.csv")

    # VP-passing candidates with Henry's data
    henrys_df = pd.read_csv(SAVED_DIR / "henrys_constant_screening.csv")

    # Standout candidates with credibility analysis
    standout_df = pd.read_csv(SAVED_DIR / "standout_candidates_credibility.csv")

    logger.info(f"Loaded {len(full_df)} total candidates")
    logger.info(f"Loaded {len(henrys_df)} VP-passing candidates with Henry's data")
    logger.info(f"Loaded {len(standout_df)} standout candidates")

    return full_df, henrys_df, standout_df


def figure1_epsilon_k_vs_fluorines(full_df):
    """Figure 1: Scatter of epsilon_k vs n_fluorine, colored by molecule class."""
    logger.info("Generating Figure 1: epsilon_k vs n_fluorine...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color mapping
    class_colors = {
        "HFO": "#1f77b4",  # blue
        "HCFO": "#ff7f0e",  # orange
        "Cl-olefin": "#2ca02c",  # green
        "other": "#d62728",  # red
    }

    for mol_class in ["HFO", "HCFO", "Cl-olefin"]:
        subset = full_df[full_df["molecule_class"] == mol_class]
        ax.scatter(
            subset["n_fluorine"],
            subset["epsilon_k"],
            c=class_colors[mol_class],
            label=mol_class,
            alpha=0.6,
            s=20,
        )

    # Reference line for cyclopentane
    ax.axhline(
        CYCLOPENTANE["epsilon_k"],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Cyclopentane (ε/k = 288.84 K)",
    )

    # Annotate top candidates (e.g., highest epsilon_k among HFOs)
    hfos = full_df[full_df["molecule_class"] == "HFO"]
    if len(hfos) > 0:
        top_hfo = hfos.nlargest(1, "epsilon_k").iloc[0]
        ax.annotate(
            top_hfo["smiles"],
            xy=(top_hfo["n_fluorine"], top_hfo["epsilon_k"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    ax.set_xlabel("Number of Fluorine Atoms", fontsize=14)
    ax.set_ylabel("ε/k (K)", fontsize=14)
    ax.set_title("Predicted Dispersion Energy vs Fluorination Level", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "epsilon_k_vs_fluorines.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 1: epsilon_k_vs_fluorines.png")


def figure2_vp_vs_henrys_funnel(full_df, henrys_df):
    """Figure 2: Two-panel showing Henry's histogram and screening funnel."""
    logger.info("Generating Figure 2: VP vs Henry's screening funnel...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Henry's ratio histogram (log scale)
    h_ratios = henrys_df["H_ratio"].dropna()
    ax1.hist(h_ratios, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.axvspan(H_RATIO_MIN, H_RATIO_MAX, alpha=0.2, color="green", label="Acceptance Region")
    ax1.set_xlabel("H / H(cyclopentane)", fontsize=14)
    ax1.set_ylabel("Count", fontsize=14)
    ax1.set_title("Henry's Constant Distribution\n(VP-Passing Candidates)", fontsize=16)
    ax1.set_xscale("log")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Right panel: Screening funnel (bar chart)
    total_candidates = len(full_df)
    vp_passing = len(henrys_df)
    henry_passing = henrys_df[
        (henrys_df["H_ratio"] >= H_RATIO_MIN)
        & (henrys_df["H_ratio"] <= H_RATIO_MAX)
    ]
    n_henry_passing = len(henry_passing)

    stages = ["Total\nEnumerated", "VP-Passing\n(298K)", "Henry-Screened\n[0.5, 2.0]"]
    counts = [total_candidates, vp_passing, n_henry_passing]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    bars = ax2.bar(stages, counts, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Number of Candidates", fontsize=14)
    ax2.set_title("Screening Funnel", fontsize=16)
    ax2.tick_params(labelsize=12)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate bars with counts
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "vp_vs_henrys_comparison.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 2: vp_vs_henrys_comparison.png")


def figure3_henrys_sensitivity(standout_df):
    """Figure 3: H ratio vs epsilon_ij sensitivity with Boltzmann curve overlay."""
    logger.info("Generating Figure 3: Henry's sensitivity analysis...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute epsilon_ij / epsilon_ij(ref) from epsilon_k
    # For Henry's law, epsilon_ij = sqrt(epsilon_i * epsilon_j)
    # Assuming solvent is hexane with epsilon_k ~ 323 K (representative alkane)
    epsilon_hexane = 323.0
    epsilon_cyclopentane = CYCLOPENTANE["epsilon_k"]

    standout_df = standout_df.copy()
    standout_df["epsilon_ij"] = np.sqrt(standout_df["epsilon_k"] * epsilon_hexane)
    standout_df["epsilon_ij_ref"] = np.sqrt(epsilon_cyclopentane * epsilon_hexane)
    standout_df["epsilon_ij_ratio"] = standout_df["epsilon_ij"] / standout_df["epsilon_ij_ref"]

    # Scatter plot
    ax.scatter(
        standout_df["epsilon_ij_ratio"],
        standout_df["H_ratio"],
        alpha=0.6,
        s=30,
        c="steelblue",
        label="Standout Candidates",
    )

    # Theoretical Boltzmann curve: H ∝ exp(epsilon_ij / (kT))
    # H_ratio = exp((epsilon_ij - epsilon_ij_ref) / (kT))
    # At T=298K, kT = 0.592 kcal/mol = 0.025 eV = 298 K * k_B
    # In units of K, kT = 298 K
    epsilon_ij_ratios = np.linspace(0.7, 1.1, 100)
    # Use a simplified form: H_ratio ≈ exp(c * (epsilon_ij_ratio - 1))
    # where c is a fitted constant. For demonstration, use exponential relationship.
    # From Boltzmann: H ∝ exp(epsilon / kT)
    # H_ratio = exp((epsilon_ij - epsilon_ij_ref) / kT)
    # epsilon_ij = epsilon_ij_ratio * epsilon_ij_ref
    # So H_ratio = exp(epsilon_ij_ref * (epsilon_ij_ratio - 1) / kT)
    kT = 298  # K
    epsilon_ij_ref = standout_df["epsilon_ij_ref"].iloc[0]
    theoretical_H = np.exp(epsilon_ij_ref * (epsilon_ij_ratios - 1) / kT)

    ax.plot(
        epsilon_ij_ratios,
        theoretical_H,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical Boltzmann (exp[ε_ij/kT])",
    )

    # Acceptance band
    ax.axhspan(H_RATIO_MIN, H_RATIO_MAX, alpha=0.2, color="green", label="Acceptance Region")

    ax.set_xlabel("ε_ij / ε_ij(ref)", fontsize=14)
    ax.set_ylabel("H / H(cyclopentane)", fontsize=14)
    ax.set_title("Henry's Constant Sensitivity to Interaction Energy", fontsize=16)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "henrys_sensitivity.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 3: henrys_sensitivity.png")


def figure4_uncertainty_envelope(standout_df):
    """Figure 4: H ratio with error bars from sensitivity analysis."""
    logger.info("Generating Figure 4: Uncertainty envelope for standout candidates...")

    # Sort by H_ratio
    standout_df = standout_df.sort_values("H_ratio").reset_index(drop=True)

    # For plotting, show top 20 candidates
    plot_df = standout_df.head(20)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Error bars from epsilon sensitivity (convert to H ratio uncertainty)
    # H_eps_plus and H_eps_minus are in Pa, need to convert to ratios
    H_ref = 51676.0  # Pa, cyclopentane reference
    if "H_eps_plus" in plot_df.columns and "H_eps_minus" in plot_df.columns:
        H_ratio_plus = plot_df["H_eps_plus"] / H_ref
        H_ratio_minus = plot_df["H_eps_minus"] / H_ref
        y_err_upper = np.abs(H_ratio_plus - plot_df["H_ratio"])
        y_err_lower = np.abs(plot_df["H_ratio"] - H_ratio_minus)
        yerr = [y_err_lower.values, y_err_upper.values]
    else:
        yerr = None

    x_pos = np.arange(len(plot_df))
    ax.errorbar(
        x_pos,
        plot_df["H_ratio"],
        yerr=yerr,
        fmt="o",
        markersize=6,
        color="steelblue",
        ecolor="gray",
        capsize=4,
        alpha=0.7,
    )

    # Acceptance band
    ax.axhspan(H_RATIO_MIN, H_RATIO_MAX, alpha=0.2, color="green", label="Acceptance Region")

    # Labels
    ax.set_xlabel("Candidate Index (sorted by H ratio)", fontsize=14)
    ax.set_ylabel("H / H(cyclopentane)", fontsize=14)
    ax.set_title("Top 20 Candidates with Uncertainty from ε Sensitivity", fontsize=16)
    ax.set_xticks(x_pos[::2])  # Show every other tick
    ax.set_xticklabels(x_pos[::2])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "uncertainty_envelope.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 4: uncertainty_envelope.png")


def figure5_pareto_frontier(henrys_df):
    """Figure 5: Pareto frontier |H_ratio - 1| vs n_fluorine."""
    logger.info("Generating Figure 5: Pareto frontier...")

    henrys_df = henrys_df.copy()

    # Compute deviation from ideal
    henrys_df["h_deviation"] = np.abs(henrys_df["H_ratio"] - 1.0)

    # Classify molecules
    henrys_df["molecule_class"] = henrys_df["smiles"].apply(classify_molecule)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by class
    class_colors = {
        "HFO": "#1f77b4",
        "HCFO": "#ff7f0e",
        "Cl-olefin": "#2ca02c",
        "other": "#d62728",
    }

    for mol_class in ["HFO", "HCFO", "Cl-olefin"]:
        subset = henrys_df[henrys_df["molecule_class"] == mol_class]
        if len(subset) > 0:
            # Need to compute n_fluorine
            n_f_list = []
            for smi in subset["smiles"]:
                mol = Chem.MolFromSmiles(smi)
                n_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9) if mol else 0
                n_f_list.append(n_f)
            subset = subset.copy()
            subset["n_fluorine"] = n_f_list

            ax.scatter(
                subset["n_fluorine"],
                subset["h_deviation"],
                c=class_colors[mol_class],
                label=mol_class,
                alpha=0.6,
                s=30,
            )

    ax.set_xlabel("Number of Fluorine Atoms", fontsize=14)
    ax.set_ylabel("|H / H(ref) - 1|", fontsize=14)
    ax.set_title("Pareto Frontier: Solubility Match vs Fluorination", fontsize=16)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=12)

    # Annotate: Lower-right is Pareto-optimal (high F, low deviation)
    ax.text(
        0.95,
        0.05,
        "Pareto-optimal\n(lower-right)",
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "pareto_frontier.png", dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 5: pareto_frontier.png")


def main():
    full_df, henrys_df, standout_df = load_data()

    figure1_epsilon_k_vs_fluorines(full_df)
    figure2_vp_vs_henrys_funnel(full_df, henrys_df)
    figure3_henrys_sensitivity(standout_df)
    figure4_uncertainty_envelope(standout_df)
    figure5_pareto_frontier(henrys_df)

    logger.info(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
