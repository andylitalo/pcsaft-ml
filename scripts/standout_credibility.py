#!/usr/bin/env python3
"""Uncertainty and applicability domain for standout candidates (Step 22.5)."""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from model.thermodynamic import HEXANE, compute_henrys_constant

# Paths
DATA_DIR = Path(__file__).parent.parent / "model" / "saved"
TRAIN_DATA = Path(__file__).parent.parent / "model" / "data" / "esper_pcsaft.csv"
INPUT_CSV = DATA_DIR / "henrys_constant_screening.csv"
OUTPUT_CSV = DATA_DIR / "standout_candidates_credibility.csv"
RF_MODEL_PATH = DATA_DIR / "random_forest_multitask.pkl"


def compute_morgan_fingerprint(smiles, radius=2, nBits=2048):
    """Compute Morgan fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)


def tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints."""
    if fp1 is None or fp2 is None:
        return np.nan
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    return intersection / union if union > 0 else 0.0


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    train_df = pd.read_csv(TRAIN_DATA)

    # Filter for Henry-screened candidates (0.5 <= H/H(ref) <= 2.0)
    henry_screened = df[(df["H_ratio"] >= 0.5) & (df["H_ratio"] <= 2.0)].copy()
    print(f"Henry-screened candidates: {len(henry_screened)}")

    # Compute Morgan fingerprints for training set
    print("\nComputing training set fingerprints...")
    train_fps = []
    train_smiles = []
    for smiles in train_df["smiles"]:
        fp = compute_morgan_fingerprint(smiles)
        if fp is not None:
            train_fps.append(fp)
            train_smiles.append(smiles)
    print(f"Training set: {len(train_fps)} molecules")

    # Compute Tanimoto similarity to nearest training molecule
    print("\nComputing Tanimoto similarities...")
    max_similarities = []
    for smiles in henry_screened["smiles"]:
        fp = compute_morgan_fingerprint(smiles)
        if fp is None:
            max_similarities.append(np.nan)
            continue

        similarities = [tanimoto_similarity(fp, train_fp) for train_fp in train_fps]
        max_similarities.append(max(similarities) if similarities else np.nan)

    henry_screened["tanimoto_to_nearest_train"] = max_similarities

    # Load RF model and compute per-tree uncertainty
    print("\nComputing RF tree disagreement...")
    try:
        import joblib

        rf_model = joblib.load(RF_MODEL_PATH)

        # Get per-tree predictions
        from model.descriptors import compute_morgan_fingerprints

        X_candidates = compute_morgan_fingerprints(
            henry_screened["smiles"].tolist(), radius=2, n_bits=2048
        )

        # Per-tree predictions for epsilon_k (most important parameter)
        tree_predictions = np.array(
            [tree.predict(X_candidates)[:, 2] for tree in rf_model.estimators_]
        )
        epsilon_k_std = np.std(tree_predictions, axis=0)

        henry_screened["epsilon_k_tree_std"] = epsilon_k_std
        print(f"Mean tree std(epsilon_k): {epsilon_k_std.mean():.2f} K")
    except Exception as e:
        print(f"Could not load RF model for uncertainty: {e}")
        henry_screened["epsilon_k_tree_std"] = np.nan

    # Sensitivity analysis: epsilon_k ± 1*std
    print("\nComputing Henry's constant sensitivity to epsilon_k...")
    H_nominal = []
    H_eps_plus = []
    H_eps_minus = []

    for idx, row in henry_screened.iterrows():
        if idx % 50 == 0:
            print(f"  Processing {idx}/{len(henry_screened)}...")

        params_nominal = {
            "m": row["m"],
            "sigma": row["sigma"],
            "epsilon_k": row["epsilon_k"],
        }

        eps_k_std = (
            row["epsilon_k_tree_std"]
            if not np.isnan(row["epsilon_k_tree_std"])
            else 10.0
        )
        params_plus = {
            "m": row["m"],
            "sigma": row["sigma"],
            "epsilon_k": row["epsilon_k"] + eps_k_std,
        }
        params_minus = {
            "m": row["m"],
            "sigma": row["sigma"],
            "epsilon_k": row["epsilon_k"] - eps_k_std,
        }

        result_nominal = compute_henrys_constant(params_nominal, HEXANE, T=298.15)
        result_plus = compute_henrys_constant(params_plus, HEXANE, T=298.15)
        result_minus = compute_henrys_constant(params_minus, HEXANE, T=298.15)

        H_nominal.append(result_nominal["H_Pa"] if result_nominal else np.nan)
        H_eps_plus.append(result_plus["H_Pa"] if result_plus else np.nan)
        H_eps_minus.append(result_minus["H_Pa"] if result_minus else np.nan)

    henry_screened["H_eps_plus"] = H_eps_plus
    henry_screened["H_eps_minus"] = H_eps_minus
    henry_screened["H_sensitivity_pct"] = (
        200
        * abs(np.array(H_eps_plus) - np.array(H_eps_minus))
        / (np.array(H_eps_plus) + np.array(H_eps_minus))
    )

    # k_ij sensitivity
    print("\nComputing Henry's constant sensitivity to k_ij...")
    H_kij_plus = []
    H_kij_minus = []

    for idx, row in henry_screened.iterrows():
        if idx % 50 == 0:
            print(f"  Processing {idx}/{len(henry_screened)}...")

        params = {"m": row["m"], "sigma": row["sigma"], "epsilon_k": row["epsilon_k"]}

        result_plus = compute_henrys_constant(params, HEXANE, T=298.15, k_ij=0.05)
        result_minus = compute_henrys_constant(params, HEXANE, T=298.15, k_ij=-0.05)

        H_kij_plus.append(result_plus["H_Pa"] if result_plus else np.nan)
        H_kij_minus.append(result_minus["H_Pa"] if result_minus else np.nan)

    henry_screened["H_kij_plus"] = H_kij_plus
    henry_screened["H_kij_minus"] = H_kij_minus
    henry_screened["H_kij_sensitivity_pct"] = (
        200
        * abs(np.array(H_kij_plus) - np.array(H_kij_minus))
        / (np.array(H_kij_plus) + np.array(H_kij_minus))
    )

    # Save results
    henry_screened.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Summary statistics
    print("\n=== Credibility Summary ===")
    print("Tanimoto similarity to training set:")
    print(f"  Min: {henry_screened['tanimoto_to_nearest_train'].min():.3f}")
    print(f"  Median: {henry_screened['tanimoto_to_nearest_train'].median():.3f}")
    print(f"  Max: {henry_screened['tanimoto_to_nearest_train'].max():.3f}")

    print("\nRF tree std(epsilon_k):")
    print(f"  Min: {henry_screened['epsilon_k_tree_std'].min():.2f} K")
    print(f"  Median: {henry_screened['epsilon_k_tree_std'].median():.2f} K")
    print(f"  Max: {henry_screened['epsilon_k_tree_std'].max():.2f} K")

    print("\nH sensitivity to epsilon_k ± 1 std:")
    print(f"  Median: {henry_screened['H_sensitivity_pct'].median():.1f}%")
    print(f"  Max: {henry_screened['H_sensitivity_pct'].max():.1f}%")

    print("\nH sensitivity to k_ij ± 0.05:")
    print(f"  Median: {henry_screened['H_kij_sensitivity_pct'].median():.1f}%")
    print(f"  Max: {henry_screened['H_kij_sensitivity_pct'].max():.1f}%")

    # Top candidates by credibility
    print("\n=== Top 10 Candidates (by Tanimoto similarity) ===")
    top_credible = henry_screened.nsmallest(10, "H_ratio").copy()
    top_credible = top_credible.sort_values("tanimoto_to_nearest_train", ascending=False)
    print(
        top_credible[
            [
                "smiles",
                "H_ratio",
                "tanimoto_to_nearest_train",
                "epsilon_k_tree_std",
                "H_sensitivity_pct",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
