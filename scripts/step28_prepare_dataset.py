#!/usr/bin/env python3
"""
Step 28: Prepare publication-ready dataset from novel PC-SAFT predictions.

Loads model/saved/novel_pcsaft_predictions.csv (4,663 rows) and enriches with:
- InChI and InChIKey (via RDKit, if missing)
- IUPAC name attempt (from RDKit)
- backbone (from HFO-centric ranking if available)
- boiling_point_K (from HFO-centric ranking if available)
- hfo_distance (from HFO-centric ranking if available)
- cyc_distance (from novel predictions column: distance)
- n_carbon count

Saves to data/pcsaft_novel_predictions_v1.csv with header comments.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOVEL_PREDICTIONS = PROJECT_ROOT / "model/saved/novel_pcsaft_predictions.csv"
HFO_RANKED = PROJECT_ROOT / "screening/results/hfo_centric_ranked.csv"
SYSTEMATIC_RANKED = PROJECT_ROOT / "screening/results/ranked_candidates_systematic.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/pcsaft_novel_predictions_v1.csv"


def compute_inchi_inchikey(smiles: str) -> tuple[str, str]:
    """Compute InChI and InChIKey from SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ""
    inchi = Chem.MolToInchi(mol)
    inchikey = Chem.MolToInchiKey(mol)
    return inchi if inchi else "", inchikey if inchikey else ""


def get_iupac_name(smiles: str) -> str:
    """Attempt to get IUPAC name from RDKit (often returns empty)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    # RDKit doesn't have built-in IUPAC naming; return empty
    # In production, could call PubChem or other service
    return ""


def count_carbons(smiles: str) -> int:
    """Count number of carbon atoms in molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")


def main():
    print("Step 28: Preparing publication-ready dataset")
    print("=" * 60)

    # Load novel predictions
    print(f"\n1. Loading {NOVEL_PREDICTIONS}")
    df_novel = pd.read_csv(NOVEL_PREDICTIONS)
    print(f"   Loaded {len(df_novel)} novel predictions")
    print(f"   Columns: {list(df_novel.columns)}")

    # Load HFO-centric rankings (if available)
    df_hfo = None
    if HFO_RANKED.exists():
        print(f"\n2. Loading {HFO_RANKED}")
        df_hfo = pd.read_csv(HFO_RANKED)
        print(f"   Loaded {len(df_hfo)} HFO-centric rankings")
        print(f"   Columns: {list(df_hfo.columns)}")
    else:
        print(f"\n2. HFO-centric rankings not found at {HFO_RANKED}")

    # Load systematic rankings (if available)
    df_systematic = None
    if SYSTEMATIC_RANKED.exists():
        print(f"\n3. Loading {SYSTEMATIC_RANKED}")
        df_systematic = pd.read_csv(SYSTEMATIC_RANKED)
        print(f"   Loaded {len(df_systematic)} systematic rankings")
        print(f"   Columns: {list(df_systematic.columns)}")
    else:
        print(f"\n3. Systematic rankings not found at {SYSTEMATIC_RANKED}")

    # Start building publication dataset
    print("\n4. Building publication dataset")
    df_pub = df_novel.copy()

    # Use canonical_smiles as primary SMILES (it's already in the data)
    if "canonical_smiles" in df_pub.columns:
        df_pub["smiles"] = df_pub["canonical_smiles"]

    # InChI and InChIKey: already present in novel predictions, but ensure inchikey exists
    if "inchikey" not in df_pub.columns:
        print("   Computing InChIKey from InChI...")
        df_pub["inchikey"] = ""
        for idx, row in df_pub.iterrows():
            if pd.isna(row.get("inchi", "")) or row.get("inchi", "") == "":
                inchi, inchikey = compute_inchi_inchikey(row["smiles"])
                df_pub.at[idx, "inchi"] = inchi
                df_pub.at[idx, "inchikey"] = inchikey
            else:
                # Compute InChIKey from existing InChI
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol:
                    inchikey = Chem.MolToInchiKey(mol)
                    df_pub.at[idx, "inchikey"] = inchikey if inchikey else ""
    elif df_pub["inchikey"].isna().any() or (df_pub["inchikey"] == "").any():
        print("   Computing InChI/InChIKey for missing entries...")
        for idx, row in df_pub.iterrows():
            if pd.isna(row.get("inchikey", "")) or row.get("inchikey", "") == "":
                inchi, inchikey = compute_inchi_inchikey(row["smiles"])
                df_pub.at[idx, "inchi"] = inchi
                df_pub.at[idx, "inchikey"] = inchikey

    # IUPAC name (placeholder)
    print("   Adding IUPAC name placeholder...")
    df_pub["name"] = df_pub["smiles"].apply(get_iupac_name)

    # Rename molecule_class to mol_class
    if "molecule_class" in df_pub.columns:
        df_pub.rename(columns={"molecule_class": "mol_class"}, inplace=True)

    # Add backbone from HFO-centric if available
    if df_hfo is not None and "backbone" in df_hfo.columns:
        print("   Merging backbone from HFO-centric rankings...")
        df_pub = df_pub.merge(
            df_hfo[["smiles", "backbone"]],
            on="smiles",
            how="left"
        )
    else:
        df_pub["backbone"] = np.nan

    # Add boiling_point_K from HFO-centric if available
    if df_hfo is not None and "boiling_point_K" in df_hfo.columns:
        print("   Merging boiling_point_K from HFO-centric rankings...")
        df_pub = df_pub.merge(
            df_hfo[["smiles", "boiling_point_K"]],
            on="smiles",
            how="left",
            suffixes=("", "_hfo")
        )
        # Use HFO value if available
        if "boiling_point_K_hfo" in df_pub.columns:
            df_pub["boiling_point_K"] = df_pub["boiling_point_K_hfo"].fillna(
                df_pub.get("boiling_point_K", np.nan)
            )
            df_pub.drop(columns=["boiling_point_K_hfo"], inplace=True)
    elif "boiling_point_K" not in df_pub.columns:
        df_pub["boiling_point_K"] = np.nan

    # Add hfo_distance from HFO-centric if available
    if df_hfo is not None and "hfo_distance" in df_hfo.columns:
        print("   Merging hfo_distance from HFO-centric rankings...")
        df_pub = df_pub.merge(
            df_hfo[["smiles", "hfo_distance"]],
            on="smiles",
            how="left"
        )
    else:
        df_pub["hfo_distance"] = np.nan

    # Rename 'distance' to 'cyc_distance' (distance to cyclopentane)
    if "distance" in df_pub.columns:
        df_pub.rename(columns={"distance": "cyc_distance"}, inplace=True)

    # Add n_carbon count
    print("   Computing n_carbon counts...")
    df_pub["n_carbon"] = df_pub["smiles"].apply(count_carbons)

    # Rename std columns to match expected names
    if "std_m" in df_pub.columns:
        df_pub.rename(columns={
            "std_m": "m_std",
            "std_sigma": "sigma_std",
            "std_epsilon_k": "epsilon_k_std"
        }, inplace=True)

    # Rename tanimoto_max to tanimoto_nn
    if "tanimoto_max" in df_pub.columns:
        df_pub.rename(columns={"tanimoto_max": "tanimoto_nn"}, inplace=True)

    # Rename ad_flag to ad_in_domain (convert to boolean)
    # ad_flag values are "in_domain", "warning", "ood" (set in package_novel_predictions.py)
    if "ad_flag" in df_pub.columns:
        df_pub["ad_in_domain"] = df_pub["ad_flag"].apply(
            lambda x: x == "in_domain" if pd.notna(x) else False
        )

    # Add parameter_source (all are model-generated)
    df_pub["parameter_source"] = "model_predicted"

    # Add vapor pressure at 298K (placeholder - would need EOS calculation)
    if "vp_ratio_to_ref" not in df_pub.columns:
        df_pub["vp_298K_Pa"] = np.nan
    else:
        # If we have vp_ratio_to_ref, we could compute absolute VP
        # For now, leave as NaN or derive from ratio if reference is known
        df_pub["vp_298K_Pa"] = np.nan

    # Add density at 298K (placeholder - would need EOS calculation)
    df_pub["density_298K_mol_m3"] = np.nan

    # Add Henry's ratio (from H_ratio if available)
    if "H_ratio" in df_pub.columns:
        df_pub["henrys_ratio"] = df_pub["H_ratio"]
    else:
        df_pub["henrys_ratio"] = np.nan

    # Select and order final columns
    required_columns = [
        "smiles", "inchi", "inchikey", "name", "mol_class", "backbone",
        "mw", "n_fluorine", "n_chlorine", "n_carbon", "has_double_bond",
        "parameter_source", "m", "sigma", "epsilon_k",
        "m_std", "sigma_std", "epsilon_k_std",
        "tanimoto_nn", "ad_in_domain", "sa_score",
        "boiling_point_K", "vp_298K_Pa", "density_298K_mol_m3",
        "henrys_ratio", "hfo_distance", "cyc_distance"
    ]

    # Ensure all columns exist (fill missing with NaN)
    for col in required_columns:
        if col not in df_pub.columns:
            print(f"   WARNING: Column '{col}' not found, filling with NaN")
            df_pub[col] = np.nan

    # Select and reorder
    df_pub = df_pub[required_columns]

    # Remove duplicates by InChIKey (if any)
    print("\n5. Removing duplicate InChIKeys...")
    initial_count = len(df_pub)
    df_pub = df_pub.drop_duplicates(subset=["inchikey"], keep="first")
    final_count = len(df_pub)
    print(f"   Removed {initial_count - final_count} duplicates")
    print(f"   Final dataset: {final_count} molecules")

    # Save to CSV with header comment
    print(f"\n6. Saving to {OUTPUT_CSV}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w") as f:
        # Write header comment
        f.write("# PC-SAFT Novel Predictions Library v1.0\n")
        f.write("# This is a MODEL-GENERATED predictive library\n")
        f.write(
            "# PC-SAFT parameters (m, sigma, epsilon_k) are PREDICTED, "
            "not experimentally measured\n"
        )
        f.write(
            "# Thermodynamic properties are EOS-derived from predicted "
            "parameters\n"
        )
        f.write(
            "# Intended for screening and prioritization, NOT direct "
            "engineering use\n"
        )
        f.write(
            "# Consult experimental data or perform validation before "
            "production deployment\n"
        )
        f.write("#\n")

        # Write DataFrame
        df_pub.to_csv(f, index=False)

    print(f"   Saved {len(df_pub)} molecules to {OUTPUT_CSV}")

    # Summary statistics
    print("\n7. Dataset summary:")
    print(f"   Total molecules: {len(df_pub)}")
    print(f"   Molecules with boiling point: {df_pub['boiling_point_K'].notna().sum()}")
    print(f"   Molecules with backbone annotation: {df_pub['backbone'].notna().sum()}")
    print(f"   Molecules with HFO distance: {df_pub['hfo_distance'].notna().sum()}")
    print(f"   Molecules with cyc distance: {df_pub['cyc_distance'].notna().sum()}")
    print(f"   Molecules in applicability domain: {df_pub['ad_in_domain'].sum()}")
    print("\n   Molecular class distribution:")
    print(df_pub["mol_class"].value_counts())
    print("\n   Parameter ranges:")
    print(f"     m: {df_pub['m'].min():.3f} - {df_pub['m'].max():.3f}")
    print(f"     sigma (Å): {df_pub['sigma'].min():.3f} - {df_pub['sigma'].max():.3f}")
    print(f"     epsilon/k (K): {df_pub['epsilon_k'].min():.3f} - {df_pub['epsilon_k'].max():.3f}")

    print("\n" + "=" * 60)
    print("Step 28 dataset preparation complete!")


if __name__ == "__main__":
    main()
