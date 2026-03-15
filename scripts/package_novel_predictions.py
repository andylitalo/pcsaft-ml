"""Package novel PC-SAFT predictions with uncertainty, AD, and screening metadata.

Assembles all 4,663 candidates from systematic enumeration with:
- Identity: SMILES, InChI
- Predicted PC-SAFT: m, sigma, epsilon_k
- Uncertainty: std_m, std_sigma, std_epsilon_k (from RF tree variance)
- AD: tanimoto_max, ad_flag
- Screening: param_distance, vp_ratio_298K, H_ratio
- Practical: MW, n_fluorine, n_chlorine, has_double_bond, sa_score
- Molecule class: HFO / HCFO / Cl-olefin

Also validates predictions against SPT-PCSAFT overlap.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, inchi

from model.registry import _compute_features, get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "model" / "data"
SAVED_DIR = REPO_ROOT / "model" / "saved"
SCREENING_DIR = REPO_ROOT / "screening" / "results"

# Reference values (cyclopentane)
CYCLOPENTANE = {"m": 2.3655, "sigma": 3.7114, "epsilon_k": 288.84}


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def compute_tanimoto_similarity(smiles_list: list[str], train_smiles: list[str]):
    """Compute max Tanimoto similarity to training set using Morgan FPs.

    Returns array of max similarity scores (one per query molecule).
    """
    from rdkit.Chem import AllChem

    # Build Morgan fingerprints for training set
    train_fps = []
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            train_fps.append(fp)

    logger.info(f"Built {len(train_fps)} training fingerprints")

    # Compute max similarity for each query
    max_sims = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            max_sims.append(np.nan)
            continue
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        sims = [DataStructs.TanimotoSimilarity(query_fp, tfp) for tfp in train_fps]
        max_sims.append(max(sims) if sims else 0.0)

    return np.array(max_sims)


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


def compute_practical_features(smiles: str) -> dict:
    """Return MW, n_fluorine, n_chlorine, has_double_bond."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "mw": np.nan,
            "n_fluorine": 0,
            "n_chlorine": 0,
            "has_double_bond": False,
        }

    n_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    n_cl = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)

    # Check for C=C double bond
    has_db = any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in mol.GetBonds()
    )

    return {
        "mw": Descriptors.MolWt(mol),
        "n_fluorine": n_f,
        "n_chlorine": n_cl,
        "has_double_bond": has_db,
    }


def main():
    logger.info("Loading candidate data...")

    # 1. Load base candidates
    ranked_df = pd.read_csv(SCREENING_DIR / "ranked_candidates_systematic.csv")
    logger.info(f"Loaded {len(ranked_df)} candidates from ranked_candidates_systematic.csv")

    # 2. Load thermodynamic validation results
    thermo_df = pd.read_csv(SAVED_DIR / "thermo_validation_systematic.csv")
    logger.info(f"Loaded {len(thermo_df)} rows from thermo_validation_systematic.csv")

    # 3. Load Henry's constant results
    henrys_df = pd.read_csv(SAVED_DIR / "henrys_constant_screening.csv")
    logger.info(f"Loaded {len(henrys_df)} rows from henrys_constant_screening.csv")

    # Merge
    df = ranked_df.merge(
        thermo_df[["smiles", "vp_ratio_to_ref"]],
        on="smiles",
        how="left",
    )
    df = df.merge(
        henrys_df[["smiles", "H_ratio"]],
        on="smiles",
        how="left",
    )

    logger.info(f"Merged data: {len(df)} rows")

    # 4. Compute InChI
    logger.info("Computing InChI identifiers...")
    df["inchi"] = df["smiles"].apply(
        lambda s: inchi.MolToInchi(Chem.MolFromSmiles(s))
        if Chem.MolFromSmiles(s)
        else None
    )

    # 5. Get RF uncertainty estimates
    logger.info("Loading RF model for uncertainty estimates...")
    rf_model = get_model("rf")
    rf_model.load()

    smiles_list = df["smiles"].tolist()
    X = _compute_features(smiles_list)

    # Get per-tree predictions for each parameter
    logger.info("Computing RF tree variance...")
    for target in ["m", "sigma", "epsilon_k"]:
        model = rf_model._models[target]
        tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
        stds = tree_preds.std(axis=0)
        df[f"std_{target}"] = stds

    # 6. Compute Tanimoto similarity to training set
    logger.info("Loading training set for Tanimoto similarity...")
    esper_df = pd.read_csv(DATA_DIR / "esper_pcsaft.csv")
    train_smiles = esper_df["smiles"].tolist()

    logger.info("Computing Tanimoto similarity to nearest training molecule...")
    tanimoto_scores = compute_tanimoto_similarity(smiles_list, train_smiles)
    df["tanimoto_max"] = tanimoto_scores

    # AD flag: >= 0.4 "in_domain", [0.3, 0.4) "warning", < 0.3 "ood"
    df["ad_flag"] = pd.cut(
        df["tanimoto_max"],
        bins=[-np.inf, 0.3, 0.4, np.inf],
        labels=["ood", "warning", "in_domain"],
    )

    # 7. Compute practical features
    logger.info("Computing practical features...")
    practical = df["smiles"].apply(compute_practical_features).apply(pd.Series)
    df = pd.concat([df, practical], axis=1)

    # 8. Classify molecules
    logger.info("Classifying molecule types...")
    df["molecule_class"] = df["smiles"].apply(classify_molecule)

    # 9. Cross-reference with SPT-PCSAFT
    logger.info("Cross-referencing with SPT-PCSAFT dataset...")
    spt_df = pd.read_csv(DATA_DIR / "spt_pcsaft.csv", skiprows=1)

    # Canonicalize SMILES in SPT-PCSAFT
    spt_df["canonical_smiles"] = spt_df["SMILES0"].apply(canonicalize_smiles)
    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)

    # Mark which candidates are in SPT-PCSAFT
    df["in_spt_pcsaft"] = df["canonical_smiles"].isin(spt_df["canonical_smiles"])

    n_overlap = df["in_spt_pcsaft"].sum()
    n_novel = (~df["in_spt_pcsaft"]).sum()
    logger.info(f"Overlap with SPT-PCSAFT: {n_overlap} molecules")
    logger.info(f"Novel predictions (not in SPT-PCSAFT): {n_novel} molecules")

    # 10. Reorder columns for clarity
    col_order = [
        # Identity
        "smiles",
        "inchi",
        "canonical_smiles",
        # Predictions
        "m",
        "sigma",
        "epsilon_k",
        # Uncertainty
        "std_m",
        "std_sigma",
        "std_epsilon_k",
        # AD
        "tanimoto_max",
        "ad_flag",
        # Screening
        "distance",
        "vp_ratio_to_ref",
        "H_ratio",
        # Practical
        "mw",
        "n_fluorine",
        "n_chlorine",
        "has_double_bond",
        "sa_score",
        "molecule_class",
        # Metadata
        "is_associating",
        "in_spt_pcsaft",
    ]

    df = df[col_order]

    # 11. Save
    out_path = SAVED_DIR / "novel_pcsaft_predictions.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} predictions to {out_path}")

    # 12. Validate against SPT-PCSAFT overlap
    if n_overlap > 0:
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION AGAINST SPT-PCSAFT OVERLAP")
        logger.info("=" * 60)

        overlap_df = df[df["in_spt_pcsaft"]].copy()

        # Merge with SPT-PCSAFT values
        spt_subset = spt_df[["canonical_smiles", "m", "sigma", "epsilon_k"]].rename(
            columns={
                "m": "m_spt",
                "sigma": "sigma_spt",
                "epsilon_k": "epsilon_k_spt",
            }
        )
        overlap_df = overlap_df.merge(spt_subset, on="canonical_smiles")

        # Compute metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        for param in ["m", "sigma", "epsilon_k"]:
            pred = overlap_df[param].values
            true = overlap_df[f"{param}_spt"].values

            mae = mean_absolute_error(true, pred)
            rmse = np.sqrt(mean_squared_error(true, pred))
            r2 = r2_score(true, pred)

            logger.info(f"\n{param}:")
            logger.info(f"  MAE:  {mae:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R²:   {r2:.4f}")

        # Save overlap validation
        overlap_out = SAVED_DIR / "spt_pcsaft_overlap_validation.csv"
        overlap_df.to_csv(overlap_out, index=False)
        logger.info(f"\nSaved overlap validation to {overlap_out}")


if __name__ == "__main__":
    main()
