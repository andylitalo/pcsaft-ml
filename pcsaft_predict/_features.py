"""Feature computation: Morgan fingerprints and RDKit descriptors for PC-SAFT prediction."""

import logging
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

logger = logging.getLogger(__name__)


def _get_descriptor_calculator() -> MoleculeDescriptors.MolecularDescriptorCalculator:
    """Return a calculator for all available RDKit 2D descriptors."""
    names = [name for name, _ in Descriptors.descList]
    return MoleculeDescriptors.MolecularDescriptorCalculator(names)


def compute_morgan_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprints as a dense bit array.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to featurize.
    radius : int
        Morgan fingerprint radius (default 2 = ECFP4 equivalent).
    n_bits : int
        Length of the bit vector (default 2048).

    Returns
    -------
    np.ndarray
        Array of shape (n_molecules, n_bits). Rows for invalid SMILES are all zeros.
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Invalid SMILES for Morgan FP: %s", smi)
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps)


def compute_rdkit_descriptors(
    smiles_list: list[str],
    rdkit_names: list[str] | None = None,
) -> np.ndarray:
    """Compute RDKit 2D descriptors.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to featurize.
    rdkit_names : list[str] | None
        If provided, select exactly these columns from the raw descriptor
        DataFrame (for inference with pre-determined feature names).
        If None, compute all RDKit descriptors.

    Returns
    -------
    np.ndarray
        Array of shape (n_molecules, n_descriptors). Invalid SMILES get NaN rows.
    """
    calc = _get_descriptor_calculator()
    all_names = list(calc.GetDescriptorNames())

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([np.nan] * len(all_names))
        else:
            rows.append(list(calc.CalcDescriptors(mol)))

    # Convert to array
    arr = np.array(rows, dtype=np.float32)

    # If rdkit_names provided, select those columns
    if rdkit_names is not None:
        # Map names to indices
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        indices = []
        for name in rdkit_names:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
            else:
                logger.warning("RDKit descriptor %s not found, using zeros", name)

        if indices:
            selected = arr[:, indices]
            # Add zero columns for missing descriptors
            if len(indices) < len(rdkit_names):
                n_missing = len(rdkit_names) - len(indices)
                missing_cols = np.zeros((len(smiles_list), n_missing), dtype=np.float32)
                selected = np.hstack([selected, missing_cols])
        else:
            # All descriptors missing, return zeros
            selected = np.zeros((len(smiles_list), len(rdkit_names)), dtype=np.float32)

        return selected

    return arr


def build_features(
    smiles_list: list[str],
    rdkit_names: list[str] | None = None,
    scaler_path: Path | None = None,
) -> np.ndarray:
    """Build combined feature matrix: Morgan FP + RDKit descriptors.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to featurize.
    rdkit_names : list[str] | None
        Pre-determined RDKit descriptor column names from training.
    scaler_path : Path | None
        Path to RDKit descriptor scaler (.joblib file). If None, no scaling.

    Returns
    -------
    np.ndarray
        2D array of shape (n_molecules, n_features).
    """
    # Compute Morgan fingerprints (2048 bits, radius 2)
    morgan_fp = compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048)

    # Compute RDKit descriptors
    rdkit_desc = compute_rdkit_descriptors(smiles_list, rdkit_names)

    # Apply scaling if scaler provided
    if scaler_path is not None and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        rdkit_desc = scaler.transform(rdkit_desc).astype(np.float32)

    # Replace inf/nan with 0
    rdkit_desc = np.nan_to_num(rdkit_desc, nan=0.0, posinf=0.0, neginf=0.0)

    # Concatenate Morgan FP + RDKit descriptors
    features = np.hstack([morgan_fp, rdkit_desc])

    return features
