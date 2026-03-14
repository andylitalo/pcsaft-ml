"""Compute RDKit 2D molecular descriptors and Morgan fingerprints from SMILES strings."""

import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

logger = logging.getLogger(__name__)


def _get_descriptor_calculator() -> MoleculeDescriptors.MolecularDescriptorCalculator:
    """Return a calculator for all available RDKit 2D descriptors."""
    names = [name for name, _ in Descriptors.descList]
    return MoleculeDescriptors.MolecularDescriptorCalculator(names)


def compute_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    """Compute ~200 RDKit 2D descriptors for a list of SMILES.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to featurize.

    Returns
    -------
    pd.DataFrame
        DataFrame with descriptor columns. Rows for invalid SMILES are all NaN.
    """
    calc = _get_descriptor_calculator()
    desc_names = list(calc.GetDescriptorNames())
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([np.nan] * len(desc_names))
        else:
            rows.append(list(calc.CalcDescriptors(mol)))
    df = pd.DataFrame(rows, columns=desc_names)
    return df


def clean_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove zero-variance and highly-correlated descriptor columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw descriptor DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned descriptor DataFrame.
    """
    # Drop columns that are all NaN or zero variance
    df = df.dropna(axis=1, how="all")
    variances = df.var(numeric_only=True)
    zero_var = variances[variances == 0].index.tolist()
    df = df.drop(columns=zero_var, errors="ignore")

    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop highly correlated features (|r| > 0.95)
    if len(df.columns) > 1 and len(df) > 2:
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        df = df.drop(columns=to_drop, errors="ignore")

    return df


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


def _get_rdkit_features(
    smiles_list: list[str], rdkit_names: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Compute RDKit descriptor features.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings.
    rdkit_names : list[str] | None
        If provided, select exactly these columns from the raw descriptor
        DataFrame (for inference with pre-determined feature names).
        If None, apply clean_descriptors to determine columns.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (feature_array, column_names)
    """
    desc_df = compute_descriptors(smiles_list)
    if rdkit_names is not None:
        # Use pre-determined columns (inference mode)
        available = set(desc_df.columns)
        cols_to_use = [c for c in rdkit_names if c in available]
        desc_df = desc_df[cols_to_use]
        # Add zero columns for any missing descriptors
        for c in rdkit_names:
            if c not in available:
                desc_df[c] = 0.0
        desc_df = desc_df[rdkit_names]  # Ensure correct order
    else:
        # Training mode: clean descriptors
        desc_df = clean_descriptors(desc_df)
    arr = desc_df.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    return arr, desc_df.columns.tolist()


def build_features(
    smiles_list: list[str],
    use_morgan: bool = True,
    use_rdkit: bool = True,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    rdkit_names: list[str] | None = None,
) -> np.ndarray:
    """Build a combined feature matrix from SMILES.

    Concatenates Morgan fingerprints (binary) and cleaned RDKit 2D descriptors
    (continuous) into a single NumPy array with no NaN values.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to featurize.
    use_morgan : bool
        Include Morgan fingerprints (default True).
    use_rdkit : bool
        Include cleaned RDKit 2D descriptors (default True).
    morgan_radius : int
        Morgan FP radius (default 2).
    morgan_bits : int
        Morgan FP bit vector length (default 2048).
    rdkit_names : list[str] | None
        Pre-determined RDKit descriptor column names from training.
        When provided, skips clean_descriptors and selects these columns
        directly for consistent inference.

    Returns
    -------
    np.ndarray
        2D array of shape (n_molecules, n_features) with no NaN values.
    """
    if not use_morgan and not use_rdkit:
        raise ValueError("At least one of use_morgan or use_rdkit must be True")

    parts = []
    if use_morgan:
        parts.append(compute_morgan_fingerprints(smiles_list, morgan_radius, morgan_bits))
    if use_rdkit:
        arr, _ = _get_rdkit_features(smiles_list, rdkit_names)
        parts.append(arr)
    return np.hstack(parts)


def build_features_with_names(
    smiles_list: list[str],
    use_morgan: bool = True,
    use_rdkit: bool = True,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    rdkit_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a combined feature matrix and return feature names.

    Same as build_features() but also returns a list of feature names for
    interpretability and artifact tracking.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to featurize.
    use_morgan : bool
        Include Morgan fingerprints (default True).
    use_rdkit : bool
        Include cleaned RDKit 2D descriptors (default True).
    morgan_radius : int
        Morgan FP radius (default 2).
    morgan_bits : int
        Morgan FP bit vector length (default 2048).
    rdkit_names : list[str] | None
        Pre-determined RDKit descriptor column names from training.
        When provided, skips clean_descriptors and selects these columns
        directly for consistent inference.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (features, names) where features is shape (n_molecules, n_features)
        and names is a list of feature name strings.
    """
    if not use_morgan and not use_rdkit:
        raise ValueError("At least one of use_morgan or use_rdkit must be True")

    parts = []
    names = []
    if use_morgan:
        parts.append(compute_morgan_fingerprints(smiles_list, morgan_radius, morgan_bits))
        names += [f"morgan_{i}" for i in range(morgan_bits)]
    if use_rdkit:
        arr, rdkit_cols = _get_rdkit_features(smiles_list, rdkit_names)
        parts.append(arr)
        names += rdkit_cols
    features = np.hstack(parts)
    return features, names
