"""Compute RDKit 2D molecular descriptors from SMILES strings."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


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
