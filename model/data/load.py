"""Load and prepare PC-SAFT training data."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent
FALLBACK_CSV = DATA_DIR / "pcsaft_data.csv"
ESPER_CSV = DATA_DIR / "esper_pcsaft.csv"

TARGETS = ["m", "sigma", "epsilon_k"]


def load_data(source: str = "auto") -> pd.DataFrame:
    """Load PC-SAFT parameter data from CSV.

    Parameters
    ----------
    source : str
        "esper" to use Esper dataset, "fallback" for curated CSV,
        "auto" to prefer Esper if available.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: smiles, m, sigma, epsilon_k (and optionally name).
    """
    if source == "auto":
        path = ESPER_CSV if ESPER_CSV.exists() else FALLBACK_CSV
    elif source == "esper":
        if not ESPER_CSV.exists():
            raise FileNotFoundError(
                f"{ESPER_CSV} not found. Run: python -m model.data.download_esper"
            )
        path = ESPER_CSV
    else:
        path = FALLBACK_CSV

    df = pd.read_csv(path)
    required = {"smiles", "m", "sigma", "epsilon_k"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_df, test_df
