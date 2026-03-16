"""Load and prepare PC-SAFT training data."""

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
FALLBACK_CSV = DATA_DIR / "pcsaft_data.csv"
ESPER_CSV = DATA_DIR / "esper_pcsaft.csv"
MLSAFT_CSV = DATA_DIR / "mlsaft_pcsaft.csv"
FLUORINATED_CSV = DATA_DIR / "fluorinated_pcsaft.csv"
SPT_PCSAFT_CSV = DATA_DIR / "spt_pcsaft.csv"

TARGETS = ["m", "sigma", "epsilon_k"]


def deduplicate_by_inchi(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Remove duplicate molecules using InChI as the canonical identifier.

    Canonical SMILES can differ between toolkits; InChI is a canonical molecular
    identifier that handles tautomers consistently.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a SMILES column.
    smiles_col : str
        Name of the SMILES column.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates (by InChI) removed.
    """
    inchis = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        inchis.append(MolToInchi(mol) if mol else None)
    df = df.copy()
    df["_inchi"] = inchis
    df = df.dropna(subset=["_inchi"]).drop_duplicates(subset=["_inchi"])
    return df.drop(columns=["_inchi"])


def _load_combined() -> pd.DataFrame:
    """Load Esper, ML-SAFT, and fluorinated datasets with InChI deduplication.

    When duplicates exist with conflicting parameter values (same molecule,
    different m/sigma/epsilon_k), Esper values are preferred, then ML-SAFT,
    then fluorinated (priority order: esper > mlsaft > fluorinated).

    Returns
    -------
    pd.DataFrame
        Combined and deduplicated dataset.
    """
    dfs = []

    if ESPER_CSV.exists():
        esper_df = pd.read_csv(ESPER_CSV)
        esper_df = esper_df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
        esper_df["_source"] = "esper"
        dfs.append(esper_df)
        logger.info("Loaded Esper dataset: %d molecules", len(esper_df))

    if MLSAFT_CSV.exists():
        mlsaft_df = pd.read_csv(MLSAFT_CSV)
        mlsaft_df = mlsaft_df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
        mlsaft_df["_source"] = "mlsaft"
        dfs.append(mlsaft_df)
        logger.info("Loaded ML-SAFT dataset: %d molecules", len(mlsaft_df))

    if FLUORINATED_CSV.exists():
        fluorinated_df = pd.read_csv(FLUORINATED_CSV)
        fluorinated_df = fluorinated_df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
        fluorinated_df["_source"] = "fluorinated"
        dfs.append(fluorinated_df)
        logger.info("Loaded fluorinated dataset: %d molecules", len(fluorinated_df))

    if not dfs:
        raise FileNotFoundError(
            "No datasets found. Run: python -m model.data.download_esper "
            "and/or python -m model.data.download_mlsaft"
        )

    if len(dfs) == 1:
        df = dfs[0].drop(columns=["_source"], errors="ignore")
        return df

    # Concat with priority: Esper > ML-SAFT > fluorinated
    combined = pd.concat(dfs, ignore_index=True)
    # Sort so higher-priority sources come first (esper < fluorinated < mlsaft alphabetically)
    combined = combined.sort_values("_source", ascending=True)
    combined = deduplicate_by_inchi(combined)
    combined = combined.drop(columns=["_source"], errors="ignore")

    logger.info("Combined dataset after InChI deduplication: %d molecules", len(combined))
    return combined


def _load_all() -> pd.DataFrame:
    """Load Esper + ML-SAFT + SPT-PCSAFT with InChI deduplication.

    Priority order for duplicate molecules: Esper > ML-SAFT > SPT-PCSAFT.
    This is the full training corpus (~13,764 molecules before dedup).
    """
    dfs = []

    if ESPER_CSV.exists():
        esper_df = pd.read_csv(ESPER_CSV)
        esper_df = esper_df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
        esper_df["_source"] = "esper"
        dfs.append(esper_df)
        logger.info("Loaded Esper dataset: %d molecules", len(esper_df))

    if MLSAFT_CSV.exists():
        mlsaft_df = pd.read_csv(MLSAFT_CSV)
        mlsaft_df = mlsaft_df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
        mlsaft_df["_source"] = "mlsaft"
        dfs.append(mlsaft_df)
        logger.info("Loaded ML-SAFT dataset: %d molecules", len(mlsaft_df))

    if SPT_PCSAFT_CSV.exists():
        spt_df = pd.read_csv(SPT_PCSAFT_CSV)
        spt_df = spt_df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
        spt_df["_source"] = "spt_pcsaft"
        dfs.append(spt_df)
        logger.info("Loaded SPT-PCSAFT dataset: %d molecules", len(spt_df))

    if not dfs:
        raise FileNotFoundError(
            "No datasets found for source='all'. Need at least one of: "
            "esper_pcsaft.csv, mlsaft_pcsaft.csv, spt_pcsaft.csv"
        )

    if len(dfs) == 1:
        df = dfs[0].drop(columns=["_source"], errors="ignore")
        return df

    combined = pd.concat(dfs, ignore_index=True)
    # Sort so higher-priority sources come first: esper < mlsaft < spt_pcsaft
    priority = {"esper": 0, "mlsaft": 1, "spt_pcsaft": 2}
    combined["_priority"] = combined["_source"].map(priority)
    combined = combined.sort_values("_priority")

    # Preserve source before deduplication (for test set analysis)
    # deduplicate_by_inchi will keep the first occurrence (highest priority source)
    combined_dedup = deduplicate_by_inchi(combined)
    # Rename _source to source for public API
    combined_dedup = combined_dedup.rename(columns={"_source": "source"})
    combined_dedup = combined_dedup.drop(columns=["_priority"], errors="ignore")

    logger.info("Full dataset (all) after InChI deduplication: %d molecules", len(combined_dedup))
    return combined_dedup


def load_data(source: str = "auto") -> pd.DataFrame:
    """Load PC-SAFT parameter data from CSV.

    Parameters
    ----------
    source : str
        "esper" to use Esper dataset,
        "mlsaft" to use ML-SAFT dataset,
        "fluorinated" to use fluorinated compounds only,
        "spt_pcsaft" to use SPT-PCSAFT dataset,
        "fallback" for curated CSV,
        "combined" to load Esper + ML-SAFT + fluorinated with deduplication,
        "all" to load Esper + ML-SAFT + SPT-PCSAFT (~13,764 molecules),
        "auto" to prefer combined if multiple exist, else single source, else fallback.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: smiles, m, sigma, epsilon_k (and optionally name, source).
    """
    if source == "all":
        return _load_all()

    if source == "combined":
        return _load_combined()

    if source == "auto":
        # Prefer combined if multiple datasets exist
        available = sum([ESPER_CSV.exists(), MLSAFT_CSV.exists(), FLUORINATED_CSV.exists()])
        if available >= 2:
            return _load_combined()
        elif ESPER_CSV.exists():
            path = ESPER_CSV
        elif MLSAFT_CSV.exists():
            path = MLSAFT_CSV
        elif FLUORINATED_CSV.exists():
            path = FLUORINATED_CSV
        else:
            path = FALLBACK_CSV
    elif source == "esper":
        if not ESPER_CSV.exists():
            raise FileNotFoundError(
                f"{ESPER_CSV} not found. Run: python -m model.data.download_esper"
            )
        path = ESPER_CSV
    elif source == "mlsaft":
        if not MLSAFT_CSV.exists():
            raise FileNotFoundError(
                f"{MLSAFT_CSV} not found. Run: python -m model.data.download_mlsaft"
            )
        path = MLSAFT_CSV
    elif source == "fluorinated":
        if not FLUORINATED_CSV.exists():
            raise FileNotFoundError(
                f"{FLUORINATED_CSV} not found. Generate with: "
                "python scripts/generate_fluorinated_data.py"
            )
        path = FLUORINATED_CSV
    elif source == "spt_pcsaft":
        if not SPT_PCSAFT_CSV.exists():
            raise FileNotFoundError(
                f"{SPT_PCSAFT_CSV} not found. Run: python -m model.data.download_spt_pcsaft"
            )
        path = SPT_PCSAFT_CSV
    elif source == "fallback":
        path = FALLBACK_CSV
    else:
        logger.warning("Unknown source %r, falling back to %s", source, FALLBACK_CSV)
        path = FALLBACK_CSV

    df = pd.read_csv(path)
    required = {"smiles", "m", "sigma", "epsilon_k"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets with optional stratification.

    Stratifies on binned epsilon_k (the hardest target parameter) to ensure
    the test set covers the full range of dispersion energies.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed.
    stratify_bins : int
        Number of epsilon_k bins for stratification. Set to 0 to disable.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    if stratify_bins and len(df) > stratify_bins * 5:
        bins = pd.qcut(
            df["epsilon_k"], q=stratify_bins, labels=False, duplicates="drop"
        )
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=bins
        )
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
    return train_df, test_df
