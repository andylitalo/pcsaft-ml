"""Download the ML-SAFT dataset of regressed PC-SAFT parameters.

Source: Felton et al. (2024), "ML-SAFT: A machine learning framework for
PCP-SAFT parameter prediction", Chemical Engineering Journal.
DOI: 10.1016/j.cej.2024.151999
GitHub: https://github.com/sustainable-processes/ml_saft

The regressed parameters live in the repo at
data/05_model_input/pcp_saft_regressed_filtered.csv (~870 molecules).

Usage:
    python -m model.data.download_mlsaft
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

MLSAFT_CSV_URL = (
    "https://raw.githubusercontent.com/sustainable-processes/ml_saft"
    "/main/data/05_model_input/pcp_saft_regressed_filtered.csv"
)

OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "mlsaft_pcsaft.csv"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ML-SAFT column names to our standard format."""
    col_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if low in ("smiles", "smiles_1", "smiles_string", "canonical_smiles"):
            col_map[col] = "smiles"
        elif low in ("m", "m_seg", "segment_number"):
            col_map[col] = "m"
        elif low in ("sigma", "seg_diameter"):
            col_map[col] = "sigma"
        elif low in ("epsilon_k", "eps_k", "epsilon/k", "dispersion_energy"):
            col_map[col] = "epsilon_k"
        elif low in ("name", "compound", "substance"):
            col_map[col] = "name"
    df = df.rename(columns=col_map)
    return df


def download_mlsaft() -> pd.DataFrame:
    """Download and parse the ML-SAFT dataset.

    Returns the DataFrame and also saves it to disk as mlsaft_pcsaft.csv.
    If the file already exists, it is loaded directly.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: smiles, m, sigma, epsilon_k.
    """
    if OUTPUT_FILE.exists():
        logger.info("ML-SAFT dataset already exists: %s", OUTPUT_FILE)
        return pd.read_csv(OUTPUT_FILE)

    print("Downloading ML-SAFT regressed parameters from GitHub...")
    resp = requests.get(MLSAFT_CSV_URL, timeout=60)
    resp.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    print(f"Raw ML-SAFT data shape: {df.shape}")

    df = _normalize_columns(df)

    required = {"smiles", "m", "sigma", "epsilon_k"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Downloaded file is missing required PC-SAFT columns: {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )

    keep = ["name", "smiles", "m", "sigma", "epsilon_k"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
    df = df.drop_duplicates(subset=["smiles"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} molecules to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    try:
        download_mlsaft()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
