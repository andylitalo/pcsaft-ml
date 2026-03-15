"""Download SPT-PCSAFT predicted parameters (Winter et al. 2025).

Source: Winter, B., Rehner, P., Esper, T., Schilling, J., & Bardow, A.
"Understanding the language of molecules: predicting pure component
parameters for the PC-SAFT equation of state from SMILES."
Digital Discovery, 2025, 4, 1142-1157. DOI: 10.1039/D4DD00077C
arXiv: https://arxiv.org/abs/2309.12404

Provides predicted PC-SAFT parameters for ~13,645 components from a
decoder-only SMILES transformer trained end-to-end on experimental
vapor pressure and liquid density data.

NOTE: These are model-predicted parameters, not experimentally fitted.
Use as augmentation data or for benchmarking, not as gold-standard labels.

Usage:
    python -m model.data.download_spt_pcsaft
"""

import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SPT_PCSAFT_URL = (
    "https://arxiv.org/src/2309.12404v1/anc/PC-SAFT-Parameter.csv"
)

OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "spt_pcsaft.csv"


def download_spt_pcsaft() -> pd.DataFrame:
    """Download the SPT-PCSAFT predicted parameter set.

    Returns the DataFrame and saves it to disk as spt_pcsaft.csv.
    If the file already exists, it is loaded directly.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: smiles, m, sigma, epsilon_k, name, mu,
        kappa_ab, epsilon_k_ab.
    """
    if OUTPUT_FILE.exists():
        logger.info("SPT-PCSAFT dataset already exists: %s", OUTPUT_FILE)
        return pd.read_csv(OUTPUT_FILE)

    print("Downloading SPT-PCSAFT predicted parameters from arXiv...")
    resp = requests.get(SPT_PCSAFT_URL, timeout=120)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    print(f"Raw SPT-PCSAFT data shape: {df.shape}")

    col_map = {
        "names": "name",
        "SMILES0": "smiles",
        "m": "m",
        "sigma": "sigma",
        "epsilon_k": "epsilon_k",
        "mu": "mu",
        "kappa_ab": "kappa_ab",
        "epsilon_k_ab": "epsilon_k_ab",
    }
    df = df.rename(columns=col_map)

    df = df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
    df = df.drop_duplicates(subset=["smiles"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} molecules to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    try:
        download_spt_pcsaft()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
