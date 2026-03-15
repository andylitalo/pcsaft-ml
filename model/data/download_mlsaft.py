"""Download the ML-SAFT dataset of PC-SAFT parameters.

This script downloads a supplementary PC-SAFT parameters dataset to augment
the Esper training set.

IMPORTANT — Dataset Source:
    The Figshare article ID originally encoded here (24689738) resolves to an
    unrelated soil-science dataset. Before running this script, verify the
    correct source for the ML-SAFT or supplementary PC-SAFT dataset you want
    to use. Candidate sources:
      - Esper et al. supplementary data (same group, extended set)
      - Rehner & Gross (2023) "FeOs" supplementary tables
      - Any published PC-SAFT parameter table with SMILES identifiers

    Update FIGSHARE_ARTICLE_URL below with the correct article ID, or
    replace this script entirely with a direct CSV download.

Usage:
    python -m model.data.download_mlsaft
"""

import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FIGSHARE_ARTICLE_URL = "https://api.figshare.com/v2/articles/24689738"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "mlsaft_pcsaft.csv"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ML-SAFT column names to our standard format."""
    col_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if low in ("smiles", "smiles_string", "canonical_smiles"):
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

    print("Fetching ML-SAFT article metadata from Figshare...")
    resp = requests.get(FIGSHARE_ARTICLE_URL, timeout=30)
    resp.raise_for_status()
    article = resp.json()

    # Find downloadable data file (CSV, XLSX, or ZIP)
    download_url = None
    filename = None
    for f in article.get("files", []):
        fname = f.get("name", "")
        if fname.endswith((".csv", ".xlsx", ".xls", ".zip")):
            download_url = f["download_url"]
            filename = fname
            break

    if download_url is None:
        raise RuntimeError(
            "No CSV/XLSX/XLS/ZIP file found in ML-SAFT Figshare article. "
            f"Available files: {[f['name'] for f in article.get('files', [])]}"
        )

    print(f"Downloading {filename} from {download_url}...")
    data_resp = requests.get(download_url, timeout=120)
    data_resp.raise_for_status()

    if filename.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(data_resp.content))
        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found inside zip. Contents: {z.namelist()}")
        print(f"  Extracting {csv_names[0]} from zip...")
        with z.open(csv_names[0]) as f:
            df = pd.read_csv(f)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data_resp.content))
    else:
        df = pd.read_csv(io.StringIO(data_resp.text))

    print(f"Raw ML-SAFT data shape: {df.shape}")
    df = _normalize_columns(df)

    required = {"smiles", "m", "sigma", "epsilon_k"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Downloaded file is missing required PC-SAFT columns: {missing}.\n"
            f"Available columns: {list(df.columns)}\n"
            "The Figshare article ID in this script may be incorrect. "
            "Update FIGSHARE_ARTICLE_URL with the correct dataset source. "
            "See the module docstring for guidance."
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
