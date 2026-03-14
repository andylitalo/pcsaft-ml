"""Download the Esper et al. PC-SAFT parameter dataset from Figshare.

Esper et al., "PCP-SAFT Parameters of Pure Substances Using Large
Experimental Databases", Ind. Eng. Chem. Res., 2023.

Figshare collection: https://acs.figshare.com/collections/6821654/1

Usage:
    python -m model.data.download_esper
"""

import hashlib
import io
import logging
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

COLLECTION_URL = "https://api.figshare.com/v2/collections/6821654/articles"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "esper_pcsaft.csv"

EXPECTED_SHA256 = "590cd5eeea8fecd34a13777beed3dfee1e9b9c830ed46417ebd189de1c068beb"


def _find_csv_article(articles: list[dict]) -> dict | None:
    """Find the article containing the PC-SAFT parameter CSV."""
    for article in articles:
        title = article.get("title", "").lower()
        if "parameter" in title or "pcp-saft" in title or "pc-saft" in title:
            return article
    return articles[0] if articles else None


def _get_download_url(article_id: int) -> tuple[str | None, str | None]:
    """Get the download URL for the first CSV file in an article."""
    resp = requests.get(
        f"https://api.figshare.com/v2/articles/{article_id}", timeout=30
    )
    resp.raise_for_status()
    details = resp.json()
    for f in details.get("files", []):
        if f["name"].endswith((".csv", ".xlsx", ".zip")):
            return f["download_url"], f["name"]
    return None, None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to our standard format."""
    col_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if low in ("smiles", "smiles_string", "canonical_smiles"):
            col_map[col] = "smiles"
        elif re.match(r"^m$|^m_seg|^segment.?number", low):
            col_map[col] = "m"
        elif re.match(r"^sigma|^seg.?diam", low):
            col_map[col] = "sigma"
        elif re.match(r"^(eps(ilon)?[_/]?k|disp.?energy)$", low):
            col_map[col] = "epsilon_k"
        elif low in ("name", "compound", "substance"):
            col_map[col] = "name"
    df = df.rename(columns=col_map)
    return df


def verify_checksum(filepath: Path, expected: str) -> bool:
    """Verify SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected:
        logger.warning("Checksum mismatch: expected %s, got %s", expected, actual)
        return False
    return True


def download_esper() -> pd.DataFrame:
    """Download and parse the Esper dataset.

    Returns the DataFrame and also saves it to disk as esper_pcsaft.csv.
    If the file already exists and passes checksum verification, it is loaded directly.
    """
    if OUTPUT_FILE.exists() and verify_checksum(OUTPUT_FILE, EXPECTED_SHA256):
        logger.info("Esper dataset already exists and checksum verified: %s", OUTPUT_FILE)
        return pd.read_csv(OUTPUT_FILE)

    print("Fetching Figshare collection metadata...")
    resp = requests.get(COLLECTION_URL, timeout=30)
    resp.raise_for_status()
    articles = resp.json()

    article = _find_csv_article(articles)
    if article is None:
        raise RuntimeError("No articles found in Figshare collection")

    article_id = article["id"]
    print(f"Found article: {article.get('title', article_id)}")

    download_url, filename = _get_download_url(article_id)
    if download_url is None:
        raise RuntimeError(f"No CSV/XLSX/ZIP file found in article {article_id}")

    print(f"Downloading data from {download_url}...")
    data_resp = requests.get(download_url, timeout=120)
    data_resp.raise_for_status()

    if filename and filename.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(data_resp.content))
        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found inside zip. Contents: {z.namelist()}")
        print(f"  Extracting {csv_names[0]} from zip...")
        with z.open(csv_names[0]) as f:
            # Esper dataset uses tab-separated values
            df = pd.read_csv(f, sep="\t")
    elif filename and filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(data_resp.content))
    else:
        df = pd.read_csv(io.StringIO(data_resp.text))

    print(f"Raw data shape: {df.shape}")
    df = _normalize_columns(df)

    required = {"smiles", "m", "sigma", "epsilon_k"}
    missing = required - set(df.columns)
    if missing:
        print(f"Warning: columns {missing} not found. Available: {list(df.columns)}")
        print("Saving raw data; you may need to manually map columns.")
        df.to_csv(OUTPUT_FILE, index=False)
        return df

    keep = ["name", "smiles", "m", "sigma", "epsilon_k"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["smiles", "m", "sigma", "epsilon_k"])
    df = df.drop_duplicates(subset=["smiles"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} molecules to {OUTPUT_FILE}")

    if not verify_checksum(OUTPUT_FILE, EXPECTED_SHA256):
        logger.warning(
            "Downloaded file checksum differs from expected. "
            "The upstream data may have changed. Proceeding with downloaded version."
        )
    return df


if __name__ == "__main__":
    try:
        download_esper()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
