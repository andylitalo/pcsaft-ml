"""Inference API: predict PC-SAFT parameters from SMILES.

Usage:
    python -m model.predict "C1CCCC1" "CC=C" "FC(=CF)C(F)F"
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from model.data.descriptors import build_features
from model.data.load import TARGETS

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent / "saved"


def _load_feature_config() -> dict:
    """Load feature configuration from saved artifacts.

    Falls back to rdkit-only mode for backward compatibility with models
    trained before feature_config.json was introduced.
    """
    config_path = SAVED_DIR / "feature_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    logger.info("No feature_config.json found; assuming rdkit-only features")
    return {
        "features": "rdkit",
        "use_morgan": False,
        "use_rdkit": True,
        "morgan_radius": 2,
        "morgan_bits": 2048,
    }


def _get_rdkit_names_from_saved() -> list[str] | None:
    """Extract the RDKit descriptor column names from saved feature_names.

    Returns the non-Morgan feature names from the saved artifact,
    or None if not available.
    """
    fnames_path = SAVED_DIR / "feature_names.joblib"
    if not fnames_path.exists():
        return None
    feature_names = joblib.load(fnames_path)
    rdkit_names = [n for n in feature_names if not n.startswith("morgan_")]
    return rdkit_names if rdkit_names else None


def predict_pcsaft(smiles_list: list[str]) -> pd.DataFrame:
    """Predict PC-SAFT parameters for a list of SMILES.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings.

    Returns
    -------
    pd.DataFrame
        Columns: smiles, m, sigma, epsilon_k
    """
    config = _load_feature_config()
    use_rdkit = config.get("use_rdkit", True)

    # Get saved RDKit descriptor names for consistent feature computation
    rdkit_names = _get_rdkit_names_from_saved() if use_rdkit else None

    X = build_features(
        smiles_list,
        use_morgan=config.get("use_morgan", False),
        use_rdkit=use_rdkit,
        morgan_radius=config.get("morgan_radius", 2),
        morgan_bits=config.get("morgan_bits", 2048),
        rdkit_names=rdkit_names,
    )

    results = pd.DataFrame({"smiles": smiles_list})
    for target in TARGETS:
        model = joblib.load(SAVED_DIR / f"rf_{target}.joblib")
        results[target] = model.predict(X)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m model.predict SMILES [SMILES ...]")
        sys.exit(1)
    smiles = sys.argv[1:]
    df = predict_pcsaft(smiles)
    print(df.to_string(index=False))
