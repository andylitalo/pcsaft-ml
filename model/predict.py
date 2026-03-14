"""Inference API: predict PC-SAFT parameters from SMILES.

Usage:
    python -m model.predict "C1CCCC1" "CC=C" "FC(=CF)C(F)F"
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from model.data.descriptors import compute_descriptors
from model.data.load import TARGETS

SAVED_DIR = Path(__file__).parent / "saved"


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
    feature_names = joblib.load(SAVED_DIR / "feature_names.joblib")
    X = compute_descriptors(smiles_list)[feature_names]

    results = pd.DataFrame({"smiles": smiles_list})
    for target in TARGETS:
        model = joblib.load(SAVED_DIR / f"rf_{target}.joblib")
        # Fill NaN with 0 for prediction (invalid SMILES get garbage predictions)
        results[target] = model.predict(X.fillna(0))

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m model.predict SMILES [SMILES ...]")
        sys.exit(1)
    smiles = sys.argv[1:]
    df = predict_pcsaft(smiles)
    print(df.to_string(index=False))
