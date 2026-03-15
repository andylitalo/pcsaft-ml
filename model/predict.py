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


def predict_rf_with_uncertainty(smiles_list: list[str]) -> dict:
    """Predict PC-SAFT parameters with RF uncertainty from tree disagreement.

    Uses the variance across individual tree predictions as a measure of
    model uncertainty.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to predict.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``m, sigma, epsilon_k, m_std, sigma_std, epsilon_k_std``.
    """
    import numpy as np

    config = _load_feature_config()
    use_rdkit = config.get("use_rdkit", True)
    rdkit_names = _get_rdkit_names_from_saved() if use_rdkit else None

    X = build_features(
        smiles_list,
        use_morgan=config.get("use_morgan", False),
        use_rdkit=use_rdkit,
        morgan_radius=config.get("morgan_radius", 2),
        morgan_bits=config.get("morgan_bits", 2048),
        rdkit_names=rdkit_names,
    )

    result: dict[str, np.ndarray] = {}
    for target in TARGETS:
        model = joblib.load(SAVED_DIR / f"rf_{target}.joblib")
        # Get per-tree predictions
        tree_preds = np.array([
            tree.predict(X) for tree in model.estimators_
        ])  # shape: (n_trees, n_samples)
        result[target] = tree_preds.mean(axis=0)
        result[f"{target}_std"] = tree_preds.std(axis=0)

    return result


def predict_pcsaft_with_ci(
    smiles_list: list[str],
    z: float = 1.96,
) -> "pd.DataFrame":
    """Predict PC-SAFT parameters with approximate confidence intervals.

    Interval formula: pred ± z * sigma_trees, where sigma_trees is the
    standard deviation of individual RF tree predictions.

    Empirical CI coverage on the Esper test set (361 molecules):
        m        : 95.3%  — near-nominal, interval is reliable
        σ        : 92.8%  — slightly under-covers; ~7% of true values escape
        ε/k      : 90.3%  — under-covers; intervals are wide (~±58 K mean)

    The intervals are useful as a relative uncertainty signal and for
    candidate filtering, but are NOT statistically guaranteed. Use with
    that limitation in mind, especially for ε/k.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to predict.
    z : float
        Normal quantile multiplier. Default 1.96 (~95% CI under normality).

    Returns
    -------
    pd.DataFrame
        Columns: smiles, m, sigma, epsilon_k,
                 m_std, sigma_std, epsilon_k_std,
                 m_lo95, m_hi95,
                 sigma_lo95, sigma_hi95,
                 eps_lo95, eps_hi95
    """

    uq = predict_rf_with_uncertainty(smiles_list)

    rows = []
    for i, smi in enumerate(smiles_list):
        row = {"smiles": smi}
        for target, col in [("m", "m"), ("sigma", "sigma"), ("epsilon_k", "epsilon_k")]:
            pred = float(uq[target][i])
            std = float(uq[f"{target}_std"][i])
            half = z * std
            row[col] = pred
            row[f"{col}_std"] = std
            lo_key = f"{col}_lo95" if col != "epsilon_k" else "eps_lo95"
            hi_key = f"{col}_hi95" if col != "epsilon_k" else "eps_hi95"
            row[lo_key] = pred - half
            row[hi_key] = pred + half
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m model.predict SMILES [SMILES ...]")
        sys.exit(1)
    smiles = sys.argv[1:]
    df = predict_pcsaft(smiles)
    print(df.to_string(index=False))
