"""Tests for simplified GC-PC-SAFT parameter estimation."""

import numpy as np
import pandas as pd

from model.gc_pcsaft import predict_gc_pcsaft, predict_gc_pcsaft_single


def test_gc_pcsaft_returns_dataframe():
    """predict_gc_pcsaft returns DataFrame with correct columns."""
    smiles = ["C", "CC", "CCC", "C1CCCC1"]
    df = predict_gc_pcsaft(smiles)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert set(df.columns) == {"smiles", "m", "sigma", "epsilon_k"}


def test_gc_pcsaft_cyclopentane():
    """Cyclopentane prediction is in the ballpark of known values.

    Known PC-SAFT parameters for cyclopentane:
        m ~ 2.37, sigma ~ 3.71, epsilon_k ~ 289
    GC method is approximate, so we use generous tolerances.
    """
    result = predict_gc_pcsaft_single("C1CCCC1")
    assert result is not None
    # Allow 50% tolerance for this simplified method
    assert 1.0 < result["m"] < 4.0, f"m={result['m']} outside reasonable range"
    assert 2.5 < result["sigma"] < 5.0, f"sigma={result['sigma']} outside reasonable range"
    assert 100 < result["epsilon_k"] < 500, f"epsilon_k={result['epsilon_k']} outside range"


def test_gc_pcsaft_invalid_smiles():
    """Invalid SMILES produce NaN row."""
    df = predict_gc_pcsaft(["INVALID_SMILES"])
    assert len(df) == 1
    assert np.isnan(df.iloc[0]["m"])
    assert np.isnan(df.iloc[0]["sigma"])
    assert np.isnan(df.iloc[0]["epsilon_k"])


def test_gc_pcsaft_fluorinated():
    """Fluorinated compounds should produce valid predictions."""
    smiles = ["FC(F)C(F)F", "FC(F)(F)C(F)F"]
    df = predict_gc_pcsaft(smiles)
    assert len(df) == 2
    for _, row in df.iterrows():
        assert row["m"] > 0, "m should be positive"
        assert row["sigma"] > 0, "sigma should be positive"
        assert row["epsilon_k"] > 0, "epsilon_k should be positive"


def test_gc_pcsaft_positive_parameters():
    """All three parameters should be positive for valid molecules."""
    smiles = ["C", "CC", "CCO", "c1ccccc1", "CC=O"]
    df = predict_gc_pcsaft(smiles)
    valid = df.dropna()
    assert len(valid) > 0
    assert (valid["m"] > 0).all()
    assert (valid["sigma"] > 0).all()
    assert (valid["epsilon_k"] > 0).all()
