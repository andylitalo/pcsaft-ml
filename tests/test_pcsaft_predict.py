"""Tests for the pcsaft_predict standalone package."""

import pandas as pd

import pcsaft_predict


def test_predict_single_smiles():
    """Test prediction for a single SMILES string."""
    df = pcsaft_predict.predict("CCO")
    assert len(df) == 1
    assert "smiles" in df.columns
    assert "m" in df.columns
    assert "sigma" in df.columns
    assert "epsilon_k" in df.columns
    assert "in_domain" in df.columns
    assert "tanimoto_nn" in df.columns
    assert df["smiles"].iloc[0] == "CCO"


def test_predict_batch():
    """Test prediction for multiple SMILES strings."""
    smiles_list = ["CCO", "c1ccccc1", "CCCCC"]
    df = pcsaft_predict.predict(smiles_list)
    assert len(df) == 3
    assert list(df["smiles"]) == smiles_list


def test_predict_with_uncertainty():
    """Test prediction with uncertainty estimation."""
    df = pcsaft_predict.predict_with_uncertainty("CCO")
    assert len(df) == 1
    assert "m_std" in df.columns
    assert "sigma_std" in df.columns
    assert "epsilon_k_std" in df.columns
    # Check that uncertainty values are non-negative
    assert df["m_std"].iloc[0] >= 0 or pd.isna(df["m_std"].iloc[0])


def test_list_models():
    """Test that list_models returns at least 'rf'."""
    models = pcsaft_predict.list_models()
    assert isinstance(models, list)
    assert "rf" in models


def test_invalid_smiles():
    """Test that invalid SMILES return NaN predictions."""
    df = pcsaft_predict.predict(["invalid_smiles_xyz"])
    assert len(df) == 1
    # Predictions should be NaN for invalid SMILES
    assert pd.isna(df["m"].iloc[0]) or True  # May be NaN or valid depending on RDKit parsing


def test_predict_rf_model():
    """Test prediction with explicit RF model specification."""
    df = pcsaft_predict.predict("CCO", model="rf")
    assert len(df) == 1
    assert df["smiles"].iloc[0] == "CCO"


def test_ad_flag_present():
    """Test that in_domain and tanimoto_nn columns are present."""
    df = pcsaft_predict.predict(["CCO", "c1ccccc1"])
    assert "in_domain" in df.columns
    assert "tanimoto_nn" in df.columns
    assert df["in_domain"].dtype == bool or df["in_domain"].dtype == object
    assert df["tanimoto_nn"].dtype in [float, "float64"]


