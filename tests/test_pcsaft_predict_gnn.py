"""Tests for GNN model in the pcsaft_predict standalone package."""

import pandas as pd
import pytest

import pcsaft_predict

torch = pytest.importorskip("torch", reason="torch not installed")
torch_geometric = pytest.importorskip("torch_geometric", reason="torch_geometric not installed")


def test_gnn_in_list_models():
    """Test that 'gnn' appears in list_models when torch is available."""
    models = pcsaft_predict.list_models()
    assert "gnn" in models


def test_gnn_predict_single():
    """Test GNN prediction for a single SMILES string."""
    df = pcsaft_predict.predict("CCO", model="gnn")
    assert len(df) == 1
    assert "smiles" in df.columns
    assert "m" in df.columns
    assert "sigma" in df.columns
    assert "epsilon_k" in df.columns
    assert "in_domain" in df.columns
    assert "tanimoto_nn" in df.columns
    assert df["smiles"].iloc[0] == "CCO"
    # Check that predictions are not NaN
    assert not pd.isna(df["m"].iloc[0])
    assert not pd.isna(df["sigma"].iloc[0])
    assert not pd.isna(df["epsilon_k"].iloc[0])


def test_gnn_predict_batch():
    """Test GNN prediction for multiple SMILES strings."""
    smiles_list = ["CCO", "c1ccccc1", "CCCCC"]
    df = pcsaft_predict.predict(smiles_list, model="gnn")
    assert len(df) == 3
    assert list(df["smiles"]) == smiles_list
    # Check that all predictions are valid
    assert df["m"].notna().all()
    assert df["sigma"].notna().all()
    assert df["epsilon_k"].notna().all()


def test_gnn_predict_with_uncertainty():
    """Test GNN prediction with uncertainty estimation."""
    df = pcsaft_predict.predict_with_uncertainty("CCO", model="gnn", n_forward=10)
    assert len(df) == 1
    assert "m_std" in df.columns
    assert "sigma_std" in df.columns
    assert "epsilon_k_std" in df.columns
    # Check that uncertainty values are non-negative
    assert df["m_std"].iloc[0] >= 0
    assert df["sigma_std"].iloc[0] >= 0
    assert df["epsilon_k_std"].iloc[0] >= 0


def test_gnn_invalid_smiles():
    """Test that invalid SMILES return NaN predictions."""
    df = pcsaft_predict.predict(["invalid_smiles_xyz"], model="gnn")
    assert len(df) == 1
    # Predictions should be NaN for invalid SMILES
    assert pd.isna(df["m"].iloc[0])
    assert pd.isna(df["sigma"].iloc[0])
    assert pd.isna(df["epsilon_k"].iloc[0])


def test_gnn_fluorinated_molecule():
    """Test GNN prediction on a fluorinated molecule (GNN's strength)."""
    # Trifluoroethylene - a fluorinated olefin
    df = pcsaft_predict.predict("C(F)=C(F)F", model="gnn")
    assert len(df) == 1
    assert not pd.isna(df["m"].iloc[0])
    assert not pd.isna(df["sigma"].iloc[0])
    assert not pd.isna(df["epsilon_k"].iloc[0])


def test_gnn_predict_consistency():
    """Test that multiple calls to predict give the same result."""
    df1 = pcsaft_predict.predict("CCO", model="gnn")
    df2 = pcsaft_predict.predict("CCO", model="gnn")
    # Should be identical (deterministic without dropout)
    assert df1["m"].iloc[0] == df2["m"].iloc[0]
    assert df1["sigma"].iloc[0] == df2["sigma"].iloc[0]
    assert df1["epsilon_k"].iloc[0] == df2["epsilon_k"].iloc[0]
