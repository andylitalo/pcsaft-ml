"""Tests for molecular descriptor computation."""

import numpy as np
import pandas as pd

from model.data.descriptors import clean_descriptors, compute_descriptors


def test_compute_descriptors_valid_smiles():
    """Compute descriptors for valid SMILES returns non-empty DataFrame."""
    smiles = ["C", "CC", "C1CCCC1"]
    df = compute_descriptors(smiles)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.shape[1] > 100  # Should have ~200 descriptors
    assert not df.isna().all().all()  # Not all NaN


def test_compute_descriptors_invalid_smiles():
    """Invalid SMILES should produce a row of NaN."""
    smiles = ["C", "INVALID_SMILES", "CC"]
    df = compute_descriptors(smiles)
    assert len(df) == 3
    assert df.iloc[1].isna().all()  # Invalid row is all NaN
    assert not df.iloc[0].isna().all()  # Valid rows are not NaN


def test_clean_descriptors_removes_zero_variance():
    """Clean should drop columns with zero variance."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5], "c": [4, 5, 6]})
    cleaned = clean_descriptors(df)
    assert "b" not in cleaned.columns
    assert "a" in cleaned.columns


def test_clean_descriptors_removes_correlated():
    """Clean should drop highly correlated columns."""
    np.random.seed(42)
    x = np.random.randn(100)
    df = pd.DataFrame({"a": x, "b": x * 1.0001 + 0.0001, "c": np.random.randn(100)})
    cleaned = clean_descriptors(df)
    # Either a or b should be dropped (one of the correlated pair)
    assert len(cleaned.columns) <= 2


def test_compute_descriptors_fluorinated():
    """Fluorinated molecules should produce valid descriptors."""
    smiles = ["FC(F)C(F)F", "FC(=CF)C(F)F", "FC(F)(F)C(F)F"]
    df = compute_descriptors(smiles)
    assert len(df) == 3
    assert not df.iloc[0].isna().all()
