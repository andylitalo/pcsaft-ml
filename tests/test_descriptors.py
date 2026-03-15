"""Tests for molecular descriptor computation."""

import numpy as np
import pandas as pd

from model.data.descriptors import (
    build_features,
    build_features_with_names,
    clean_descriptors,
    compute_descriptors,
)


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


def test_build_features_with_gc_pcsaft():
    """Build features with GC-PC-SAFT predictions added."""
    smiles = ["C", "CC", "CCC"]
    # Get baseline feature count without GC
    features_no_gc = build_features(smiles, use_morgan=True, use_rdkit=True, use_gc_pcsaft=False)
    # With GC should add exactly 3 features
    features_with_gc = build_features(smiles, use_morgan=True, use_rdkit=True, use_gc_pcsaft=True)
    assert features_with_gc.shape[0] == 3
    assert features_with_gc.shape[1] == features_no_gc.shape[1] + 3
    # Last 3 columns should be GC features (m, sigma, epsilon_k)
    assert np.isfinite(features_with_gc[:, -3:]).all()


def test_build_features_gc_only():
    """Build features with only GC-PC-SAFT predictions."""
    smiles = ["C", "CC", "CCC"]
    features = build_features(smiles, use_morgan=False, use_rdkit=False, use_gc_pcsaft=True)
    assert features.shape == (3, 3)  # Exactly 3 GC features
    assert np.isfinite(features).all()


def test_build_features_with_names_gc_pcsaft():
    """Build features with names should include GC feature labels."""
    smiles = ["C", "CC"]
    features, names = build_features_with_names(
        smiles, use_morgan=False, use_rdkit=True, use_gc_pcsaft=True
    )
    # Should have RDKit + 3 GC features
    assert "gc_m" in names
    assert "gc_sigma" in names
    assert "gc_epsilon_k" in names
    assert len(names) == features.shape[1]


def test_build_features_gc_handles_nan():
    """GC features with NaN (unparseable SMILES) should be filled with zeros."""
    # Include an invalid SMILES that GC-PC-SAFT cannot parse
    smiles = ["C", "INVALID_SMILES", "CC"]
    features = build_features(smiles, use_morgan=False, use_rdkit=False, use_gc_pcsaft=True)
    # Should have no NaN (filled with 0)
    assert np.isfinite(features).all()
    # Invalid SMILES should have zero GC features
    assert (features[1, :] == 0).all()
