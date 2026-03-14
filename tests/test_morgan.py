"""Tests for Morgan fingerprint computation and combined feature builder."""

import numpy as np
import pytest

from model.data.descriptors import (
    build_features,
    build_features_with_names,
    compute_morgan_fingerprints,
)


def test_compute_morgan_fingerprints_shape():
    """Valid SMILES produce correct shape (n_molecules, n_bits)."""
    smiles = ["C", "CC", "C1CCCC1"]
    fps = compute_morgan_fingerprints(smiles, radius=2, n_bits=2048)
    assert fps.shape == (3, 2048)
    assert fps.dtype == np.float32


def test_compute_morgan_fingerprints_invalid():
    """Invalid SMILES produce a zero row."""
    smiles = ["C", "INVALID_SMILES", "CC"]
    fps = compute_morgan_fingerprints(smiles)
    assert fps.shape == (3, 2048)
    # Invalid SMILES row should be all zeros
    assert fps[1].sum() == 0.0
    # Valid SMILES should have at least some bits set
    assert fps[0].sum() > 0.0
    assert fps[2].sum() > 0.0


def test_compute_morgan_fingerprints_binary():
    """Morgan fingerprint values should be 0 or 1."""
    smiles = ["CCO", "c1ccccc1", "FC(F)(F)C(F)F"]
    fps = compute_morgan_fingerprints(smiles)
    assert set(np.unique(fps)).issubset({0.0, 1.0})


def test_build_features_combined():
    """Combined features have >2000 columns (2048 Morgan + ~100 RDKit)."""
    smiles = ["C", "CC", "CCC", "C1CCCC1"]
    X = build_features(smiles, use_morgan=True, use_rdkit=True)
    assert X.shape[0] == 4
    assert X.shape[1] > 2000  # 2048 Morgan bits + cleaned RDKit descriptors


def test_build_features_no_nan():
    """No NaN in output from build_features."""
    smiles = ["C", "CC", "INVALID", "C1CCCC1"]
    X = build_features(smiles, use_morgan=True, use_rdkit=True)
    assert not np.any(np.isnan(X))


def test_build_features_morgan_only():
    """Morgan-only mode produces exactly n_bits columns."""
    smiles = ["C", "CC", "CCC"]
    X = build_features(smiles, use_morgan=True, use_rdkit=False, morgan_bits=1024)
    assert X.shape == (3, 1024)


def test_build_features_rdkit_only():
    """RDKit-only mode produces <500 columns (no Morgan)."""
    smiles = ["C", "CC", "CCC"]
    X = build_features(smiles, use_morgan=False, use_rdkit=True)
    assert X.shape[0] == 3
    assert X.shape[1] < 500  # Cleaned RDKit descriptors only


def test_build_features_with_names_returns_names():
    """build_features_with_names returns matching feature names."""
    smiles = ["C", "CC", "CCC"]
    X, names = build_features_with_names(smiles, use_morgan=True, use_rdkit=True)
    assert len(names) == X.shape[1]
    # Morgan names should start with "morgan_"
    morgan_names = [n for n in names if n.startswith("morgan_")]
    assert len(morgan_names) == 2048


def test_build_features_neither_raises():
    """Passing use_morgan=False, use_rdkit=False should raise."""
    with pytest.raises(ValueError, match="At least one"):
        build_features(["C"], use_morgan=False, use_rdkit=False)


def test_compute_morgan_fingerprints_fluorinated():
    """Fluorinated molecules produce valid fingerprints."""
    smiles = ["FC(F)C(F)F", "FC(=CF)C(F)F", "FC(F)(F)C(F)F"]
    fps = compute_morgan_fingerprints(smiles)
    assert fps.shape == (3, 2048)
    for i in range(3):
        assert fps[i].sum() > 0.0  # Non-zero for valid molecules
