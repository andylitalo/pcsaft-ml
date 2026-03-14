"""Tests for combined data loading, InChI deduplication, and stratified splitting."""

import numpy as np
import pandas as pd

from model.data.load import deduplicate_by_inchi, split_data


def test_deduplicate_by_inchi():
    """Duplicate molecules (same InChI) are removed."""
    df = pd.DataFrame({
        "smiles": ["CCO", "OCC", "C"],  # CCO and OCC are the same molecule (ethanol)
        "m": [2.38, 2.40, 1.00],
        "sigma": [3.18, 3.20, 3.70],
        "epsilon_k": [198.2, 200.0, 150.0],
    })
    deduped = deduplicate_by_inchi(df)
    assert len(deduped) == 2  # CCO and OCC should collapse to one


def test_deduplicate_by_inchi_preserves_first():
    """First occurrence is kept when duplicates exist."""
    df = pd.DataFrame({
        "smiles": ["CCO", "OCC"],
        "m": [2.38, 9.99],
        "sigma": [3.18, 9.99],
        "epsilon_k": [198.2, 999.0],
    })
    deduped = deduplicate_by_inchi(df)
    assert len(deduped) == 1
    assert deduped.iloc[0]["m"] == 2.38  # First row preserved


def test_deduplicate_by_inchi_handles_invalid():
    """Invalid SMILES are dropped during deduplication."""
    df = pd.DataFrame({
        "smiles": ["CCO", "INVALID_SMILES", "C"],
        "m": [2.38, 1.0, 1.0],
        "sigma": [3.18, 3.0, 3.70],
        "epsilon_k": [198.2, 100.0, 150.0],
    })
    deduped = deduplicate_by_inchi(df)
    # Invalid SMILES gets None InChI, dropped by dropna
    assert len(deduped) == 2


def test_stratified_split():
    """Test set covers epsilon_k range when using stratified splitting."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "smiles": [f"C{'C' * i}" for i in range(n)],
        "m": np.random.uniform(1, 10, n),
        "sigma": np.random.uniform(2, 5, n),
        "epsilon_k": np.concatenate([
            np.random.uniform(100, 200, n // 4),
            np.random.uniform(200, 300, n // 4),
            np.random.uniform(300, 400, n // 4),
            np.random.uniform(400, 600, n // 4),
        ]),
    })
    train_df, test_df = split_data(df, stratify_bins=5)

    # Test set should cover the range of epsilon_k
    assert test_df["epsilon_k"].min() < 200, "Test set missing low epsilon_k"
    assert test_df["epsilon_k"].max() > 400, "Test set missing high epsilon_k"

    # Roughly 80/20 split
    assert 0.7 < len(train_df) / n < 0.9


def test_stratified_split_small_dataset():
    """Small datasets fall back to non-stratified splitting."""
    df = pd.DataFrame({
        "smiles": ["C", "CC", "CCC"],
        "m": [1.0, 1.6, 2.0],
        "sigma": [3.7, 3.5, 3.6],
        "epsilon_k": [150.0, 191.4, 208.1],
    })
    # With only 3 rows and stratify_bins=5, should fall back (3 < 5*5)
    train_df, test_df = split_data(df, stratify_bins=5)
    assert len(train_df) + len(test_df) == 3
