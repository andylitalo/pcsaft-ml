"""End-to-end smoke test for the core ML pipeline.

Exercises: load_data -> split_data -> build_features -> train (1 tree) -> predict
on the mini fixture to verify integration between modules.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from model.data.descriptors import (
    build_features_with_names,
    clean_descriptors,
    compute_descriptors,
)
from model.data.load import TARGETS


@pytest.fixture
def mini_pipeline_data(mini_dataset):
    """Prepare train/test split from mini dataset."""
    train_df = mini_dataset.iloc[:8].copy()
    test_df = mini_dataset.iloc[8:].copy()
    return train_df, test_df


def test_full_pipeline_smoke(mini_pipeline_data, tmp_model_dir):
    """Smoke test: the full pipeline from features to prediction runs without error."""
    train_df, test_df = mini_pipeline_data

    # Compute features on all data together to ensure consistent columns,
    # then split -- this mimics what build_features does internally with
    # clean_descriptors being data-dependent.
    all_smiles = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    X_all, feature_names = build_features_with_names(
        all_smiles, use_morgan=True, use_rdkit=True
    )
    X_train = X_all[: len(train_df)]
    X_test = X_all[len(train_df) :]

    assert X_train.shape[0] == len(train_df)
    assert X_test.shape[0] == len(test_df)
    assert len(feature_names) == X_train.shape[1]
    assert len(feature_names) > 2000  # Morgan + RDKit

    models = {}
    for target in TARGETS:
        rf = RandomForestRegressor(n_estimators=1, random_state=42)
        rf.fit(X_train, train_df[target].values)
        models[target] = rf

    for target in TARGETS:
        preds = models[target].predict(X_test)
        assert len(preds) == len(X_test)
        assert all(pd.notna(preds))


def test_full_pipeline_smoke_rdkit_only(mini_pipeline_data, tmp_model_dir):
    """Smoke test: pipeline works with RDKit-only features (backward compat)."""
    train_df, test_df = mini_pipeline_data

    X_train_raw = compute_descriptors(train_df["smiles"].tolist())
    X_test_raw = compute_descriptors(test_df["smiles"].tolist())

    assert X_train_raw.shape[0] == len(train_df)
    assert X_test_raw.shape[0] == len(test_df)

    X_train = clean_descriptors(X_train_raw)
    feature_names = X_train.columns.tolist()
    assert len(feature_names) > 0

    X_test = X_test_raw[feature_names]

    train_mask = X_train.notna().all(axis=1)
    test_mask = X_test.notna().all(axis=1)
    X_train = X_train[train_mask]
    X_test = X_test[test_mask]
    train_df = train_df.iloc[train_mask.values]
    test_df = test_df.iloc[test_mask.values]

    models = {}
    for target in TARGETS:
        rf = RandomForestRegressor(n_estimators=1, random_state=42)
        rf.fit(X_train, train_df[target].values)
        models[target] = rf

    for target in TARGETS:
        preds = models[target].predict(X_test.fillna(0))
        assert len(preds) == len(X_test)
        assert all(pd.notna(preds))


def test_predictions_have_correct_schema(mini_pipeline_data, tmp_model_dir):
    """Verify prediction output contains expected columns and value ranges."""
    train_df, test_df = mini_pipeline_data

    # Use combined data for consistent feature dimensions
    all_smiles = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    X_all, _ = build_features_with_names(
        all_smiles, use_morgan=True, use_rdkit=True
    )
    X_train = X_all[: len(train_df)]
    X_test = X_all[len(train_df) :]

    models = {}
    for target in TARGETS:
        rf = RandomForestRegressor(n_estimators=1, random_state=42)
        rf.fit(X_train, train_df[target].values)
        models[target] = rf

    results = pd.DataFrame({"smiles": test_df["smiles"].tolist()})
    for target in TARGETS:
        results[target] = models[target].predict(X_test)

    assert set(results.columns) == {"smiles", "m", "sigma", "epsilon_k"}
    assert (results["m"] > 0).all(), "m should be positive"
    assert (results["sigma"] > 0).all(), "sigma should be positive"
    assert (results["epsilon_k"] > 0).all(), "epsilon_k should be positive"


def test_manifest_written_after_train(mini_dataset, tmp_model_dir):
    """Verify that train() writes MANIFEST.json with artifact metadata."""
    from model.train import train

    with patch("model.train.SAVED_DIR", tmp_model_dir):
        with patch("model.data.load.ESPER_CSV", Path("/nonexistent")):
            with patch("model.data.load.MLSAFT_CSV", Path("/nonexistent")):
                train(source="fallback", tune=False, features="combined")

    manifest_path = tmp_model_dir / "MANIFEST.json"
    assert manifest_path.exists(), "MANIFEST.json should be created by train()"
    manifest = json.loads(manifest_path.read_text())
    assert "artifacts" in manifest
    filenames = [a["filename"] for a in manifest["artifacts"]]
    assert "rf_m.joblib" in filenames
    assert "rf_sigma.joblib" in filenames
    assert "rf_epsilon_k.joblib" in filenames
    assert "feature_names.joblib" in filenames
    assert "feature_config.json" in filenames
    assert "test_set.csv" in filenames
