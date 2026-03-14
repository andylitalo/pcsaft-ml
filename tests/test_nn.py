"""Tests for the PCSAFTNet neural network module.

Smoke tests verify that model components work on a mini dataset
without crashing.  They do NOT assert high accuracy.
"""

import numpy as np
import pytest
import torch

from model.nn.architecture import TARGETS, PCSAFTNet
from model.nn.dataset import PCSAFTDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_features(mini_dataset):
    """Build a small random feature matrix matching mini_dataset size."""
    n = len(mini_dataset)
    rng = np.random.RandomState(42)
    return rng.randn(n, 100).astype(np.float32)


@pytest.fixture
def mini_targets(mini_dataset):
    """Extract targets from the mini dataset."""
    return {t: mini_dataset[t].values.astype(np.float32) for t in TARGETS}


@pytest.fixture
def small_model():
    """Small PCSAFTNet for unit tests (narrow layers for speed)."""
    return PCSAFTNet(
        input_dim=100,
        trunk_dims=[32, 16],
        head_dims=[8],
        dropout=0.2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pcsaft_net_forward_shape(small_model):
    """Forward pass returns correct output shapes for each target."""
    x = torch.randn(5, 100)
    preds = small_model(x)
    assert isinstance(preds, dict)
    for target in TARGETS:
        assert target in preds
        assert preds[target].shape == (5,), (
            f"Expected shape (5,) for {target}, got {preds[target].shape}"
        )


def test_pcsaft_net_predict_with_uncertainty(small_model):
    """MC Dropout returns both means and stds for all targets."""
    x = np.random.randn(5, 100).astype(np.float32)
    result = small_model.predict_with_uncertainty(x, n_forward=10)

    for target in TARGETS:
        assert target in result
        assert f"{target}_std" in result
        assert result[target].shape == (5,)
        assert result[f"{target}_std"].shape == (5,)
        # Std should be non-negative
        assert (result[f"{target}_std"] >= 0).all()

    # After predict_with_uncertainty, model should be back in eval mode
    assert not small_model.training


def test_pcsaft_dataset_length(mini_features, mini_targets):
    """Dataset reports the correct number of samples."""
    ds = PCSAFTDataset(mini_features, mini_targets)
    assert len(ds) == len(mini_features)


def test_pcsaft_dataset_getitem(mini_features, mini_targets):
    """Dataset __getitem__ returns tensors with correct shapes."""
    ds = PCSAFTDataset(mini_features, mini_targets)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (100,)
    assert isinstance(y, dict)
    for target in TARGETS:
        assert target in y
        assert y[target].shape == ()


def test_train_nn_smoke(mini_features, mini_targets, tmp_path, monkeypatch):
    """Quick smoke test: training runs for 2 epochs without error."""
    # Redirect saved dir to tmp_path
    import model.nn.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "SAVED_DIR", tmp_path)

    feature_config = {
        "features": "combined",
        "use_morgan": True,
        "use_rdkit": True,
        "n_features": 100,
    }

    model = trainer_module.train_nn(
        mini_features,
        mini_targets,
        feature_config,
        max_epochs=2,
        patience=5,
        batch_size=4,
        trunk_dims=[16, 8],
        head_dims=[4],
        val_fraction=0.2,
    )

    assert isinstance(model, PCSAFTNet)
    # Check artifacts were saved
    assert (tmp_path / "nn_pcsaft.pt").exists()
    assert (tmp_path / "nn_history.json").exists()


def test_ad_model_predictions(mini_features, tmp_path, monkeypatch):
    """Isolation Forest returns a boolean array of the correct shape."""
    import model.nn.ad as ad_module

    monkeypatch.setattr(ad_module, "SAVED_DIR", tmp_path)

    # Train
    ad_module.train_ad_model(mini_features)
    assert (tmp_path / "ad_model.joblib").exists()

    # Predict
    in_domain = ad_module.check_applicability_domain(mini_features)
    assert in_domain.shape == (len(mini_features),)
    assert in_domain.dtype == bool


def test_model_get_config(small_model):
    """get_config returns a serializable dict with all constructor args."""
    config = small_model.get_config()
    assert config["input_dim"] == 100
    assert config["trunk_dims"] == [32, 16]
    assert config["head_dims"] == [8]
    assert config["dropout"] == 0.2
