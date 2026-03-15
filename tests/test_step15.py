"""Tests for Step 15: XGBoost and chemprop D-MPNN wrappers.

Smoke tests verify model components work on mini data without crashing.
They do NOT assert high accuracy.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_features_xgb():
    """Small random feature matrix for XGBoost tests."""
    rng = np.random.RandomState(42)
    return rng.randn(20, 50).astype(np.float32)


@pytest.fixture
def mini_targets_multi():
    """Small random target matrix (n_samples, 3)."""
    rng = np.random.RandomState(42)
    # Physically plausible ranges: m ~ 1-5, sigma ~ 3-4, epsilon_k ~ 150-350
    targets = np.column_stack([
        rng.uniform(1, 5, 20),
        rng.uniform(3, 4, 20),
        rng.uniform(150, 350, 20),
    ]).astype(np.float32)
    return targets


@pytest.fixture
def mini_smiles_short():
    """A handful of valid SMILES for chemprop tests."""
    return [
        "C", "CC", "CCC", "CCCC", "C1CCCC1",
        "c1ccccc1", "CCO", "CC=O", "CC(C)C", "C1CCCCC1",
        "CCCCC", "CCCCCC", "C(CC)CC", "C1CCC1", "CC(=O)O",
        "CCOC", "CCC(C)C", "CCN", "CC(C)(C)C", "CCCCCCC",
    ]


# ---------------------------------------------------------------------------
# XGBoost Tests
# ---------------------------------------------------------------------------

class TestXGBoostPCSAFT:
    """Tests for the XGBoost model wrapper."""

    def test_xgboost_in_registry(self):
        """XGBoost is registered in the model registry."""
        import model.xgb  # noqa: F401
        from model.registry import MODELS

        assert "xgboost" in MODELS

    def test_xgboost_instantiation(self):
        """XGBoostPCSAFT can be instantiated."""
        from model.xgb.xgb_model import XGBoostPCSAFT

        m = XGBoostPCSAFT()
        assert m.model_ is None
        assert m.best_params_ is None

    def test_xgboost_fit_predict(self, mini_features_xgb, mini_targets_multi):
        """XGBoost fits on mini data and returns correct shape."""
        from model.xgb.xgb_model import XGBoostPCSAFT

        m = XGBoostPCSAFT()
        # Use a tiny param grid for speed
        m.PARAM_GRID = {
            "estimator__n_estimators": [10],
            "estimator__max_depth": [3],
            "estimator__learning_rate": [0.1],
            "estimator__subsample": [0.8],
            "estimator__colsample_bytree": [0.8],
        }
        m.fit(
            mini_features_xgb, mini_targets_multi,
            feature_names=[f"f{i}" for i in range(50)],
            cv_folds=2,
        )

        preds = m.predict(mini_features_xgb)
        assert preds.shape == mini_targets_multi.shape

    def test_xgboost_predict_with_uncertainty(self, mini_features_xgb, mini_targets_multi):
        """predict_with_uncertainty returns mean and std arrays."""
        from model.xgb.xgb_model import XGBoostPCSAFT

        m = XGBoostPCSAFT()
        m.PARAM_GRID = {
            "estimator__n_estimators": [10],
            "estimator__max_depth": [3],
            "estimator__learning_rate": [0.1],
            "estimator__subsample": [0.8],
            "estimator__colsample_bytree": [0.8],
        }
        m.fit(mini_features_xgb, mini_targets_multi, cv_folds=2)

        means, stds = m.predict_with_uncertainty(mini_features_xgb)
        assert means.shape == mini_targets_multi.shape
        assert stds.shape == mini_targets_multi.shape
        # Stds should be non-negative
        assert (stds >= 0).all()

    def test_xgboost_feature_importances(self, mini_features_xgb, mini_targets_multi):
        """Feature importances are computed after fitting."""
        from model.xgb.xgb_model import XGBoostPCSAFT

        m = XGBoostPCSAFT()
        m.PARAM_GRID = {
            "estimator__n_estimators": [10],
            "estimator__max_depth": [3],
            "estimator__learning_rate": [0.1],
            "estimator__subsample": [0.8],
            "estimator__colsample_bytree": [0.8],
        }
        m.fit(
            mini_features_xgb, mini_targets_multi,
            feature_names=[f"f{i}" for i in range(50)],
            cv_folds=2,
        )

        imp_avg = m.feature_importances()
        assert imp_avg.shape == (50,)
        assert imp_avg.sum() > 0

        imp_t0 = m.feature_importances(target_idx=0)
        assert imp_t0.shape == (50,)

    def test_xgboost_save_load(self, mini_features_xgb, mini_targets_multi, tmp_path):
        """Model can be saved and reloaded."""
        from model.xgb.xgb_model import XGBoostPCSAFT

        m = XGBoostPCSAFT()
        m.PARAM_GRID = {
            "estimator__n_estimators": [10],
            "estimator__max_depth": [3],
            "estimator__learning_rate": [0.1],
            "estimator__subsample": [0.8],
            "estimator__colsample_bytree": [0.8],
        }
        m.fit(mini_features_xgb, mini_targets_multi, cv_folds=2)

        save_path = tmp_path / "xgb_test.joblib"
        m.save(save_path)
        assert save_path.exists()

        m2 = XGBoostPCSAFT()
        m2.load(save_path)
        preds1 = m.predict(mini_features_xgb)
        preds2 = m2.predict(mini_features_xgb)
        np.testing.assert_array_almost_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# chemprop Tests
# ---------------------------------------------------------------------------

class TestChempropPCSAFT:
    """Tests for the chemprop D-MPNN wrapper."""

    def test_chemprop_in_registry(self):
        """chemprop is registered in the model registry."""
        import model.chemprop_model  # noqa: F401
        from model.registry import MODELS

        assert "chemprop" in MODELS

    def test_chemprop_instantiation(self):
        """ChempropPCSAFT can be instantiated."""
        from model.chemprop_model.chemprop_wrapper import ChempropPCSAFT

        m = ChempropPCSAFT()
        assert m.model_ is None

    @pytest.mark.slow
    def test_chemprop_fit_predict(self, mini_smiles_short, mini_targets_multi):
        """chemprop fits on mini data and predicts correct shape."""
        from model.chemprop_model.chemprop_wrapper import ChempropPCSAFT

        m = ChempropPCSAFT()
        m.fit(
            mini_smiles_short,
            mini_targets_multi,
            max_epochs=2,
            batch_size=8,
            patience=2,
            hidden_dim=32,
            depth=1,
        )

        preds = m.predict(mini_smiles_short[:5])
        assert preds.shape == (5, 3)
        # At least some predictions should be finite (valid SMILES)
        assert np.isfinite(preds).any()

    @pytest.mark.slow
    def test_chemprop_predict_with_uncertainty(self, mini_smiles_short, mini_targets_multi):
        """predict_with_uncertainty returns mean and std arrays."""
        from model.chemprop_model.chemprop_wrapper import ChempropPCSAFT

        m = ChempropPCSAFT()
        m.fit(
            mini_smiles_short,
            mini_targets_multi,
            max_epochs=2,
            batch_size=8,
            patience=2,
            hidden_dim=32,
            depth=1,
        )

        means, stds = m.predict_with_uncertainty(mini_smiles_short[:5], n_forward=3)
        assert means.shape == (5, 3)
        assert stds.shape == (5, 3)
        # Stds should be non-negative where finite
        valid = np.isfinite(stds)
        if valid.any():
            assert (stds[valid] >= 0).all()

    @pytest.mark.slow
    def test_chemprop_save_load(self, mini_smiles_short, mini_targets_multi, tmp_path):
        """chemprop model can be saved and reloaded."""
        from model.chemprop_model.chemprop_wrapper import ChempropPCSAFT

        m = ChempropPCSAFT()
        m.fit(
            mini_smiles_short,
            mini_targets_multi,
            max_epochs=2,
            batch_size=8,
            patience=2,
            hidden_dim=300,
            depth=3,
        )

        save_path = tmp_path / "chemprop_test.pt"
        m.save(save_path)
        assert save_path.exists()

        m2 = ChempropPCSAFT()
        m2.load(save_path)
        preds1 = m.predict(mini_smiles_short[:3])
        preds2 = m2.predict(mini_smiles_short[:3])
        # Predictions should be close (same model weights)
        valid = np.isfinite(preds1) & np.isfinite(preds2)
        if valid.any():
            np.testing.assert_array_almost_equal(preds1[valid], preds2[valid], decimal=2)

    def test_chemprop_unfitted_raises(self):
        """Calling predict on an unfitted model raises RuntimeError."""
        from model.chemprop_model.chemprop_wrapper import ChempropPCSAFT

        m = ChempropPCSAFT()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict(["CCO"])

    @pytest.mark.slow
    def test_chemprop_invalid_smiles(self, mini_smiles_short, mini_targets_multi):
        """Invalid SMILES return NaN predictions."""
        from model.chemprop_model.chemprop_wrapper import ChempropPCSAFT

        m = ChempropPCSAFT()
        m.fit(
            mini_smiles_short,
            mini_targets_multi,
            max_epochs=1,
            batch_size=8,
            hidden_dim=32,
            depth=1,
        )

        preds = m.predict(["INVALID_SMILES", "CCO"])
        assert preds.shape == (2, 3)
        # Invalid SMILES should have NaN
        assert np.isnan(preds[0]).all()
        # Valid SMILES should have finite values
        assert np.isfinite(preds[1]).all()
