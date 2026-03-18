"""Smoke tests for the Step 47 Ridge regression wrapper."""

import numpy as np
import pytest

pytest.importorskip("sklearn")


@pytest.fixture
def mini_features_ridge():
    """Small random feature matrix for Ridge smoke tests."""
    rng = np.random.RandomState(7)
    return rng.randn(18, 24).astype(np.float32)


@pytest.fixture
def mini_targets_multi():
    """Small random target matrix (n_samples, 3)."""
    rng = np.random.RandomState(11)
    return np.column_stack([
        rng.uniform(1, 5, 18),
        rng.uniform(3, 4, 18),
        rng.uniform(150, 350, 18),
    ]).astype(np.float32)


class TestRidgePCSAFT:
    """Tests for the Ridge regression model wrapper."""

    def test_ridge_in_registry(self):
        """Ridge is registered in the model registry."""
        import model.linear.ridge_model  # noqa: F401
        from model.registry import MODELS

        assert "ridge" in MODELS

    def test_ridge_fit_predict_and_save_load(
        self, mini_features_ridge, mini_targets_multi, tmp_path
    ):
        """Target-wise Ridge fits, predicts, and reloads."""
        from model.linear.ridge_model import RidgePCSAFT

        mdl = RidgePCSAFT()
        mdl.PARAM_GRID = {"regressor__alpha": [1.0]}
        mdl.fit(
            mini_features_ridge,
            mini_targets_multi,
            feature_names=[f"f{i}" for i in range(mini_features_ridge.shape[1])],
            cv_folds=2,
        )

        preds = mdl.predict(mini_features_ridge)
        assert preds.shape == mini_targets_multi.shape
        assert set(mdl.best_params_) == {"m", "sigma", "epsilon_k"}
        assert set(mdl.cv_best_scores_) == {"m", "sigma", "epsilon_k"}

        save_path = tmp_path / "ridge_test.joblib"
        mdl.save(save_path)
        assert save_path.exists()

        reloaded = RidgePCSAFT()
        reloaded.load(save_path)
        np.testing.assert_array_almost_equal(preds, reloaded.predict(mini_features_ridge))

    def test_ridge_single_target_mode(self, mini_features_ridge, mini_targets_multi):
        """Single-target fitting works for epsilon_k-only tuning."""
        from model.linear.ridge_model import RidgePCSAFT

        mdl = RidgePCSAFT(target_names=["epsilon_k"])
        mdl.PARAM_GRID = {"regressor__alpha": [1.0]}
        mdl.fit(
            mini_features_ridge,
            mini_targets_multi[:, 2],
            cv_folds=2,
        )

        preds = mdl.predict(mini_features_ridge)
        assert preds.shape == (mini_features_ridge.shape[0],)
        assert "epsilon_k" in mdl.best_params_
