"""Smoke tests for the Step 47 SVR wrapper."""

import numpy as np
import pytest

pytest.importorskip("sklearn")


@pytest.fixture
def mini_features_svm():
    """Small random feature matrix for SVR smoke tests."""
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


class TestSVMPCSAFT:
    """Tests for the SVR model wrapper."""

    def test_svm_in_registry(self):
        """SVM is registered in the model registry."""
        import model.svm  # noqa: F401
        from model.registry import MODELS

        assert "svm" in MODELS

    def test_svm_fit_predict_and_save_load(self, mini_features_svm, mini_targets_multi, tmp_path):
        """Target-wise SVR fits, predicts, and reloads."""
        from model.svm.svm_model import SVMPCSAFT

        model = SVMPCSAFT()
        model.PARAM_GRID = {
            "regressor__C": [1],
            "regressor__gamma": ["scale"],
            "regressor__epsilon": [0.1],
        }
        model.fit(
            mini_features_svm,
            mini_targets_multi,
            feature_names=[f"f{i}" for i in range(mini_features_svm.shape[1])],
            cv_folds=2,
        )

        preds = model.predict(mini_features_svm)
        assert preds.shape == mini_targets_multi.shape
        assert set(model.best_params_) == {"m", "sigma", "epsilon_k"}
        assert set(model.cv_best_scores_) == {"m", "sigma", "epsilon_k"}

        save_path = tmp_path / "svm_test.joblib"
        model.save(save_path)
        assert save_path.exists()

        reloaded = SVMPCSAFT()
        reloaded.load(save_path)
        np.testing.assert_array_almost_equal(preds, reloaded.predict(mini_features_svm))

    def test_svm_single_target_mode(self, mini_features_svm, mini_targets_multi):
        """Single-target fitting works for epsilon_k-only tuning."""
        from model.svm.svm_model import SVMPCSAFT

        model = SVMPCSAFT(target_names=["epsilon_k"])
        model.PARAM_GRID = {
            "regressor__C": [1],
            "regressor__gamma": ["scale"],
            "regressor__epsilon": [0.1],
        }
        model.fit(
            mini_features_svm,
            mini_targets_multi[:, 2],
            cv_folds=2,
        )

        preds = model.predict(mini_features_svm)
        assert preds.shape == (mini_features_svm.shape[0],)
        assert "epsilon_k" in model.best_params_
