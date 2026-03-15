"""Tests for the uncertainty-weighted ensemble model.

Uses mocking to avoid requiring actual saved model files.  Each mock
constituent model returns predictable predictions and uncertainties so
that the inverse-variance weighting logic can be verified analytically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from model.data.load import TARGETS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(
    preds: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
) -> MagicMock:
    """Create a mock model that returns the given predictions/uncertainties."""
    model = MagicMock()
    result = {}
    for t in TARGETS:
        result[t] = preds[t]
        result[f"{t}_std"] = stds[t]
    model.predict_with_uncertainty.return_value = result
    model.predict.return_value = {t: preds[t] for t in TARGETS}
    return model


def _mock_get_model(mock_models: dict[str, MagicMock]):
    """Return a patched get_model function that returns mock models."""
    def _get(name):
        if name not in mock_models:
            raise KeyError(f"Unknown model {name!r}")
        m = mock_models[name]
        # Return a wrapper that behaves like registry output
        wrapper = MagicMock()
        wrapper.load = MagicMock()
        wrapper.predict = m.predict
        wrapper.predict_with_uncertainty = m.predict_with_uncertainty
        return wrapper
    return _get


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWeightedEnsemble:
    """Tests for WeightedEnsemble."""

    def test_normal_two_model_ensemble(self):
        """Two models with different uncertainties produce weighted mean."""
        n = 5
        # Model A: predictions = 10, sigma = 1  -> weight = 1
        # Model B: predictions = 20, sigma = 2  -> weight = 0.25
        # Weighted mean = (1*10 + 0.25*20) / (1 + 0.25) = 15/1.25 = 12
        preds_a = {t: np.full(n, 10.0) for t in TARGETS}
        stds_a = {t: np.full(n, 1.0) for t in TARGETS}
        preds_b = {t: np.full(n, 20.0) for t in TARGETS}
        stds_b = {t: np.full(n, 2.0) for t in TARGETS}

        mock_a = _make_mock_model(preds_a, stds_a)
        mock_b = _make_mock_model(preds_b, stds_b)

        mocks = {"model_a": mock_a, "model_b": mock_b}

        with patch("model.ensemble.weighted.get_model", side_effect=_mock_get_model(mocks)):
            from model.ensemble.weighted import WeightedEnsemble
            ens = WeightedEnsemble(["model_a", "model_b"])

        result = ens.predict(["C"] * n)

        expected = 12.0  # (1*10 + 0.25*20) / 1.25
        for t in TARGETS:
            np.testing.assert_allclose(result[t], expected, atol=1e-6)

    def test_predict_with_uncertainty_returns_std_keys(self):
        """predict_with_uncertainty returns both predictions and _std keys."""
        n = 3
        preds_a = {t: np.full(n, 5.0) for t in TARGETS}
        stds_a = {t: np.full(n, 0.5) for t in TARGETS}

        mock_a = _make_mock_model(preds_a, stds_a)
        mocks = {"model_a": mock_a}

        with patch("model.ensemble.weighted.get_model", side_effect=_mock_get_model(mocks)):
            from model.ensemble.weighted import WeightedEnsemble
            ens = WeightedEnsemble(["model_a"])

        result = ens.predict_with_uncertainty(["C"] * n)

        for t in TARGETS:
            assert t in result, f"Missing key {t}"
            assert f"{t}_std" in result, f"Missing key {t}_std"
            assert len(result[t]) == n
            assert len(result[f"{t}_std"]) == n

    def test_nan_handling_invalid_smiles(self):
        """If a model returns NaN for a molecule, exclude it from weighting."""
        # Model A: all valid
        preds_a = {t: np.array([10.0, 10.0, 10.0]) for t in TARGETS}
        stds_a = {t: np.array([1.0, 1.0, 1.0]) for t in TARGETS}
        # Model B: second molecule is NaN
        preds_b = {t: np.array([20.0, np.nan, 20.0]) for t in TARGETS}
        stds_b = {t: np.array([1.0, np.nan, 1.0]) for t in TARGETS}

        mock_a = _make_mock_model(preds_a, stds_a)
        mock_b = _make_mock_model(preds_b, stds_b)
        mocks = {"model_a": mock_a, "model_b": mock_b}

        with patch("model.ensemble.weighted.get_model", side_effect=_mock_get_model(mocks)):
            from model.ensemble.weighted import WeightedEnsemble
            ens = WeightedEnsemble(["model_a", "model_b"])

        result = ens.predict(["C", "INVALID", "CC"])

        # Molecule 1 and 3: equal weights (sigma=1 for both) -> mean = 15
        np.testing.assert_allclose(result["m"][0], 15.0, atol=1e-6)
        np.testing.assert_allclose(result["m"][2], 15.0, atol=1e-6)

        # Molecule 2: only model A valid -> fallback to model A = 10
        np.testing.assert_allclose(result["m"][1], 10.0, atol=1e-6)

    def test_single_valid_model_fallback(self):
        """If only one model loads, ensemble falls back to that model."""
        n = 4
        preds_a = {t: np.full(n, 7.0) for t in TARGETS}
        stds_a = {t: np.full(n, 0.3) for t in TARGETS}

        mock_a = _make_mock_model(preds_a, stds_a)

        def _get(name):
            if name == "model_a":
                wrapper = MagicMock()
                wrapper.load = MagicMock()
                wrapper.predict = mock_a.predict
                wrapper.predict_with_uncertainty = mock_a.predict_with_uncertainty
                return wrapper
            raise KeyError(f"Unknown model {name!r}")

        with patch("model.ensemble.weighted.get_model", side_effect=_get):
            from model.ensemble.weighted import WeightedEnsemble
            ens = WeightedEnsemble(["model_a", "model_missing"])

        # Should have 1 model loaded
        assert len(ens.models) == 1

        result = ens.predict(["C"] * n)
        for t in TARGETS:
            np.testing.assert_allclose(result[t], 7.0, atol=1e-6)

    def test_ensemble_registered_in_registry(self):
        """The ensemble is registered as 'ensemble' in the model registry."""
        from model.registry import MODELS, get_model

        assert "ensemble" in MODELS
        model = get_model("ensemble")
        assert hasattr(model, "load")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_with_uncertainty")

    def test_weight_computation_correctness(self):
        """Verify inverse-variance weights are computed correctly."""
        n = 2
        # Model A: sigma = 1  -> w = 1/1 = 1
        # Model B: sigma = 3  -> w = 1/9
        # Normalized: A = 1/(1+1/9) = 9/10, B = (1/9)/(1+1/9) = 1/10
        preds_a = {t: np.full(n, 100.0) for t in TARGETS}
        stds_a = {t: np.full(n, 1.0) for t in TARGETS}
        preds_b = {t: np.full(n, 200.0) for t in TARGETS}
        stds_b = {t: np.full(n, 3.0) for t in TARGETS}

        mock_a = _make_mock_model(preds_a, stds_a)
        mock_b = _make_mock_model(preds_b, stds_b)
        mocks = {"model_a": mock_a, "model_b": mock_b}

        with patch("model.ensemble.weighted.get_model", side_effect=_mock_get_model(mocks)):
            from model.ensemble.weighted import WeightedEnsemble
            ens = WeightedEnsemble(["model_a", "model_b"])

        weights = ens.get_per_model_weights(["C"] * n)

        # Check weights for model_a
        expected_a = 1.0 / (1.0 + 1.0 / 9.0)  # 9/10 = 0.9
        expected_b = (1.0 / 9.0) / (1.0 + 1.0 / 9.0)  # 1/10 = 0.1

        for t in TARGETS:
            np.testing.assert_allclose(
                weights["model_a"][t], expected_a, atol=1e-6,
            )
            np.testing.assert_allclose(
                weights["model_b"][t], expected_b, atol=1e-6,
            )

    def test_sigma_floor_prevents_division_by_zero(self):
        """Zero std is clamped to floor (1e-6) to avoid div-by-zero."""
        n = 2
        preds_a = {t: np.full(n, 5.0) for t in TARGETS}
        stds_a = {t: np.full(n, 0.0) for t in TARGETS}  # zero!

        mock_a = _make_mock_model(preds_a, stds_a)
        mocks = {"model_a": mock_a}

        with patch("model.ensemble.weighted.get_model", side_effect=_mock_get_model(mocks)):
            from model.ensemble.weighted import WeightedEnsemble
            ens = WeightedEnsemble(["model_a"])

        result = ens.predict_with_uncertainty(["C"] * n)

        for t in TARGETS:
            assert np.all(np.isfinite(result[t]))
            assert np.all(np.isfinite(result[f"{t}_std"]))

    def test_all_nan_returns_nan(self):
        """If all models return NaN for a molecule, ensemble returns NaN."""
        n = 1
        preds_a = {t: np.full(n, np.nan) for t in TARGETS}
        stds_a = {t: np.full(n, np.nan) for t in TARGETS}

        mock_a = _make_mock_model(preds_a, stds_a)
        mocks = {"model_a": mock_a}

        with patch("model.ensemble.weighted.get_model", side_effect=_mock_get_model(mocks)):
            from model.ensemble.weighted import WeightedEnsemble
            ens = WeightedEnsemble(["model_a"])

        result = ens.predict(["INVALID_SMILES"])
        for t in TARGETS:
            assert np.isnan(result[t][0])

    def test_predict_disagreement(self):
        """Disagreement is max pairwise absolute difference."""
        # Model A predicts 10, model B predicts 20 -> disagreement = 10
        preds_a = {t: np.array([10.0, 10.0]) for t in TARGETS}
        stds_a = {t: np.array([1.0, 1.0]) for t in TARGETS}
        preds_b = {t: np.array([20.0, 15.0]) for t in TARGETS}
        stds_b = {t: np.array([1.0, 1.0]) for t in TARGETS}

        mock_a = _make_mock_model(preds_a, stds_a)
        mock_b = _make_mock_model(preds_b, stds_b)
        mocks = {"model_a": mock_a, "model_b": mock_b}

        with patch("model.ensemble.weighted.get_model", side_effect=_mock_get_model(mocks)):
            from model.ensemble.weighted import WeightedEnsemble
            ens = WeightedEnsemble(["model_a", "model_b"])

        dis = ens.predict_disagreement(["C", "CC"])
        for t in TARGETS:
            np.testing.assert_allclose(dis[t][0], 10.0, atol=1e-6)
            np.testing.assert_allclose(dis[t][1], 5.0, atol=1e-6)

    def test_no_models_loaded_raises(self):
        """If no models can be loaded, constructor raises RuntimeError."""
        def _fail(name):
            raise KeyError(f"Unknown model {name!r}")

        with patch("model.ensemble.weighted.get_model", side_effect=_fail):
            from model.ensemble.weighted import WeightedEnsemble
            with pytest.raises(RuntimeError, match="No constituent models"):
                WeightedEnsemble(["nonexistent1", "nonexistent2"])
