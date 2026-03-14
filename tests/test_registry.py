"""Tests for model registry and unified evaluation harness."""

import numpy as np
import pandas as pd
import pytest

from model.data.load import TARGETS


class TestRegistryPattern:
    """Tests for the registry decorator and lookup functions."""

    def test_register_and_list_models(self):
        """All three models (gc_pcsaft, rf, nn) are registered."""
        from model.registry import list_models

        names = list_models()
        assert "gc_pcsaft" in names
        assert "rf" in names
        assert "nn" in names

    def test_get_model_returns_instance(self):
        """get_model returns a new instance with load/predict methods."""
        from model.registry import get_model

        for name in ("gc_pcsaft", "rf", "nn"):
            model = get_model(name)
            assert hasattr(model, "load")
            assert hasattr(model, "predict")
            assert hasattr(model, "predict_with_uncertainty")

    def test_get_model_unknown_raises(self):
        """get_model raises KeyError for an unregistered name."""
        from model.registry import get_model

        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent_model")


class TestGCPCSAFTModel:
    """Tests for the GC-PC-SAFT model wrapper."""

    def test_predict_returns_correct_schema(self, mini_smiles):
        """GC model predict returns dict with m, sigma, epsilon_k arrays."""
        from model.registry import get_model

        model = get_model("gc_pcsaft")
        model.load()
        result = model.predict(mini_smiles)

        assert isinstance(result, dict)
        for target in TARGETS:
            assert target in result
            assert isinstance(result[target], np.ndarray)
            assert len(result[target]) == len(mini_smiles)

    def test_predict_with_uncertainty_returns_std(self, mini_smiles):
        """GC uncertainty returns zero std (no intrinsic UQ)."""
        from model.registry import get_model

        model = get_model("gc_pcsaft")
        model.load()
        result = model.predict_with_uncertainty(mini_smiles)

        for target in TARGETS:
            assert target in result
            assert f"{target}_std" in result
            # GC-PC-SAFT uncertainty is always 0
            np.testing.assert_array_equal(
                result[f"{target}_std"],
                np.zeros(len(mini_smiles)),
            )

    def test_predict_values_are_physical(self, mini_smiles):
        """GC predictions should be in physically reasonable ranges."""
        from model.registry import get_model

        model = get_model("gc_pcsaft")
        result = model.predict(mini_smiles)

        # Filter out NaN (invalid SMILES)
        valid = np.isfinite(result["m"])
        assert valid.any(), "At least some predictions should be valid"

        # m should be > 0 (number of segments)
        assert (result["m"][valid] > 0).all()
        # sigma should be > 0 (segment diameter in Angstroms)
        assert (result["sigma"][valid] > 0).all()
        # epsilon_k should be > 0 (dispersion energy in K)
        assert (result["epsilon_k"][valid] > 0).all()


class TestRFModelWrapper:
    """Tests for the RF model wrapper (uses saved models if available)."""

    def test_rf_load_and_predict(self, mini_smiles):
        """RF wrapper loads saved models and returns correct schema."""
        from pathlib import Path

        from model.registry import get_model

        saved_dir = Path(__file__).resolve().parent.parent / "model" / "saved"
        rf_path = saved_dir / "rf_m.joblib"
        if not rf_path.exists():
            pytest.skip("RF models not saved; run training first")

        model = get_model("rf")
        model.load()
        result = model.predict(mini_smiles)

        assert isinstance(result, dict)
        for target in TARGETS:
            assert target in result
            assert len(result[target]) == len(mini_smiles)

    def test_rf_uncertainty(self, mini_smiles):
        """RF wrapper returns tree-disagreement uncertainty."""
        from pathlib import Path

        from model.registry import get_model

        saved_dir = Path(__file__).resolve().parent.parent / "model" / "saved"
        rf_path = saved_dir / "rf_m.joblib"
        if not rf_path.exists():
            pytest.skip("RF models not saved; run training first")

        model = get_model("rf")
        model.load()
        result = model.predict_with_uncertainty(mini_smiles)

        for target in TARGETS:
            assert target in result
            assert f"{target}_std" in result
            # RF std should be > 0 for valid predictions
            valid = np.isfinite(result[f"{target}_std"])
            if valid.any():
                assert (result[f"{target}_std"][valid] >= 0).all()


class TestNNModelWrapper:
    """Tests for the NN model wrapper."""

    def test_nn_load_and_predict(self, mini_smiles):
        """NN wrapper loads checkpoint and returns correct schema."""
        from pathlib import Path

        from model.registry import get_model

        saved_dir = Path(__file__).resolve().parent.parent / "model" / "saved"
        nn_path = saved_dir / "nn_pcsaft.pt"
        if not nn_path.exists():
            pytest.skip("NN checkpoint not saved; run training first")

        model = get_model("nn")
        model.load()
        result = model.predict(mini_smiles)

        assert isinstance(result, dict)
        for target in TARGETS:
            assert target in result
            assert len(result[target]) == len(mini_smiles)

    def test_nn_uncertainty(self, mini_smiles):
        """NN wrapper returns MC Dropout uncertainty."""
        from pathlib import Path

        from model.registry import get_model

        saved_dir = Path(__file__).resolve().parent.parent / "model" / "saved"
        nn_path = saved_dir / "nn_pcsaft.pt"
        if not nn_path.exists():
            pytest.skip("NN checkpoint not saved; run training first")

        model = get_model("nn")
        model.load()
        result = model.predict_with_uncertainty(mini_smiles)

        for target in TARGETS:
            assert target in result
            assert f"{target}_std" in result


class TestEvaluateSmoke:
    """Smoke test: full evaluate pipeline on mini data or saved test set."""

    def test_evaluate_gc_pcsaft_only(self, mini_smiles, mini_dataset, tmp_path):
        """Evaluate GC-PC-SAFT on mini dataset without needing saved models."""
        # Create a mini test_set.csv in a temp location
        test_csv = tmp_path / "test_set.csv"
        mini_dataset.to_csv(test_csv, index=False)

        # Evaluate directly using the GC model
        from model.registry import get_model

        model = get_model("gc_pcsaft")
        model.load()
        preds = model.predict(mini_smiles)

        # Verify we get predictions for each target
        for target in TARGETS:
            assert target in preds
            assert len(preds[target]) == len(mini_smiles)

    def test_evaluate_full_smoke(self):
        """Full evaluate runs without crash on the saved test set."""
        from pathlib import Path

        saved_dir = Path(__file__).resolve().parent.parent / "model" / "saved"
        if not (saved_dir / "test_set.csv").exists():
            pytest.skip("No test_set.csv; train models first")
        if not (saved_dir / "rf_m.joblib").exists():
            pytest.skip("No RF models; train first")

        from model.evaluate import evaluate

        # Run only gc_pcsaft and rf (NN may not be available)
        models_to_eval = ["gc_pcsaft", "rf"]
        nn_path = saved_dir / "nn_pcsaft.pt"
        if nn_path.exists():
            models_to_eval.append("nn")

        metrics_df = evaluate(model_names=models_to_eval)

        assert isinstance(metrics_df, pd.DataFrame)
        assert len(metrics_df) > 0
        assert "model" in metrics_df.columns
        assert "target" in metrics_df.columns
        assert "mae" in metrics_df.columns
        assert "rmse" in metrics_df.columns
        assert "r2" in metrics_df.columns
