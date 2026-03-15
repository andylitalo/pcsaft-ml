"""Tests for the FastAPI serving layer."""

import csv
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient with the model loaded."""
    saved_dir = Path(__file__).resolve().parent.parent / "model" / "saved"
    rf_path = saved_dir / "rf_m.joblib"
    if not rf_path.exists():
        pytest.skip("RF models not saved; run training first")

    from serving.app import app

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "rf"
        assert "version" in data


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_valid_smiles(self, client):
        resp = client.post("/predict", json={"smiles": ["C1CCCC1", "CCCCCC"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "rf"
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert pred["valid"] is True
            assert pred["m"] > 0
            assert pred["sigma"] > 0
            assert pred["epsilon_k"] > 0

    def test_predict_invalid_smiles_returns_valid_false(self, client):
        resp = client.post(
            "/predict", json={"smiles": ["C1CCCC1", "NOT_A_SMILES_STRING"]}
        )
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        # First should be valid
        assert preds[0]["valid"] is True
        # Second should be invalid (not a 500)
        assert preds[1]["valid"] is False

    def test_predict_empty_list_returns_422(self, client):
        resp = client.post("/predict", json={"smiles": []})
        assert resp.status_code == 422

    def test_predict_includes_uncertainty(self, client):
        resp = client.post("/predict", json={"smiles": ["C1CCCC1"]})
        assert resp.status_code == 200
        pred = resp.json()["predictions"][0]
        # RF provides tree-disagreement uncertainty
        if pred["uncertainty"] is not None:
            assert "m_std" in pred["uncertainty"]
            assert "sigma_std" in pred["uncertainty"]
            assert "epsilon_k_std" in pred["uncertainty"]

    def test_predict_includes_association_flag(self, client):
        # Ethanol has OH group -> associating
        resp = client.post("/predict", json={"smiles": ["CCO"]})
        assert resp.status_code == 200
        pred = resp.json()["predictions"][0]
        assert pred["is_associating"] is True

        # Pentane has no association sites
        resp = client.post("/predict", json={"smiles": ["CCCCC"]})
        assert resp.status_code == 200
        pred = resp.json()["predictions"][0]
        assert pred["is_associating"] is False

    def test_predict_includes_tanimoto_nn(self, client):
        """Test that predictions include Tanimoto similarity if TanimotoAD model is available."""
        resp = client.post("/predict", json={"smiles": ["C1CCCC1"]})
        assert resp.status_code == 200
        pred = resp.json()["predictions"][0]
        # tanimoto_nn may be None if model not available, or a float in [0, 1]
        if pred["tanimoto_nn"] is not None:
            assert 0.0 <= pred["tanimoto_nn"] <= 1.0


class TestSubmitDataEndpoint:
    """Tests for POST /submit-data."""

    def test_submit_data_valid(self, client, tmp_path, monkeypatch):
        csv_path = tmp_path / "submissions.csv"
        monkeypatch.setattr("serving.app.settings.submissions_path", str(csv_path))

        resp = client.post(
            "/submit-data",
            json={
                "smiles": "C1CCCC1",
                "m": 2.3655,
                "sigma": 3.7114,
                "epsilon_k": 288.84,
                "source": "Gross2001",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["smiles"] == "C1CCCC1"

        # Verify CSV was written
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["smiles"] == "C1CCCC1"

    def test_submit_data_negative_sigma_returns_422(self, client):
        resp = client.post(
            "/submit-data",
            json={
                "smiles": "C1CCCC1",
                "m": 2.3655,
                "sigma": -1.0,
                "epsilon_k": 288.84,
                "source": "test",
            },
        )
        assert resp.status_code == 422


class TestPredictionsMatchOffline:
    """Verify API predictions match direct model predictions."""

    def test_predictions_match_offline_model(self, client):
        from model.registry import get_model

        smiles_list = ["C1CCCC1", "CCCCCC", "c1ccccc1"]
        model = get_model("rf")
        model.load()
        offline = model.predict(smiles_list)

        resp = client.post("/predict", json={"smiles": smiles_list})
        assert resp.status_code == 200
        preds = resp.json()["predictions"]

        for i, smi in enumerate(smiles_list):
            if preds[i]["valid"]:
                np.testing.assert_allclose(
                    preds[i]["m"], offline["m"][i], rtol=1e-5,
                    err_msg=f"m mismatch for {smi}",
                )
                np.testing.assert_allclose(
                    preds[i]["sigma"], offline["sigma"][i], rtol=1e-5,
                    err_msg=f"sigma mismatch for {smi}",
                )
                np.testing.assert_allclose(
                    preds[i]["epsilon_k"], offline["epsilon_k"][i], rtol=1e-5,
                    err_msg=f"epsilon_k mismatch for {smi}",
                )
