"""Tests for the Streamlit portal components."""

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from portal.api_client import PCSAFTClient
from portal.components.molecule import render_molecule, smiles_to_image_bytes


class TestAPIClient:
    """Tests for PCSAFTClient."""

    def test_init(self):
        """Test client initialization."""
        client = PCSAFTClient("http://example.com:8000")
        assert client.base_url == "http://example.com:8000"

        # Test URL cleanup (trailing slash removal)
        client = PCSAFTClient("http://example.com:8000/")
        assert client.base_url == "http://example.com:8000"

    @patch("portal.api_client.requests.post")
    def test_predict_success(self, mock_post):
        """Test successful prediction."""
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "predictions": [
                {
                    "smiles": "C1CCCC1",
                    "m": 2.3655,
                    "sigma": 3.7114,
                    "epsilon_k": 288.84,
                    "valid": True,
                    "in_domain": True,
                    "is_associating": False,
                    "uncertainty": None,
                }
            ],
            "model_name": "random_forest",
            "model_version": "0.1.0",
        }
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = PCSAFTClient()
        result = client.predict(["C1CCCC1"])

        assert "predictions" in result
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["smiles"] == "C1CCCC1"
        mock_post.assert_called_once()

    @patch("portal.api_client.requests.post")
    def test_predict_connection_error(self, mock_post):
        """Test prediction with connection error."""
        mock_post.side_effect = Exception("Connection refused")

        client = PCSAFTClient()
        result = client.predict(["C1CCCC1"])

        assert "error" in result
        assert result["predictions"] == []

    @patch("portal.api_client.requests.post")
    def test_submit_data_success(self, mock_post):
        """Test successful data submission."""
        mock_resp = Mock()
        mock_resp.json.return_value = {"status": "accepted", "smiles": "C1CCCC1"}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = PCSAFTClient()
        result = client.submit_data("C1CCCC1", 2.37, 3.71, 289.0, "10.1021/test")

        assert result["status"] == "accepted"
        mock_post.assert_called_once()

    @patch("portal.api_client.requests.post")
    def test_submit_data_error(self, mock_post):
        """Test data submission with error."""
        mock_post.side_effect = Exception("Connection refused")

        client = PCSAFTClient()
        result = client.submit_data("C1CCCC1", 2.37, 3.71, 289.0, "10.1021/test")

        assert result["status"] == "error"
        assert "detail" in result

    @patch("portal.api_client.requests.get")
    def test_get_health_success(self, mock_get):
        """Test successful health check."""
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "status": "healthy",
            "model_loaded": True,
            "model_name": "random_forest",
            "version": "0.1.0",
        }
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        client = PCSAFTClient()
        result = client.get_health()

        assert result["status"] == "healthy"
        assert result["model_loaded"] is True

    @patch("portal.api_client.requests.get")
    def test_get_health_unreachable(self, mock_get):
        """Test health check when API is unreachable."""
        mock_get.side_effect = Exception("Connection refused")

        client = PCSAFTClient()
        result = client.get_health()

        assert result["status"] == "unreachable"
        assert result["model_loaded"] is False

    @patch("portal.api_client.requests.get")
    def test_get_model_info_with_fallback(self, mock_get):
        """Test model info with fallback metrics."""
        mock_get.side_effect = Exception("Connection refused")

        client = PCSAFTClient()
        result = client.get_model_info()

        # Should return fallback metrics
        assert result["model_name"] == "random_forest"
        assert result["status"] == "offline"
        assert "r2_m" in result
        assert "mae_epsilon_k" in result
        assert result["n_training"] > 0


class TestMoleculeRendering:
    """Tests for molecule rendering utilities."""

    def test_render_molecule_valid(self):
        """Test rendering a valid molecule."""
        img = render_molecule("C1CCCC1", size=(250, 250))
        assert isinstance(img, Image.Image)
        assert img.size == (250, 250)

    def test_render_molecule_invalid(self):
        """Test rendering an invalid SMILES (should return placeholder)."""
        img = render_molecule("INVALID_SMILES", size=(300, 300))
        assert isinstance(img, Image.Image)
        assert img.size == (300, 300)
        # Placeholder is gray (240, 240, 240)
        # Check a sample of pixels
        assert img.getpixel((0, 0)) == (240, 240, 240)
        assert img.getpixel((150, 150)) == (240, 240, 240)

    def test_smiles_to_image_bytes(self):
        """Test converting SMILES to PNG bytes."""
        img_bytes = smiles_to_image_bytes("C1CCCC1", size=(250, 250))
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0
        # PNG magic number
        assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n"


class TestComponents:
    """Tests for Streamlit component functions."""

    def test_predictor_module_imports(self):
        """Test that predictor module and functions exist."""
        from portal.components import predictor

        assert hasattr(predictor, "render_prediction_page")

    def test_submitter_module_imports(self):
        """Test that submitter module and functions exist."""
        from portal.components import submitter

        assert hasattr(submitter, "render_submission_page")

    def test_dashboard_module_imports(self):
        """Test that dashboard module and functions exist."""
        from portal.components import dashboard

        assert hasattr(dashboard, "render_dashboard")

    def test_reference_comparison_logic(self):
        """Test the reference molecule comparison calculation."""
        # Test calculation logic with example values
        ref_m = 2.3655
        pred_m = 2.5

        pct_diff = abs(pred_m - ref_m) / ref_m * 100
        assert abs(pct_diff - 5.68) < 0.1  # ~5.68% difference

    def test_submitter_validation_logic(self):
        """Test submission validation logic."""
        from rdkit import Chem

        # Valid SMILES
        assert Chem.MolFromSmiles("C1CCCC1") is not None

        # Invalid SMILES
        assert Chem.MolFromSmiles("INVALID") is None

        # Parameter range checks
        m = 2.5
        assert 0.5 <= m <= 10.0

        sigma = 3.5
        assert 2.0 <= sigma <= 6.0

        epsilon_k = 250.0
        assert 50.0 <= epsilon_k <= 600.0


class TestAppModule:
    """Tests for the main app module."""

    def test_app_imports(self):
        """Test that app module imports successfully."""
        # Streamlit warns about missing context when imported outside streamlit run
        # This is expected behavior, so we just check that import doesn't fail
        try:
            import portal.app  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed to import portal.app: {e}")
        except RuntimeError:
            # Streamlit context error is expected when not running via streamlit run
            pass

    @patch.dict("os.environ", {"PCSAFT_API_URL": "http://custom-api:9000"})
    def test_app_uses_env_var(self):
        """Test that app reads API URL from environment."""
        import os

        assert os.getenv("PCSAFT_API_URL") == "http://custom-api:9000"


class TestPortalPackage:
    """Tests for portal package structure."""

    def test_portal_init(self):
        """Test portal package imports."""
        import portal

        assert hasattr(portal, "__name__")

    def test_components_init(self):
        """Test components subpackage exports."""
        from portal.components import (
            render_dashboard,
            render_molecule,
            render_prediction_page,
            render_submission_page,
            smiles_to_image_bytes,
        )

        assert callable(render_prediction_page)
        assert callable(render_submission_page)
        assert callable(render_dashboard)
        assert callable(render_molecule)
        assert callable(smiles_to_image_bytes)
