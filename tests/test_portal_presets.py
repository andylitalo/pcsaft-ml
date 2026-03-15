"""Tests for portal presets and generalized screening functionality."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from portal.api_client import PCSAFTClient
from portal.components.screener import _apply_filters, _rank_molecules
from portal.presets import PRESETS, get_preset, list_presets


class TestPresets:
    """Tests for application domain presets."""

    def test_presets_all_have_required_keys(self):
        """Test that all presets have required keys."""
        required_keys = [
            "name",
            "description",
            "default_reference",
            "filters",
            "ranking_metric",
            "soft_preferences",
            "caution",
        ]

        for preset_key, preset in PRESETS.items():
            for key in required_keys:
                assert key in preset, f"Preset {preset_key} missing key {key}"

    def test_general_preset_has_no_filters(self):
        """Test that general preset has minimal filters."""
        preset = get_preset("general")

        assert preset["filters"]["require_in_domain"] is False
        assert preset["filters"]["min_tanimoto"] == 0.0
        assert "m_range" not in preset["filters"]
        assert "sigma_range" not in preset["filters"]
        assert "epsilon_k_range" not in preset["filters"]

    def test_list_presets(self):
        """Test that list_presets returns all preset names."""
        presets = list_presets()

        assert "blowing_agents" in presets
        assert "refrigerants" in presets
        assert "solvents" in presets
        assert "general" in presets
        assert len(presets) == 4

    def test_get_preset_unknown(self):
        """Test that get_preset raises KeyError for unknown preset."""
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_blowing_agents_preset_filters(self):
        """Test blowing agents preset has sensible filter ranges."""
        preset = get_preset("blowing_agents")

        assert preset["filters"]["m_range"] == (1.8, 3.5)
        assert preset["filters"]["sigma_range"] == (3.0, 4.5)
        assert preset["filters"]["epsilon_k_range"] == (200, 400)
        assert preset["filters"]["max_mw"] == 200
        assert preset["filters"]["require_in_domain"] is True
        assert preset["ranking_metric"] == "distance_to_reference"


class TestLocalFallback:
    """Tests for local fallback mode (no API)."""

    def test_local_fallback_predict(self):
        """Test prediction using local pcsaft_predict library."""
        # Mock predict_with_uncertainty return
        mock_df = pd.DataFrame({
            "smiles": ["CCO"],
            "m": [2.123],
            "sigma": [3.456],
            "epsilon_k": [234.5],
            "m_std": [0.045],
            "sigma_std": [0.023],
            "epsilon_k_std": [12.3],
            "in_domain": [True],
            "tanimoto_nn": [0.85],
        })

        with patch(
            "pcsaft_predict.predict_with_uncertainty", return_value=mock_df,
        ) as mock_predict:
            # Create client in local mode
            client = PCSAFTClient(base_url=None)
            assert client._use_local is True

            # Predict
            result = client.predict(["CCO"])

            # Check result format
            assert "predictions" in result
            assert len(result["predictions"]) == 1
            pred = result["predictions"][0]
            assert pred["smiles"] == "CCO"
            assert pred["m"] == 2.123
            assert pred["in_domain"] is True
            assert pred["uncertainty"]["m_std"] == 0.045

            # Verify mock was called
            mock_predict.assert_called_once_with(["CCO"])

    def test_local_fallback_reference_molecules(self):
        """Test that local mode returns hardcoded reference molecules."""
        client = PCSAFTClient(base_url=None)
        refs = client.get_reference_molecules()

        assert len(refs) > 0
        assert any(r["name"] == "Cyclopentane" for r in refs)
        assert any(r["name"] == "R-134a" for r in refs)

        # Check structure
        for ref in refs:
            assert "name" in ref
            assert "smiles" in ref
            assert "m" in ref
            assert "sigma" in ref
            assert "epsilon_k" in ref
            assert "source" in ref


class TestBatchScreening:
    """Tests for batch screening functionality."""

    def test_batch_screening_smoke(self):
        """Test batch screening filter and ranking pipeline."""
        # Create mock predictions DataFrame
        df = pd.DataFrame({
            "smiles": ["C1CCCC1", "CCCCCC", "c1ccccc1"],
            "m": [2.3, 3.5, 2.5],
            "sigma": [3.7, 4.0, 3.6],
            "epsilon_k": [290, 350, 285],
            "in_domain": [True, True, False],
            "tanimoto_nn": [0.9, 0.5, 0.2],
        })

        # Apply filters (blowing agents preset)
        filters = {
            "m_range": (1.8, 3.5),
            "sigma_range": (3.0, 4.5),
            "epsilon_k_range": (200, 400),
            "require_in_domain": True,
            "min_tanimoto": 0.3,
        }

        df_filtered = _apply_filters(df, filters)

        # Check that filtering worked
        assert "passed_filters" in df_filtered.columns
        assert "filter_reason" in df_filtered.columns

        # Cyclopentane should pass all filters
        assert df_filtered.loc[df_filtered["smiles"] == "C1CCCC1", "passed_filters"].iloc[0]

        # Benzene should fail (out of domain, low tanimoto)
        assert not df_filtered.loc[df_filtered["smiles"] == "c1ccccc1", "passed_filters"].iloc[0]

    def test_ranking_distance_to_reference(self):
        """Test ranking by distance to reference molecule."""
        df = pd.DataFrame({
            "smiles": ["C1CCCC1", "CCCCCC"],
            "m": [2.3, 3.0],
            "sigma": [3.7, 4.0],
            "epsilon_k": [290, 350],
        })

        ref_params = {"m": 2.3, "sigma": 3.7, "epsilon_k": 290}

        df_ranked = _rank_molecules(df, "distance_to_reference", ref_params)

        assert "rank_score" in df_ranked.columns

        # Cyclopentane should have rank_score ~ 0 (exact match)
        assert df_ranked.loc[df_ranked["smiles"] == "C1CCCC1", "rank_score"].iloc[0] < 0.01

        # Hexane should have higher rank_score (farther from reference)
        assert df_ranked.loc[df_ranked["smiles"] == "CCCCCC", "rank_score"].iloc[0] > 0.1

    def test_ranking_tanimoto(self):
        """Test ranking by Tanimoto similarity."""
        df = pd.DataFrame({
            "smiles": ["C1CCCC1", "CCCCCC"],
            "m": [2.3, 3.0],
            "sigma": [3.7, 4.0],
            "epsilon_k": [290, 350],
            "tanimoto_nn": [0.9, 0.5],
        })

        df_ranked = _rank_molecules(df, "tanimoto", None)

        assert "rank_score" in df_ranked.columns

        # Higher Tanimoto → lower rank_score (better)
        assert (
            df_ranked.loc[df_ranked["smiles"] == "C1CCCC1", "rank_score"].iloc[0]
            < df_ranked.loc[df_ranked["smiles"] == "CCCCCC", "rank_score"].iloc[0]
        )


class TestPresetReferenceMolecules:
    """Tests for preset reference molecules."""

    @patch("portal.api_client.requests.get")
    def test_preset_reference_molecule_exists(self, mock_get):
        """Test that default reference molecules from presets exist in reference list."""
        # Mock API response with reference molecules
        mock_resp = Mock()
        mock_resp.json.return_value = [
            {
                "name": "Cyclopentane",
                "smiles": "C1CCCC1",
                "m": 2.3655,
                "sigma": 3.7114,
                "epsilon_k": 288.84,
                "source": "Gross & Sadowski (2001)",
            },
            {
                "name": "R-134a",
                "smiles": "FC(F)C(F)F",
                "m": 2.696,
                "sigma": 3.174,
                "epsilon_k": 201.7,
                "source": "Tumakaka et al. (2002)",
            },
            {
                "name": "Ethyl acetate",
                "smiles": "CCOC(C)=O",
                "m": 2.870,
                "sigma": 3.307,
                "epsilon_k": 230.8,
                "source": "Gross & Sadowski (2001)",
            },
        ]
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        client = PCSAFTClient()
        refs = client.get_reference_molecules()
        ref_names = [r["name"] for r in refs]

        # Check that preset default references exist
        for preset in PRESETS.values():
            default_ref = preset["default_reference"]
            # Allow "Custom..." as default (used for user-defined references)
            if default_ref != "Custom...":
                assert default_ref in ref_names or default_ref == "Cyclopentane", (
                    f"Default reference {default_ref} not found in reference list"
                )

    def test_preset_reference_molecule_exists_local_mode(self):
        """Test that default reference molecules exist in local mode."""
        client = PCSAFTClient(base_url=None)
        refs = client.get_reference_molecules()
        ref_names = [r["name"] for r in refs]

        # Check key defaults
        assert "Cyclopentane" in ref_names
        assert "R-134a" in ref_names
        assert "Ethyl acetate" in ref_names


class TestAPIClientLocalMode:
    """Tests for API client local mode initialization."""

    def test_local_mode_init(self):
        """Test that client initializes in local mode when base_url is None."""
        client = PCSAFTClient(base_url=None)
        assert client.base_url is None
        assert client._use_local is True

    def test_api_mode_init(self):
        """Test that client initializes in API mode when base_url is provided."""
        client = PCSAFTClient(base_url="http://example.com:8000")
        assert client.base_url == "http://example.com:8000"
        assert client._use_local is False
