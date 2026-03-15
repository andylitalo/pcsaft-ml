"""Tests for reference molecule functionality."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def test_reference_molecules_endpoint_returns_list():
    """GET /reference-molecules should return a list of molecules."""
    from serving.app import app

    client = TestClient(app)
    response = client.get("/reference-molecules")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 20


def test_reference_molecules_schema():
    """Each reference molecule should have required fields."""
    from serving.app import app

    client = TestClient(app)
    response = client.get("/reference-molecules")
    data = response.json()

    required_fields = ["name", "smiles", "m", "sigma", "epsilon_k", "source"]
    for molecule in data:
        for field in required_fields:
            assert field in molecule, f"Missing field: {field}"
        assert isinstance(molecule["name"], str)
        assert isinstance(molecule["smiles"], str)
        assert isinstance(molecule["m"], (int, float))
        assert isinstance(molecule["sigma"], (int, float))
        assert isinstance(molecule["epsilon_k"], (int, float))
        assert isinstance(molecule["source"], str)


def test_reference_molecules_includes_cyclopentane():
    """Cyclopentane should be in the reference list."""
    from serving.app import app

    client = TestClient(app)
    response = client.get("/reference-molecules")
    data = response.json()

    names = [m["name"] for m in data]
    assert "Cyclopentane" in names

    cyclopentane = next(m for m in data if m["name"] == "Cyclopentane")
    assert cyclopentane["smiles"] == "C1CCCC1"
    assert cyclopentane["m"] == pytest.approx(2.25774, abs=0.001)
    assert cyclopentane["sigma"] == pytest.approx(3.75308, abs=0.001)
    assert cyclopentane["epsilon_k"] == pytest.approx(273.50029, abs=0.01)


def test_reference_molecules_diverse_categories():
    """Reference list should include diverse molecular families."""
    from serving.app import app

    client = TestClient(app)
    response = client.get("/reference-molecules")
    data = response.json()

    names = [m["name"].lower() for m in data]

    # Check for alkanes
    assert any("propane" in n or "butane" in n or "pentane" in n for n in names)

    # Check for cycloalkanes
    assert any("cyclo" in n for n in names)

    # Check for aromatics
    assert any("benzene" in n or "toluene" in n for n in names)

    # Check for fluorocarbons
    assert any("r-" in n or "hfo" in n or "fluoro" in n for n in names)

    # Check for ethers
    assert any("ether" in n or "mtbe" in n for n in names)

    # Check for ketones
    assert any("acetone" in n or "ketone" in n or "mek" in n for n in names)


def test_api_client_get_reference_molecules():
    """PCSAFTClient.get_reference_molecules should fetch from API."""
    from portal.api_client import PCSAFTClient

    client = PCSAFTClient(base_url="http://testserver")

    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "name": "Propane",
            "smiles": "CCC",
            "m": 1.98602,
            "sigma": 3.6244,
            "epsilon_k": 209.08586,
            "source": "Esper",
        }
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        result = client.get_reference_molecules()

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["name"] == "Propane"


def test_api_client_get_reference_molecules_error_handling():
    """get_reference_molecules should return empty list on error."""
    from portal.api_client import PCSAFTClient

    client = PCSAFTClient(base_url="http://nonexistent:9999")

    # Should not raise exception, just return empty list
    result = client.get_reference_molecules()
    assert result == []


def test_reference_molecule_schema_validation():
    """ReferenceMolecule schema should validate correctly."""
    from serving.schemas import ReferenceMolecule

    # Valid molecule
    mol = ReferenceMolecule(
        name="Test", smiles="CCC", m=2.0, sigma=3.5, epsilon_k=200.0, source="Esper"
    )
    assert mol.name == "Test"
    assert mol.source == "Esper"

    # Default source
    mol2 = ReferenceMolecule(
        name="Test2", smiles="CCCC", m=2.1, sigma=3.6, epsilon_k=210.0
    )
    assert mol2.source == "Esper"
