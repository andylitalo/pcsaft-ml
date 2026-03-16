"""Tests for similarity search endpoint."""

import pytest
from fastapi.testclient import TestClient

from serving.app import app


# Use context manager to ensure lifespan startup/shutdown is called
@pytest.fixture(scope="module")
def client():
    """Create a test client with lifespan context."""
    with TestClient(app) as c:
        yield c


def test_similar_by_parameter(client):
    """Query with known PC-SAFT parameters, verify closest neighbor is reasonable."""
    # Query with cyclopentane-like parameters (slightly perturbed)
    response = client.post(
        "/similar",
        json={
            "m": 2.26,
            "sigma": 3.75,
            "epsilon_k": 273.0,
            "k": 5,
            "corpus": "reference",
            "metric": "parameter",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["metric"] == "parameter"
    assert data["corpus"] == "reference"
    assert len(data["neighbors"]) <= 5

    # First neighbor should be cyclopentane (closest to query)
    top = data["neighbors"][0]
    assert top["smiles"] == "C1CCCC1"
    assert top["parameter_distance"] is not None
    assert top["parameter_distance"] < 0.05  # Very close


def test_similar_by_smiles(client):
    """Query with a SMILES in the corpus, verify it returns itself as closest match."""
    response = client.post(
        "/similar",
        json={
            "smiles": "C1CCCC1",  # Cyclopentane
            "k": 5,
            "corpus": "reference",
            "metric": "parameter",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["query_smiles"] == "C1CCCC1"
    assert len(data["neighbors"]) <= 5

    # First neighbor should be cyclopentane (very close to query)
    # SMILES may be canonicalized differently, so check by distance
    top = data["neighbors"][0]
    assert top["parameter_distance"] < 0.05  # Very close to query


def test_similar_tanimoto(client):
    """Query with known SMILES, verify Tanimoto similarity is in valid range."""
    response = client.post(
        "/similar",
        json={
            "smiles": "c1ccccc1",  # Benzene
            "k": 5,
            "corpus": "reference",
            "metric": "tanimoto",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["metric"] == "tanimoto"
    assert len(data["neighbors"]) <= 5

    # All Tanimoto similarities should be in [0, 1]
    for neighbor in data["neighbors"]:
        assert 0.0 <= neighbor["tanimoto_similarity"] <= 1.0

    # First neighbor should be itself with Tanimoto ~1.0
    top = data["neighbors"][0]
    assert top["smiles"] == "c1ccccc1"
    assert top["tanimoto_similarity"] > 0.99


def test_similar_k_limit(client):
    """Verify at most k neighbors returned."""
    response = client.post(
        "/similar",
        json={
            "smiles": "CCCC",
            "k": 3,
            "corpus": "reference",  # Use reference corpus which has fewer molecules
            "metric": "parameter",
        },
    )
    assert response.status_code == 200
    data = response.json()

    # Should return at most k neighbors (or fewer if corpus is smaller)
    assert len(data["neighbors"]) <= 3
    assert len(data["neighbors"]) > 0


def test_similar_invalid_smiles(client):
    """Query with invalid SMILES, verify graceful error."""
    response = client.post(
        "/similar",
        json={
            "smiles": "INVALID_SMILES_XYZ",
            "k": 5,
            "corpus": "reference",
            "metric": "parameter",
        },
    )
    # Should get 400 error for invalid SMILES
    assert response.status_code == 400
    assert "Invalid SMILES" in response.json()["detail"]


def test_similar_parameter_only(client):
    """Query with parameters only (no SMILES), verify parameter distance works."""
    response = client.post(
        "/similar",
        json={
            "m": 3.0,
            "sigma": 3.8,
            "epsilon_k": 240.0,
            "k": 5,
            "corpus": "reference",
            "metric": "parameter",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["query_smiles"] is None
    assert data["query_params"] == {"m": 3.0, "sigma": 3.8, "epsilon_k": 240.0}
    assert len(data["neighbors"]) <= 5

    # All neighbors should have parameter_distance computed
    for neighbor in data["neighbors"]:
        assert neighbor["parameter_distance"] is not None


def test_similar_corpus_selection(client):
    """Verify reference, novel, and all return the expected source set."""
    # Reference corpus
    ref_response = client.post(
        "/similar",
        json={
            "smiles": "C1CCCC1",
            "k": 5,
            "corpus": "reference",
            "metric": "parameter",
        },
    )
    assert ref_response.status_code == 200
    ref_data = ref_response.json()
    assert ref_data["corpus"] == "reference"
    assert ref_data["corpus_version"] == "reference_molecules_v1"
    # All neighbors should be from reference corpus
    for neighbor in ref_data["neighbors"]:
        assert neighbor["corpus"] == "reference"

    # Novel corpus
    novel_response = client.post(
        "/similar",
        json={
            "smiles": "C1CCCC1",
            "k": 5,
            "corpus": "novel",
            "metric": "parameter",
        },
    )
    assert novel_response.status_code == 200
    novel_data = novel_response.json()
    assert novel_data["corpus"] == "novel"
    assert novel_data["corpus_version"] == "pcsaft_novel_predictions_v1"
    # All neighbors should be from novel corpus
    for neighbor in novel_data["neighbors"]:
        assert neighbor["corpus"] == "novel"

    # All corpus
    all_response = client.post(
        "/similar",
        json={
            "smiles": "C1CCCC1",
            "k": 10,
            "corpus": "all",
            "metric": "parameter",
        },
    )
    assert all_response.status_code == 200
    all_data = all_response.json()
    assert all_data["corpus"] == "all"
    assert "reference_molecules_v1+pcsaft_novel_predictions_v1" in all_data["corpus_version"]


def test_similar_response_versioning(client):
    """Verify corpus_version is returned and matches expected format."""
    response = client.post(
        "/similar",
        json={
            "smiles": "C1CCCC1",
            "k": 5,
            "corpus": "novel",
            "metric": "parameter",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert "corpus_version" in data
    assert data["corpus_version"] == "pcsaft_novel_predictions_v1"


def test_similar_both_metric(client):
    """Test 'both' metric returns both parameter_distance and tanimoto_similarity."""
    response = client.post(
        "/similar",
        json={
            "smiles": "c1ccccc1",
            "k": 5,
            "corpus": "reference",
            "metric": "both",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["metric"] == "both"

    # All neighbors should have both distance and similarity
    for neighbor in data["neighbors"]:
        assert neighbor["parameter_distance"] is not None
        assert neighbor["tanimoto_similarity"] is not None


def test_similar_missing_inputs(client):
    """Verify error when neither smiles nor parameters are provided."""
    response = client.post(
        "/similar",
        json={
            "k": 5,
            "corpus": "reference",
            "metric": "parameter",
        },
    )
    assert response.status_code == 400
    assert "Must provide" in response.json()["detail"]


def test_similar_tanimoto_without_smiles(client):
    """Verify error when using tanimoto metric without providing smiles."""
    response = client.post(
        "/similar",
        json={
            "m": 2.5,
            "sigma": 3.7,
            "epsilon_k": 280.0,
            "k": 5,
            "corpus": "reference",
            "metric": "tanimoto",
        },
    )
    assert response.status_code == 400
    assert "Tanimoto similarity requires 'smiles'" in response.json()["detail"]
