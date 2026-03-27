"""Shared test fixtures for the ml-chem test suite."""

import os
from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_dataset():
    """Load the 10-molecule mini PC-SAFT dataset for smoke tests."""
    return pd.read_csv(FIXTURES_DIR / "mini_pcsaft.csv")


@pytest.fixture
def mini_smiles(mini_dataset):
    """List of SMILES from the mini dataset."""
    return mini_dataset["smiles"].tolist()


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary directory for model artifacts during testing."""
    model_dir = tmp_path / "model_saved"
    model_dir.mkdir()
    return model_dir


@pytest.fixture(autouse=True)
def _no_network_in_tests(monkeypatch):
    """Prevent accidental network calls in unit tests.

    Tests that intentionally need network access should use
    @pytest.mark.usefixtures() to override, or mock the specific call.
    """
    if os.environ.get("ALLOW_NETWORK_TESTS"):
        return
    try:
        import requests
    except ModuleNotFoundError:
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "Network access blocked in tests. "
            "Mock the call or set ALLOW_NETWORK_TESTS=1."
        )

    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)
