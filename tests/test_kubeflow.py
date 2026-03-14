"""Tests for Kubeflow Pipeline components and pipeline definition."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.components.compare import compare_models_logic
from pipeline.components.validate import validate_data_logic


@pytest.fixture
def tmp_submissions_csv(tmp_path):
    """Create a temporary submissions CSV with valid and invalid data."""
    data = {
        "smiles": [
            "CCO",  # valid ethanol
            "INVALID_SMILES",  # invalid SMILES
            "CCCC",  # valid butane
            "C1CCCCC1",  # valid cyclohexane, but epsilon_k out of range
            "CC(C)C",  # valid isobutane, but m out of range
        ],
        "m": [1.5, 2.0, 1.8, 2.5, 15.0],  # last one out of range
        "sigma": [3.2, 3.5, 3.8, 3.6, 3.3],
        "epsilon_k": [150.0, 200.0, 180.0, 700.0, 250.0],  # 4th out of range
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "submissions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tmp_valid_csv(tmp_path):
    """Create a temporary validated CSV."""
    return tmp_path / "valid.csv"


@pytest.fixture
def tmp_rejected_csv(tmp_path):
    """Create a temporary rejected CSV."""
    return tmp_path / "rejected.csv"


def test_validate_data_logic(tmp_submissions_csv, tmp_valid_csv, tmp_rejected_csv):
    """Test validate_data_logic correctly validates and rejects submissions."""
    n_valid = validate_data_logic(
        str(tmp_submissions_csv),
        str(tmp_valid_csv),
        str(tmp_rejected_csv),
    )

    # Should have 2 valid rows (CCO and CCCC)
    assert n_valid == 2

    # Check valid CSV
    valid_df = pd.read_csv(tmp_valid_csv)
    assert len(valid_df) == 2
    assert "CCO" in valid_df["smiles"].values
    assert "CCCC" in valid_df["smiles"].values

    # Check rejected CSV
    rejected_df = pd.read_csv(tmp_rejected_csv)
    assert len(rejected_df) == 3

    # Check rejection reasons
    reasons = rejected_df["reject_reason"].tolist()
    assert "invalid_smiles" in reasons
    assert "epsilon_k_out_of_range" in reasons
    assert "m_out_of_range" in reasons


def test_validate_data_all_valid(tmp_path):
    """Test validate_data_logic with all valid submissions."""
    # Create all-valid submissions
    data = {
        "smiles": ["CCO", "CCCC", "CC(C)C"],
        "m": [1.5, 1.8, 2.0],
        "sigma": [3.2, 3.8, 3.5],
        "epsilon_k": [150.0, 180.0, 200.0],
    }
    df = pd.DataFrame(data)
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    valid_csv = tmp_path / "valid.csv"
    rejected_csv = tmp_path / "rejected.csv"

    n_valid = validate_data_logic(str(input_csv), str(valid_csv), str(rejected_csv))

    assert n_valid == 3
    assert len(pd.read_csv(valid_csv)) == 3
    # Check rejected file is empty
    with open(rejected_csv) as f:
        content = f.read()
    assert content == ""


def test_validate_data_range_checks(tmp_path):
    """Test parameter range validation."""
    # Test edge cases for each parameter
    test_cases = [
        # (m, sigma, epsilon_k, should_pass)
        (0.5, 3.0, 100.0, True),  # min values (valid)
        (10.0, 6.0, 600.0, True),  # max values (valid)
        (0.4, 3.0, 100.0, False),  # m too low
        (10.1, 3.0, 100.0, False),  # m too high
        (2.0, 1.9, 100.0, False),  # sigma too low
        (2.0, 6.1, 100.0, False),  # sigma too high
        (2.0, 3.0, 49.0, False),  # epsilon_k too low
        (2.0, 3.0, 601.0, False),  # epsilon_k too high
    ]

    for m, sigma, epsilon_k, should_pass in test_cases:
        data = {
            "smiles": ["CCO"],
            "m": [m],
            "sigma": [sigma],
            "epsilon_k": [epsilon_k],
        }
        df = pd.DataFrame(data)
        input_csv = tmp_path / f"input_{m}_{sigma}_{epsilon_k}.csv"
        df.to_csv(input_csv, index=False)

        valid_csv = tmp_path / f"valid_{m}_{sigma}_{epsilon_k}.csv"
        rejected_csv = tmp_path / f"rejected_{m}_{sigma}_{epsilon_k}.csv"

        n_valid = validate_data_logic(str(input_csv), str(valid_csv), str(rejected_csv))

        if should_pass:
            assert n_valid == 1, f"Expected valid for {m}, {sigma}, {epsilon_k}"
        else:
            assert n_valid == 0, f"Expected invalid for {m}, {sigma}, {epsilon_k}"


def test_compare_models_logic_champion_wins(tmp_path):
    """Test compare_models_logic when champion is better."""
    champion_metrics = {
        "m": {"r2": 0.65},
        "sigma": {"r2": 0.35},
        "epsilon_k": {"r2": 0.30},
    }
    challenger_metrics = {
        "m": {"r2": 0.64},
        "sigma": {"r2": 0.34},
        "epsilon_k": {"r2": 0.29},
    }

    champion_path = tmp_path / "champion.json"
    challenger_path = tmp_path / "challenger.json"

    champion_path.write_text(json.dumps(champion_metrics))
    challenger_path.write_text(json.dumps(challenger_metrics))

    should_promote = compare_models_logic(
        str(champion_path),
        str(challenger_path),
        improvement_threshold=0.005,
    )

    # Challenger is worse, should not promote
    assert should_promote is False


def test_compare_models_logic_challenger_wins(tmp_path):
    """Test compare_models_logic when challenger is better."""
    champion_metrics = {
        "m": {"r2": 0.65},
        "sigma": {"r2": 0.35},
        "epsilon_k": {"r2": 0.30},
    }
    challenger_metrics = {
        "m": {"r2": 0.68},  # +0.03 improvement
        "sigma": {"r2": 0.36},  # +0.01 improvement
        "epsilon_k": {"r2": 0.31},  # +0.01 improvement
    }
    # Average improvement: (0.03 + 0.01 + 0.01) / 3 = 0.0167 > 0.005

    champion_path = tmp_path / "champion.json"
    challenger_path = tmp_path / "challenger.json"

    champion_path.write_text(json.dumps(champion_metrics))
    challenger_path.write_text(json.dumps(challenger_metrics))

    should_promote = compare_models_logic(
        str(champion_path),
        str(challenger_path),
        improvement_threshold=0.005,
    )

    # Challenger is better, should promote
    assert should_promote is True


def test_compare_models_logic_marginal_improvement(tmp_path):
    """Test compare_models_logic with improvement below threshold."""
    champion_metrics = {
        "m": {"r2": 0.65},
        "sigma": {"r2": 0.35},
        "epsilon_k": {"r2": 0.30},
    }
    challenger_metrics = {
        "m": {"r2": 0.651},
        "sigma": {"r2": 0.351},
        "epsilon_k": {"r2": 0.301},
    }
    # Average improvement: 0.003 < 0.005 threshold

    champion_path = tmp_path / "champion.json"
    challenger_path = tmp_path / "challenger.json"

    champion_path.write_text(json.dumps(champion_metrics))
    challenger_path.write_text(json.dumps(challenger_metrics))

    should_promote = compare_models_logic(
        str(champion_path),
        str(challenger_path),
        improvement_threshold=0.005,
    )

    # Improvement is below threshold, should not promote
    assert should_promote is False


def test_pipeline_compiles():
    """Test that the pipeline compiles to valid YAML."""
    pipeline_yaml = Path("pipeline/pcsaft_retrain_pipeline.yaml")
    assert pipeline_yaml.exists(), "Pipeline YAML file does not exist"

    # Check that it's a valid YAML file
    import yaml

    with open(pipeline_yaml) as f:
        pipeline_spec = yaml.safe_load(f)

    # Check basic structure
    assert "pipelineInfo" in pipeline_spec
    assert "root" in pipeline_spec
    assert pipeline_spec["pipelineInfo"]["name"] == "pcsaft-retrain-pipeline"


def test_pipeline_has_all_components():
    """Test that the compiled pipeline contains all expected components."""
    import yaml

    pipeline_yaml = Path("pipeline/pcsaft_retrain_pipeline.yaml")
    with open(pipeline_yaml) as f:
        pipeline_spec = yaml.safe_load(f)

    # Extract component names from the pipeline spec
    components = pipeline_spec.get("components", {})

    # Check for expected component names (KFP transforms names)
    expected_components = [
        "validate-data",
        "merge-datasets",
        "retrain-model",
        "evaluate-model",
        "compare-models",
    ]

    for component in expected_components:
        assert any(
            component in comp_name.lower() for comp_name in components.keys()
        ), f"Component {component} not found in pipeline"


@patch("kfp.Client")
def test_trigger_insufficient_samples(mock_client, tmp_path):
    """Test trigger does not run pipeline with insufficient samples."""
    from pipeline.trigger import check_and_trigger

    # Create submissions file with < 10 samples
    data = {
        "smiles": ["CCO", "CCCC"],
        "m": [1.5, 1.8],
        "sigma": [3.2, 3.8],
        "epsilon_k": [150, 180],
    }
    df = pd.DataFrame(data)
    submissions_csv = tmp_path / "submissions.csv"
    df.to_csv(submissions_csv, index=False)

    result = check_and_trigger(
        submissions_csv=str(submissions_csv),
        pipeline_yaml="pipeline/pcsaft_retrain_pipeline.yaml",
        min_samples=10,
    )

    # Should not trigger
    assert result is None
    mock_client.assert_not_called()


@patch("kfp.Client")
def test_trigger_sufficient_samples(mock_client, tmp_path):
    """Test trigger runs pipeline with sufficient samples."""
    from pipeline.trigger import check_and_trigger

    # Create submissions file with >= 10 samples
    data = {
        "smiles": [f"C{i}" for i in range(15)],
        "m": [2.0] * 15,
        "sigma": [3.5] * 15,
        "epsilon_k": [200.0] * 15,
    }
    df = pd.DataFrame(data)
    submissions_csv = tmp_path / "submissions.csv"
    df.to_csv(submissions_csv, index=False)

    # Mock the KFP client
    mock_client_instance = MagicMock()
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "test-experiment-id"
    mock_client_instance.get_experiment.return_value = mock_experiment
    mock_run = MagicMock()
    mock_run.run_id = "test-run-id"
    mock_client_instance.create_run_from_pipeline_package.return_value = mock_run
    mock_client.return_value = mock_client_instance

    result = check_and_trigger(
        submissions_csv=str(submissions_csv),
        pipeline_yaml="pipeline/pcsaft_retrain_pipeline.yaml",
        min_samples=10,
    )

    # Should trigger
    assert result == "test-run-id"
    mock_client.assert_called_once()
    mock_client_instance.create_run_from_pipeline_package.assert_called_once()


@patch("kfp.Client")
def test_trigger_no_submissions_file(mock_client, tmp_path):
    """Test trigger handles missing submissions file gracefully."""
    from pipeline.trigger import check_and_trigger

    result = check_and_trigger(
        submissions_csv=str(tmp_path / "nonexistent.csv"),
        pipeline_yaml="pipeline/pcsaft_retrain_pipeline.yaml",
        min_samples=10,
    )

    # Should not trigger
    assert result is None
    mock_client.assert_not_called()
