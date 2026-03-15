#!/usr/bin/env python3
"""
Tests for Step 28: Dataset and Model Card Publication.

Validates that the publication-ready dataset and documentation meet quality standards.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = PROJECT_ROOT / "data/pcsaft_novel_predictions_v1.csv"
DATASHEET = PROJECT_ROOT / "data/DATASHEET.md"
MODEL_CARDS = [
    PROJECT_ROOT / "docs/model_cards/gnn.md",
    PROJECT_ROOT / "docs/model_cards/rf.md",
    PROJECT_ROOT / "docs/model_cards/ensemble.md",
]


@pytest.fixture
def dataset():
    """Load the publication dataset, skipping comment lines."""
    if not DATASET_CSV.exists():
        pytest.skip(f"Dataset not found: {DATASET_CSV}")

    # Skip lines starting with #
    df = pd.read_csv(DATASET_CSV, comment="#")
    return df


def test_dataset_csv_columns(dataset):
    """Verify all required columns are present."""
    required_columns = [
        "smiles", "inchi", "inchikey", "name", "mol_class", "backbone",
        "mw", "n_fluorine", "n_chlorine", "n_carbon", "has_double_bond",
        "parameter_source", "m", "sigma", "epsilon_k",
        "m_std", "sigma_std", "epsilon_k_std",
        "tanimoto_nn", "ad_in_domain", "sa_score",
        "boiling_point_K", "vp_298K_Pa", "density_298K_mol_m3",
        "henrys_ratio", "hfo_distance", "cyc_distance"
    ]

    missing = set(required_columns) - set(dataset.columns)
    assert not missing, f"Missing required columns: {missing}"

    # Check for duplicate InChIKeys
    duplicates = dataset[dataset.duplicated(subset=["inchikey"], keep=False)]
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate InChIKeys"


def test_dataset_no_null_smiles(dataset):
    """Verify no null SMILES entries."""
    null_smiles = dataset["smiles"].isna().sum()
    assert null_smiles == 0, f"Found {null_smiles} null SMILES entries"

    empty_smiles = (dataset["smiles"] == "").sum()
    assert empty_smiles == 0, f"Found {empty_smiles} empty SMILES entries"


def test_dataset_parameter_ranges(dataset):
    """Verify PC-SAFT parameters are in physically plausible ranges."""
    # m (segment number): typically 1-10 for small molecules
    assert (dataset["m"] > 0).all(), "Found non-positive m values"
    assert (dataset["m"] < 20).all(), f"Found unrealistic m > 20: max={dataset['m'].max()}"

    # sigma (segment diameter): typically 2.5-5.5 Å
    assert (dataset["sigma"] > 0).all(), "Found non-positive sigma values"
    max_sigma = dataset["sigma"].max()
    assert (dataset["sigma"] < 10).all(), \
        f"Found unrealistic sigma > 10: max={max_sigma}"

    # epsilon_k (dispersion energy): typically 100-500 K for small molecules
    assert (dataset["epsilon_k"] > 0).all(), "Found non-positive epsilon_k values"
    max_epsilon = dataset["epsilon_k"].max()
    assert (dataset["epsilon_k"] < 1000).all(), \
        f"Found unrealistic epsilon_k > 1000: max={max_epsilon}"


def test_dataset_parameter_source(dataset):
    """Verify all parameters are marked as model-predicted."""
    # All entries should have parameter_source = "model_predicted"
    if "parameter_source" in dataset.columns:
        sources = dataset["parameter_source"].unique()
        assert all(s == "model_predicted" for s in sources if pd.notna(s)), \
            f"Found unexpected parameter sources: {sources}"


def test_model_card_files_exist():
    """Verify all three model cards exist with required sections."""
    required_sections = [
        "Model Details",
        "Intended Use",
        "Training Data",
        "Evaluation Results",
        "Limitations and Biases",
        "Ethical Considerations"
    ]

    for card_path in MODEL_CARDS:
        assert card_path.exists(), f"Model card not found: {card_path}"

        with open(card_path, "r") as f:
            content = f.read()

        # Check for required sections (case-insensitive)
        for section in required_sections:
            # Look for "## Section Name" headers
            pattern = rf"##\s+{re.escape(section)}"
            assert re.search(pattern, content, re.IGNORECASE), \
                f"Model card {card_path.name} missing section: {section}"


def test_model_card_gnn_metrics():
    """Verify GNN model card contains expected performance metrics."""
    gnn_card = MODEL_CARDS[0]
    if not gnn_card.exists():
        pytest.skip(f"GNN model card not found: {gnn_card}")

    with open(gnn_card, "r") as f:
        content = f.read()

    # Check for R² values in expected range (0.70-0.80)
    assert "0.76" in content or "0.762" in content, "Missing expected R²(m)=0.76"
    assert "0.77" in content or "0.774" in content, "Missing expected R²(σ)=0.77"
    assert "0.73" in content or "0.731" in content, "Missing expected R²(ε/k)=0.73"


def test_model_card_rf_metrics():
    """Verify RF model card contains expected performance metrics."""
    rf_card = MODEL_CARDS[1]
    if not rf_card.exists():
        pytest.skip(f"RF model card not found: {rf_card}")

    with open(rf_card, "r") as f:
        content = f.read()

    # Check for R² values in expected range (0.30-0.65)
    assert "0.62" in content or "0.620" in content, "Missing expected R²(m)=0.62"
    assert "0.35" in content or "0.354" in content, "Missing expected R²(σ)=0.35"
    assert "0.33" in content or "0.332" in content, "Missing expected R²(ε/k)=0.33"


def test_model_card_ensemble_warning():
    """Verify ensemble model card contains warning about underperformance."""
    ensemble_card = MODEL_CARDS[2]
    if not ensemble_card.exists():
        pytest.skip(f"Ensemble model card not found: {ensemble_card}")

    with open(ensemble_card, "r") as f:
        content = f.read()

    # Check for explicit warning about ensemble underperformance
    warning_phrases = [
        "underperform",
        "DO NOT USE",
        "worse than",
        "DEPRECATED",
        "not recommended"
    ]

    found_warning = any(phrase.lower() in content.lower() for phrase in warning_phrases)
    assert found_warning, "Ensemble model card missing warning about underperformance"


def test_datasheet_exists_and_complete():
    """Verify datasheet exists and contains key sections."""
    assert DATASHEET.exists(), f"Datasheet not found: {DATASHEET}"

    required_sections = [
        "Motivation",
        "Composition",
        "Collection Process",
        "Preprocessing",
        "Uses",
        "Distribution",
        "Maintenance"
    ]

    with open(DATASHEET, "r") as f:
        content = f.read()

    # Check for Gebru et al. framework sections
    for section in required_sections:
        assert section in content, f"Datasheet missing section: {section}"

    # Check for critical warnings
    assert "model-generated" in content.lower() or "model generated" in content.lower(), \
        "Datasheet missing 'model-generated' warning"
    assert "NOT" in content or "not experimental" in content.lower(), \
        "Datasheet missing experimental data disclaimer"


def test_citation_cff_exists():
    """Verify CITATION.cff exists at repo root."""
    citation_file = PROJECT_ROOT / "CITATION.cff"
    assert citation_file.exists(), f"CITATION.cff not found: {citation_file}"

    with open(citation_file, "r") as f:
        content = f.read()

    # Check for required CFF fields
    assert "cff-version: 1.2.0" in content, "Missing cff-version"
    assert "title:" in content, "Missing title"
    assert "license: MIT" in content, "Missing license"
    assert "version: 1.0.0" in content, "Missing version"
