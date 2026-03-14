"""Tests for thermodynamic property computation module."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.thermodynamic import (
    CYCLOPENTANE,
    compute_properties,
    make_pcsaft_model,
    validate_candidates,
)


def test_make_pcsaft_model():
    """Test that PC-SAFT model creation works."""
    model = make_pcsaft_model(
        CYCLOPENTANE["m"], CYCLOPENTANE["sigma"], CYCLOPENTANE["epsilon_k"]
    )
    assert model is not None
    # teqp models don't expose much of an API, but we can check it's callable
    assert hasattr(model, "pure_VLE_T")


def test_compute_properties_returns_dict():
    """Test that compute_properties returns expected structure."""
    props = compute_properties(
        CYCLOPENTANE["m"], CYCLOPENTANE["sigma"], CYCLOPENTANE["epsilon_k"]
    )
    assert isinstance(props, dict)
    assert "vapor_pressure_Pa" in props
    assert "liquid_density_mol_m3" in props


def test_compute_properties_cyclopentane():
    """Test that cyclopentane gives reasonable properties."""
    props = compute_properties(
        CYCLOPENTANE["m"], CYCLOPENTANE["sigma"], CYCLOPENTANE["epsilon_k"]
    )

    # Should get positive, finite values
    assert props["vapor_pressure_Pa"] > 0
    assert props["liquid_density_mol_m3"] > 0
    assert np.isfinite(props["vapor_pressure_Pa"])
    assert np.isfinite(props["liquid_density_mol_m3"])

    # Sanity checks: vapor pressure at 298K should be O(10-100 kPa)
    # and liquid density should be O(1000-10000 mol/m³)
    assert 1000 < props["vapor_pressure_Pa"] < 1e6  # 1-1000 kPa in Pa
    assert 1000 < props["liquid_density_mol_m3"] < 20000


def test_compute_properties_handles_unphysical_params():
    """Test that unphysical parameters return NaN gracefully."""
    # Very small sigma should cause problems
    props = compute_properties(2.0, 0.01, 300.0)
    assert np.isnan(props["vapor_pressure_Pa"])
    assert np.isnan(props["liquid_density_mol_m3"])

    # Negative m should cause problems
    props = compute_properties(-1.0, 3.7, 300.0)
    assert np.isnan(props["vapor_pressure_Pa"])
    assert np.isnan(props["liquid_density_mol_m3"])

    # Zero epsilon_k should cause problems
    props = compute_properties(2.0, 3.7, 0.0)
    assert np.isnan(props["vapor_pressure_Pa"])
    assert np.isnan(props["liquid_density_mol_m3"])


def test_validate_candidates_adds_columns():
    """Test that validate_candidates adds expected columns."""
    # Create minimal test dataframe
    test_df = pd.DataFrame(
        {
            "smiles": ["C1CCCC1", "c1ccccc1"],
            "m": [CYCLOPENTANE["m"], 2.5],
            "sigma": [CYCLOPENTANE["sigma"], 3.8],
            "epsilon_k": [CYCLOPENTANE["epsilon_k"], 300.0],
            "distance": [0.0, 0.2],
        }
    )

    validated, ref_props = validate_candidates(test_df)

    # Check new columns exist
    expected_cols = [
        "vapor_pressure_Pa",
        "liquid_density_mol_m3",
        "vp_ratio_to_ref",
        "rho_ratio_to_ref",
        "property_distance",
    ]
    for col in expected_cols:
        assert col in validated.columns

    # Check reference props are returned
    assert "vapor_pressure_Pa" in ref_props
    assert "liquid_density_mol_m3" in ref_props


def test_property_distance_zero_for_reference():
    """Test that property distance is 0 for cyclopentane itself."""
    test_df = pd.DataFrame(
        {
            "smiles": ["C1CCCC1"],
            "m": [CYCLOPENTANE["m"]],
            "sigma": [CYCLOPENTANE["sigma"]],
            "epsilon_k": [CYCLOPENTANE["epsilon_k"]],
            "distance": [0.0],
        }
    )

    validated, _ = validate_candidates(test_df)

    # Property distance should be very close to 0
    assert validated["property_distance"].iloc[0] < 1e-10


def test_validate_candidates_ratios():
    """Test that ratios to reference are computed correctly."""
    test_df = pd.DataFrame(
        {
            "smiles": ["C1CCCC1"],
            "m": [CYCLOPENTANE["m"]],
            "sigma": [CYCLOPENTANE["sigma"]],
            "epsilon_k": [CYCLOPENTANE["epsilon_k"]],
            "distance": [0.0],
        }
    )

    validated, _ = validate_candidates(test_df)

    # Ratios should be ~1.0 for cyclopentane
    assert abs(validated["vp_ratio_to_ref"].iloc[0] - 1.0) < 1e-10
    assert abs(validated["rho_ratio_to_ref"].iloc[0] - 1.0) < 1e-10


def test_spearman_correlation_calculation():
    """Test that Spearman correlation can be computed on results."""
    from scipy.stats import spearmanr

    # Create test data with known correlation
    test_df = pd.DataFrame(
        {
            "smiles": ["C1CCCC1", "c1ccccc1", "CC", "CCC"],
            "m": [2.3, 2.5, 1.8, 2.0],
            "sigma": [3.7, 3.8, 3.5, 3.6],
            "epsilon_k": [288, 300, 250, 270],
            "distance": [0.1, 0.2, 0.3, 0.4],
        }
    )

    validated, _ = validate_candidates(test_df)

    # Filter out any NaN values
    valid = validated.dropna(subset=["property_distance"])

    if len(valid) > 1:
        rho, p_value = spearmanr(valid["distance"], valid["property_distance"])
        # Should get finite correlation (unless data has no variation)
        # It's OK if p_value is NaN if there's no variation in the data
        assert np.isfinite(rho) or len(valid) < 3


def test_validation_script_runs():
    """Test that the validation script executes without error."""
    # Create a small test CSV
    test_csv = Path("test_screening_results.csv")
    test_df = pd.DataFrame(
        {
            "smiles": ["C1CCCC1", "c1ccccc1"],
            "sa_score": [2.7, 3.5],
            "m": [CYCLOPENTANE["m"], 2.5],
            "sigma": [CYCLOPENTANE["sigma"], 3.8],
            "epsilon_k": [CYCLOPENTANE["epsilon_k"], 300.0],
            "distance": [0.0, 0.2],
            "is_associating": [False, False],
        }
    )
    test_df.to_csv(test_csv, index=False)

    test_output = Path("test_thermo_validation.csv")

    try:
        # Run the script
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_thermo_validation.py",
                "--screening-results",
                str(test_csv),
                "--output",
                str(test_output),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should complete successfully
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Output file should exist
        assert test_output.exists()

        # Check output has expected columns
        output_df = pd.read_csv(test_output)
        assert "vapor_pressure_Pa" in output_df.columns
        assert "property_distance" in output_df.columns

    finally:
        # Cleanup
        if test_csv.exists():
            test_csv.unlink()
        if test_output.exists():
            test_output.unlink()
        # Clean up any generated figures
        test_fig_dir = Path("figures/09_thermodynamic_validation")
        if test_fig_dir.exists():
            for fig in test_fig_dir.glob("*.png"):
                # Don't delete if they're from the real run
                if fig.stat().st_size < 1000:  # Likely empty/test file
                    fig.unlink()
