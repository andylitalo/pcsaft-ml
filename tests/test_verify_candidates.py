"""Tests for Step 12: Multi-criteria candidate verification."""

import pandas as pd
import pytest

from scripts.verify_candidates import (
    PARAM_TOP_N,
    SA_THRESHOLD,
    TEMPERATURES,
    VP_RATIO_BOUNDS,
    run_verification,
)


@pytest.fixture
def mock_thermo_csv(tmp_path):
    """Create a mock thermodynamic validation CSV for testing."""
    data = {
        "smiles": [
            "F[C@H]1C[C@@H](F)C1",  # Good candidate (verified)
            "C=C(Cl)CC",  # Good candidate (verified)
            "ClC1=CC1",  # Fails EOS
            "C1=CCCC1",  # Cyclopentane (good params, high VP)
            "CCCCCCCC",  # Far in param space (fails criterion 2)
        ],
        "sa_score": [3.3, 2.8, 2.9, 2.7, 2.0],
        "m": [2.33, 2.35, 2.33, 2.31, 3.50],
        "sigma": [3.55, 3.45, 3.55, 3.66, 4.20],
        "epsilon_k": [299.6, 301.0, 299.6, 276.0, 350.0],
        "distance": [0.127, 0.160, 0.079, 0.082, 0.900],
        "is_associating": [False] * 5,
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "thermo_validation.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_run_verification_structure(mock_thermo_csv, tmp_path):
    """Test that verification produces expected output structure."""
    output_csv = tmp_path / "verified.csv"
    _ = run_verification(mock_thermo_csv, output_csv)

    # Check output file exists
    assert output_csv.exists()

    # Load full results
    df = pd.read_csv(output_csv)

    # Check all expected columns present
    expected_cols = [
        "in_domain",
        "vp_Pa_273K",
        "vp_Pa_298K",
        "vp_Pa_323K",
        "rho_273K",
        "rho_298K",
        "rho_323K",
        "vp_ratio_273K",
        "vp_ratio_298K",
        "vp_ratio_323K",
        "criterion_ad",
        "criterion_param",
        "criterion_eos",
        "criterion_vp",
        "criterion_sa",
        "all_criteria",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_run_verification_criteria_logic(mock_thermo_csv, tmp_path):
    """Test that criteria are applied correctly."""
    output_csv = tmp_path / "verified.csv"
    run_verification(mock_thermo_csv, output_csv)

    df = pd.read_csv(output_csv)

    # All candidates should have SA ≤ 4.5 in our mock data
    assert df["criterion_sa"].all()

    # Check that all_criteria requires all five criteria
    for _, row in df.iterrows():
        expected = (
            row["criterion_ad"]
            and row["criterion_param"]
            and row["criterion_eos"]
            and row["criterion_vp"]
            and row["criterion_sa"]
        )
        assert row["all_criteria"] == expected


def test_multi_temperature_computation(mock_thermo_csv, tmp_path):
    """Test that properties are computed at all three temperatures."""
    output_csv = tmp_path / "verified.csv"
    run_verification(mock_thermo_csv, output_csv)

    df = pd.read_csv(output_csv)

    # Check that we have VP and density at all temperatures
    for T in TEMPERATURES:
        temp_k = int(T)
        assert f"vp_Pa_{temp_k}K" in df.columns
        assert f"rho_{temp_k}K" in df.columns
        assert f"vp_ratio_{temp_k}K" in df.columns

    # VP should generally increase with temperature (for valid candidates)
    for _, row in df.iterrows():
        vp_273 = row["vp_Pa_273K"]
        vp_298 = row["vp_Pa_298K"]
        vp_323 = row["vp_Pa_323K"]

        # Skip if any NaN (EOS failure)
        if pd.isna(vp_273) or pd.isna(vp_298) or pd.isna(vp_323):
            continue

        # VP should increase with temperature
        assert vp_273 < vp_298 < vp_323, f"VP not monotonic for {row['smiles']}"


def test_parameter_proximity_criterion(mock_thermo_csv, tmp_path):
    """Test that parameter proximity criterion selects top-N candidates."""
    output_csv = tmp_path / "verified.csv"
    run_verification(mock_thermo_csv, output_csv)

    df = pd.read_csv(output_csv)

    # Get threshold
    param_threshold = df["distance"].nsmallest(PARAM_TOP_N).max()

    # Check that criterion_param matches this threshold
    for _, row in df.iterrows():
        expected = row["distance"] <= param_threshold
        assert row["criterion_param"] == expected


def test_vp_ratio_bounds_criterion(mock_thermo_csv, tmp_path):
    """Test that VP ratio criterion checks all temperatures."""
    output_csv = tmp_path / "verified.csv"
    run_verification(mock_thermo_csv, output_csv)

    df = pd.read_csv(output_csv)

    for _, row in df.iterrows():
        if not row["criterion_vp"]:
            # At least one temperature should be out of bounds or NaN
            ratios = [row[f"vp_ratio_{int(T)}K"] for T in TEMPERATURES]
            assert any(
                pd.isna(r) or r < VP_RATIO_BOUNDS[0] or r > VP_RATIO_BOUNDS[1]
                for r in ratios
            )


def test_eos_convergence_criterion(mock_thermo_csv, tmp_path):
    """Test that EOS convergence checks all temperatures."""
    output_csv = tmp_path / "verified.csv"
    run_verification(mock_thermo_csv, output_csv)

    df = pd.read_csv(output_csv)

    for _, row in df.iterrows():
        vps = [row[f"vp_Pa_{int(T)}K"] for T in TEMPERATURES]
        expected = all(not pd.isna(vp) for vp in vps)
        assert row["criterion_eos"] == expected


def test_sa_threshold_criterion(mock_thermo_csv, tmp_path):
    """Test that SA criterion uses correct threshold."""
    output_csv = tmp_path / "verified.csv"
    run_verification(mock_thermo_csv, output_csv)

    df = pd.read_csv(output_csv)

    for _, row in df.iterrows():
        expected = row["sa_score"] <= SA_THRESHOLD
        assert row["criterion_sa"] == expected


def test_verified_candidates_return(mock_thermo_csv, tmp_path):
    """Test that run_verification returns only verified candidates."""
    output_csv = tmp_path / "verified.csv"
    verified = run_verification(mock_thermo_csv, output_csv)

    # All returned candidates should have all_criteria = True
    assert verified["all_criteria"].all()

    # Load full results
    full_df = pd.read_csv(output_csv)

    # Number of verified should match all_criteria sum
    assert len(verified) == full_df["all_criteria"].sum()


def test_in_domain_column_added(mock_thermo_csv, tmp_path):
    """Test that in_domain column is computed and added."""
    output_csv = tmp_path / "verified.csv"
    run_verification(mock_thermo_csv, output_csv)

    df = pd.read_csv(output_csv)

    # Column should exist
    assert "in_domain" in df.columns

    # Should be boolean-like (0 or 1 after CSV roundtrip)
    assert df["in_domain"].isin([0, 1, True, False]).all()


def test_empty_verification_result(tmp_path):
    """Test handling of case where no candidates pass all criteria."""
    # Create mock data where all candidates fail at least one criterion
    data = {
        "smiles": ["CCCCCCCCCCCC", "c1ccccc1"],  # Far from cyclopentane
        "sa_score": [2.0, 2.0],
        "m": [5.0, 3.0],
        "sigma": [5.0, 4.0],
        "epsilon_k": [400.0, 350.0],
        "distance": [2.0, 1.5],  # Very far
        "is_associating": [False, False],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "bad_candidates.csv"
    df.to_csv(csv_path, index=False)

    output_csv = tmp_path / "verified.csv"
    verified = run_verification(csv_path, output_csv)

    # Should return empty DataFrame
    assert len(verified) == 0

    # But output file should still exist with all candidates
    full_df = pd.read_csv(output_csv)
    assert len(full_df) == 2
    assert not full_df["all_criteria"].any()
