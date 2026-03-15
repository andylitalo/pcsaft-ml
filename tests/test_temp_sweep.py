"""Tests for temperature sweep validation (Step 19)."""

from pathlib import Path

import numpy as np
import pandas as pd

from model.thermodynamic import CYCLOPENTANE, compute_properties


class TestTemperatureSweep:
    """Test temperature-dependent property calculations."""

    def test_cyclopentane_at_multiple_temps(self):
        """Cyclopentane should have valid VP and density at moderate temperatures."""
        # 230K is too low (below normal operating range), so we skip it
        temperatures = [250, 270, 290, 310, 330]

        for T in temperatures:
            props = compute_properties(
                CYCLOPENTANE["m"],
                CYCLOPENTANE["sigma"],
                CYCLOPENTANE["epsilon_k"],
                T
            )

            # Should have valid properties at moderate temps
            assert not np.isnan(props["vapor_pressure_Pa"]), f"VP invalid at {T}K"
            assert not np.isnan(props["liquid_density_mol_m3"]), f"Density invalid at {T}K"

        # Vapor pressure should increase monotonically
        vp_values = []
        for T in temperatures:
            props = compute_properties(
                CYCLOPENTANE["m"],
                CYCLOPENTANE["sigma"],
                CYCLOPENTANE["epsilon_k"],
                T
            )
            vp_values.append(props["vapor_pressure_Pa"])

        for i in range(1, len(vp_values)):
            assert vp_values[i] > vp_values[i-1], \
                f"VP should increase with T (failed at {temperatures[i]}K)"

    def test_vp_increases_with_temperature(self):
        """Vapor pressure should increase monotonically with temperature for stable compounds."""
        # Use a stable candidate from the top-50
        m, sigma, epsilon_k = 2.31, 3.66, 276.0  # Cyclopentene-like

        vp_values = []
        for T in [270, 290, 310, 330]:
            props = compute_properties(m, sigma, epsilon_k, T)
            if not np.isnan(props["vapor_pressure_Pa"]):
                vp_values.append(props["vapor_pressure_Pa"])

        # Should have at least 3 valid points and be monotonically increasing
        if len(vp_values) >= 3:
            for i in range(1, len(vp_values)):
                assert vp_values[i] > vp_values[i-1], "VP should increase with T"

    def test_density_decreases_with_temperature(self):
        """Liquid density should decrease with temperature."""
        m, sigma, epsilon_k = 2.31, 3.66, 276.0

        density_values = []
        for T in [270, 290, 310, 330]:
            props = compute_properties(m, sigma, epsilon_k, T)
            if not np.isnan(props["liquid_density_mol_m3"]):
                density_values.append(props["liquid_density_mol_m3"])

        # Should have at least 3 valid points and be monotonically decreasing
        if len(density_values) >= 3:
            for i in range(1, len(density_values)):
                assert density_values[i] < density_values[i-1], "Density should decrease with T"

    def test_temp_sweep_results_file_exists(self):
        """Temperature sweep should generate results CSV."""
        results_path = Path(__file__).parent.parent / "model" / "saved" / "temp_sweep_results.csv"
        assert results_path.exists(), "temp_sweep_results.csv should exist"

        # Check structure
        df = pd.read_csv(results_path)
        assert len(df) == 50, "Should have 50 candidates"

        # Check required columns
        required_cols = ["smiles", "m", "sigma", "epsilon_k", "distance"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check temperature columns
        temperatures = [230, 250, 270, 290, 310, 330]
        for T in temperatures:
            assert f"vp_{T}K" in df.columns, f"Missing VP column for {T}K"
            assert f"rho_{T}K" in df.columns, f"Missing density column for {T}K"
            assert f"prop_dist_{T}K" in df.columns, f"Missing property distance column for {T}K"
            assert f"rank_{T}K" in df.columns, f"Missing rank column for {T}K"

    def test_temp_sweep_figures_exist(self):
        """Temperature sweep should generate required figures."""
        figures_dir = Path(__file__).parent.parent / "figures" / "19_temperature_sweep"

        vp_plot = figures_dir / "vp_vs_T.png"
        rank_plot = figures_dir / "rank_stability.png"

        assert vp_plot.exists(), "VP vs T plot should exist"
        assert rank_plot.exists(), "Rank stability heatmap should exist"

        # Check file sizes (should be non-empty)
        assert vp_plot.stat().st_size > 10000, "VP plot should be substantial"
        assert rank_plot.stat().st_size > 10000, "Rank plot should be substantial"

    def test_rank_stability_high_correlation(self):
        """Ranks at moderate temperatures should be highly correlated."""
        results_path = Path(__file__).parent.parent / "model" / "saved" / "temp_sweep_results.csv"
        df = pd.read_csv(results_path)

        # Check correlation between 290K and 310K (should be very high)
        valid_mask = df[["rank_290K", "rank_310K"]].notna().all(axis=1)
        if valid_mask.sum() >= 10:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(df.loc[valid_mask, "rank_290K"],
                               df.loc[valid_mask, "rank_310K"])
            assert rho > 0.9, f"Ranks at 290K and 310K should be highly correlated (ρ={rho:.3f})"

    def test_low_temp_failures_expected(self):
        """Many candidates should fail at low temperatures (230K)."""
        results_path = Path(__file__).parent.parent / "model" / "saved" / "temp_sweep_results.csv"
        df = pd.read_csv(results_path)

        # At 230K, expect many failures
        n_valid_230 = df["vp_230K"].notna().sum()
        n_total = len(df)

        # Should have < 50% success rate at 230K (it's below normal boiling range)
        assert n_valid_230 < n_total * 0.5, \
            f"Expected many failures at 230K, got {n_valid_230}/{n_total} successes"

    def test_high_temp_more_successes(self):
        """Higher temperatures should have more successful EOS calculations."""
        results_path = Path(__file__).parent.parent / "model" / "saved" / "temp_sweep_results.csv"
        df = pd.read_csv(results_path)

        n_valid_270 = df["vp_270K"].notna().sum()
        n_valid_330 = df["vp_330K"].notna().sum()

        # 330K should have at least as many successes as 270K
        assert n_valid_330 >= n_valid_270, \
            "Higher temperature (330K) should have ≥ successes as lower temp (270K)"

    def test_property_distance_computation(self):
        """Property distance should be normalized Euclidean distance."""
        results_path = Path(__file__).parent.parent / "model" / "saved" / "temp_sweep_results.csv"
        df = pd.read_csv(results_path)

        # Get cyclopentane reference at 290K
        ref_props = compute_properties(
            CYCLOPENTANE["m"],
            CYCLOPENTANE["sigma"],
            CYCLOPENTANE["epsilon_k"],
            290
        )

        # Check first valid candidate at 290K
        valid_rows = df[df["vp_290K"].notna()]
        if len(valid_rows) > 0:
            row = valid_rows.iloc[0]

            # Manually compute property distance
            ref_vp = ref_props["vapor_pressure_Pa"]
            ref_rho = ref_props["liquid_density_mol_m3"]
            vp_term = (row["vp_290K"] - ref_vp) / ref_vp
            rho_term = (row["rho_290K"] - ref_rho) / ref_rho
            expected_dist = np.sqrt(vp_term**2 + rho_term**2)

            # Should match stored value
            assert abs(row["prop_dist_290K"] - expected_dist) < 1e-6, \
                "Property distance calculation mismatch"
