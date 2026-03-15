"""Tests for NIST experimental validation."""

import numpy as np
import pandas as pd
import pytest

from model.data.load import DATA_DIR
from model.predict import predict_pcsaft
from model.thermodynamic import T_REF, compute_properties
from scripts.run_nist_validation import compute_mare, within_tolerance


class TestNISTValidationDataset:
    """Tests for NIST validation dataset."""

    def test_dataset_exists(self):
        """NIST validation CSV should exist."""
        nist_csv = DATA_DIR / "nist_validation.csv"
        assert nist_csv.exists()

    def test_dataset_structure(self):
        """NIST validation dataset should have required columns."""
        nist_csv = DATA_DIR / "nist_validation.csv"
        df = pd.read_csv(nist_csv)

        required_cols = ['smiles', 'name', 'cas_number', 'vp_exp_Pa',
                        'rho_exp_kg_m3', 'T_K', 'source']
        assert all(col in df.columns for col in required_cols)

    def test_dataset_size(self):
        """NIST validation dataset should have >= 50 molecules."""
        nist_csv = DATA_DIR / "nist_validation.csv"
        df = pd.read_csv(nist_csv)
        assert len(df) >= 50

    def test_dataset_has_experimental_data(self):
        """Dataset should have molecules with both VP and density data."""
        nist_csv = DATA_DIR / "nist_validation.csv"
        df = pd.read_csv(nist_csv)
        valid = df.dropna(subset=['vp_exp_Pa', 'rho_exp_kg_m3'])
        assert len(valid) >= 40, "Should have at least 40 molecules with complete data"


class TestValidationMetrics:
    """Tests for validation metric calculations."""

    def test_compute_mare_basic(self):
        """MARE should be computed correctly for simple cases."""
        predicted = np.array([100, 200, 300])
        experimental = np.array([100, 200, 300])
        mare = compute_mare(predicted, experimental)
        assert mare == 0.0

    def test_compute_mare_with_error(self):
        """MARE should compute percentage error correctly."""
        predicted = np.array([110, 220, 330])
        experimental = np.array([100, 200, 300])
        mare = compute_mare(predicted, experimental)
        assert np.isclose(mare, 10.0)

    def test_compute_mare_handles_nan(self):
        """MARE should skip NaN values."""
        predicted = np.array([110, np.nan, 330])
        experimental = np.array([100, 200, 300])
        mare = compute_mare(predicted, experimental)
        # Only first and third values: (10% + 10%) / 2 = 10%
        assert np.isclose(mare, 10.0)

    def test_within_tolerance_all_good(self):
        """within_tolerance should return 1.0 when all within bounds."""
        predicted = np.array([110, 210, 310])
        experimental = np.array([100, 200, 300])
        frac = within_tolerance(predicted, experimental, tolerance=0.20)
        assert frac == 1.0

    def test_within_tolerance_partial(self):
        """within_tolerance should compute fraction correctly."""
        predicted = np.array([110, 250, 310])
        experimental = np.array([100, 200, 300])
        # First and third are within 20%, second is not (25% error)
        frac = within_tolerance(predicted, experimental, tolerance=0.20)
        assert np.isclose(frac, 2/3)


class TestPredictionWorkflow:
    """Tests for end-to-end prediction workflow."""

    @pytest.fixture
    def sample_molecules(self):
        """Sample molecules for testing."""
        return ['CCCC', 'CCCCC', 'c1ccccc1']

    def test_rf_predictions(self, sample_molecules):
        """RF should predict parameters for sample molecules."""
        df = predict_pcsaft(sample_molecules)
        assert len(df) == len(sample_molecules)
        assert all(col in df.columns for col in ['smiles', 'm', 'sigma', 'epsilon_k'])
        assert df['m'].notna().all()
        assert df['sigma'].notna().all()
        assert df['epsilon_k'].notna().all()

    def test_thermodynamic_properties(self, sample_molecules):
        """Thermodynamic properties should be computable from predictions."""
        df = predict_pcsaft(sample_molecules)
        props = []

        for _, row in df.iterrows():
            p = compute_properties(row['m'], row['sigma'], row['epsilon_k'], T_REF)
            props.append(p)

        # At least some should succeed (non-polar molecules)
        success = sum(1 for p in props if not np.isnan(p['vapor_pressure_Pa']))
        assert success >= len(sample_molecules) / 2

    def test_density_conversion(self):
        """Density conversion from mol/m³ to kg/m³ should work."""
        from rdkit import Chem

        smiles = 'CCCC'
        mol = Chem.MolFromSmiles(smiles)
        mw = Chem.Descriptors.MolWt(mol)  # g/mol

        rho_mol_m3 = 10000  # mol/m³
        rho_kg_m3 = rho_mol_m3 * (mw / 1000)  # kg/m³

        # Butane MW ≈ 58 g/mol → 10000 * 0.058 = 580 kg/m³
        assert 500 < rho_kg_m3 < 650


class TestSPTComparison:
    """Tests for SPT-PCSAFT comparison."""

    def test_spt_dataset_exists(self):
        """SPT-PCSAFT CSV should exist."""
        spt_csv = DATA_DIR / "spt_pcsaft.csv"
        assert spt_csv.exists()

    def test_spt_dataset_structure(self):
        """SPT-PCSAFT dataset should have required columns."""
        spt_csv = DATA_DIR / "spt_pcsaft.csv"
        df = pd.read_csv(spt_csv, skiprows=1)  # Skip license header

        required_cols = ['SMILES0', 'm', 'sigma', 'epsilon_k']
        assert all(col in df.columns for col in required_cols)

    def test_spt_overlap_with_validation_set(self):
        """SPT dataset should overlap with validation set."""
        nist_csv = DATA_DIR / "nist_validation.csv"
        spt_csv = DATA_DIR / "spt_pcsaft.csv"

        nist_df = pd.read_csv(nist_csv)
        spt_df = pd.read_csv(spt_csv, skiprows=1)

        # Merge on SMILES
        merged = nist_df.merge(spt_df, left_on='smiles', right_on='SMILES0', how='inner')
        assert len(merged) >= 20, "Should have at least 20 molecules in common"
