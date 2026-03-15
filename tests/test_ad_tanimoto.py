"""Tests for TanimotoAD and Williams plot modules."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from model.ad_tanimoto import TanimotoAD
from model.ad_williams import compute_leverage, plot_williams


class TestTanimotoAD:
    """Test TanimotoAD functionality."""

    def test_init(self):
        """Test TanimotoAD initialization."""
        ad = TanimotoAD(radius=2, n_bits=2048)
        assert ad.radius == 2
        assert ad.n_bits == 2048
        assert ad.train_fps_ is None

    def test_fit(self):
        """Test fitting TanimotoAD on training SMILES."""
        ad = TanimotoAD()
        train_smiles = ["C1CCCC1", "CCCCCC", "c1ccccc1"]
        ad.fit(train_smiles)
        assert ad.train_fps_ is not None
        assert len(ad.train_fps_) == 3

    def test_fit_with_invalid_smiles(self):
        """Test that invalid SMILES are skipped during fit."""
        ad = TanimotoAD()
        train_smiles = ["C1CCCC1", "INVALID", "CCCCCC"]
        ad.fit(train_smiles)
        assert ad.train_fps_ is not None
        # Only 2 valid molecules
        assert len(ad.train_fps_) == 2

    def test_tanimoto_nn_identical(self):
        """Test Tanimoto NN returns 1.0 for identical molecule."""
        ad = TanimotoAD()
        train_smiles = ["C1CCCC1", "CCCCCC"]
        ad.fit(train_smiles)
        # Query with same molecule
        similarity = ad.tanimoto_nn("C1CCCC1")
        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_tanimoto_nn_similar(self):
        """Test Tanimoto NN for similar molecules."""
        ad = TanimotoAD()
        train_smiles = ["C1CCCC1"]  # cyclopentane
        ad.fit(train_smiles)
        # Cyclohexane is very similar (both are cycloalkanes)
        similarity = ad.tanimoto_nn("C1CCCCC1")
        assert 0.7 < similarity <= 1.0

    def test_tanimoto_nn_dissimilar(self):
        """Test Tanimoto NN for dissimilar molecules."""
        ad = TanimotoAD()
        train_smiles = ["CCC"]  # propane
        ad.fit(train_smiles)
        # Benzene is very different
        similarity = ad.tanimoto_nn("c1ccccc1")
        assert similarity < 0.3

    def test_tanimoto_nn_invalid_smiles(self):
        """Test Tanimoto NN returns 0.0 for invalid SMILES."""
        ad = TanimotoAD()
        train_smiles = ["C1CCCC1"]
        ad.fit(train_smiles)
        similarity = ad.tanimoto_nn("INVALID_SMILES")
        assert similarity == 0.0

    def test_tanimoto_nn_not_fitted(self):
        """Test that tanimoto_nn raises error if not fitted."""
        ad = TanimotoAD()
        with pytest.raises(ValueError, match="not fitted"):
            ad.tanimoto_nn("C1CCCC1")

    def test_in_domain(self):
        """Test in_domain classification."""
        ad = TanimotoAD()
        train_smiles = ["C1CCCC1", "CCCCCC", "c1ccccc1"]
        ad.fit(train_smiles)

        # Same molecule should be in domain
        assert ad.in_domain("C1CCCC1") is True

        # Very dissimilar molecule should be out of domain
        # Use a complex fluorinated molecule
        assert ad.in_domain("FC(F)(F)C(F)(F)C(F)(F)F") is False

    def test_thresholds(self):
        """Test that threshold constants are set correctly."""
        assert TanimotoAD.IN_DOMAIN_THRESHOLD == 0.4
        assert TanimotoAD.WARNING_THRESHOLD == 0.3


class TestWilliamsPlot:
    """Test Williams plot functions."""

    def test_compute_leverage_basic(self):
        """Test basic leverage computation."""
        # Simple 2D case
        X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        X_test = np.array([[1.5, 2.5], [10.0, 10.0]])

        leverage = compute_leverage(X_train, X_test, pca_components=None)
        assert leverage.shape == (2,)
        # Test point close to training data should have moderate leverage
        # Test point far from training data should have high leverage
        assert leverage[1] > leverage[0]

    def test_compute_leverage_with_pca(self):
        """Test leverage computation with PCA dimension reduction."""
        # High-dimensional case
        np.random.seed(42)
        X_train = np.random.randn(50, 100)
        X_test = np.random.randn(10, 100)

        leverage = compute_leverage(X_train, X_test, pca_components=10)
        assert leverage.shape == (10,)
        assert np.all(leverage >= 0)

    def test_compute_leverage_no_pca_when_low_dim(self):
        """Test that PCA is skipped when dimensionality is low."""
        X_train = np.random.randn(50, 10)
        X_test = np.random.randn(5, 10)

        # Request PCA with 50 components but only have 10 features
        leverage = compute_leverage(X_train, X_test, pca_components=50)
        assert leverage.shape == (5,)

    def test_plot_williams(self):
        """Test Williams plot generation."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        X_test = np.random.randn(20, 50)
        y_test = np.random.randn(20)
        y_pred = y_test + np.random.randn(20) * 0.1  # Add some noise

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "williams_test.png"
            plot_williams(
                X_train,
                X_test,
                y_test,
                y_pred,
                target_name="test_param",
                output_path=str(output_path),
            )
            assert output_path.exists()
            # Check file is not empty
            assert output_path.stat().st_size > 1000

    def test_plot_williams_with_pca(self):
        """Test Williams plot with PCA dimension reduction."""
        np.random.seed(42)
        X_train = np.random.randn(100, 200)  # High dimensional
        X_test = np.random.randn(20, 200)
        y_test = np.random.randn(20)
        y_pred = y_test + np.random.randn(20) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "williams_pca_test.png"
            plot_williams(
                X_train,
                X_test,
                y_test,
                y_pred,
                target_name="test_param",
                output_path=str(output_path),
                pca_components=50,
            )
            assert output_path.exists()
