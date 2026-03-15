"""Random Forest model wrapper for PC-SAFT parameter prediction."""

import logging

import joblib
import numpy as np

from pcsaft_predict._features import build_features
from pcsaft_predict._weights import check_model_exists, get_cache_dir, get_model_path

logger = logging.getLogger(__name__)

TARGETS = ["m", "sigma", "epsilon_k"]


class RFModel:
    """Random Forest model wrapper for PC-SAFT prediction.

    Loads three separate RF models (one per target: m, sigma, epsilon_k)
    and associated feature configuration artifacts.
    """

    def __init__(self):
        self._models = None
        self._rdkit_names = None
        self._scaler_path = None
        self._loaded = False

    def load(self):
        """Load RF models and feature configuration."""
        if self._loaded:
            return

        cache_dir = get_cache_dir()

        # Load RF models
        self._models = {}
        for target in TARGETS:
            model_file = f"rf_{target}.joblib"
            if not check_model_exists(model_file):
                raise FileNotFoundError(
                    f"RF model not found: {get_model_path(model_file)}. "
                    "Ensure model files are in the cache directory."
                )
            self._models[target] = joblib.load(get_model_path(model_file))
            logger.info("Loaded RF model for %s", target)

        # Load feature names
        feature_names_path = cache_dir / "feature_names.joblib"
        if feature_names_path.exists():
            feature_names = joblib.load(feature_names_path)
            # Extract non-Morgan feature names (RDKit descriptors)
            self._rdkit_names = [n for n in feature_names if not n.startswith("morgan_")]
        else:
            logger.warning("feature_names.joblib not found; using all RDKit descriptors")
            self._rdkit_names = None

        # Load scaler
        scaler_path = cache_dir / "rdkit_scaler.joblib"
        if scaler_path.exists():
            self._scaler_path = scaler_path
        else:
            logger.warning("rdkit_scaler.joblib not found; skipping feature scaling")
            self._scaler_path = None

        self._loaded = True
        logger.info("RF model loaded successfully")

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        """Predict PC-SAFT parameters for a list of SMILES.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings to predict.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys: m, sigma, epsilon_k. Each value is a 1D array
            of predictions. Invalid SMILES get NaN predictions.
        """
        if not self._loaded:
            self.load()

        # Build features
        features = build_features(smiles_list, self._rdkit_names, self._scaler_path)

        # Check for valid features (no NaN/inf in entire row)
        valid_mask = np.isfinite(features).all(axis=1)

        # Initialize results with NaN
        result = {}
        for target in TARGETS:
            preds = np.full(len(smiles_list), np.nan)
            if valid_mask.any():
                preds[valid_mask] = self._models[target].predict(features[valid_mask])
            result[target] = preds

        return result

    def predict_with_uncertainty(
        self, smiles_list: list[str], n_forward: int = 30
    ) -> dict[str, np.ndarray]:
        """Predict PC-SAFT parameters with RF uncertainty from tree disagreement.

        Uses the variance across individual tree predictions as a measure of
        model uncertainty.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings to predict.
        n_forward : int
            Unused (for API compatibility with NN/GNN models).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys: m, sigma, epsilon_k, m_std, sigma_std, epsilon_k_std.
            Invalid SMILES get NaN predictions and uncertainties.
        """
        if not self._loaded:
            self.load()

        # Build features
        features = build_features(smiles_list, self._rdkit_names, self._scaler_path)

        # Check for valid features
        valid_mask = np.isfinite(features).all(axis=1)

        # Initialize results
        result = {}
        for target in TARGETS:
            preds = np.full(len(smiles_list), np.nan)
            stds = np.full(len(smiles_list), np.nan)

            if valid_mask.any():
                model = self._models[target]
                # Get per-tree predictions
                tree_preds = np.array(
                    [tree.predict(features[valid_mask]) for tree in model.estimators_]
                )
                preds[valid_mask] = tree_preds.mean(axis=0)
                stds[valid_mask] = tree_preds.std(axis=0)

            result[target] = preds
            result[f"{target}_std"] = stds

        return result
