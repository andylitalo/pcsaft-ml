"""Singleton model server: loads a registry model once and caches it."""

import logging

import numpy as np
from rdkit import Chem

from model.data.load import TARGETS
from model.registry import get_model

logger = logging.getLogger(__name__)

_AD_MODEL = None
_AD_AVAILABLE = False


def _load_ad_model():
    """Load descriptor-space Isolation Forest AD model; return None on failure."""
    global _AD_MODEL, _AD_AVAILABLE
    try:
        from model.nn.ad import load_ad_model

        _AD_MODEL = load_ad_model()
        _AD_AVAILABLE = True
        logger.info("Descriptor-space AD model loaded successfully")
    except (FileNotFoundError, ImportError):
        _AD_MODEL = None
        _AD_AVAILABLE = False
        logger.info(
            "Descriptor-space AD model not available; "
            "will use model-specific AD if available"
        )


class ModelServer:
    """Wraps a registry model with prediction, uncertainty, and AD logic."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self._model = get_model(model_type)
        self._model.load()

        # Load descriptor-space AD for RF and NN
        if model_type in {"rf", "nn"}:
            _load_ad_model()

        self._warmup()
        logger.info("ModelServer ready: model_type=%s", model_type)

    def _warmup(self):
        """Run a throwaway prediction to surface load-time errors early."""
        self._model.predict(["C"])  # methane
        logger.info("Warm-up prediction complete")

    def predict(self, smiles_list: list[str]) -> list[dict]:
        """Run predictions with uncertainty, AD check, and association flag.

        Returns a list of dicts (one per input SMILES), each with keys:
        smiles, m, sigma, epsilon_k, uncertainty, in_domain, is_associating, valid.
        Invalid SMILES get valid=False and NaN parameters.
        """
        # Identify invalid SMILES up front
        valid_mask = [Chem.MolFromSmiles(s) is not None for s in smiles_list]

        # Predict with uncertainty for all SMILES (registry handles NaN for bad ones)
        try:
            result = self._model.predict_with_uncertainty(smiles_list)
        except Exception:
            logger.exception("predict_with_uncertainty failed, falling back to predict")
            result = self._model.predict(smiles_list)

        # AD check (model-specific)
        in_domain = np.ones(len(smiles_list), dtype=bool)
        if self.model_type == "chemberta":
            # Use ChemBERTa embedding-space AD
            try:
                in_domain = self._model.predict_in_domain(smiles_list)
            except Exception:
                logger.exception("ChemBERTa AD check failed; marking all as in-domain")
        elif self.model_type in {"rf", "nn"} and _AD_AVAILABLE and _AD_MODEL is not None:
            # Use descriptor-space AD for RF and NN
            try:
                from model.registry import _compute_features

                X = _compute_features(smiles_list)
                feature_valid = np.isfinite(X).all(axis=1)
                for i in range(len(smiles_list)):
                    if feature_valid[i]:
                        labels = _AD_MODEL.predict(X[i : i + 1])
                        in_domain[i] = labels[0] == 1
                    else:
                        in_domain[i] = False
            except Exception:
                logger.exception("Descriptor-space AD check failed; marking all as in-domain")

        # Association check
        from screening.filters import is_associating

        predictions = []
        for i, smi in enumerate(smiles_list):
            if not valid_mask[i] or not np.isfinite(result["m"][i]):
                predictions.append(
                    {
                        "smiles": smi,
                        "m": 0.0,
                        "sigma": 0.0,
                        "epsilon_k": 0.0,
                        "uncertainty": None,
                        "in_domain": False,
                        "is_associating": False,
                        "valid": False,
                    }
                )
                continue

            uncertainty = None
            if f"{TARGETS[0]}_std" in result:
                m_std = result["m_std"][i]
                sigma_std = result["sigma_std"][i]
                ek_std = result["epsilon_k_std"][i]
                if np.isfinite(m_std) and np.isfinite(sigma_std) and np.isfinite(ek_std):
                    uncertainty = {
                        "m_std": float(m_std),
                        "sigma_std": float(sigma_std),
                        "epsilon_k_std": float(ek_std),
                    }

            predictions.append(
                {
                    "smiles": smi,
                    "m": float(result["m"][i]),
                    "sigma": float(result["sigma"][i]),
                    "epsilon_k": float(result["epsilon_k"][i]),
                    "uncertainty": uncertainty,
                    "in_domain": bool(in_domain[i]),
                    "is_associating": is_associating(smi),
                    "valid": True,
                }
            )

        return predictions
