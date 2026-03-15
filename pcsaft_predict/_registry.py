"""Simplified model registry for PC-SAFT prediction.

Provides high-level API functions: predict, predict_with_uncertainty, list_models, load_model.
"""

import logging

import pandas as pd

from pcsaft_predict._ad import TanimotoAD
from pcsaft_predict._rf import RFModel

logger = logging.getLogger(__name__)

# Global registry of available models
MODELS = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls):
        MODELS[name] = cls
        return cls

    return decorator


# Register the RF model
MODELS["rf"] = RFModel


def list_models() -> list[str]:
    """Return list of available model names.

    Returns
    -------
    list[str]
        List of registered model names.
    """
    return list(MODELS.keys())


def load_model(name: str):
    """Load a model by name.

    Parameters
    ----------
    name : str
        Model name (e.g., "rf").

    Returns
    -------
    Model instance
        Loaded model object.

    Raises
    ------
    KeyError
        If model name is not registered.
    """
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {list_models()}")

    model = MODELS[name]()
    model.load()
    return model


def predict(smiles, model="rf") -> pd.DataFrame:
    """Predict PC-SAFT parameters from SMILES.

    Parameters
    ----------
    smiles : str | list[str]
        Single SMILES string or list of SMILES strings.
    model : str
        Model name to use (default: "rf").

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - smiles: Input SMILES
        - m: Number of segments
        - sigma: Segment diameter (Angstrom)
        - epsilon_k: Dispersion energy (K)
        - in_domain: Boolean flag for applicability domain
        - tanimoto_nn: Max Tanimoto similarity to training set
    """
    # Convert single SMILES to list
    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    # Load model and predict
    model_obj = load_model(model)
    preds = model_obj.predict(smiles_list)

    # Load AD model and compute in_domain flags
    ad_model = TanimotoAD()
    ad_model.load()
    in_domain_flags, tanimoto_scores = ad_model.batch_predict(smiles_list)

    # Build result DataFrame
    df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "m": preds["m"],
            "sigma": preds["sigma"],
            "epsilon_k": preds["epsilon_k"],
            "in_domain": in_domain_flags,
            "tanimoto_nn": tanimoto_scores,
        }
    )

    return df


def predict_with_uncertainty(smiles, model="rf", n_forward=30) -> pd.DataFrame:
    """Predict PC-SAFT parameters with uncertainty estimates.

    Parameters
    ----------
    smiles : str | list[str]
        Single SMILES string or list of SMILES strings.
    model : str
        Model name to use (default: "rf").
    n_forward : int
        Number of forward passes for uncertainty estimation (default: 30).
        For RF models, this parameter is unused (uncertainty comes from tree variance).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - smiles: Input SMILES
        - m, sigma, epsilon_k: Predicted parameters
        - m_std, sigma_std, epsilon_k_std: Uncertainty estimates
        - in_domain: Boolean flag for applicability domain
        - tanimoto_nn: Max Tanimoto similarity to training set
    """
    # Convert single SMILES to list
    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    # Load model and predict with uncertainty
    model_obj = load_model(model)
    preds = model_obj.predict_with_uncertainty(smiles_list, n_forward=n_forward)

    # Load AD model and compute in_domain flags
    ad_model = TanimotoAD()
    ad_model.load()
    in_domain_flags, tanimoto_scores = ad_model.batch_predict(smiles_list)

    # Build result DataFrame
    df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "m": preds["m"],
            "sigma": preds["sigma"],
            "epsilon_k": preds["epsilon_k"],
            "m_std": preds["m_std"],
            "sigma_std": preds["sigma_std"],
            "epsilon_k_std": preds["epsilon_k_std"],
            "in_domain": in_domain_flags,
            "tanimoto_nn": tanimoto_scores,
        }
    )

    return df
