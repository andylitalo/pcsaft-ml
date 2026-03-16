"""PC-SAFT parameter prediction from SMILES.

A standalone pip-installable package for predicting PC-SAFT equation-of-state
parameters (m, sigma, epsilon_k) from molecular SMILES strings.

Example usage:

    >>> import pcsaft_predict
    >>> df = pcsaft_predict.predict("CCO")  # ethanol
    >>> print(df)
       smiles      m  sigma  epsilon_k  in_domain  tanimoto_nn
    0     CCO  2.123  3.456      234.5       True         0.85

    >>> # Predict with uncertainty
    >>> df = pcsaft_predict.predict_with_uncertainty(["CCO", "c1ccccc1"])
    >>> print(df[["smiles", "m", "m_std", "in_domain"]])
         smiles      m  m_std  in_domain
    0       CCO  2.123  0.045       True
    1  c1ccccc1  2.567  0.062       True

    >>> # Use GNN model (requires torch and torch_geometric)
    >>> df = pcsaft_predict.predict("C(F)=C(F)F", model="gnn")
    >>> print(df[["smiles", "m", "sigma", "epsilon_k"]])
            smiles      m  sigma  epsilon_k
    0  C(F)=C(F)F  2.345  3.567      245.6

Available functions:
- predict(smiles, model="rf"): Predict parameters (model can be "rf" or "gnn")
- predict_with_uncertainty(smiles, model="rf", n_forward=30): Predict with uncertainty
- list_models(): List available models
- load_model(name): Load a specific model
"""

from pcsaft_predict._registry import (
    list_models,
    load_model,
    predict,
    predict_with_uncertainty,
)

__all__ = ["predict", "predict_with_uncertainty", "list_models", "load_model"]
__version__ = "1.0.0"
