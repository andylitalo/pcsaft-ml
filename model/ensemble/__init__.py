"""Ensemble methods for PC-SAFT parameter prediction.

Combines multiple models (RF, GNN, optionally ChemBERTa) using
inverse-variance weighting to produce lower-variance predictions.
"""

from model.ensemble.weighted import WeightedEnsemble

__all__ = ["WeightedEnsemble"]
