"""Uncertainty-weighted ensemble of PC-SAFT prediction models.

Combines predictions from multiple models using inverse-variance weighting:

    w_i = 1 / sigma_i^2
    y_hat_ensemble = sum(w_i * y_hat_i) / sum(w_i)
    sigma_ensemble = 1 / sqrt(sum(w_i))

Each constituent model is loaded via the model registry and must expose
``predict_with_uncertainty(smiles_list) -> dict[str, np.ndarray]`` returning
keys ``{target}`` and ``{target}_std`` for each target in TARGETS.
"""

from __future__ import annotations

import logging

import numpy as np

from model.data.load import TARGETS
from model.registry import get_model

logger = logging.getLogger(__name__)

# Minimum standard deviation to avoid division by zero in inverse-variance
_SIGMA_FLOOR = 1e-6


class WeightedEnsemble:
    """Uncertainty-weighted ensemble of PC-SAFT prediction models.

    Parameters
    ----------
    model_names : list[str] | None
        Registry names of constituent models.  Defaults to ``["rf", "gnn"]``.
        Models that fail to load are silently skipped with a warning.
    """

    def __init__(self, model_names: list[str] | None = None):
        names = model_names or ["rf", "gnn"]
        self.requested_names: list[str] = list(names)
        self.models: dict[str, object] = {}

        for name in names:
            try:
                m = get_model(name)
                m.load()
                self.models[name] = m
                logger.info("Ensemble: loaded model %r", name)
            except Exception:
                logger.warning(
                    "Ensemble: skipping model %r (failed to load)", name,
                    exc_info=True,
                )

        if not self.models:
            raise RuntimeError(
                f"No constituent models could be loaded from {names}"
            )
        logger.info(
            "Ensemble ready with %d model(s): %s",
            len(self.models),
            list(self.models.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        """Return inverse-variance-weighted mean predictions.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings to predict.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are target names (m, sigma, epsilon_k).
        """
        combined = self.predict_with_uncertainty(smiles_list)
        return {t: combined[t] for t in TARGETS}

    def predict_with_uncertainty(
        self, smiles_list: list[str],
    ) -> dict[str, np.ndarray]:
        """Return combined predictions and uncertainty estimates.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings to predict.

        Returns
        -------
        dict[str, np.ndarray]
            Keys ``{target}`` (mean prediction) and ``{target}_std``
            (combined standard deviation) for each target.
        """
        n = len(smiles_list)

        # Collect per-model predictions and uncertainties
        model_results: dict[str, dict[str, np.ndarray]] = {}
        for name, model in self.models.items():
            try:
                res = model.predict_with_uncertainty(smiles_list)
                model_results[name] = res
            except Exception:
                logger.warning(
                    "Ensemble: model %r failed during prediction; skipping",
                    name,
                    exc_info=True,
                )

        if not model_results:
            # All models failed -- return NaN
            result: dict[str, np.ndarray] = {}
            for t in TARGETS:
                result[t] = np.full(n, np.nan)
                result[f"{t}_std"] = np.full(n, np.nan)
            return result

        return self._combine(model_results, n)

    def predict_disagreement(
        self, smiles_list: list[str],
    ) -> dict[str, np.ndarray]:
        """Return max pairwise disagreement per molecule per target.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are target names; values are max |y_i - y_j| arrays.
        """
        n = len(smiles_list)
        model_results: dict[str, dict[str, np.ndarray]] = {}
        for name, model in self.models.items():
            try:
                res = model.predict_with_uncertainty(smiles_list)
                model_results[name] = res
            except Exception:
                pass

        disagreement: dict[str, np.ndarray] = {}
        for t in TARGETS:
            preds = []
            for name in model_results:
                if t in model_results[name]:
                    preds.append(model_results[name][t])
            if len(preds) < 2:
                disagreement[t] = np.zeros(n)
            else:
                # Max pairwise absolute difference
                max_diff = np.zeros(n)
                for i in range(len(preds)):
                    for j in range(i + 1, len(preds)):
                        diff = np.abs(preds[i] - preds[j])
                        # NaN-safe max
                        max_diff = np.fmax(max_diff, np.nan_to_num(diff, nan=0.0))
                disagreement[t] = max_diff
        return disagreement

    def get_per_model_weights(
        self, smiles_list: list[str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return the normalized inverse-variance weights per model per target.

        Useful for diagnostics: shows which model dominates for each molecule.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            Outer key = model name, inner key = target name,
            value = weight array of shape (n_molecules,).
        """
        n = len(smiles_list)
        model_results: dict[str, dict[str, np.ndarray]] = {}
        for name, model in self.models.items():
            try:
                res = model.predict_with_uncertainty(smiles_list)
                model_results[name] = res
            except Exception:
                pass

        weights_out: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in model_results
        }
        for t in TARGETS:
            # Collect sigmas
            sigmas: dict[str, np.ndarray] = {}
            for name in model_results:
                std_key = f"{t}_std"
                if std_key in model_results[name]:
                    sigma = model_results[name][std_key].copy()
                    sigma = np.where(np.isfinite(sigma), sigma, np.nan)
                    sigmas[name] = sigma

            # Compute inverse-variance weights
            inv_var: dict[str, np.ndarray] = {}
            for name, sigma in sigmas.items():
                clamped = np.maximum(sigma, _SIGMA_FLOOR)
                iv = 1.0 / (clamped ** 2)
                # Mask out NaN predictions
                pred = model_results[name].get(t, np.full(n, np.nan))
                valid = np.isfinite(pred) & np.isfinite(sigma)
                iv = np.where(valid, iv, 0.0)
                inv_var[name] = iv

            total_iv = sum(inv_var.values())
            total_iv = np.where(total_iv > 0, total_iv, np.nan)

            for name in inv_var:
                weights_out[name][t] = inv_var[name] / total_iv

        return weights_out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _combine(
        self,
        model_results: dict[str, dict[str, np.ndarray]],
        n: int,
    ) -> dict[str, np.ndarray]:
        """Inverse-variance-weighted combination of model predictions."""
        result: dict[str, np.ndarray] = {}

        for t in TARGETS:
            weighted_sum = np.zeros(n)
            weight_total = np.zeros(n)

            for name, res in model_results.items():
                pred = res.get(t)
                sigma = res.get(f"{t}_std")
                if pred is None or sigma is None:
                    continue

                # Clamp sigma to floor
                sigma = np.maximum(np.abs(sigma), _SIGMA_FLOOR)

                # Determine valid entries (finite prediction AND finite sigma)
                valid = np.isfinite(pred) & np.isfinite(sigma)

                # Inverse-variance weight
                w = np.zeros(n)
                w[valid] = 1.0 / (sigma[valid] ** 2)

                weighted_sum += np.where(valid, w * pred, 0.0)
                weight_total += w

            # Compute ensemble mean
            has_weight = weight_total > 0
            ensemble_mean = np.full(n, np.nan)
            ensemble_mean[has_weight] = (
                weighted_sum[has_weight] / weight_total[has_weight]
            )

            # Ensemble std: 1 / sqrt(sum(w_i))
            ensemble_std = np.full(n, np.nan)
            ensemble_std[has_weight] = 1.0 / np.sqrt(weight_total[has_weight])

            result[t] = ensemble_mean
            result[f"{t}_std"] = ensemble_std

        return result
