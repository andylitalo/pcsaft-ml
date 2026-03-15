"""API client for communicating with the PC-SAFT prediction service."""

import logging

import requests

logger = logging.getLogger(__name__)


class PCSAFTClient:
    """Client for the PC-SAFT parameter prediction API.

    Parameters
    ----------
    base_url : str | None
        Base URL of the FastAPI service (e.g., http://localhost:8000).
        If None, falls back to direct pcsaft_predict library calls.
    """

    def __init__(self, base_url: str | None = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/") if base_url else None
        self._use_local = self.base_url is None

    def predict(self, smiles_list: list[str]) -> dict:
        """Predict PC-SAFT parameters for a batch of molecules.

        Parameters
        ----------
        smiles_list : list[str]
            List of SMILES strings to predict.

        Returns
        -------
        dict
            Prediction response with keys: predictions, model_name, model_version.
            On error, returns: {"error": str, "predictions": []}.
        """
        # Local fallback mode
        if self._use_local:
            try:
                from rdkit import Chem

                import pcsaft_predict

                logger.info("Using local pcsaft_predict library (no API)")

                df = pcsaft_predict.predict_with_uncertainty(smiles_list)

                # Convert DataFrame to API-compatible format
                predictions = []
                for _, row in df.iterrows():
                    mol = Chem.MolFromSmiles(row["smiles"])
                    is_valid = mol is not None

                    # Check for association sites (OH, NH, COOH)
                    is_associating = False
                    if mol:
                        smarts_patterns = ["[OH]", "[NH]", "[NH2]", "C(=O)[OH]"]
                        is_associating = any(
                            mol.HasSubstructMatch(Chem.MolFromSmarts(p))
                            for p in smarts_patterns
                        )

                    predictions.append({
                        "smiles": row["smiles"],
                        "m": float(row["m"]),
                        "sigma": float(row["sigma"]),
                        "epsilon_k": float(row["epsilon_k"]),
                        "valid": is_valid,
                        "in_domain": bool(row["in_domain"]),
                        "is_associating": is_associating,
                        "tanimoto_nn": float(row["tanimoto_nn"]),
                        "uncertainty": {
                            "m_std": float(row["m_std"]),
                            "sigma_std": float(row["sigma_std"]),
                            "epsilon_k_std": float(row["epsilon_k_std"]),
                        },
                    })

                return {
                    "predictions": predictions,
                    "model_name": "random_forest",
                    "model_version": "local-1.0.0",
                }
            except Exception as e:
                logger.error("Local prediction failed: %s", e)
                return {"error": f"Local prediction error: {e}", "predictions": []}

        # API mode
        try:
            resp = requests.post(
                f"{self.base_url}/predict",
                json={"smiles": smiles_list},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            logger.error("API connection failed: %s", self.base_url)
            return {"error": "Cannot connect to API server", "predictions": []}
        except requests.HTTPError as e:
            logger.error("API error: %s", e)
            return {"error": f"API error: {e.response.status_code}", "predictions": []}
        except requests.Timeout:
            logger.error("API timeout")
            return {"error": "API request timed out", "predictions": []}
        except Exception as e:
            logger.error("Unexpected error during prediction: %s", e)
            return {"error": str(e), "predictions": []}

    def submit_data(
        self, smiles: str, m: float, sigma: float, epsilon_k: float, source: str
    ) -> dict:
        """Submit experimental PC-SAFT data.

        Parameters
        ----------
        smiles : str
            Molecular SMILES string.
        m : float
            Segment number.
        sigma : float
            Segment diameter (Angstrom).
        epsilon_k : float
            Dispersion energy (K).
        source : str
            Publication DOI or lab identifier.

        Returns
        -------
        dict
            Submission response with key "status" (accepted/rejected).
            On error, returns: {"status": "error", "detail": str}.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/submit-data",
                json={
                    "smiles": smiles,
                    "m": m,
                    "sigma": sigma,
                    "epsilon_k": epsilon_k,
                    "source": source,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            logger.error("API connection failed: %s", self.base_url)
            return {"status": "error", "detail": "Cannot connect to API server"}
        except requests.HTTPError as e:
            logger.error("Submission error: %s", e)
            return {
                "status": "error",
                "detail": f"API rejected submission: {e.response.status_code}",
            }
        except requests.Timeout:
            logger.error("API timeout")
            return {"status": "error", "detail": "API request timed out"}
        except Exception as e:
            logger.error("Unexpected error during submission: %s", e)
            return {"status": "error", "detail": str(e)}

    def get_health(self) -> dict:
        """Check API health status.

        Returns
        -------
        dict
            Health response with keys: status, model_loaded, model_name, version.
            On error, returns: {"status": "unreachable"}.
        """
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.error("Health check failed for %s", self.base_url)
            return {"status": "unreachable", "model_loaded": False}

    def get_reference_molecules(self) -> list[dict]:
        """Get list of reference molecules with known PC-SAFT parameters.

        Returns
        -------
        list[dict]
            List of reference molecules with keys: name, smiles, m, sigma, epsilon_k, source.
            Returns empty list on error.
        """
        # Local fallback mode - hardcoded common references
        if self._use_local:
            return [
                {
                    "name": "Cyclopentane",
                    "smiles": "C1CCCC1",
                    "m": 2.3655,
                    "sigma": 3.7114,
                    "epsilon_k": 288.84,
                    "source": "Gross & Sadowski (2001)",
                },
                {
                    "name": "R-134a",
                    "smiles": "FC(F)C(F)F",
                    "m": 2.696,
                    "sigma": 3.174,
                    "epsilon_k": 201.7,
                    "source": "Tumakaka et al. (2002)",
                },
                {
                    "name": "Ethyl acetate",
                    "smiles": "CCOC(C)=O",
                    "m": 2.870,
                    "sigma": 3.307,
                    "epsilon_k": 230.8,
                    "source": "Gross & Sadowski (2001)",
                },
                {
                    "name": "Benzene",
                    "smiles": "c1ccccc1",
                    "m": 2.465,
                    "sigma": 3.648,
                    "epsilon_k": 287.4,
                    "source": "Gross & Sadowski (2001)",
                },
                {
                    "name": "Toluene",
                    "smiles": "Cc1ccccc1",
                    "m": 2.816,
                    "sigma": 3.717,
                    "epsilon_k": 285.7,
                    "source": "Gross & Sadowski (2001)",
                },
            ]

        # API mode
        try:
            resp = requests.get(f"{self.base_url}/reference-molecules", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.error("Failed to fetch reference molecules from %s", self.base_url)
            return []

    def get_model_info(self) -> dict:
        """Get model information and metrics.

        Currently reuses the /health endpoint. In a production system, this
        would hit a dedicated /model-info endpoint with R², MAE, training history.

        Returns
        -------
        dict
            Model info. Falls back to hardcoded baseline metrics if API is unreachable.
        """
        health = self.get_health()
        if health.get("status") == "unreachable":
            # Fallback to hardcoded RF baseline metrics
            logger.warning("API unreachable, using fallback metrics")
            return {
                "model_name": "random_forest",
                "model_loaded": False,
                "r2_m": 0.62,
                "r2_sigma": 0.35,
                "r2_epsilon_k": 0.33,
                "mae_m": 0.63,
                "mae_sigma": 0.19,
                "mae_epsilon_k": 26.1,
                "n_training": 1445,
                "status": "offline",
            }

        # For now, return health data + hardcoded metrics
        # Step 9+ would extend /health or add /model-info with real metrics
        return {
            "model_name": health.get("model_name", "unknown"),
            "model_loaded": health.get("model_loaded", False),
            "r2_m": 0.62,
            "r2_sigma": 0.35,
            "r2_epsilon_k": 0.33,
            "mae_m": 0.63,
            "mae_sigma": 0.19,
            "mae_epsilon_k": 26.1,
            "n_training": 1445,
            "status": health.get("status", "unknown"),
        }
