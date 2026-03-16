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

    def find_similar(
        self,
        smiles: str | None = None,
        m: float | None = None,
        sigma: float | None = None,
        epsilon_k: float | None = None,
        k: int = 10,
        corpus: str = "all",
        metric: str = "parameter",
    ) -> dict:
        """Find similar molecules by PC-SAFT parameters or structural similarity.

        Parameters
        ----------
        smiles : str | None
            Query SMILES string (for Tanimoto or parameter prediction)
        m : float | None
            Query m parameter
        sigma : float | None
            Query sigma parameter
        epsilon_k : float | None
            Query epsilon_k parameter
        k : int
            Number of neighbors to return
        corpus : str
            Search corpus: 'reference', 'novel', or 'all'
        metric : str
            Distance metric: 'parameter', 'tanimoto', or 'both'

        Returns
        -------
        dict
            Similarity response with keys: query_smiles, query_params, corpus,
            corpus_version, neighbors, metric.
            On error, returns: {"error": str, "neighbors": []}.
        """
        # Local fallback mode
        if self._use_local:
            try:
                import pandas as pd
                from rdkit import Chem
                from rdkit.Chem import AllChem, DataStructs

                import pcsaft_predict

                logger.info("Using local similarity search (no API)")

                # Load search corpora
                from pathlib import Path

                predictions_csv = Path("data/pcsaft_novel_predictions_v1.csv")
                novel_db = pd.read_csv(predictions_csv, comment="#")
                ref_db = pd.DataFrame([
                    {
                        "smiles": "C1CCCC1",
                        "m": 2.3655,
                        "sigma": 3.7114,
                        "epsilon_k": 288.84,
                        "mol_class": "cycloalkane",
                        "name": "Cyclopentane",
                    }
                ])

                # Select corpus
                if corpus == "reference":
                    db = ref_db.copy()
                    corpus_version = "reference_molecules_v1"
                elif corpus == "novel":
                    db = novel_db.copy()
                    corpus_version = "pcsaft_novel_predictions_v1"
                else:
                    db = pd.concat([ref_db, novel_db], ignore_index=True)
                    corpus_version = "reference_molecules_v1+pcsaft_novel_predictions_v1"
                    corpus_labels = ["reference"] * len(ref_db) + ["novel"] * len(novel_db)
                    db["corpus"] = corpus_labels

                if "corpus" not in db.columns:
                    db["corpus"] = corpus

                # Resolve query parameters
                has_smiles = smiles is not None
                has_params = all([m is not None, sigma is not None, epsilon_k is not None])

                if has_smiles and not has_params:
                    pred_df = pcsaft_predict.predict_pcsaft([smiles])
                    query_params = {
                        "m": float(pred_df.iloc[0]["m"]),
                        "sigma": float(pred_df.iloc[0]["sigma"]),
                        "epsilon_k": float(pred_df.iloc[0]["epsilon_k"]),
                    }
                elif has_params:
                    query_params = {"m": m, "sigma": sigma, "epsilon_k": epsilon_k}
                else:
                    return {"error": "Must provide smiles or parameters", "neighbors": []}

                # Compute distances
                if metric in ("parameter", "both"):
                    q_m = query_params["m"]
                    q_s = query_params["sigma"]
                    q_e = query_params["epsilon_k"]
                    db["parameter_distance"] = (
                        ((db["m"] - q_m) / q_m) ** 2
                        + ((db["sigma"] - q_s) / q_s) ** 2
                        + ((db["epsilon_k"] - q_e) / q_e) ** 2
                    ) ** 0.5

                if metric in ("tanimoto", "both") and has_smiles:
                    query_mol = Chem.MolFromSmiles(smiles)
                    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

                    def tanimoto_sim(smi):
                        mol = Chem.MolFromSmiles(smi)
                        if mol is None:
                            return 0.0
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        return DataStructs.TanimotoSimilarity(query_fp, fp)

                    db["tanimoto_similarity"] = db["smiles"].apply(tanimoto_sim)

                # Sort
                if metric == "tanimoto":
                    results = db.sort_values("tanimoto_similarity", ascending=False)
                elif metric == "both":
                    results = db.sort_values(
                        ["parameter_distance", "tanimoto_similarity"],
                        ascending=[True, False],
                    )
                else:
                    results = db.sort_values("parameter_distance", ascending=True)

                top_k = results.head(k)

                neighbors = []
                for _, row in top_k.iterrows():
                    param_dist = (
                        float(row["parameter_distance"])
                        if "parameter_distance" in row
                        else None
                    )
                    tanimoto_sim = (
                        float(row["tanimoto_similarity"])
                        if "tanimoto_similarity" in row
                        else None
                    )
                    bp_k = (
                        float(row["boiling_point_K"])
                        if pd.notna(row.get("boiling_point_K"))
                        else None
                    )
                    neighbors.append({
                        "smiles": row["smiles"],
                        "corpus": row["corpus"],
                        "m": float(row["m"]),
                        "sigma": float(row["sigma"]),
                        "epsilon_k": float(row["epsilon_k"]),
                        "parameter_distance": param_dist,
                        "tanimoto_similarity": tanimoto_sim,
                        "mol_class": row.get("mol_class"),
                        "boiling_point_K": bp_k,
                    })

                return {
                    "query_smiles": smiles,
                    "query_params": query_params,
                    "corpus": corpus,
                    "corpus_version": corpus_version,
                    "neighbors": neighbors,
                    "metric": metric,
                }

            except Exception as e:
                logger.error("Local similarity search failed: %s", e)
                return {"error": f"Local search error: {e}", "neighbors": []}

        # API mode
        try:
            resp = requests.post(
                f"{self.base_url}/similar",
                json={
                    "smiles": smiles,
                    "m": m,
                    "sigma": sigma,
                    "epsilon_k": epsilon_k,
                    "k": k,
                    "corpus": corpus,
                    "metric": metric,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            logger.error("API connection failed: %s", self.base_url)
            return {"error": "Cannot connect to API server", "neighbors": []}
        except requests.HTTPError as e:
            logger.error("API error: %s", e)
            return {"error": f"API error: {e.response.status_code}", "neighbors": []}
        except requests.Timeout:
            logger.error("API timeout")
            return {"error": "API request timed out", "neighbors": []}
        except Exception as e:
            logger.error("Unexpected error during similarity search: %s", e)
            return {"error": str(e), "neighbors": []}
