"""Applicability domain checking using Tanimoto nearest-neighbor."""

import logging

import joblib
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles

from pcsaft_predict._weights import check_model_exists, get_model_path

logger = logging.getLogger(__name__)


class TanimotoAD:
    """Applicability domain model based on max Tanimoto similarity to training set.

    Uses Morgan fingerprints (radius=2, 2048 bits) to compute Tanimoto similarity
    between query molecules and the training set. Molecules with max similarity
    >= 0.4 are considered in-domain.
    """

    IN_DOMAIN_THRESHOLD = 0.4  # max similarity >= 0.4 → in domain
    WARNING_THRESHOLD = 0.3  # max similarity in [0.3, 0.4) → warning

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits
        self.train_fps_ = None
        self._loaded = False

    def load(self):
        """Load precomputed AD model from disk."""
        if self._loaded:
            return

        ad_model_file = "tanimoto_ad.joblib"
        if not check_model_exists(ad_model_file):
            logger.warning(
                "AD model not found: %s. All molecules will be marked as in-domain.",
                get_model_path(ad_model_file),
            )
            self.train_fps_ = []
            self._loaded = True
            return

        # Load the AD model (should contain train_fps_ attribute)
        ad_model = joblib.load(get_model_path(ad_model_file))
        if hasattr(ad_model, "train_fps_"):
            self.train_fps_ = ad_model.train_fps_
        else:
            logger.warning("AD model missing train_fps_; marking all as in-domain")
            self.train_fps_ = []

        self._loaded = True
        logger.info("Tanimoto AD model loaded with %d training fingerprints", len(self.train_fps_))

    def tanimoto_nn(self, smiles: str) -> float:
        """Return max Tanimoto similarity to any training molecule.

        Parameters
        ----------
        smiles : str
            Query SMILES string.

        Returns
        -------
        float
            Maximum Tanimoto similarity in range [0, 1].
        """
        if not self._loaded:
            self.load()

        if not self.train_fps_:
            # No training FPs loaded; return 1.0 (in-domain)
            return 1.0

        mol = MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        sims = DataStructs.BulkTanimotoSimilarity(fp, self.train_fps_)
        return float(max(sims)) if sims else 0.0

    def in_domain(self, smiles: str) -> bool:
        """Check if molecule is in domain using IN_DOMAIN_THRESHOLD.

        Parameters
        ----------
        smiles : str
            Query SMILES string.

        Returns
        -------
        bool
            True if max Tanimoto similarity >= IN_DOMAIN_THRESHOLD.
        """
        return self.tanimoto_nn(smiles) >= self.IN_DOMAIN_THRESHOLD

    def batch_predict(self, smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predict AD status and Tanimoto NN similarity for a batch of SMILES.

        Parameters
        ----------
        smiles_list : list[str]
            List of SMILES strings.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (in_domain, tanimoto_nn) where:
            - in_domain: boolean array (True = in domain, False = OOD)
            - tanimoto_nn: float array of max Tanimoto similarities
        """
        if not self._loaded:
            self.load()

        tanimoto_scores = np.array([self.tanimoto_nn(smi) for smi in smiles_list])
        in_domain_flags = tanimoto_scores >= self.IN_DOMAIN_THRESHOLD

        return in_domain_flags, tanimoto_scores
