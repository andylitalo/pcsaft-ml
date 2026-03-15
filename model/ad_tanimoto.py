"""Tanimoto nearest-neighbor applicability domain."""
from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles


class TanimotoAD:
    """AD model based on max Tanimoto similarity to training set."""

    IN_DOMAIN_THRESHOLD = 0.4  # max similarity >= 0.4 → in domain
    WARNING_THRESHOLD = 0.3  # max similarity in [0.3, 0.4) → warning

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits
        self.train_fps_ = None

    def fit(self, smiles_list: list[str]):
        """Fit the AD model on training SMILES.

        Parameters
        ----------
        smiles_list : list[str]
            Training set SMILES strings.

        Returns
        -------
        self
            Fitted model.
        """
        self.train_fps_ = [
            AllChem.GetMorganFingerprintAsBitVect(
                MolFromSmiles(s), self.radius, nBits=self.n_bits
            )
            for s in smiles_list
            if MolFromSmiles(s) is not None
        ]
        return self

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
        if self.train_fps_ is None:
            raise ValueError("TanimotoAD model not fitted. Call .fit() first.")
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
