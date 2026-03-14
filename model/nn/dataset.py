"""PyTorch Dataset for PC-SAFT parameter prediction."""

import numpy as np
import torch
from torch.utils.data import Dataset


class PCSAFTDataset(Dataset):
    """Dataset wrapping feature arrays and target dicts for PC-SAFT training.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    targets : dict[str, np.ndarray]
        Mapping from target name (``m``, ``sigma``, ``epsilon_k``) to 1D arrays.
    """

    def __init__(self, features: np.ndarray, targets: dict[str, np.ndarray]):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.X[idx], {k: v[idx] for k, v in self.y.items()}
