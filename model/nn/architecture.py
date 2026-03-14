"""PCSAFTNet: Multi-task neural network for PC-SAFT parameter prediction.

Shared trunk learns a common molecular representation, then branches into
task-specific heads for m (segments), sigma (segment diameter), and
epsilon_k (dispersion energy).
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TARGETS = ("m", "sigma", "epsilon_k")


class PCSAFTNet(nn.Module):
    """Multi-task neural network predicting three PC-SAFT parameters.

    Architecture:
        - Shared trunk: ``[input_dim] -> 512 -> 256 -> 128`` with
          BatchNorm + ReLU + Dropout after each layer.
        - Task-specific heads: one per target, each ``[128] -> 64 -> ReLU -> 1``.

    Parameters
    ----------
    input_dim : int
        Number of input features (Morgan FP + RDKit descriptors).
    trunk_dims : list[int]
        Hidden dimensions for the shared trunk layers.
    head_dims : list[int]
        Hidden dimensions for each task-specific head (before the final output).
    dropout : float
        Dropout probability used throughout the network.
    """

    def __init__(
        self,
        input_dim: int,
        trunk_dims: list[int] | None = None,
        head_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if trunk_dims is None:
            trunk_dims = [512, 256, 128]
        if head_dims is None:
            head_dims = [64]

        self.input_dim = input_dim
        self.trunk_dims = trunk_dims
        self.head_dims = head_dims
        self.dropout = dropout

        # Shared trunk
        trunk_layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in trunk_dims:
            trunk_layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Task-specific heads
        self.heads = nn.ModuleDict()
        for target in TARGETS:
            head_layers: list[nn.Module] = []
            hprev = prev_dim
            for hdim in head_dims:
                head_layers += [nn.Linear(hprev, hdim), nn.ReLU()]
                hprev = hdim
            head_layers.append(nn.Linear(hprev, 1))
            self.heads[target] = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through trunk and all task heads.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, input_dim)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from target name to predictions of shape ``(batch,)``.
        """
        shared = self.trunk(x)
        return {
            target: head(shared).squeeze(-1)
            for target, head in self.heads.items()
        }

    def predict_with_uncertainty(
        self,
        x: np.ndarray | torch.Tensor,
        n_forward: int = 30,
    ) -> dict[str, np.ndarray]:
        """MC Dropout uncertainty estimation.

        Enables dropout at inference time, runs ``n_forward`` stochastic
        forward passes, and returns per-target mean and standard deviation.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input features of shape ``(n_samples, input_dim)``.
        n_forward : int
            Number of stochastic forward passes.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``m, sigma, epsilon_k, m_std, sigma_std, epsilon_k_std``.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        device = next(self.parameters()).device
        x = x.to(device)

        # Enable dropout for MC sampling
        self.train()
        predictions: dict[str, list[np.ndarray]] = {t: [] for t in TARGETS}

        with torch.no_grad():
            for _ in range(n_forward):
                preds = self.forward(x)
                for target in TARGETS:
                    predictions[target].append(preds[target].cpu().numpy())

        # Restore eval mode
        self.eval()

        result: dict[str, np.ndarray] = {}
        for target in TARGETS:
            stacked = np.stack(predictions[target], axis=0)  # (n_forward, n_samples)
            result[target] = stacked.mean(axis=0)
            result[f"{target}_std"] = stacked.std(axis=0)

        return result

    def get_config(self) -> dict:
        """Return model configuration dict for serialization."""
        return {
            "input_dim": self.input_dim,
            "trunk_dims": self.trunk_dims,
            "head_dims": self.head_dims,
            "dropout": self.dropout,
        }
