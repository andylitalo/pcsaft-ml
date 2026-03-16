"""Graph Neural Network model wrapper for PC-SAFT parameter prediction."""

import logging

import numpy as np

from pcsaft_predict._weights import check_model_exists, get_model_path

logger = logging.getLogger(__name__)

TARGETS = ["m", "sigma", "epsilon_k"]


class GNNModel:
    """GIN-based Graph Neural Network wrapper for PC-SAFT prediction.

    Loads the GNN model checkpoint and provides prediction and uncertainty
    estimation through MC Dropout.
    """

    def __init__(self):
        self._net = None
        self._target_scalers = None
        self._loaded = False

    def load(self):
        """Load GNN model and target scalers."""
        if self._loaded:
            return

        # Lazy import torch and torch_geometric
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "GNN model requires torch and torch_geometric. "
                "Install with: pip install torch torch_geometric"
            ) from e

        try:
            from pcsaft_predict._gnn_architecture import PCSAFTGraphNet
        except ImportError as e:
            raise ImportError(
                "GNN model requires torch_geometric. "
                "Install with: pip install torch_geometric"
            ) from e

        model_file = "gnn_pcsaft.pt"
        if not check_model_exists(model_file):
            raise FileNotFoundError(
                f"GNN model not found: {get_model_path(model_file)}. "
                "Ensure model files are in the cache directory."
            )

        checkpoint = torch.load(get_model_path(model_file), map_location="cpu", weights_only=False)

        # Reconstruct model from config
        cfg = checkpoint["model_config"]
        self._net = PCSAFTGraphNet(
            node_dim=cfg["node_dim"],
            edge_dim=cfg["edge_dim"],
            hidden_dim=cfg.get("hidden_dim", 256),
            num_layers=cfg.get("num_layers", 4),
            dropout=cfg.get("dropout", 0.1),
            num_targets=cfg.get("num_targets", 3),
        )
        self._net.load_state_dict(checkpoint["model_state_dict"])
        self._net.eval()

        # Load target scalers for denormalization
        self._target_scalers = checkpoint["target_scalers"]

        self._loaded = True
        logger.info("GNN model loaded successfully")

    def _denormalize(self, target: str, values: np.ndarray) -> np.ndarray:
        """Denormalize predictions using target scalers."""
        info = self._target_scalers[target]
        return values * info["std"] + info["mean"]

    def _graphs_from_smiles(self, smiles_list: list[str]):
        """Convert SMILES to batched PyG data.

        Returns
        -------
        tuple
            (batch, valid_indices) where batch is a PyG Batch object or None,
            and valid_indices is a list of indices of valid SMILES.
        """
        from torch_geometric.data import Batch

        from pcsaft_predict._gnn_featurizer import smiles_to_graph

        graphs = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            g = smiles_to_graph(smi)
            if g is not None:
                graphs.append(g)
                valid_indices.append(i)
        if not graphs:
            return None, valid_indices
        batch = Batch.from_data_list(graphs)
        return batch, valid_indices

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        """Predict PC-SAFT parameters for a list of SMILES.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings to predict.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys: m, sigma, epsilon_k. Each value is a 1D array
            of predictions. Invalid SMILES get NaN predictions.
        """
        if not self._loaded:
            self.load()

        # Lazy import torch
        import torch

        batch, valid_indices = self._graphs_from_smiles(smiles_list)

        # Initialize results with NaN
        result = {}
        for target in TARGETS:
            result[target] = np.full(len(smiles_list), np.nan)

        if batch is None:
            return result

        self._net.eval()
        with torch.no_grad():
            preds = self._net(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            ).cpu().numpy()

        # Denormalize and assign to result array
        for i_target, target in enumerate(TARGETS):
            denormed = self._denormalize(target, preds[:, i_target])
            for j, idx in enumerate(valid_indices):
                result[target][idx] = denormed[j]

        return result

    def predict_with_uncertainty(
        self, smiles_list: list[str], n_forward: int = 30
    ) -> dict[str, np.ndarray]:
        """Predict PC-SAFT parameters with MC Dropout uncertainty.

        Uses dropout at inference time to estimate model uncertainty via
        Monte Carlo sampling.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings to predict.
        n_forward : int
            Number of MC Dropout forward passes (default: 30).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys: m, sigma, epsilon_k, m_std, sigma_std, epsilon_k_std.
            Invalid SMILES get NaN predictions and uncertainties.
        """
        if not self._loaded:
            self.load()

        batch, valid_indices = self._graphs_from_smiles(smiles_list)

        # Initialize results
        result = {}
        for target in TARGETS:
            result[target] = np.full(len(smiles_list), np.nan)
            result[f"{target}_std"] = np.full(len(smiles_list), np.nan)

        if batch is None:
            return result

        mean_pred, std_pred = self._net.predict_with_uncertainty(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            n_forward=n_forward,
        )
        mean_np = mean_pred.cpu().numpy()
        std_np = std_pred.cpu().numpy()

        # Denormalize and assign
        for i_target, target in enumerate(TARGETS):
            denormed_mean = self._denormalize(target, mean_np[:, i_target])
            info = self._target_scalers[target]
            denormed_std = std_np[:, i_target] * info["std"]
            for j, idx in enumerate(valid_indices):
                result[target][idx] = denormed_mean[j]
                result[f"{target}_std"][idx] = denormed_std[j]

        return result
