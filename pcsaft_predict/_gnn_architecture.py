"""Graph Isomorphism Network (GIN) for multi-task PC-SAFT regression.

Architecture:
    - GINEConv message-passing layers (edge-feature-aware GIN)
    - Batch normalization + ReLU between layers
    - Global mean + max pooling (concatenated)
    - Multi-task MLP head → (m, σ, ε/k)

References:
    Xu et al. (2019) "How Powerful are Graph Neural Networks?" (GIN)
    Hu et al. (2020) "Strategies for Pre-training Graph Neural Networks" (GIN+edge)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool


class PCSAFTGraphNet(nn.Module):
    """GIN-based GNN for predicting PC-SAFT parameters from molecular graphs.

    Parameters
    ----------
    node_dim : int
        Input node feature dimension.
    edge_dim : int
        Input edge feature dimension.
    hidden_dim : int
        Hidden dimension for GIN layers.
    num_layers : int
        Number of GIN message-passing layers.
    dropout : float
        Dropout rate applied after each GIN layer and in the MLP head.
    num_targets : int
        Number of regression targets (default 3: m, sigma, epsilon_k).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_targets: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_targets = num_targets

        # Initial projection to hidden dim
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # GIN message-passing layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Readout head: global_mean + global_max → 2 * hidden_dim
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Node features (num_nodes, node_dim).
        edge_index : Tensor
            Edge connectivity (2, num_edges).
        edge_attr : Tensor
            Edge features (num_edges, edge_dim).
        batch : Tensor
            Batch assignment vector (num_nodes,).

        Returns
        -------
        Tensor
            Predictions of shape (batch_size, num_targets).
        """
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, e)
            h_new = self.bns[i](h_new)
            h_new = functional.relu(h_new)
            h_new = functional.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual connection

        # Graph-level readout
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        graph_repr = torch.cat([h_mean, h_max], dim=-1)

        return self.head(graph_repr)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        n_forward: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout uncertainty estimation.

        Returns
        -------
        tuple[Tensor, Tensor]
            (mean_predictions, std_predictions) each of shape (batch_size, num_targets).
        """
        self.train()  # enable dropout
        preds = []
        for _ in range(n_forward):
            out = self.forward(x, edge_index, edge_attr, batch)
            preds.append(out)
        preds = torch.stack(preds, dim=0)
        self.eval()
        return preds.mean(dim=0), preds.std(dim=0)
