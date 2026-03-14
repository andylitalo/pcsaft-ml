"""ChemBERTa model with regression head for PC-SAFT parameter prediction.

Uses the pretrained ``seyonec/ChemBERTa-zinc-base-v1`` encoder (RoBERTa-base
trained on ~77M SMILES from ZINC) with a two-layer MLP head that predicts
the three PC-SAFT parameters (m, sigma, epsilon_k).
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

TARGETS = ("m", "sigma", "epsilon_k")

DEFAULT_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class ChemBERTaForPCSAFT(nn.Module):
    """ChemBERTa encoder + regression head for PC-SAFT parameters.

    Architecture:
        - Encoder: pretrained RoBERTa-base (768 hidden dim)
        - Head: 768 -> 256 -> ReLU -> Dropout(0.1) -> 64 -> ReLU -> 3

    Parameters
    ----------
    model_name : str
        HuggingFace Hub model identifier for the pretrained encoder.
    dropout : float
        Dropout probability in the regression head.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for RoBERTa-base

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # m, sigma, epsilon_k
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encoder -> [CLS] pooling -> regression head.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor
            Attention mask of shape ``(batch, seq_len)``.
        labels : torch.Tensor | None
            Target values of shape ``(batch, 3)`` for computing MSE loss.

        Returns
        -------
        dict
            ``{"loss": ..., "predictions": ...}`` where predictions has
            shape ``(batch, 3)`` corresponding to (m, sigma, epsilon_k).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        predictions = self.regression_head(cls_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(predictions, labels)

        return {"loss": loss, "predictions": predictions}

    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_forward: int = 30,
    ) -> dict[str, np.ndarray]:
        """MC Dropout uncertainty estimation.

        Enables dropout at inference time, runs multiple stochastic forward
        passes, and returns per-target mean and standard deviation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor
            Attention mask of shape ``(batch, seq_len)``.
        n_forward : int
            Number of stochastic forward passes.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``m, sigma, epsilon_k, m_std, sigma_std, epsilon_k_std``.
        """
        self.train()  # Enable dropout
        all_preds = []

        with torch.no_grad():
            for _ in range(n_forward):
                result = self.forward(input_ids, attention_mask)
                all_preds.append(result["predictions"].cpu().numpy())

        self.eval()  # Restore eval mode

        stacked = np.stack(all_preds, axis=0)  # (n_forward, batch, 3)
        means = stacked.mean(axis=0)  # (batch, 3)
        stds = stacked.std(axis=0)  # (batch, 3)

        output: dict[str, np.ndarray] = {}
        for i, target in enumerate(TARGETS):
            output[target] = means[:, i]
            output[f"{target}_std"] = stds[:, i]

        return output
