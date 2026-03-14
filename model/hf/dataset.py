"""HuggingFace-compatible Dataset for PC-SAFT SMILES regression."""

import logging

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class PCSAFTSmilesDataset(Dataset):
    """Dataset wrapping tokenized SMILES and PC-SAFT target values.

    Parameters
    ----------
    smiles : list[str]
        SMILES strings to tokenize.
    targets : torch.Tensor | None
        Target values of shape ``(n_samples, 3)`` for (m, sigma, epsilon_k).
        If None, the dataset is used for inference only.
    tokenizer : AutoTokenizer | None
        Pretrained tokenizer. If None, loads from *model_name*.
    model_name : str
        HuggingFace Hub model name (used only if *tokenizer* is None).
    max_length : int
        Maximum token sequence length.
    """

    def __init__(
        self,
        smiles: list[str],
        targets: torch.Tensor | None = None,
        tokenizer: AutoTokenizer | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 128,
    ):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.encodings = tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = targets  # (n_samples, 3) or None

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item
