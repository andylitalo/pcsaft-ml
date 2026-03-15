"""Fit ChemBERTa applicability domain detector on existing trained model.

This script is a one-time utility to fit the AD detector for an already-trained
ChemBERTa model. Normally, AD fitting is part of the training pipeline.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from model.data.load import load_data, split_data
from model.hf.ad import fit_chemberta_ad
from model.hf.chemberta_model import ChemBERTaForPCSAFT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent / "model" / "saved"
CHEMBERTA_DIR = SAVED_DIR / "chemberta"


def main():
    # Load data
    logger.info("Loading data...")
    df = load_data("auto")
    train_df, _ = split_data(df, stratify_bins=5)
    logger.info("Training set: %d molecules", len(train_df))

    # Load trained model
    logger.info("Loading ChemBERTa model...")
    model = ChemBERTaForPCSAFT()
    weights_path = CHEMBERTA_DIR / "chemberta_pcsaft.pt"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Load tokenizer
    tokenizer_dir = CHEMBERTA_DIR / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Extract CLS embeddings from training set
    logger.info("Extracting CLS embeddings...")
    train_smiles = train_df["smiles"].tolist()
    batch_size = 64
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(train_smiles), batch_size):
            batch_smiles = train_smiles[i : i + batch_size]
            encodings = tokenizer(
                batch_smiles,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            cls_embeddings = model.encode(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )
            all_embeddings.append(cls_embeddings.cpu().numpy())

    train_embeddings = np.vstack(all_embeddings)
    logger.info(
        "Extracted %d CLS embeddings (dim=%d)",
        train_embeddings.shape[0],
        train_embeddings.shape[1],
    )

    # Fit AD model
    logger.info("Fitting applicability domain detector...")
    fit_chemberta_ad(train_embeddings, contamination=0.05)

    # Update metadata with correct source
    metadata_path = CHEMBERTA_DIR / "chemberta_ad_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["train_source"] = "esper"  # Update if using different source
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    # Save embeddings for visualization
    embeddings_path = CHEMBERTA_DIR / "train_cls_embeddings.npy"
    np.save(embeddings_path, train_embeddings)
    logger.info("Saved training embeddings to %s", embeddings_path)

    logger.info("ChemBERTa AD fitting complete!")


if __name__ == "__main__":
    main()
