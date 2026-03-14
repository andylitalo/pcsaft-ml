"""Fine-tuning script for ChemBERTa on PC-SAFT parameter regression.

Loads the Esper dataset, tokenizes SMILES, normalizes targets to zero mean /
unit variance, and fine-tunes ``seyonec/ChemBERTa-zinc-base-v1`` with a
regression head using the HuggingFace Trainer.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from model.data.load import TARGETS, load_data, split_data
from model.hf.chemberta_model import DEFAULT_MODEL_NAME, ChemBERTaForPCSAFT
from model.hf.dataset import PCSAFTSmilesDataset

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved"
CHEMBERTA_DIR = SAVED_DIR / "chemberta"


def _normalize_targets(
    train_df, test_df,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float]]]:
    """Normalize targets to zero mean / unit variance.

    Returns
    -------
    train_targets : torch.Tensor of shape (n_train, 3)
    test_targets : torch.Tensor of shape (n_test, 3)
    scaler_info : dict mapping target name to {"mean": float, "std": float}
    """
    scaler_info: dict[str, dict[str, float]] = {}
    train_cols = []
    test_cols = []

    for target in TARGETS:
        scaler = StandardScaler()
        train_vals = scaler.fit_transform(
            train_df[target].values.reshape(-1, 1)
        ).ravel()
        test_vals = scaler.transform(
            test_df[target].values.reshape(-1, 1)
        ).ravel()
        train_cols.append(train_vals)
        test_cols.append(test_vals)
        scaler_info[target] = {
            "mean": float(scaler.mean_[0]),
            "std": float(scaler.scale_[0]),
        }

    train_targets = torch.tensor(
        np.column_stack(train_cols), dtype=torch.float32
    )
    test_targets = torch.tensor(
        np.column_stack(test_cols), dtype=torch.float32
    )
    return train_targets, test_targets, scaler_info


def compute_metrics(eval_pred):
    """Compute per-target MAE and R2 for the Trainer."""
    predictions, labels = eval_pred
    metrics = {}
    for i, target in enumerate(TARGETS):
        pred_col = predictions[:, i]
        true_col = labels[:, i]
        metrics[f"mae_{target}"] = float(mean_absolute_error(true_col, pred_col))
        metrics[f"r2_{target}"] = float(r2_score(true_col, pred_col))
    return metrics


class ChemBERTaTrainer(Trainer):
    """Custom Trainer that calls model.forward with the right signature."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        result = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        loss = result["loss"]
        return (loss, result) if return_outputs else loss


def train_chemberta(
    source: str = "auto",
    stratify_bins: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
    num_train_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    patience: int = 5,
) -> None:
    """Fine-tune ChemBERTa for PC-SAFT parameter prediction.

    Parameters
    ----------
    source : str
        Data source: "esper", "fallback", "combined", or "auto".
    stratify_bins : int
        Number of epsilon_k bins for stratified splitting.
    model_name : str
        HuggingFace Hub pretrained model identifier.
    num_train_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Per-device training batch size.
    learning_rate : float
        Peak learning rate.
    warmup_ratio : float
        Fraction of total steps for linear warmup.
    weight_decay : float
        Weight decay for AdamW.
    patience : int
        Early stopping patience (evaluation epochs without improvement).
    """
    # --- Load data ---
    logger.info("Loading data (source=%s)...", source)
    df = load_data(source)
    logger.info("Loaded %d molecules", len(df))

    train_df, test_df = split_data(df, stratify_bins=stratify_bins)
    logger.info("Train: %d, Test: %d", len(train_df), len(test_df))

    # --- Normalize targets ---
    train_targets, test_targets, scaler_info = _normalize_targets(
        train_df, test_df
    )
    logger.info("Target scalers: %s", scaler_info)

    # --- Tokenize SMILES ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()

    train_dataset = PCSAFTSmilesDataset(
        train_smiles, targets=train_targets, tokenizer=tokenizer
    )
    eval_dataset = PCSAFTSmilesDataset(
        test_smiles, targets=test_targets, tokenizer=tokenizer
    )

    # --- Model ---
    model = ChemBERTaForPCSAFT(model_name=model_name)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d total, %d trainable", n_params, n_trainable)

    # --- Training arguments ---
    output_dir = str(CHEMBERTA_DIR)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        fp16=False,  # CPU-only
        save_total_limit=3,
        report_to="none",
    )

    # --- Trainer ---
    trainer = ChemBERTaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    logger.info("Starting ChemBERTa fine-tuning...")
    trainer.train()

    # --- Save artifacts ---
    CHEMBERTA_DIR.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = CHEMBERTA_DIR / "chemberta_pcsaft.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model weights to %s", model_path)

    # Save tokenizer
    tokenizer.save_pretrained(str(CHEMBERTA_DIR / "tokenizer"))
    logger.info("Saved tokenizer to %s", CHEMBERTA_DIR / "tokenizer")

    # Save scaler info for denormalization at inference
    scaler_path = CHEMBERTA_DIR / "target_scalers.json"
    scaler_path.write_text(json.dumps(scaler_info, indent=2) + "\n")
    logger.info("Saved target scalers to %s", scaler_path)

    # Save training history from trainer state
    log_history = trainer.state.log_history
    _save_training_history(log_history)

    # Save test set for evaluation
    test_df.to_csv(SAVED_DIR / "test_set.csv", index=False)
    logger.info("Saved test set (%d molecules)", len(test_df))

    logger.info("ChemBERTa training complete. Artifacts saved to %s", CHEMBERTA_DIR)


def _save_training_history(log_history: list[dict]) -> None:
    """Extract and save train/eval loss curves from Trainer log history."""
    train_losses = []
    eval_losses = []
    epochs = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            train_losses.append(
                {"epoch": entry["epoch"], "loss": entry["loss"]}
            )
        if "eval_loss" in entry and "epoch" in entry:
            eval_losses.append(
                {"epoch": entry["epoch"], "eval_loss": entry["eval_loss"]}
            )
            epochs.append(entry["epoch"])

    history = {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "epochs": epochs,
    }
    history_path = CHEMBERTA_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    logger.info("Saved training history to %s", history_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    train_chemberta()
