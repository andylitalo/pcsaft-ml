"""chemprop D-MPNN wrapper for PC-SAFT parameter prediction.

Uses chemprop v2.x with Lightning for training a Directed Message-Passing
Neural Network (D-MPNN) on molecular graphs.  Registered as ``chemprop``
in the model registry.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from model.registry import register_model

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved" / "chemprop"


def _smiles_to_mol(smi):
    """Convert SMILES to RDKit Mol, returning None on failure."""
    mol = Chem.MolFromSmiles(smi)
    return mol


@register_model("chemprop")
class ChempropPCSAFT:
    """D-MPNN (chemprop v2) for multi-target PC-SAFT prediction.

    Trains a message-passing neural network directly on molecular graphs,
    predicting m, sigma, and epsilon_k simultaneously.
    """

    def __init__(self):
        self.model_ = None
        self.trainer_ = None
        self.target_scalers_ = None  # {target: {"mean": float, "std": float}}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        smiles_train,
        y_train,
        smiles_val=None,
        y_val=None,
        max_epochs=20,
        batch_size=64,
        patience=5,
        hidden_dim=300,
        depth=3,
        lr=1e-3,
    ):
        """Train a chemprop D-MPNN on SMILES + targets.

        Parameters
        ----------
        smiles_train : list[str]
            Training SMILES.
        y_train : np.ndarray
            Target matrix of shape ``(n_samples, n_targets)``.
        smiles_val : list[str] | None
            Validation SMILES. If None, a 10% split is used.
        y_val : np.ndarray | None
            Validation targets.
        max_epochs : int
            Maximum training epochs.
        batch_size : int
            Batch size for training.
        patience : int
            Early stopping patience.
        hidden_dim : int
            Hidden dimension for message passing.
        depth : int
            Message passing depth.
        lr : float
            Maximum learning rate.

        Returns
        -------
        self
        """
        import lightning as L
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
        from chemprop.models import MPNN
        from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

        n_targets = y_train.shape[1]

        # Normalize targets
        self.target_scalers_ = {}
        y_train_norm = np.zeros_like(y_train, dtype=np.float32)
        from model.data.load import TARGETS
        for i, name in enumerate(TARGETS):
            col = y_train[:, i]
            mean_val = float(np.nanmean(col))
            std_val = float(np.nanstd(col))
            if std_val < 1e-9:
                std_val = 1.0
            self.target_scalers_[name] = {"mean": mean_val, "std": std_val}
            y_train_norm[:, i] = (col - mean_val) / std_val

        # Build training data, filtering invalid SMILES
        train_datapoints = []
        for smi, y in zip(smiles_train, y_train_norm):
            mol = _smiles_to_mol(smi)
            if mol is not None:
                train_datapoints.append(MoleculeDatapoint(mol, y=y.astype(np.float32)))

        if len(train_datapoints) < 10:
            raise ValueError(
                f"Only {len(train_datapoints)} valid training molecules. Need >= 10."
            )

        train_dataset = MoleculeDataset(train_datapoints)
        train_loader = build_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Validation data
        val_loader = None
        if smiles_val is not None and y_val is not None:
            y_val_norm = np.zeros_like(y_val, dtype=np.float32)
            for i, name in enumerate(TARGETS):
                sc = self.target_scalers_[name]
                y_val_norm[:, i] = (y_val[:, i] - sc["mean"]) / sc["std"]

            val_datapoints = []
            for smi, y in zip(smiles_val, y_val_norm):
                mol = _smiles_to_mol(smi)
                if mol is not None:
                    val_datapoints.append(MoleculeDatapoint(mol, y=y.astype(np.float32)))

            if len(val_datapoints) >= 2:
                val_dataset = MoleculeDataset(val_datapoints)
                val_loader = build_dataloader(
                    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
                )

        # Build model
        mp = BondMessagePassing(d_h=hidden_dim, depth=depth)
        agg = MeanAggregation()
        ffn = RegressionFFN(
            n_tasks=n_targets,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=1,
            dropout=0.1,
        )
        model = MPNN(mp, agg, ffn, max_lr=lr)

        # Callbacks
        callbacks = []
        if val_loader is not None:
            early_stop = L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, mode="min"
            )
            callbacks.append(early_stop)

        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator="cpu",
            enable_progress_bar=True,
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
        )

        if val_loader is not None:
            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_loader)

        self.model_ = model
        self.trainer_ = trainer
        logger.info(
            "chemprop training complete: %d train molecules, %d epochs max",
            len(train_datapoints),
            max_epochs,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, smiles_list):
        """Predict PC-SAFT parameters for a list of SMILES.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings.

        Returns
        -------
        np.ndarray
            Predictions of shape ``(n_samples, n_targets)`` with NaN for
            invalid SMILES.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

        from model.data.load import TARGETS

        n_targets = len(TARGETS)
        result = np.full((len(smiles_list), n_targets), np.nan, dtype=np.float32)

        # Build datapoints for valid SMILES
        valid_indices = []
        datapoints = []
        for i, smi in enumerate(smiles_list):
            mol = _smiles_to_mol(smi)
            if mol is not None:
                datapoints.append(MoleculeDatapoint(mol))
                valid_indices.append(i)

        if not datapoints:
            return result

        dataset = MoleculeDataset(datapoints)
        loader = build_dataloader(
            dataset, batch_size=64, shuffle=False, num_workers=0
        )

        self.model_.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                # batch is a TrainingBatch; extract the BatchMolGraph
                preds = self.model_(batch.bmg, None, None)
                all_preds.append(preds.cpu().numpy())

        if all_preds:
            all_preds = np.concatenate(all_preds, axis=0)  # (n_valid, n_targets)

            # Denormalize
            for i, name in enumerate(TARGETS):
                sc = self.target_scalers_[name]
                all_preds[:, i] = all_preds[:, i] * sc["std"] + sc["mean"]

            for j, idx in enumerate(valid_indices):
                result[idx] = all_preds[j]

        return result

    def predict_with_uncertainty(self, smiles_list, n_forward=10):
        """MC Dropout uncertainty estimate.

        Runs ``n_forward`` forward passes with dropout enabled.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean_predictions, std_predictions) each of shape
            ``(n_samples, n_targets)``.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

        from model.data.load import TARGETS

        n_targets = len(TARGETS)
        n = len(smiles_list)
        mean_result = np.full((n, n_targets), np.nan, dtype=np.float32)
        std_result = np.full((n, n_targets), np.nan, dtype=np.float32)

        valid_indices = []
        datapoints = []
        for i, smi in enumerate(smiles_list):
            mol = _smiles_to_mol(smi)
            if mol is not None:
                datapoints.append(MoleculeDatapoint(mol))
                valid_indices.append(i)

        if not datapoints:
            return mean_result, std_result

        dataset = MoleculeDataset(datapoints)
        loader = build_dataloader(
            dataset, batch_size=64, shuffle=False, num_workers=0
        )

        # Enable dropout for MC
        self.model_.train()
        all_runs = []
        with torch.no_grad():
            for _ in range(n_forward):
                run_preds = []
                for batch in loader:
                    # batch is a TrainingBatch; extract the BatchMolGraph
                    preds = self.model_(batch.bmg, None, None)
                    run_preds.append(preds.cpu().numpy())
                run_preds = np.concatenate(run_preds, axis=0)
                all_runs.append(run_preds)

        self.model_.eval()
        all_runs = np.array(all_runs)  # (n_forward, n_valid, n_targets)
        means = all_runs.mean(axis=0)
        stds = all_runs.std(axis=0)

        # Denormalize
        for i, name in enumerate(TARGETS):
            sc = self.target_scalers_[name]
            means[:, i] = means[:, i] * sc["std"] + sc["mean"]
            stds[:, i] = stds[:, i] * sc["std"]

        for j, idx in enumerate(valid_indices):
            mean_result[idx] = means[j]
            std_result[idx] = stds[j]

        return mean_result, std_result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path=None):
        """Save model checkpoint."""
        if path is None:
            SAVED_DIR.mkdir(parents=True, exist_ok=True)
            path = SAVED_DIR / "chemprop_model.pt"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model_.state_dict(),
            "target_scalers": self.target_scalers_,
        }
        torch.save(checkpoint, path)
        logger.info("chemprop model saved to %s", path)
        return path

    def load(self, path=None):
        """Load model checkpoint."""
        from chemprop.models import MPNN
        from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

        if path is None:
            path = SAVED_DIR / "chemprop_model.pt"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"chemprop model not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.target_scalers_ = checkpoint["target_scalers"]

        n_targets = len(self.target_scalers_)
        # Reconstruct model with default architecture
        mp = BondMessagePassing(d_h=300, depth=3)
        agg = MeanAggregation()
        ffn = RegressionFFN(
            n_tasks=n_targets,
            input_dim=300,
            hidden_dim=300,
            n_layers=1,
            dropout=0.1,
        )
        model = MPNN(mp, agg, ffn)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model_ = model
        logger.info("chemprop model loaded from %s", path)
        return self
