"""Model registry for unified evaluation of PC-SAFT prediction models.

Provides a decorator-based registry pattern so that any model can be
registered once and then loaded / predicted through a common interface.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from model.data.descriptors import build_features
from model.data.load import TARGETS
from model.gc_pcsaft import predict_gc_pcsaft

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent / "saved"

MODELS: dict[str, type] = {}


def register_model(name: str):
    """Class decorator that registers a model wrapper under *name*."""

    def decorator(cls):
        MODELS[name] = cls
        return cls

    return decorator


def get_model(name: str):
    """Instantiate a registered model wrapper by name."""
    if name not in MODELS:
        raise KeyError(
            f"Unknown model {name!r}. Registered: {list(MODELS.keys())}"
        )
    return MODELS[name]()


def list_models() -> list[str]:
    """Return the names of all registered models."""
    return list(MODELS.keys())


# ---------------------------------------------------------------------------
# Helper: load feature config & compute features (shared by RF and NN)
# ---------------------------------------------------------------------------

def _load_feature_config() -> dict:
    config_path = SAVED_DIR / "feature_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {
        "features": "rdkit",
        "use_morgan": False,
        "use_rdkit": True,
        "morgan_radius": 2,
        "morgan_bits": 2048,
    }


def _get_rdkit_names_from_saved() -> list[str] | None:
    fnames_path = SAVED_DIR / "feature_names.joblib"
    if not fnames_path.exists():
        return None
    feature_names = joblib.load(fnames_path)
    rdkit_names = [n for n in feature_names if not n.startswith("morgan_")]
    return rdkit_names if rdkit_names else None


def _compute_features(smiles_list: list[str]) -> np.ndarray:
    """Compute features using the saved feature configuration."""
    config = _load_feature_config()
    use_rdkit = config.get("use_rdkit", True)
    rdkit_names = _get_rdkit_names_from_saved() if use_rdkit else None
    return build_features(
        smiles_list,
        use_morgan=config.get("use_morgan", False),
        use_rdkit=use_rdkit,
        morgan_radius=config.get("morgan_radius", 2),
        morgan_bits=config.get("morgan_bits", 2048),
        rdkit_names=rdkit_names,
    )


# ---------------------------------------------------------------------------
# GC-PC-SAFT wrapper (stateless)
# ---------------------------------------------------------------------------

@register_model("gc_pcsaft")
class GCPCSAFTModel:
    """Group-contribution PC-SAFT baseline (no saved artifacts needed)."""

    def load(self) -> None:
        """No-op: GC-PC-SAFT is stateless."""

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        df = predict_gc_pcsaft(smiles_list)
        return {t: df[t].values for t in TARGETS}

    def predict_with_uncertainty(
        self, smiles_list: list[str]
    ) -> dict[str, np.ndarray]:
        """GC-PC-SAFT has no intrinsic uncertainty; return zeros."""
        preds = self.predict(smiles_list)
        result = dict(preds)
        for t in TARGETS:
            result[f"{t}_std"] = np.zeros_like(preds[t])
        return result


# ---------------------------------------------------------------------------
# Random Forest wrapper
# ---------------------------------------------------------------------------

@register_model("rf")
class RFModel:
    """Random Forest wrapper (one model per target)."""

    def __init__(self):
        self._models: dict | None = None

    def load(self) -> None:
        self._models = {}
        for target in TARGETS:
            path = SAVED_DIR / f"rf_{target}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"RF model not found: {path}")
            self._models[target] = joblib.load(path)

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        if self._models is None:
            self.load()
        X = _compute_features(smiles_list)
        valid_mask = np.isfinite(X).all(axis=1)
        result: dict[str, np.ndarray] = {}
        for target in TARGETS:
            preds = np.full(len(smiles_list), np.nan)
            if valid_mask.any():
                preds[valid_mask] = self._models[target].predict(X[valid_mask])
            result[target] = preds
        return result

    def predict_with_uncertainty(
        self, smiles_list: list[str]
    ) -> dict[str, np.ndarray]:
        if self._models is None:
            self.load()
        X = _compute_features(smiles_list)
        valid_mask = np.isfinite(X).all(axis=1)
        result: dict[str, np.ndarray] = {}
        for target in TARGETS:
            preds = np.full(len(smiles_list), np.nan)
            stds = np.full(len(smiles_list), np.nan)
            if valid_mask.any():
                model = self._models[target]
                tree_preds = np.array(
                    [tree.predict(X[valid_mask]) for tree in model.estimators_]
                )
                preds[valid_mask] = tree_preds.mean(axis=0)
                stds[valid_mask] = tree_preds.std(axis=0)
            result[target] = preds
            result[f"{target}_std"] = stds
        return result


# ---------------------------------------------------------------------------
# Neural Network wrapper
# ---------------------------------------------------------------------------

@register_model("nn")
class NNModel:
    """PCSAFTNet wrapper with MC Dropout uncertainty."""

    def __init__(self):
        self._net = None
        self._feature_scaler: StandardScaler | None = None
        self._target_scalers: dict[str, dict[str, float]] | None = None

    def load(self) -> None:
        import torch
        from model.nn.architecture import PCSAFTNet

        path = SAVED_DIR / "nn_pcsaft.pt"
        if not path.exists():
            raise FileNotFoundError(f"NN checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        cfg = checkpoint["model_config"]
        self._net = PCSAFTNet(
            input_dim=cfg["input_dim"],
            trunk_dims=cfg.get("trunk_dims"),
            head_dims=cfg.get("head_dims"),
            dropout=cfg.get("dropout", 0.2),
        )
        self._net.load_state_dict(checkpoint["model_state_dict"])
        self._net.eval()

        # Feature scaler
        fs = checkpoint["feature_scaler"]
        self._feature_scaler = StandardScaler()
        self._feature_scaler.mean_ = np.array(fs["mean"])
        self._feature_scaler.scale_ = np.array(fs["std"])
        self._feature_scaler.var_ = self._feature_scaler.scale_ ** 2
        self._feature_scaler.n_features_in_ = len(fs["mean"])

        # Target scalers (dict of {mean, std} per target)
        self._target_scalers = checkpoint["target_scalers"]

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        return self._feature_scaler.transform(X).astype(np.float32)

    def _denormalize(self, target: str, values: np.ndarray) -> np.ndarray:
        info = self._target_scalers[target]
        return values * info["std"] + info["mean"]

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        import torch
        if self._net is None:
            self.load()
        X = _compute_features(smiles_list)
        valid_mask = np.isfinite(X).all(axis=1)
        result: dict[str, np.ndarray] = {}
        for t in TARGETS:
            result[t] = np.full(len(smiles_list), np.nan)

        if valid_mask.any():
            X_scaled = self._scale_features(X[valid_mask])
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            self._net.eval()
            with torch.no_grad():
                preds = self._net(x_tensor)
            for t in TARGETS:
                result[t][valid_mask] = self._denormalize(
                    t, preds[t].cpu().numpy()
                )
        return result

    def predict_with_uncertainty(
        self, smiles_list: list[str], n_forward: int = 30
    ) -> dict[str, np.ndarray]:
        if self._net is None:
            self.load()
        X = _compute_features(smiles_list)
        valid_mask = np.isfinite(X).all(axis=1)
        result: dict[str, np.ndarray] = {}
        for t in TARGETS:
            result[t] = np.full(len(smiles_list), np.nan)
            result[f"{t}_std"] = np.full(len(smiles_list), np.nan)

        if valid_mask.any():
            X_scaled = self._scale_features(X[valid_mask])
            # MC Dropout operates on the raw numpy array
            mc_result = self._net.predict_with_uncertainty(
                X_scaled, n_forward=n_forward
            )
            for t in TARGETS:
                result[t][valid_mask] = self._denormalize(t, mc_result[t])
                # Scale std by target std (denormalize spread)
                info = self._target_scalers[t]
                result[f"{t}_std"][valid_mask] = mc_result[f"{t}_std"] * info["std"]
        return result


# ---------------------------------------------------------------------------
# GNN wrapper
# ---------------------------------------------------------------------------

@register_model("gnn")
class GNNModel:
    """GIN-based Graph Neural Network with MC Dropout uncertainty."""

    def __init__(self):
        self._net = None
        self._target_scalers: dict[str, dict[str, float]] | None = None

    def load(self) -> None:
        import torch
        from model.gnn.architecture import PCSAFTGraphNet

        path = SAVED_DIR / "gnn_pcsaft.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"GNN checkpoint not found: {path}. "
                "Train with: python -m model.gnn.train_gnn"
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
        self._target_scalers = checkpoint["target_scalers"]

    def _denormalize(self, target: str, values: np.ndarray) -> np.ndarray:
        info = self._target_scalers[target]
        return values * info["std"] + info["mean"]

    def _graphs_from_smiles(self, smiles_list: list[str]):
        """Convert SMILES to batched PyG data."""
        from torch_geometric.data import Batch

        from model.gnn.graph_featurizer import smiles_to_graph

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
        import torch
        if self._net is None:
            self.load()

        batch, valid_indices = self._graphs_from_smiles(smiles_list)
        result: dict[str, np.ndarray] = {}
        for t in TARGETS:
            result[t] = np.full(len(smiles_list), np.nan)

        if batch is None:
            return result

        self._net.eval()
        with torch.no_grad():
            preds = self._net(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            ).cpu().numpy()

        for i_target, t in enumerate(TARGETS):
            denormed = self._denormalize(t, preds[:, i_target])
            for j, idx in enumerate(valid_indices):
                result[t][idx] = denormed[j]

        return result

    def predict_with_uncertainty(
        self, smiles_list: list[str], n_forward: int = 30
    ) -> dict[str, np.ndarray]:
        if self._net is None:
            self.load()

        batch, valid_indices = self._graphs_from_smiles(smiles_list)
        result: dict[str, np.ndarray] = {}
        for t in TARGETS:
            result[t] = np.full(len(smiles_list), np.nan)
            result[f"{t}_std"] = np.full(len(smiles_list), np.nan)

        if batch is None:
            return result

        mean_pred, std_pred = self._net.predict_with_uncertainty(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            n_forward=n_forward,
        )
        mean_np = mean_pred.cpu().numpy()
        std_np = std_pred.cpu().numpy()

        for i_target, t in enumerate(TARGETS):
            denormed_mean = self._denormalize(t, mean_np[:, i_target])
            info = self._target_scalers[t]
            denormed_std = std_np[:, i_target] * info["std"]
            for j, idx in enumerate(valid_indices):
                result[t][idx] = denormed_mean[j]
                result[f"{t}_std"][idx] = denormed_std[j]

        return result


# ---------------------------------------------------------------------------
# ChemBERTa wrapper
# ---------------------------------------------------------------------------

@register_model("chemberta")
class ChemBERTaModel:
    """ChemBERTa fine-tuned wrapper with MC Dropout uncertainty."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._target_scalers: dict[str, dict[str, float]] | None = None

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer

        from model.hf.chemberta_model import ChemBERTaForPCSAFT

        chemberta_dir = SAVED_DIR / "chemberta"

        # Load model weights
        weights_path = chemberta_dir / "chemberta_pcsaft.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"ChemBERTa weights not found: {weights_path}")

        self._model = ChemBERTaForPCSAFT()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(state_dict)
        self._model.eval()

        # Load tokenizer
        tokenizer_dir = chemberta_dir / "tokenizer"
        if tokenizer_dir.exists():
            self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "seyonec/ChemBERTa-zinc-base-v1"
            )

        # Load target scalers
        scaler_path = chemberta_dir / "target_scalers.json"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Target scalers not found: {scaler_path}")
        self._target_scalers = json.loads(scaler_path.read_text())

    def _tokenize(self, smiles_list: list[str]):
        return self._tokenizer(
            smiles_list,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    def _denormalize(self, target: str, values: np.ndarray) -> np.ndarray:
        info = self._target_scalers[target]
        return values * info["std"] + info["mean"]

    def predict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        import torch
        if self._model is None:
            self.load()
        encodings = self._tokenize(smiles_list)
        self._model.eval()
        with torch.no_grad():
            result = self._model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )
        preds = result["predictions"].cpu().numpy()  # (n, 3)

        output: dict[str, np.ndarray] = {}
        for i, target in enumerate(TARGETS):
            output[target] = self._denormalize(target, preds[:, i])
        return output

    def predict_with_uncertainty(
        self, smiles_list: list[str], n_forward: int = 30
    ) -> dict[str, np.ndarray]:
        if self._model is None:
            self.load()
        encodings = self._tokenize(smiles_list)
        raw = self._model.predict_with_uncertainty(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            n_forward=n_forward,
        )
        output: dict[str, np.ndarray] = {}
        for target in TARGETS:
            output[target] = self._denormalize(target, raw[target])
            info = self._target_scalers[target]
            output[f"{target}_std"] = raw[f"{target}_std"] * info["std"]
        return output

    def embed(self, smiles_list: list[str]) -> np.ndarray:
        """Return CLS embeddings for a list of SMILES.

        Parameters
        ----------
        smiles_list : list[str]
            List of SMILES strings.

        Returns
        -------
        np.ndarray
            CLS embeddings of shape ``(n_molecules, hidden_size)``.
        """
        import torch
        if self._model is None:
            self.load()
        encodings = self._tokenize(smiles_list)
        self._model.eval()
        with torch.no_grad():
            emb = self._model.encode(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )
        return emb.cpu().numpy()

    def predict_in_domain(self, smiles_list: list[str]) -> np.ndarray:
        """Return boolean array using ChemBERTa embedding-space AD.

        Parameters
        ----------
        smiles_list : list[str]
            List of SMILES strings.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_molecules,)``. ``True`` means
            in-domain, ``False`` means out-of-domain.
        """
        try:
            from model.hf.ad import load_chemberta_ad_model, predict_chemberta_ad

            ad_model = load_chemberta_ad_model()
            embeddings = self.embed(smiles_list)
            return predict_chemberta_ad(ad_model, embeddings)
        except FileNotFoundError:
            # Fallback: mark all as in-domain if AD artifact missing
            logger.warning("ChemBERTa AD model not found; marking all as in-domain")
            return np.ones(len(smiles_list), dtype=bool)


# ---------------------------------------------------------------------------
# Ensemble wrapper (lazy-loads constituent models)
# ---------------------------------------------------------------------------

@register_model("ensemble")
class EnsembleModel:
    """Uncertainty-weighted ensemble of multiple PC-SAFT models.

    Default constituents are RF + GNN (always available without heavy deps).
    ChemBERTa is included if ``transformers`` is installed and weights exist.
    """

    def __init__(self):
        self._ensemble = None

    def load(self, model_names=None):
        """Load constituent models via the WeightedEnsemble.

        Parameters
        ----------
        model_names : list[str] | None
            Registry names to include.  Defaults to ``["rf", "gnn"]``.
        """
        from model.ensemble.weighted import WeightedEnsemble

        names = model_names or ["rf", "gnn"]
        self._ensemble = WeightedEnsemble(names)

    def predict(self, smiles_list):
        """Return inverse-variance-weighted predictions."""
        if self._ensemble is None:
            self.load()
        return self._ensemble.predict(smiles_list)

    def predict_with_uncertainty(self, smiles_list):
        """Return combined predictions and uncertainty estimates."""
        if self._ensemble is None:
            self.load()
        return self._ensemble.predict_with_uncertainty(smiles_list)
