"""Model registry for unified evaluation of PC-SAFT prediction models.

Provides a decorator-based registry pattern so that any model can be
registered once and then loaded / predicted through a common interface.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from model.data.descriptors import build_features
from model.data.load import TARGETS
from model.gc_pcsaft import predict_gc_pcsaft
from model.nn.architecture import PCSAFTNet

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
        self._net: PCSAFTNet | None = None
        self._feature_scaler: StandardScaler | None = None
        self._target_scalers: dict[str, dict[str, float]] | None = None

    def load(self) -> None:
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
