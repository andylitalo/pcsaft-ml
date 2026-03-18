"""Ridge regression baseline for PC-SAFT parameter prediction.

Trains one StandardScaler + Ridge pipeline per target with GridSearchCV
hyperparameter tuning. Registered as ``ridge`` in the model registry.
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.data.load import TARGETS
from model.registry import register_model

logger = logging.getLogger(__name__)

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import pickle

    class _JoblibCompat:
        @staticmethod
        def dump(obj, path):
            with Path(path).open("wb") as fh:
                pickle.dump(obj, fh)

        @staticmethod
        def load(path):
            with Path(path).open("rb") as fh:
                return pickle.load(fh)

    joblib = _JoblibCompat()

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved" / "linear"


@register_model("ridge")
class RidgePCSAFT:
    """Target-wise Ridge regressor for PC-SAFT parameters.

    Each target is tuned independently with
    ``Pipeline([StandardScaler(), Ridge()])`` and ``GridSearchCV``.
    """

    PARAM_GRID = {
        "regressor__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
    }

    def __init__(self, target_names=None):
        self.model_ = None
        self.best_params_ = None
        self.cv_best_scores_ = None
        self.feature_names_ = None
        self.target_names_ = list(target_names) if target_names is not None else list(TARGETS)

    def _build_pipeline(self):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge()),
        ])

    def _prepare_targets(self, y_train):
        """Normalize targets to a dict keyed by target name."""
        y_arr = np.asarray(y_train)
        if y_arr.ndim == 1:
            if len(self.target_names_) != 1:
                raise ValueError(
                    "1D targets require exactly one target name; "
                    f"got {self.target_names_!r}"
                )
            return {self.target_names_[0]: y_arr}

        if y_arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D targets, got shape {y_arr.shape}")

        if y_arr.shape[1] != len(self.target_names_):
            raise ValueError(
                "Number of target columns does not match target names: "
                f"{y_arr.shape[1]} vs {len(self.target_names_)}"
            )
        return {
            target_name: y_arr[:, idx]
            for idx, target_name in enumerate(self.target_names_)
        }

    def fit(self, X_train, y_train, feature_names=None, cv_folds=3, verbose=0):
        """Fit with GridSearchCV to select best hyperparameters.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y_train : np.ndarray
            Target matrix of shape ``(n_samples, n_targets)``.
        feature_names : list[str] | None
            Optional feature names for artifact tracking.
        cv_folds : int
            Number of CV folds (default 3 for speed).
        verbose : int
            Verbosity passed to GridSearchCV.

        Returns
        -------
        self
        """
        targets = self._prepare_targets(y_train)
        self.model_ = {}
        self.best_params_ = {}
        self.cv_best_scores_ = {}

        for target_name, y_target in targets.items():
            cv = GridSearchCV(
                self._build_pipeline(),
                self.PARAM_GRID,
                cv=cv_folds,
                scoring="r2",
                n_jobs=-1,
                refit=True,
                verbose=verbose,
            )
            cv.fit(X_train, y_target)
            self.model_[target_name] = cv.best_estimator_
            self.best_params_[target_name] = cv.best_params_
            self.cv_best_scores_[target_name] = float(cv.best_score_)
            logger.info(
                "Best Ridge params for %s: %s (CV R2=%.4f)",
                target_name,
                cv.best_params_,
                cv.best_score_,
            )

        self.feature_names_ = feature_names
        return self

    def predict(self, X):
        """Return predictions of shape ``(n_samples, n_targets)``."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")
        preds = [self.model_[target_name].predict(X) for target_name in self.target_names_]
        pred_arr = np.column_stack(preds)
        if pred_arr.shape[1] == 1:
            return pred_arr[:, 0]
        return pred_arr

    def save(self, path=None):
        """Save model to disk via joblib."""
        if path is None:
            SAVED_DIR.mkdir(parents=True, exist_ok=True)
            path = SAVED_DIR / "ridge_model.joblib"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model_,
                "best_params": self.best_params_,
                "cv_best_scores": self.cv_best_scores_,
                "feature_names": self.feature_names_,
                "target_names": self.target_names_,
            },
            path,
        )
        logger.info("Ridge model saved to %s", path)
        return path

    def load(self, path=None):
        """Load model from disk."""
        if path is None:
            path = SAVED_DIR / "ridge_model.joblib"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Ridge model not found: {path}")
        data = joblib.load(path)
        self.model_ = data["model"]
        self.best_params_ = data["best_params"]
        self.cv_best_scores_ = data.get("cv_best_scores", {})
        self.feature_names_ = data.get("feature_names")
        self.target_names_ = data.get("target_names", list(TARGETS))
        logger.info("Ridge model loaded from %s", path)
        return self
