"""SVR-based PC-SAFT parameter predictor.

Trains one StandardScaler + SVR pipeline per target with GridSearchCV
hyperparameter tuning. Registered as ``svm`` in the model registry.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved" / "svm"


@register_model("svm")
class SVMPCSAFT:
    """Target-wise SVR regressor for PC-SAFT parameters.

    Each target is tuned independently with
    ``Pipeline([StandardScaler(), SVR(kernel='rbf')])`` and ``GridSearchCV``.
    This keeps the `epsilon_k` benchmark target from being under-tuned by
    averaging across easier outputs.
    """

    PARAM_GRID = {
        "regressor__C": [1, 10, 100],
        "regressor__gamma": ["scale", "auto", 0.01, 0.001],
        "regressor__epsilon": [0.1, 0.01],
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
            ("regressor", SVR(kernel="rbf", max_iter=10000)),
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
        self.n_convergence_warnings_ = {}
        all_cv_results = {}

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
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                cv.fit(X_train, y_target)
            n_conv_warn = sum(
                1 for w in caught_warnings if issubclass(w.category, ConvergenceWarning)
            )
            self.n_convergence_warnings_[target_name] = n_conv_warn
            if n_conv_warn > 0:
                logger.warning(
                    "SVR %s: %d convergence warning(s) during GridSearchCV "
                    "(max_iter=10000; consider increasing if warnings persist)",
                    target_name,
                    n_conv_warn,
                )

            self.model_[target_name] = cv.best_estimator_
            self.best_params_[target_name] = cv.best_params_
            self.cv_best_scores_[target_name] = float(cv.best_score_)
            logger.info(
                "Best SVR params for %s: %s (CV R2=%.4f, convergence_warnings=%d)",
                target_name,
                cv.best_params_,
                cv.best_score_,
                n_conv_warn,
            )

            # Check for grid boundary hits and warn
            best = cv.best_params_
            grid = self.PARAM_GRID
            for param_key, grid_vals in grid.items():
                numeric_vals = [v for v in grid_vals if isinstance(v, (int, float))]
                if not numeric_vals:
                    continue
                best_val = best.get(param_key)
                if best_val == max(numeric_vals) or best_val == min(numeric_vals):
                    logger.warning(
                        "SVR %s: best %s=%s is at grid boundary %s — consider extending grid",
                        target_name,
                        param_key,
                        best_val,
                        numeric_vals,
                    )

            # Serialize cv_results_ for later use (Step 49 convergence diagnostics)
            cv_results_serializable = {}
            for k, v in cv.cv_results_.items():
                if k == "params":
                    cv_results_serializable[k] = [
                        {pk: str(pv) for pk, pv in p.items()} for p in v
                    ]
                elif hasattr(v, "tolist"):
                    cv_results_serializable[k] = v.tolist()
                else:
                    cv_results_serializable[k] = v
            cv_results_serializable["best_params"] = {
                k: str(v) for k, v in cv.best_params_.items()
            }
            cv_results_serializable["best_score"] = float(cv.best_score_)
            cv_results_serializable["n_convergence_warnings"] = n_conv_warn
            all_cv_results[target_name] = cv_results_serializable

        # Persist combined cv_results for Step 49 (convergence diagnostics)
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        cv_results_path = SAVED_DIR / "cv_results.json"
        cv_results_path.write_text(json.dumps(all_cv_results, indent=2) + "\n")
        logger.info("SVM CV results saved to %s", cv_results_path)

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
            path = SAVED_DIR / "svm_model.joblib"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model_,
                "best_params": self.best_params_,
                "cv_best_scores": self.cv_best_scores_,
                "n_convergence_warnings": getattr(self, "n_convergence_warnings_", {}),
                "feature_names": self.feature_names_,
                "target_names": self.target_names_,
            },
            path,
        )
        logger.info("SVM model saved to %s", path)
        return path

    def load(self, path=None):
        """Load model from disk."""
        if path is None:
            path = SAVED_DIR / "svm_model.joblib"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SVM model not found: {path}")
        data = joblib.load(path)
        self.model_ = data["model"]
        self.best_params_ = data["best_params"]
        self.cv_best_scores_ = data.get("cv_best_scores", {})
        self.n_convergence_warnings_ = data.get("n_convergence_warnings", {})
        self.feature_names_ = data.get("feature_names")
        self.target_names_ = data.get("target_names", list(TARGETS))
        logger.info("SVM model loaded from %s", path)
        return self
