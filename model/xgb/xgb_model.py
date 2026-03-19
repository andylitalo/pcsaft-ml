"""XGBoost-based PC-SAFT parameter predictor.

Uses MultiOutputRegressor wrapping XGBRegressor with GridSearchCV for
hyperparameter tuning.  Registered as ``xgboost`` in the model registry.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from model.registry import register_model

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved" / "xgb"


@register_model("xgboost")
class XGBoostPCSAFT:
    """Multi-output XGBoost regressor for PC-SAFT parameters.

    Wraps ``sklearn.multioutput.MultiOutputRegressor(XGBRegressor(...))``
    with a reduced-size ``GridSearchCV`` grid (max 8 combinations) to keep
    wall-clock time reasonable on CPU.
    """

    # Reduced grid: 2 x 2 x 2 = 8 combinations
    PARAM_GRID = {
        "estimator__n_estimators": [200, 500],
        "estimator__max_depth": [4, 6],
        "estimator__learning_rate": [0.05, 0.1],
        "estimator__subsample": [0.8],
        "estimator__colsample_bytree": [0.8],
    }

    def __init__(self):
        self.model_ = None
        self.best_params_ = None
        self.feature_names_ = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, feature_names=None, cv_folds=3):
        """Fit with GridSearchCV to select best hyperparameters.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y_train : np.ndarray
            Target matrix of shape ``(n_samples, n_targets)``.
        feature_names : list[str] | None
            Optional feature names for interpretability.
        cv_folds : int
            Number of CV folds (default 3 for speed).

        Returns
        -------
        self
        """
        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        wrapped = MultiOutputRegressor(base)
        cv = GridSearchCV(
            wrapped,
            self.PARAM_GRID,
            cv=cv_folds,
            scoring="r2",
            n_jobs=1,
            refit=True,
            verbose=1,
        )
        cv.fit(X_train, y_train)
        self.model_ = cv.best_estimator_
        self.best_params_ = cv.best_params_
        self.feature_names_ = feature_names
        logger.info("Best XGBoost params: %s", self.best_params_)

        self._save_cv_results(cv)
        return self

    def _save_cv_results(self, cv: GridSearchCV) -> None:
        """Persist GridSearchCV results for downstream diagnostics."""
        cv_results = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in cv.cv_results_.items()
            if not k.startswith("param_")
        }
        cv_results["params"] = [
            {k: str(v) for k, v in p.items()} for p in cv.cv_results_["params"]
        ]
        cv_results["best_params"] = {
            k: str(v) for k, v in cv.best_params_.items()
        }
        cv_results["best_score"] = float(cv.best_score_)

        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        cv_path = SAVED_DIR / "cv_results.json"
        cv_path.write_text(json.dumps(cv_results, indent=2) + "\n")
        logger.info("XGBoost CV results saved to %s", cv_path)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        """Return predictions of shape ``(n_samples, n_targets)``."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")
        return self.model_.predict(X)

    def predict_with_uncertainty(self, X):
        """Per-target uncertainty via disagreement of sub-estimators.

        Each ``MultiOutputRegressor.estimators_`` is an XGBRegressor for one
        target.  We collect predictions from each and compute per-target std
        across the *internal* ensemble of boosting iterations by averaging
        XGBoost tree outputs.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean_predictions, std_predictions) each of shape
            ``(n_samples, n_targets)``.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        means = []
        stds = []
        for est in self.model_.estimators_:
            # Get individual tree predictions
            booster = est.get_booster()
            import xgboost as xgb

            dmat = xgb.DMatrix(X)
            # Predict with each tree iteration
            n_trees = est.n_estimators
            sample_trees = min(n_trees, 50)
            step = max(1, n_trees // sample_trees)
            tree_preds = []
            for t in range(0, n_trees, step):
                tree_preds.append(
                    booster.predict(dmat, iteration_range=(0, t + 1))
                )
            tree_preds = np.array(tree_preds)  # (n_sampled_trees, n_samples)
            means.append(tree_preds[-1])  # full ensemble prediction
            stds.append(tree_preds.std(axis=0))

        mean_arr = np.column_stack(means)  # (n_samples, n_targets)
        std_arr = np.column_stack(stds)
        return mean_arr, std_arr

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self, target_idx=None):
        """Return feature importances for a specific target or averaged.

        Parameters
        ----------
        target_idx : int | None
            Target index (0=m, 1=sigma, 2=epsilon_k). If None, average
            across all targets.

        Returns
        -------
        np.ndarray
            Feature importances of shape ``(n_features,)``.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        if target_idx is not None:
            return self.model_.estimators_[target_idx].feature_importances_
        importances = np.array(
            [est.feature_importances_ for est in self.model_.estimators_]
        )
        return importances.mean(axis=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path=None):
        """Save model to disk via joblib."""
        if path is None:
            SAVED_DIR.mkdir(parents=True, exist_ok=True)
            path = SAVED_DIR / "xgb_model.joblib"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model_,
                "best_params": self.best_params_,
                "feature_names": self.feature_names_,
            },
            path,
        )
        logger.info("XGBoost model saved to %s", path)
        return path

    def load(self, path=None):
        """Load model from disk."""
        if path is None:
            path = SAVED_DIR / "xgb_model.joblib"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {path}")
        data = joblib.load(path)
        self.model_ = data["model"]
        self.best_params_ = data["best_params"]
        self.feature_names_ = data.get("feature_names")
        logger.info("XGBoost model loaded from %s", path)
        return self
