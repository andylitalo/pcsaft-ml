"""Applicability Domain (AD) check using Isolation Forest.

Determines whether new molecules fall within the training set's feature-space
distribution. Molecules flagged as out-of-domain may produce unreliable
predictions regardless of the model type (RF or NN).
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent.parent / "saved"


def train_ad_model(
    X_train: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest on the training feature matrix.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape ``(n_train, n_features)``.
    contamination : float
        Expected proportion of outliers in the training set.
    random_state : int
        Random seed.

    Returns
    -------
    IsolationForest
        Fitted Isolation Forest model.
    """
    ad_model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    ad_model.fit(X_train)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    ad_path = SAVED_DIR / "ad_model.joblib"
    joblib.dump(ad_model, ad_path)
    logger.info("Saved AD model to %s", ad_path)

    return ad_model


def load_ad_model() -> IsolationForest:
    """Load a previously saved Isolation Forest AD model.

    Returns
    -------
    IsolationForest
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If the saved model does not exist.
    """
    ad_path = SAVED_DIR / "ad_model.joblib"
    if not ad_path.exists():
        raise FileNotFoundError(
            f"AD model not found at {ad_path}. "
            "Train it first with: python -m model.train --model nn"
        )
    return joblib.load(ad_path)


def check_applicability_domain(X: np.ndarray) -> np.ndarray:
    """Check whether input molecules are within the applicability domain.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_samples,)``. ``True`` means in-domain,
        ``False`` means out-of-domain (OOD).
    """
    ad_model = load_ad_model()
    # IsolationForest.predict returns 1 for inliers, -1 for outliers
    labels = ad_model.predict(X)
    return labels == 1
