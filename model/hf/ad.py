"""ChemBERTa Applicability Domain (AD) check using Isolation Forest.

Determines whether new molecules fall within the training set's CLS embedding
distribution. Uses the fine-tuned ChemBERTa encoder's representation space
instead of descriptor-space features, providing a more architecturally
consistent AD boundary for transformer-based predictions.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

CHEMBERTA_DIR = Path(__file__).resolve().parent.parent / "saved" / "chemberta"


def fit_chemberta_ad(
    embeddings: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest on ChemBERTa CLS embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        CLS embedding matrix of shape ``(n_train, embedding_dim)``.
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
    ad_model.fit(embeddings)

    # Save model
    CHEMBERTA_DIR.mkdir(parents=True, exist_ok=True)
    ad_path = CHEMBERTA_DIR / "chemberta_ad.joblib"
    joblib.dump(ad_model, ad_path)
    logger.info("Saved ChemBERTa AD model to %s", ad_path)

    # Save metadata
    metadata = {
        "detector_type": "IsolationForest",
        "contamination": contamination,
        "embedding_dim": int(embeddings.shape[1]),
        "n_train_molecules": int(embeddings.shape[0]),
        "train_source": "esper",  # Will be updated by training script if needed
    }
    metadata_path = CHEMBERTA_DIR / "chemberta_ad_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Saved ChemBERTa AD metadata to %s", metadata_path)

    return ad_model


def load_chemberta_ad_model() -> IsolationForest:
    """Load a previously saved ChemBERTa Isolation Forest AD model.

    Returns
    -------
    IsolationForest
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If the saved model does not exist.
    """
    ad_path = CHEMBERTA_DIR / "chemberta_ad.joblib"
    if not ad_path.exists():
        raise FileNotFoundError(
            f"ChemBERTa AD model not found at {ad_path}. "
            "Train ChemBERTa first with: python -m model.hf.train_chemberta"
        )
    return joblib.load(ad_path)


def predict_chemberta_ad(
    ad_model: IsolationForest,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Check whether molecules are within the applicability domain.

    Parameters
    ----------
    ad_model : IsolationForest
        Fitted Isolation Forest model.
    embeddings : np.ndarray
        CLS embedding matrix of shape ``(n_samples, embedding_dim)``.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_samples,)``. ``True`` means in-domain,
        ``False`` means out-of-domain (OOD).
    """
    # IsolationForest.predict returns 1 for inliers, -1 for outliers
    labels = ad_model.predict(embeddings)
    return labels == 1
