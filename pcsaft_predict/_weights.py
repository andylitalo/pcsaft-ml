"""Weight download and cache manager for PC-SAFT prediction models."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get the cache directory for model weights.

    Priority:
    1. PCSAFT_PREDICT_CACHE_DIR env var
    2. ~/.cache/pcsaft_predict/
    3. <repo_root>/model/saved/ (for development)

    Returns
    -------
    Path
        Cache directory path.
    """
    # Check environment variable
    if "PCSAFT_PREDICT_CACHE_DIR" in os.environ:
        cache_dir = Path(os.environ["PCSAFT_PREDICT_CACHE_DIR"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    # For development: check if we're in the repo
    pkg_dir = Path(__file__).parent
    repo_saved = pkg_dir.parent / "model" / "saved"
    if repo_saved.exists():
        return repo_saved

    # Fallback to user cache
    cache_dir = Path.home() / ".cache" / "pcsaft_predict"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_path(filename: str) -> Path:
    """Get the full path to a model file.

    Parameters
    ----------
    filename : str
        Model filename (e.g., "rf_m.joblib").

    Returns
    -------
    Path
        Full path to the model file.
    """
    cache_dir = get_cache_dir()
    return cache_dir / filename


def check_model_exists(filename: str) -> bool:
    """Check if a model file exists.

    Parameters
    ----------
    filename : str
        Model filename.

    Returns
    -------
    bool
        True if file exists, False otherwise.
    """
    return get_model_path(filename).exists()
