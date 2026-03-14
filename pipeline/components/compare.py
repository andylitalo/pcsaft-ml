"""Compare champion and challenger models."""

from kfp import dsl


def compare_models_logic(
    champion_metrics: str,
    challenger_metrics: str,
    improvement_threshold: float = 0.005,
) -> bool:
    """Core logic for comparing champion and challenger models.

    Parameters
    ----------
    champion_metrics : str
        Path to champion model metrics JSON.
    challenger_metrics : str
        Path to challenger model metrics JSON.
    improvement_threshold : float
        Minimum R² improvement required to promote (default 0.005).

    Returns
    -------
    bool
        True if challenger should be promoted.
    """
    import json

    with open(champion_metrics) as f:
        champion = json.load(f)

    with open(challenger_metrics) as f:
        challenger = json.load(f)

    # Compute average R² across all targets
    champion_avg_r2 = sum(
        champion[t]["r2"] for t in ["m", "sigma", "epsilon_k"]
    ) / 3.0

    challenger_avg_r2 = sum(
        challenger[t]["r2"] for t in ["m", "sigma", "epsilon_k"]
    ) / 3.0

    improvement = challenger_avg_r2 - champion_avg_r2

    # Require improvement > threshold (not just noise)
    return improvement > improvement_threshold


@dsl.component(
    base_image="python:3.11-slim",
)
def compare_models(
    champion_metrics: str,
    challenger_metrics: str,
    improvement_threshold: float = 0.005,
) -> bool:
    """Compare champion and challenger models.

    Promotes challenger if average R² improvement > threshold.

    Parameters
    ----------
    champion_metrics : str
        Path to champion model evaluation metrics JSON.
    challenger_metrics : str
        Path to challenger model evaluation metrics JSON.
    improvement_threshold : float
        Minimum R² improvement to promote (default 0.005).

    Returns
    -------
    bool
        True if challenger should be promoted to production.
    """
    return compare_models_logic(
        champion_metrics,
        challenger_metrics,
        improvement_threshold,
    )
