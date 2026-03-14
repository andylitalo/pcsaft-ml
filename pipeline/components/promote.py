"""Promote challenger model to production."""

from kfp import dsl


def promote_model_logic(
    challenger_model: str,
    production_path: str,
) -> str:
    """Core logic for promoting model to production.

    Parameters
    ----------
    challenger_model : str
        Path to challenger model artifact.
    production_path : str
        Path to production model location.

    Returns
    -------
    str
        Confirmation message.
    """
    import shutil
    from pathlib import Path

    # Ensure production directory exists
    Path(production_path).parent.mkdir(parents=True, exist_ok=True)

    # Copy challenger to production path
    shutil.copy2(challenger_model, production_path)

    return f"Model promoted to {production_path}"


@dsl.component(
    base_image="python:3.11-slim",
)
def promote_model(
    challenger_model: str,
    production_path: str = "/models/production/pcsaft_rf.joblib",
) -> str:
    """Promote challenger model to production.

    Copies challenger model artifact to production serving path.
    In a real deployment, this would trigger a K8s rollout restart.

    Parameters
    ----------
    challenger_model : str
        Path to challenger model artifact to promote.
    production_path : str
        Production model path (default: /models/production/pcsaft_rf.joblib).

    Returns
    -------
    str
        Confirmation message.
    """
    return promote_model_logic(challenger_model, production_path)
