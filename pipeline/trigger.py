"""Cron-based pipeline trigger with submission count check."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_NEW_SAMPLES = 10


def check_and_trigger(
    submissions_csv: str = "serving/submissions.csv",
    pipeline_yaml: str = "pipeline/pcsaft_retrain_pipeline.yaml",
    kfp_host: str = "http://localhost:8080",
    experiment_name: str = "pcsaft-retraining",
    min_samples: int = MIN_NEW_SAMPLES,
) -> str | None:
    """Check submission count and trigger pipeline if threshold met.

    Parameters
    ----------
    submissions_csv : str
        Path to submissions CSV file.
    pipeline_yaml : str
        Path to compiled pipeline YAML.
    kfp_host : str
        Kubeflow Pipelines API host.
    experiment_name : str
        KFP experiment name.
    min_samples : int
        Minimum new samples required to trigger pipeline.

    Returns
    -------
    str | None
        Run ID if pipeline was triggered, None otherwise.
    """
    import pandas as pd
    from kfp import Client

    # Check if submissions file exists and has enough samples
    submissions_path = Path(submissions_csv)
    if not submissions_path.exists():
        logger.info("No submissions file found at %s", submissions_csv)
        return None

    df = pd.read_csv(submissions_path)
    n_submissions = len(df)

    if n_submissions < min_samples:
        logger.info(
            "Insufficient submissions: %d < %d (threshold). Skipping retrain.",
            n_submissions,
            min_samples,
        )
        return None

    # Trigger pipeline
    logger.info("Triggering retrain pipeline with %d new samples", n_submissions)

    client = Client(host=kfp_host)

    # Create or get experiment
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
    except Exception:
        experiment = client.create_experiment(name=experiment_name)

    # Create pipeline run
    run = client.create_run_from_pipeline_package(
        pipeline_file=pipeline_yaml,
        experiment_id=experiment.experiment_id,
        arguments={
            "submissions_csv": submissions_csv,
        },
    )

    logger.info("Pipeline run created: %s", run.run_id)
    return run.run_id


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    check_and_trigger()
