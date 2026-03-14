"""Kubeflow Pipeline definition for PC-SAFT model retraining."""

from kfp import dsl

from pipeline.components.compare import compare_models
from pipeline.components.evaluate import evaluate_model
from pipeline.components.merge import merge_datasets
from pipeline.components.promote import promote_model
from pipeline.components.retrain import retrain_model
from pipeline.components.validate import validate_data


@dsl.pipeline(
    name="pcsaft-retrain-pipeline",
    description="Automated PC-SAFT model retraining with champion/challenger comparison",
)
def pcsaft_retrain_pipeline(
    submissions_csv: str = "serving/submissions.csv",
    existing_csv: str = "model/data/esper_pcsaft.csv",
    champion_test_csv: str = "model/saved/test_set.csv",
    champion_metrics_path: str = "model/saved/champion_metrics.json",
    production_model_path: str = "/models/production/pcsaft_rf.joblib",
    improvement_threshold: float = 0.005,
):
    """PC-SAFT model retraining pipeline.

    Pipeline DAG:
    1. validate_data: Validate submissions (SMILES, parameter ranges)
    2. merge_datasets: Merge validated data with existing training data
    3. retrain_model: Train new RF model (challenger)
    4. evaluate_model: Evaluate challenger on test set
    5. compare_models: Compare champion vs challenger metrics
    6. promote_model: (conditional) Promote challenger to production

    Parameters
    ----------
    submissions_csv : str
        Path to submitted PC-SAFT data CSV.
    existing_csv : str
        Path to existing training data CSV.
    champion_test_csv : str
        Path to champion model's test set.
    champion_metrics_path : str
        Path to champion model's metrics JSON.
    production_model_path : str
        Production model deployment path.
    improvement_threshold : float
        Minimum R² improvement for promotion (default 0.005).
    """
    # Step 1: Validate submissions
    validate_task = validate_data(input_csv=submissions_csv)

    # Step 2: Merge validated data with existing training data
    merge_task = merge_datasets(
        valid_csv=validate_task.outputs["valid_csv"],
        existing_csv=existing_csv,
        run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )

    # Step 3: Retrain model on merged data
    retrain_task = retrain_model(
        merged_csv=merge_task.outputs["merged_csv"],
    )

    # Step 4: Evaluate challenger model on test set
    evaluate_task = evaluate_model(
        model_artifact=retrain_task.outputs["model_artifact"],
        test_csv=champion_test_csv,
    )

    # Step 5: Compare champion vs challenger
    compare_task = compare_models(
        champion_metrics=champion_metrics_path,
        challenger_metrics=evaluate_task.outputs["eval_metrics_artifact"],
        improvement_threshold=improvement_threshold,
    )

    # Step 6: Conditionally promote challenger to production
    with dsl.If(compare_task.output == True, name="promote-if-better"):  # noqa: E712
        promote_model(
            challenger_model=retrain_task.outputs["model_artifact"],
            production_path=production_model_path,
        )


def compile_pipeline(output_path: str = "pipeline/pcsaft_retrain_pipeline.yaml"):
    """Compile the pipeline to YAML.

    Parameters
    ----------
    output_path : str
        Path to save compiled pipeline YAML.
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=pcsaft_retrain_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled to {output_path}")


if __name__ == "__main__":
    compile_pipeline()
