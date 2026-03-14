# Step 7: Kubeflow Retrain Pipeline

## Purpose

Build a Kubeflow Pipelines DAG that implements the online learning loop: when enough new experimental data has been submitted through the API (Step 5), automatically validate the data, merge it with the existing training set, retrain the model, evaluate it against the current champion, and promote the challenger if it wins.

This is the "experimental validation loop" from the original `next_steps.md` -- but automated and running on Kubernetes instead of done by hand.

## What It Adds Over Step 6

| After Step 6 | After This Step |
|--------------|----------------|
| Static model, never updated after deployment | Model improves automatically as new data arrives |
| No data quality validation | Pipeline rejects bad submissions before they corrupt training |
| Manual retrain-and-redeploy cycle | Automated retrain → evaluate → promote workflow |
| Single model version | Champion/challenger model comparison with rollback safety |

## Skills Demonstrated

- **Kubeflow Pipelines SDK**: `@dsl.component`, `@dsl.pipeline`, compiling to YAML
- **MLOps patterns**: Champion/challenger evaluation, data versioning, model registry
- **Pipeline design**: Conditional execution, artifact passing, retry logic
- **Active learning concepts**: When to retrain, how to validate contributed data

## Architecture

```
                   ┌──────────────┐
  /submit-data ───▶│  Submissions  │
                   │     Store     │
                   └──────┬───────┘
                          │  trigger: N new points
                          ▼
               ┌─────────────────────┐
               │  Kubeflow Pipeline   │
               │                     │
               │  1. validate_data   │
               │         │           │
               │  2. merge_datasets  │
               │         │           │
               │  3. retrain_model   │
               │         │           │
               │  4. evaluate_model  │
               │         │           │
               │  5. compare_models  │──── challenger loses → stop
               │         │           │
               │  6. promote_model   │──── challenger wins → deploy
               └─────────────────────┘
```

## Implementation Guide

### 1. Project structure

```
pipeline/
    __init__.py
    components/
        validate.py
        merge.py
        retrain.py
        evaluate.py
        compare.py
        promote.py
    pipeline.py         # Pipeline definition
    trigger.py          # Checks if retrain is needed
```

### 2. Pipeline components

Each component is a self-contained Python function decorated with `@dsl.component`. Kubeflow runs each in its own container.

#### `pipeline/components/validate.py`

```python
from kfp import dsl

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "rdkit"],
)
def validate_data(
    submissions_path: dsl.Input[dsl.Dataset],
    validated_path: dsl.Output[dsl.Dataset],
    rejected_path: dsl.Output[dsl.Dataset],
) -> int:
    """Validate submitted PC-SAFT data.

    Checks:
    - SMILES parses to a valid molecule
    - m > 0 (typical range: 1–8)
    - sigma in [2.0, 6.0] Å
    - epsilon_k in [50, 600] K
    - No exact duplicates of existing training data
    """
    import pandas as pd
    from rdkit import Chem

    df = pd.read_csv(submissions_path.path)
    valid_mask = (
        df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        & df["m"].between(0.5, 10)
        & df["sigma"].between(2.0, 6.0)
        & df["epsilon_k"].between(50, 600)
    )

    df[valid_mask].to_csv(validated_path.path, index=False)
    df[~valid_mask].to_csv(rejected_path.path, index=False)

    return int(valid_mask.sum())
```

#### `pipeline/components/retrain.py`

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["torch", "pandas", "rdkit", "scikit-learn", "numpy"],
)
def retrain_model(
    merged_data: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Output[dsl.Model],
    metrics_artifact: dsl.Output[dsl.Metrics],
):
    """Retrain the PyTorch multi-task NN on the merged dataset."""
    import pandas as pd
    import torch
    # ... import model architecture, feature builder, trainer ...

    df = pd.read_csv(merged_data.path)
    # ... build features, split, train, evaluate ...

    torch.save(model.state_dict(), model_artifact.path)
    # ... log metrics to metrics_artifact ...
```

#### `pipeline/components/compare.py`

```python
@dsl.component(base_image="python:3.11-slim")
def compare_models(
    champion_metrics: dsl.Input[dsl.Metrics],
    challenger_metrics: dsl.Input[dsl.Metrics],
) -> bool:
    """Return True if challenger beats champion on aggregate R²."""
    # Average R² across all three targets
    champion_r2 = (
        champion_metrics.metadata["r2_m"]
        + champion_metrics.metadata["r2_sigma"]
        + champion_metrics.metadata["r2_epsilon_k"]
    ) / 3

    challenger_r2 = (
        challenger_metrics.metadata["r2_m"]
        + challenger_metrics.metadata["r2_sigma"]
        + challenger_metrics.metadata["r2_epsilon_k"]
    ) / 3

    # Require meaningful improvement (not just noise)
    return challenger_r2 > champion_r2 + 0.005
```

#### `pipeline/components/promote.py`

```python
@dsl.component(base_image="python:3.11-slim")
def promote_model(
    challenger_model: dsl.Input[dsl.Model],
    production_model_path: str,
):
    """Copy challenger model to the production serving path.

    The serving pod watches this path (or a K8s ConfigMap) and reloads.
    """
    import shutil
    shutil.copy(challenger_model.path, production_model_path)
    # Optionally: kubectl rollout restart deployment/pcsaft-api -n pcsaft
```

### 3. Pipeline definition (`pipeline/pipeline.py`)

```python
from kfp import dsl

@dsl.pipeline(
    name="PC-SAFT Retrain Pipeline",
    description="Validate new data, retrain model, promote if improved",
)
def pcsaft_retrain_pipeline(
    submissions_path: str,
    existing_data_path: str,
    champion_model_path: str,
    production_model_path: str,
):
    validate_task = validate_data(submissions_path=submissions_path)

    merge_task = merge_datasets(
        validated_data=validate_task.outputs["validated_path"],
        existing_data=existing_data_path,
    )

    retrain_task = retrain_model(merged_data=merge_task.outputs["merged_path"])

    compare_task = compare_models(
        champion_metrics=...,  # loaded from champion model
        challenger_metrics=retrain_task.outputs["metrics_artifact"],
    )

    with dsl.Condition(compare_task.output == True):
        promote_model(
            challenger_model=retrain_task.outputs["model_artifact"],
            production_model_path=production_model_path,
        )
```

### 4. Compile and upload

```bash
pip install kfp

# Compile to YAML
python -c "
from kfp import compiler
from pipeline.pipeline import pcsaft_retrain_pipeline
compiler.Compiler().compile(pcsaft_retrain_pipeline, 'pipeline.yaml')
"

# Upload to Kubeflow (if running locally via kind + Kubeflow standalone)
kfp pipeline upload -p 'PC-SAFT Retrain' pipeline.yaml
```

### 5. Trigger mechanism

Two options:

**Option A: Cron-based** (simpler)

A CronJob checks every hour if there are N or more new submissions. If so, create a pipeline run:

```python
# pipeline/trigger.py
MIN_NEW_SAMPLES = 10

def check_and_trigger():
    new_count = count_new_submissions()
    if new_count >= MIN_NEW_SAMPLES:
        client = kfp.Client()
        client.create_run_from_pipeline_package(
            "pipeline.yaml",
            arguments={...},
        )
```

**Option B: Event-driven** (more sophisticated)

The `/submit-data` endpoint publishes an event; a Kubernetes event watcher triggers the pipeline when the threshold is met.

Start with Option A.

### 6. Data versioning

Each pipeline run should snapshot the training data used:

```python
# In the merge component
merged_df.to_csv(f"data/versions/train_v{run_id}.csv", index=False)
```

This lets you trace which data produced which model, which is essential for debugging regressions.

### 7. Kubeflow on kind (local development)

```bash
# Install Kubeflow Pipelines standalone
export PIPELINE_VERSION=2.0.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

# Port-forward the UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
# Visit http://localhost:8080
```

## Evaluation & Success Criteria

### Pipeline checks

- [ ] Pipeline compiles to valid YAML without errors
- [ ] Each component runs independently in a test container
- [ ] Pipeline executes end-to-end on Kubeflow (even with synthetic data)
- [ ] Validation correctly rejects out-of-range submissions
- [ ] Champion/challenger comparison correctly promotes a better model
- [ ] Champion/challenger comparison correctly blocks a worse model

### Integration checks

- [ ] Submitted data from the API (Step 5) flows into the pipeline
- [ ] A promoted model is picked up by the serving pod (Step 6)
- [ ] Data versions are saved and traceable

### What success looks like

- You can describe the full cycle: "A researcher submits experimental data → it gets validated → when enough accumulates, the model retrains → if it's better, it goes live"
- The pipeline is visible in the Kubeflow UI with clear step status
- You can show a model that improved after incorporating new data

### When to move to Step 8

You're ready for Step 8 when:

1. The retrain pipeline runs end-to-end (at least with synthetic test data)
2. Model promotion updates the serving endpoint
3. You can explain every component and why it exists
4. The pipeline handles failure gracefully (a bad retrain doesn't corrupt the production model)
