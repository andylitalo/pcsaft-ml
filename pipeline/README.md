# Kubeflow Pipeline for PC-SAFT Model Retraining

## Overview

This directory contains the Kubeflow Pipelines implementation for automated PC-SAFT model retraining with champion/challenger comparison.

## Directory Structure

```
pipeline/
├── components/           # KFP component definitions
│   ├── validate.py      # Validate submitted PC-SAFT data
│   ├── merge.py         # Merge validated data with existing training data
│   ├── retrain.py       # Retrain RF model on merged data
│   ├── evaluate.py      # Evaluate challenger model on test set
│   ├── compare.py       # Champion vs challenger comparison
│   └── promote.py       # Promote challenger to production
├── pipeline.py          # Pipeline definition and compilation
├── trigger.py           # Cron-based trigger with submission count check
└── pcsaft_retrain_pipeline.yaml  # Compiled pipeline (22KB)
```

## Pipeline DAG

```
validate_data
    ↓
merge_datasets
    ↓
retrain_model
    ↓
evaluate_model
    ↓
compare_models
    ↓
promote_model (conditional: if challenger R² > champion R² + 0.005)
```

## Usage

### Compile Pipeline

```bash
python -m pipeline.pipeline
```

This generates `pcsaft_retrain_pipeline.yaml`.

### Trigger Pipeline

#### Manual Trigger

```python
from kfp import Client

client = Client(host="http://localhost:8080")
experiment = client.create_experiment(name="pcsaft-retraining")

run = client.create_run_from_pipeline_package(
    pipeline_file="pipeline/pcsaft_retrain_pipeline.yaml",
    experiment_id=experiment.experiment_id,
    arguments={
        "submissions_csv": "serving/submissions.csv",
    },
)
```

#### Cron-based Trigger

```bash
python -m pipeline.trigger
```

This checks if `serving/submissions.csv` has >= 10 samples. If so, triggers the pipeline.

### Deploy as K8s CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: pcsaft-retrain-trigger
  namespace: pcsaft
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trigger
            image: python:3.11-slim
            command: ["python", "-m", "pipeline.trigger"]
            env:
            - name: KFP_HOST
              value: "http://ml-pipeline.kubeflow:8888"
```

## Component Details

### validate_data

**Inputs**:
- `input_csv`: Submitted PC-SAFT data CSV

**Outputs**:
- `valid_csv`: Validated submissions
- `rejected_csv`: Rejected submissions with reasons

**Validation Rules**:
- SMILES must be valid (RDKit)
- m ∈ [0.5, 10]
- sigma ∈ [2.0, 6.0]
- epsilon_k ∈ [50, 600]

### merge_datasets

**Inputs**:
- `valid_csv`: Validated submissions
- `existing_csv`: Current training data
- `run_id`: Pipeline run ID for versioning

**Outputs**:
- `merged_csv`: Deduplicated combined dataset
- `version_csv`: Versioned snapshot

**Deduplication**: Uses InChI (canonical molecular identifier) to handle duplicate molecules with different SMILES.

### retrain_model

**Inputs**:
- `merged_csv`: Combined training data

**Outputs**:
- `model_artifact`: Joblib file with dict of 3 RF models (m, sigma, epsilon_k)
- `metrics_artifact`: Training metrics JSON

**Features**: Morgan fingerprints (2048-bit) + cleaned RDKit descriptors

### evaluate_model

**Inputs**:
- `model_artifact`: Trained model
- `test_csv`: Test set

**Outputs**:
- `eval_metrics_artifact`: Evaluation metrics (MAE, RMSE, R²)

### compare_models

**Inputs**:
- `champion_metrics`: Current production model metrics
- `challenger_metrics`: New model metrics
- `improvement_threshold`: Minimum improvement required (default 0.005)

**Output**: Boolean (True if challenger should be promoted)

**Logic**: `avg_r2(challenger) - avg_r2(champion) > threshold`

### promote_model

**Inputs**:
- `challenger_model`: Model artifact to promote
- `production_path`: Production model location

**Action**: Copies model to production path. In real deployment, triggers K8s rollout restart.

## Testing

Run tests:

```bash
python -m pytest tests/test_kubeflow.py -v
```

11 tests covering:
- Component logic (validation, comparison)
- Pipeline structure (YAML validity, component presence)
- Trigger logic (threshold checks, KFP client mocking)

## Integration

### With FastAPI (Step 05)

Pipeline reads submissions from `serving/submissions.csv`, written by the `/submit-data` endpoint.

### With Kubernetes (Step 06)

Pipeline writes promoted models to `/models/production/pcsaft_rf.joblib`, mounted in the API deployment.

## Parameters

Pipeline accepts these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `submissions_csv` | `serving/submissions.csv` | Submitted data CSV |
| `existing_csv` | `model/data/esper_pcsaft.csv` | Current training data |
| `champion_test_csv` | `model/saved/test_set.csv` | Test set for evaluation |
| `champion_metrics_path` | `model/saved/champion_metrics.json` | Champion model metrics |
| `production_model_path` | `/models/production/pcsaft_rf.joblib` | Production model path |
| `improvement_threshold` | `0.005` | Minimum R² improvement for promotion |

## Design Decisions

### Why Random Forest?

- Fast training (~seconds)
- No GPU required
- Deterministic with fixed random_state
- Current deployment model (Step 05/06)

### Why 0.005 Improvement Threshold?

Balances:
- Noise tolerance (avoid promoting marginal improvements)
- Continuous improvement (allow meaningful gains)
- Stability (prevent frequent model churn)

### Why Self-Contained Components?

KFP components run in isolated containers. Self-containment (all imports inside function, explicit package versions) ensures reproducibility.

## Next Steps

See `docs/reports/07_kubeflow_retrain_pipeline.md` for full results report.
