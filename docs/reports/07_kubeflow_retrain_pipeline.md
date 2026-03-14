# Results Report: Kubeflow Retrain Pipeline

## Overview

Implemented an automated PC-SAFT model retraining pipeline using Kubeflow Pipelines (KFP). The pipeline provides continuous model improvement by validating researcher-submitted data, merging it with existing training data, retraining the model, and promoting improved models to production through a champion/challenger comparison.

## Pipeline Architecture

The pipeline consists of 6 components organized in a sequential DAG with conditional promotion:

```
1. validate_data
   └─> 2. merge_datasets
       └─> 3. retrain_model
           └─> 4. evaluate_model
               └─> 5. compare_models
                   └─> 6. promote_model (conditional)
```

### Pipeline Flow

1. **Data Validation**: Incoming submissions are validated for SMILES validity and parameter ranges
2. **Dataset Merging**: Valid submissions are merged with existing training data using InChI-based deduplication
3. **Model Retraining**: A new Random Forest model (challenger) is trained on the merged dataset
4. **Model Evaluation**: The challenger is evaluated on the held-out test set
5. **Champion/Challenger Comparison**: Average R² across all three parameters is compared
6. **Conditional Promotion**: If challenger R² > champion R² + 0.005, promote to production

## Component Details

### 1. validate_data (`pipeline/components/validate.py`)

Validates submitted PC-SAFT data against quality criteria:

- **SMILES Validation**: Uses RDKit to verify molecular structure validity
- **Range Checks**:
  - `m` (segments): [0.5, 10]
  - `sigma` (Å): [2.0, 6.0]
  - `epsilon_k` (K): [50, 600]

**Outputs**:
- `valid_csv`: Validated submissions ready for training
- `rejected_csv`: Rejected submissions with rejection reasons
- Returns count of valid rows

### 2. merge_datasets (`pipeline/components/merge.py`)

Merges validated submissions with existing training data:

- **InChI Deduplication**: Uses canonical InChI identifiers to handle duplicate molecules with different SMILES representations
- **Versioned Snapshots**: Saves a timestamped snapshot to `data/versions/train_v{run_id}.csv` for reproducibility

**Outputs**:
- `merged_csv`: Deduplicated combined dataset
- `version_csv`: Versioned snapshot
- Returns total row count

### 3. retrain_model (`pipeline/components/retrain.py`)

Trains a new Random Forest model on merged data:

- **Feature Engineering**: Combines Morgan fingerprints (2048-bit) + cleaned RDKit descriptors
- **Multi-Target Training**: Trains one RF model per parameter (m, sigma, epsilon_k)
- **Train/Test Split**: 80/20 split with random_state=42 for reproducibility

**Outputs**:
- `model_artifact`: Joblib file containing dict of 3 RF models
- `metrics_artifact`: JSON with training metrics (R², sample counts)
- Returns average test R²

### 4. evaluate_model (`pipeline/components/evaluate.py`)

Evaluates the retrained model on held-out test set:

- **Metrics**: Computes MAE, RMSE, R² for each parameter
- **Test Set**: Uses champion model's test set for fair comparison

**Outputs**:
- `eval_metrics_artifact`: JSON with evaluation metrics
- Returns average R² across all three parameters

### 5. compare_models (`pipeline/components/compare.py`)

Implements champion/challenger comparison logic:

- **Metric**: Average R² across m, sigma, epsilon_k
- **Improvement Threshold**: Requires improvement > 0.005 (not just noise)
- **Decision**: Returns boolean for promotion

**Logic**:
```python
challenger_avg_r2 - champion_avg_r2 > 0.005
```

### 6. promote_model (`pipeline/components/promote.py`)

Promotes challenger to production (conditional):

- **Copy Artifact**: Copies model to production path
- **Integration Point**: In real deployment, triggers K8s rollout restart to load new model

## Champion/Challenger Logic

The pipeline uses a champion/challenger pattern to ensure model quality:

1. **Champion**: Current production model (baseline)
2. **Challenger**: Newly retrained model on expanded dataset
3. **Comparison Metric**: Average R² across all three PC-SAFT parameters
4. **Promotion Criteria**: Challenger must improve by at least 0.005 R² points

### Why 0.005 Threshold?

This threshold balances:
- **Noise Tolerance**: Avoids promoting models with marginal, statistically insignificant improvements
- **Continuous Improvement**: Allows meaningful improvements (e.g., 0.30 → 0.31 R²) to be deployed
- **Stability**: Prevents frequent model churn from random variations

## Integration Points

### With FastAPI (Step 05)

The pipeline consumes data from the `/submit-data` endpoint:

```python
# serving/app.py
@app.post("/submit-data")
async def submit_data(submission: PCSAFTSubmission):
    # Append to serving/submissions.csv
    # Pipeline reads this file via `submissions_csv` parameter
```

### With Kubernetes (Step 06)

The pipeline integrates with K8s in two ways:

1. **Model Artifacts**: Writes to persistent volume mounted at `/models/production/`
2. **Deployment Rollout**: In production, `promote_model` would trigger:
   ```bash
   kubectl rollout restart deployment/pcsaft-api -n pcsaft
   ```

### Trigger Mechanism (`pipeline/trigger.py`)

Cron-based trigger with submission count check:

- **Check Interval**: Runs on schedule (e.g., daily via K8s CronJob)
- **Threshold**: Only triggers if >= 10 new submissions exist
- **KFP Client**: Uses `kfp.Client` to create pipeline run

```python
# Example cron configuration (k8s/cronjob.yaml):
# schedule: "0 2 * * *"  # 2 AM daily
```

## Compiled Pipeline

Pipeline compiles to `pipeline/pcsaft_retrain_pipeline.yaml` (22KB):

- **Format**: Kubeflow Pipelines IR YAML
- **Components**: 6 components (5 sequential + 1 conditional)
- **Container Images**: Uses `python:3.11-slim` base with explicit package versions
- **Reproducibility**: All components are self-contained with pinned dependencies

## Key Design Decisions

### 1. Self-Contained Components

Each KFP component is fully self-contained:
- All imports inside component function body
- Explicit package versions in `packages_to_install`
- No external module dependencies

**Rationale**: KFP components run in isolated containers. Self-containment ensures reproducibility and avoids dependency conflicts.

### 2. OutputPath vs Output[Dataset]

Components use `OutputPath(str)` instead of `Output[Dataset]`:

```python
def validate_data(
    input_csv: str,
    valid_csv: dsl.OutputPath(str),
    rejected_csv: dsl.OutputPath(str),
) -> int:
```

**Rationale**: Simpler type system, easier testing, and avoids KFP type compatibility issues with pipeline parameters.

### 3. Logic Extraction for Testing

Each component has a `*_logic()` function extracted from the KFP decorator:

```python
def validate_data_logic(input_csv: str, valid_csv: str, rejected_csv: str) -> int:
    # Core validation logic

@dsl.component(...)
def validate_data(input_csv: str, valid_csv: dsl.OutputPath(str), ...):
    return validate_data_logic(input_csv, valid_csv, rejected_csv)
```

**Rationale**: Enables unit testing of component logic without requiring KFP runtime.

### 4. Random Forest as Retrain Target

Pipeline retrains RF models, not NN or ChemBERTa:

**Rationale**:
- RF is the deployment model (Step 05, Step 06)
- Fast training (~seconds vs. minutes for NN)
- No GPU requirements
- Deterministic with fixed random_state

### 5. Champion Metrics Path as Parameter

Champion metrics are passed as a file path parameter, not loaded from persistent storage:

```python
def pcsaft_retrain_pipeline(
    champion_metrics_path: str = "model/saved/champion_metrics.json",
    ...
):
```

**Rationale**: Allows flexibility in champion definition (could point to different baseline models for A/B testing).

## Testing Strategy

Comprehensive test coverage in `tests/test_kubeflow.py`:

### Component Logic Tests
- `test_validate_data_logic`: SMILES validation and range checks
- `test_validate_data_all_valid`: All submissions pass validation
- `test_validate_data_range_checks`: Edge cases for parameter bounds
- `test_compare_models_logic_champion_wins`: Champion is better
- `test_compare_models_logic_challenger_wins`: Challenger is better
- `test_compare_models_logic_marginal_improvement`: Below threshold

### Pipeline Tests
- `test_pipeline_compiles`: Pipeline YAML exists and is valid
- `test_pipeline_has_all_components`: All 6 components present

### Trigger Tests
- `test_trigger_insufficient_samples`: < 10 samples, no trigger
- `test_trigger_sufficient_samples`: >= 10 samples, triggers pipeline
- `test_trigger_no_submissions_file`: Missing file handled gracefully

All 11 tests pass. Total test suite: 99 tests pass.

## Key Findings

### What Worked Well

1. **KFP Component Pattern**: Self-contained components with explicit dependencies worked flawlessly
2. **Logic Extraction**: Separating business logic from KFP decorators made testing straightforward
3. **Champion/Challenger Pattern**: Simple, effective model promotion logic
4. **Pipeline Compilation**: KFP compiler validated the pipeline DAG without requiring a running cluster

### Challenges Encountered

1. **Type System Complexity**: Initial attempt used `Input[Dataset]` / `Output[Model]` types, which caused incompatibilities with pipeline string parameters. Switching to `str` and `OutputPath(str)` resolved this.

2. **Placeholder API Changes**: `PIPELINE_RUN_ID_PLACEHOLDER` was renamed to `PIPELINE_JOB_ID_PLACEHOLDER` in newer KFP versions.

3. **Empty DataFrame Handling**: Pandas `read_csv` fails on empty files. Fixed by writing empty string when no rejections exist.

### Opportunities for Enhancement

1. **A/B Testing**: Pipeline could support parallel challenger training (e.g., RF vs. NN vs. ChemBERTa)
2. **Hyperparameter Tuning**: Retrain component could use GridSearchCV for auto-tuning
3. **Drift Detection**: Add component to detect dataset drift before retraining
4. **Model Registry**: Integrate with MLflow or KFServe for centralized artifact tracking
5. **Rollback Logic**: Automated rollback if new model performs poorly in production

## Figures

No figures for this step (infrastructure/orchestration).

## Deviations from Step Guide

None. All specified components and tests were implemented as described.

## Readiness Check

### Success Criteria

- [x] Pipeline compiles to valid YAML without errors
- [x] All 6 components implemented (validate, merge, retrain, evaluate, compare, promote)
- [x] Champion/challenger comparison correctly promotes better model
- [x] Champion/challenger comparison blocks worse model
- [x] Pipeline YAML file exists after compilation
- [x] Trigger logic checks submission count threshold
- [x] Tests cover component logic (validation, comparison, trigger)
- [x] Tests verify pipeline structure (components present, valid YAML)
- [x] All tests pass (11 KFP tests, 99 total tests)
- [x] Ruff code quality checks pass

### Integration Verification

- [x] Pipeline consumes FastAPI submissions CSV (`serving/submissions.csv`)
- [x] Pipeline writes to K8s production model path (`/models/production/pcsaft_rf.joblib`)
- [x] Trigger uses KFP Client API for pipeline execution
- [x] Components use versioned snapshots for reproducibility

## Next Steps

**Step 08: Streamlit Portal** — Build a researcher-facing web UI for:
- Submitting new PC-SAFT measurements
- Visualizing model predictions
- Viewing screening results
- Monitoring retrain pipeline status

The Kubeflow pipeline provides the automated infrastructure for continuous model improvement. Step 08 will provide the user interface for researchers to interact with the system.
