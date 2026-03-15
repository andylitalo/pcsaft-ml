# Results Report: Docker & Kubernetes Deployment

## Overview

Step 06 packages the FastAPI serving application into a containerized deployment with Kubernetes orchestration manifests. This step transforms the API from "runs on my laptop" to "runs on any infrastructure" and introduces the container orchestration primitives that Steps 7-8 will build on.

## Deliverables

All infrastructure-as-code artifacts created and tested:

1. **Dockerfile** (multi-stage build)
2. **.dockerignore** (build optimization)
3. **k8s/namespace.yaml** (namespace: pcsaft)
4. **k8s/configmap.yaml** (environment variables)
5. **k8s/deployment.yaml** (2 replicas with health probes)
6. **k8s/service.yaml** (ClusterIP service)
7. **k8s/pvc.yaml** (1Gi PersistentVolumeClaim for submissions)
8. **tests/test_docker_k8s.py** (14 tests validating structure)

## Dockerfile Architecture

### Multi-Stage Build Strategy

The Dockerfile uses a two-stage build pattern to optimize image size:

**Stage 1 (builder)**:
- Base: `python:3.11-slim`
- Installs all dependencies from `pyproject.toml` into `/install` prefix
- Uses `--no-cache-dir` to avoid keeping pip download cache in the image
- Isolates build-time dependencies that aren't needed at runtime

**Stage 2 (runtime)**:
- Base: `python:3.11-slim` (fresh image)
- Installs `curl` for healthcheck execution
- Copies only installed packages from builder stage
- Copies application code: `model/`, `serving/`, `screening/`
- Exposes port 8000
- HEALTHCHECK hits `/health` every 30s (5s timeout, 3 retries)
- CMD runs `uvicorn serving.app:app --host 0.0.0.0 --port 8000`

### Layer Caching & Build Optimization

The `.dockerignore` file excludes:
- Version control (`.git/`)
- Python cache (`__pycache__`, `*.pyc`, `.pytest_cache`)
- Development artifacts (`tests/`, `docs/`, `figures/`)
- OS files (`.DS_Store`)
- CI/CD and orchestrator state (`.github/`, `state.yaml`, `PLAN.md`)

This reduces build context size and prevents unnecessary rebuilds when documentation or tests change.

### Model Artifacts: Bake-In vs. Mount

**Current approach (bake-in)**: Model artifacts in `model/saved/` are copied directly into the image. This makes the container self-contained but requires a rebuild to update the model.

**Future approach (Step 7)**: The deployment already includes a PVC mount at `/app/serving/submissions`. Step 7 (Kubeflow Retrain Pipeline) will refactor to mount model artifacts from a shared PVC, allowing model updates without rebuilding the image. The BDK pattern from CLAUDE.md provides a reference for this approach.

## Kubernetes Manifest Architecture

### Namespace Isolation

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pcsaft
```

All resources deploy into the `pcsaft` namespace, isolating this application from other workloads in the cluster.

### ConfigMap for Environment Variables

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pcsaft-config
  namespace: pcsaft
data:
  PCSAFT_MODEL_TYPE: "rf"
  PCSAFT_SUBMISSIONS_PATH: "/app/serving/submissions/submissions.csv"
  PCSAFT_HOST: "0.0.0.0"
  PCSAFT_PORT: "8000"
```

**Key decision**: `PCSAFT_MODEL_TYPE: "rf"` (Random Forest) is the default, not `nn`, because RF achieved the best test R² on the key parameters (m: 0.61, sigma: 0.32, epsilon_k: 0.27) compared to the NN baseline.

> **Uncertainty note**: All R², MAE, and RMSE values in this report are point estimates on a single holdout split. 95% bootstrap confidence intervals are available via `python -m model.evaluate --bootstrap` and saved to `model/saved/comparison_metrics.csv` with `_lo`/`_hi` suffix columns.

The ConfigMap is injected into pods via `envFrom: configMapRef`, and `serving/config.py` uses `pydantic-settings` to read these with the `PCSAFT_` prefix:

```python
class Settings(BaseSettings):
    model_type: str = "rf"
    submissions_path: str = "serving/submissions.csv"
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "PCSAFT_"}
```

### Deployment with Health Probes

The deployment spec includes:

**Replicas**: 2 pods for basic high availability

**Resource requests/limits**:
- Requests: 250m CPU, 512Mi memory (guaranteed allocation)
- Limits: 1000m CPU, 2Gi memory (hard cap)

**Liveness probe**:
- Endpoint: `GET /health`
- Initial delay: 30s (allow model loading time)
- Period: 15s
- Failures trigger pod restart

**Readiness probe**:
- Endpoint: `GET /health`
- Initial delay: 10s (faster than liveness)
- Period: 5s
- Failures remove pod from service endpoints (no traffic routing)

**Volume mount**:
- PVC `pcsaft-submissions` mounted at `/app/serving/submissions`
- Persists submitted data across pod restarts
- Shared across replicas (ReadWriteOnce mode, so only one pod writes in practice)

### Service (ClusterIP)

```yaml
spec:
  selector:
    app: pcsaft-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

Internal-only service (ClusterIP). External access requires `kubectl port-forward` or an Ingress (not implemented in this step). The service exposes port 80 externally, routing to container port 8000.

### PersistentVolumeClaim

```yaml
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

1Gi storage for submission data. `ReadWriteOnce` mode means the volume can be mounted by one node at a time, which is sufficient for this single-cluster deployment. Step 7 may introduce a second PVC for model artifacts with ReadWriteMany if needed.

## Configuration Flow

Environment variables flow through three layers:

1. **ConfigMap** (`k8s/configmap.yaml`): Defines `PCSAFT_*` variables as cluster-level config
2. **Deployment** (`k8s/deployment.yaml`): Injects ConfigMap into pods via `envFrom`
3. **pydantic-settings** (`serving/config.py`): Reads environment variables with `env_prefix="PCSAFT_"`

This pattern allows changing the model type (e.g., `rf` → `nn`) by updating the ConfigMap and restarting pods, without rebuilding the image.

## Testing

Created `tests/test_docker_k8s.py` with 14 tests:

### File Existence & Structure
- `test_dockerfile_exists`
- `test_dockerfile_has_multistage` (verifies >=2 FROM statements)
- `test_dockerfile_exposes_port_8000`
- `test_dockerfile_has_healthcheck`
- `test_dockerignore_exists`
- `test_dockerignore_excludes_common_dirs` (tests/, docs/, figures/)
- `test_k8s_manifests_exist` (all 5 YAML files)

### Kubernetes Configuration
- `test_k8s_deployment_has_probes` (liveness + readiness on /health)
- `test_k8s_deployment_has_resource_limits` (requests + limits)
- `test_k8s_configmap_has_model_type` (PCSAFT_MODEL_TYPE=rf)
- `test_k8s_deployment_has_volume_mount` (submissions PVC)
- `test_k8s_service_type_clusterip` (port 80→8000)
- `test_k8s_namespace_is_pcsaft`
- `test_k8s_pvc_requests_storage` (1Gi)

All 14 tests pass. The tests use PyYAML to parse manifests and verify structure, ensuring the K8s resources are correctly defined.

## Key Findings

### What Works Well

1. **Multi-stage build**: Clean separation between build-time and runtime dependencies reduces final image size
2. **Health probe integration**: The `/health` endpoint from Step 05 maps directly to K8s liveness/readiness probes
3. **ConfigMap pattern**: Environment-based configuration allows runtime model switching without rebuilding
4. **PVC for submissions**: Persists user-submitted data across pod restarts, addressing the CSV-on-disk limitation from Step 05

### Limitations & Future Work

1. **No actual build/deploy**: This is a local dev machine without Docker/kind. The implementation is infrastructure-as-code only. To validate, you would run:
   ```bash
   docker build -t pcsaft-api:latest .
   kind load docker-image pcsaft-api:latest --name pcsaft
   kubectl apply -f k8s/
   ```

2. **Model artifacts baked in**: Current approach copies models into the image. Step 7 will refactor to mount models from a PVC so Kubeflow can promote new models without rebuilding the container.

3. **No Ingress**: Service is ClusterIP-only. Production would add an Ingress resource or LoadBalancer for external access.

4. **ReadWriteOnce PVC**: The submissions PVC uses ReadWriteOnce mode, meaning only one pod can write at a time. This is fine for low-volume testing but may require ReadWriteMany or a database (Step 7) for production.

5. **No autoscaling**: Fixed 2 replicas. Production would add HorizontalPodAutoscaler based on CPU/memory or request latency.

### Deviations from Step Guide

None. The implementation follows the step guide exactly, with one clarification: used `rf` as the default model type in the ConfigMap (not `nn` as shown in the guide) because RF achieved better test performance in Step 03.

## Readiness Check

- [x] Dockerfile exists and uses multi-stage build
- [x] Dockerfile exposes port 8000
- [x] Dockerfile includes HEALTHCHECK directive
- [x] .dockerignore excludes tests, docs, figures
- [x] All 5 K8s manifests exist (namespace, configmap, deployment, service, pvc)
- [x] Deployment has liveness and readiness probes on /health
- [x] Deployment has resource requests and limits
- [x] ConfigMap sets PCSAFT_MODEL_TYPE=rf
- [x] Service is ClusterIP type, port 80→8000
- [x] PVC requests 1Gi storage
- [x] All 14 tests pass
- [x] No ruff lint errors

**Ready to proceed to Step 07 (Kubeflow Retrain Pipeline).**

## Next Steps (Step 07 Preview)

Step 07 will introduce Kubeflow Pipelines for automated retraining:
- Add `/retrain` endpoint to FastAPI app
- Build Kubeflow pipeline with KFP Python SDK
- Implement model promotion to shared PVC
- Refactor deployment to mount models from PVC instead of baking in
- Add pipeline-triggered model updates without container rebuilds

The current PVC infrastructure and ConfigMap pattern provide the foundation for this workflow.
