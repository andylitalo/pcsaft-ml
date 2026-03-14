# Step 6: Docker & Kubernetes Deployment

## Purpose

Package the FastAPI serving application into a Docker container and deploy it to Kubernetes. This step takes the API from "runs on my laptop" to "runs on any infrastructure" and introduces the container orchestration primitives that Steps 7–8 build on.

## What It Adds Over Step 5

| After Step 5 | After This Step |
|--------------|----------------|
| Runs via `uvicorn` on your machine | Runs in a container on any Docker-compatible host |
| Dependencies installed in your venv | Dependencies frozen in a reproducible image |
| No resource limits or scaling | K8s manages CPU/memory limits, replicas, restarts |
| No health-based restarts | K8s liveness/readiness probes use `/health` endpoint |
| Model artifacts live on local disk | Model artifacts baked into image or mounted via PVC |

## Skills Demonstrated

- **Docker**: Multi-stage builds, layer caching, `.dockerignore`, image size optimization
- **Kubernetes manifests**: Deployment, Service, ConfigMap, PersistentVolumeClaim
- **Container orchestration**: Resource requests/limits, health probes, rolling updates
- **Infrastructure as code**: Declarative manifest files checked into version control

## Implementation Guide

### 1. Multi-stage Dockerfile

```dockerfile
# Stage 1: Build dependencies (including RDKit compilation if needed)
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY model/ model/
COPY serving/ serving/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**RDKit note**: The `rdkit` pip package includes prebuilt wheels for Linux, so `pip install rdkit` works in the slim image. If you hit issues, use `continuumio/miniconda3` as the base and install via conda.

### 2. `.dockerignore`

```
.git
.pytest_cache
__pycache__
*.egg-info
figures/
docs/
tests/
.DS_Store
*.pyc
```

### 3. Build and test locally

```bash
docker build -t pcsaft-api:latest .
docker run -p 8000:8000 pcsaft-api:latest

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["C1CCCC1"]}'
```

### 4. Kubernetes manifests

Create a `k8s/` directory at the project root:

```
k8s/
    namespace.yaml
    configmap.yaml
    deployment.yaml
    service.yaml
    pvc.yaml            # for submitted data persistence
```

#### `k8s/namespace.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pcsaft
```

#### `k8s/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pcsaft-config
  namespace: pcsaft
data:
  PCSAFT_MODEL_TYPE: "nn"
  PCSAFT_MODEL_PATH: "model/saved/nn_pcsaft.pt"
  PCSAFT_HOST: "0.0.0.0"
  PCSAFT_PORT: "8000"
```

#### `k8s/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pcsaft-api
  namespace: pcsaft
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pcsaft-api
  template:
    metadata:
      labels:
        app: pcsaft-api
    spec:
      containers:
        - name: pcsaft-api
          image: pcsaft-api:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: pcsaft-config
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          volumeMounts:
            - name: submissions
              mountPath: /app/serving/submissions
      volumes:
        - name: submissions
          persistentVolumeClaim:
            claimName: pcsaft-submissions
```

#### `k8s/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: pcsaft-api
  namespace: pcsaft
spec:
  selector:
    app: pcsaft-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

#### `k8s/pvc.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pcsaft-submissions
  namespace: pcsaft
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

### 5. Deploy to a local kind cluster

```bash
# Create cluster (if not already running)
kind create cluster --name pcsaft

# Load image into kind (no registry push needed)
kind load docker-image pcsaft-api:latest --name pcsaft

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# Verify
kubectl get pods -n pcsaft
kubectl logs -n pcsaft -l app=pcsaft-api

# Port-forward for local testing
kubectl port-forward -n pcsaft svc/pcsaft-api 8000:80
curl http://localhost:8000/health
```

### 6. Image size optimization

Target: keep the image under 1.5 GB. Key strategies:

- Multi-stage build (don't include build tools in runtime image)
- `--no-cache-dir` on pip installs
- Don't copy test files, docs, or figures
- Consider `python:3.11-slim` over full `python:3.11`

### 7. Model artifacts: bake in vs. mount

**Bake in** (copy model files into the image):
- Simpler, self-contained
- Requires rebuild to update model
- Good for stable models

**Mount via PVC** (model files on a persistent volume):
- Model can be updated without rebuilding the image
- Required for Step 7 (Kubeflow promotes new models to a shared volume)
- More complex but more production-realistic

Start with bake-in for simplicity; refactor to PVC mounting when you build Step 7.

## Evaluation & Success Criteria

### Deployment checks

- [ ] `docker build` completes without errors
- [ ] Container starts and `/health` responds within 30 seconds
- [ ] `kubectl get pods -n pcsaft` shows 2/2 Running
- [ ] `kubectl port-forward` + `curl /predict` returns correct results
- [ ] Pod restarts automatically if liveness probe fails (test by killing the uvicorn process)
- [ ] ConfigMap changes propagate (change model type, restart pods, verify)

### Resource checks

- [ ] Image size < 1.5 GB
- [ ] Pod memory usage < 1 GB under load (check with `kubectl top pods -n pcsaft`)
- [ ] Startup time < 30 seconds from container start to first successful `/health`

### What success looks like

- You can demo the full flow: build image → deploy to K8s → query the API
- The deployment survives pod deletion (K8s recreates it)
- You can articulate why each manifest field exists (resource limits, probe timings, volume mounts)

### When to move to Step 7

You're ready for Step 7 when:

1. The API runs stably on Kubernetes
2. You understand the PVC model-mount pattern (needed for model promotion)
3. Health probes are working and you've tested pod restart behavior
4. You have a clear mental picture of where Kubeflow fits alongside this deployment
