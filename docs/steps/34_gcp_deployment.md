# Step 34: Portal Dockerfile and GCP/GKE Deployment

## Objective

Create a Dockerfile for the Streamlit portal, add GKE-specific Kubernetes manifests, switch the default model to GNN, and deploy both the API and portal to a GKE cluster. Validate locally with kind before targeting GKE.

## Motivation

The primary deliverable of this project is a **hosted web UI** where researchers can predict PC-SAFT parameters, search for similar molecules, and screen candidates. The current infrastructure covers only the FastAPI API (Dockerfile, k8s manifests for a ClusterIP service). Missing pieces:

1. **Portal Dockerfile**: The Streamlit portal has no container image
2. **External access**: No Ingress or load balancer for external HTTPS traffic
3. **Portal k8s manifests**: No Deployment, Service, or resource limits for the portal
4. **Default model**: ConfigMap still uses `PCSAFT_MODEL_TYPE=rf` instead of GNN
5. **Autoscaling**: No HPA for the API pods

## Dependencies

- Step 31 completed: GNN retrained and validated (for setting GNN as default)
- Step 33 completed: `/similar` endpoint exists in the API
- `Dockerfile` exists for the API (from Step 06)
- `k8s/` manifests exist (namespace, configmap, deployment, service, PVC)
- `portal/app.py` exists with prediction, screening, and dashboard tabs

## Implementation Guide

### 34.1 Create Portal Dockerfile

Create `Dockerfile.portal` (or `portal/Dockerfile`) in the project root:

```dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --extra portal --no-dev

# Copy application code
COPY portal/ portal/
COPY model/ model/
COPY serving/ serving/
COPY screening/ screening/
COPY data/pcsaft_novel_predictions_v1.csv data/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "portal/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

Key design decisions:
- Use `python:3.12-slim` to match the API Dockerfile
- Include the predictions CSV so the portal can run similarity search in local fallback mode
- Include model weights so the portal can predict locally if the API is unreachable
- Expose port 8501 (Streamlit default)

### 34.2 Update ConfigMap to use GNN

Update `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pcsaft-config
  namespace: pcsaft
data:
  PCSAFT_MODEL_TYPE: "gnn"
  PCSAFT_API_HOST: "0.0.0.0"
  PCSAFT_API_PORT: "8000"
```

Also update `serving/config.py` to change the default:

```python
model_type: str = "gnn"  # Changed from "rf"
```

### 34.3 Add Portal k8s manifests

Create `k8s/portal-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pcsaft-portal
  namespace: pcsaft
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pcsaft-portal
  template:
    metadata:
      labels:
        app: pcsaft-portal
    spec:
      containers:
      - name: portal
        image: pcsaft-portal:latest
        ports:
        - containerPort: 8501
        env:
        - name: PCSAFT_API_URL
          value: "http://pcsaft-api:80"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 15
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 10
          periodSeconds: 10
```

Create `k8s/portal-service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: pcsaft-portal
  namespace: pcsaft
spec:
  selector:
    app: pcsaft-portal
  ports:
  - port: 80
    targetPort: 8501
  type: ClusterIP
```

### 34.4 Add Ingress for external access

Create `k8s/ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pcsaft-ingress
  namespace: pcsaft
  annotations:
    # For GKE managed certificates:
    # networking.gke.io/managed-certificates: pcsaft-cert
    # For kind/local testing, no annotations needed
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pcsaft-portal
            port:
              number: 80
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: pcsaft-api
            port:
              number: 80
```

For GKE, this should use the GKE Ingress controller. For kind/local testing, install nginx-ingress controller.

### 34.5 Add HorizontalPodAutoscaler for the API

Create `k8s/api-hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pcsaft-api-hpa
  namespace: pcsaft
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pcsaft-api
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 34.6 Local validation with kind

Before targeting GKE, test locally:

```bash
# Build images
docker build -t pcsaft-api:latest -f Dockerfile .
docker build -t pcsaft-portal:latest -f Dockerfile.portal .

# Create kind cluster (if not exists)
kind create cluster --name pcsaft

# Load images into kind
kind load docker-image pcsaft-api:latest --name pcsaft
kind load docker-image pcsaft-portal:latest --name pcsaft

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/portal-deployment.yaml
kubectl apply -f k8s/portal-service.yaml
kubectl apply -f k8s/ingress.yaml

# Verify pods are running
kubectl get pods -n pcsaft
kubectl logs -n pcsaft -l app=pcsaft-portal
kubectl logs -n pcsaft -l app=pcsaft-api

# Port-forward to test
kubectl port-forward -n pcsaft svc/pcsaft-portal 8501:80
# Open http://localhost:8501
```

### 34.7 GKE deployment documentation

Create `docs/gke_deployment.md` with:

1. **Cluster setup**: Recommended machine type (e2-standard-2 or n1-standard-2), node pool size (1-3 nodes), region
2. **Container registry**: Push images to GCR or Artifact Registry
3. **DNS**: Configure a domain or use the GKE external IP
4. **TLS**: Use GKE managed certificates for HTTPS
5. **Monitoring**: Enable GKE monitoring and logging

### 34.8 Tests

Add to `tests/test_docker_k8s.py` or create new test file:

1. `test_portal_dockerfile_exists`: Verify `Dockerfile.portal` exists
2. `test_portal_deployment_yaml`: Verify portal deployment manifest is valid YAML with required fields
3. `test_ingress_yaml`: Verify ingress manifest has correct paths
4. `test_configmap_uses_gnn`: Verify configmap sets `PCSAFT_MODEL_TYPE: "gnn"`
5. `test_hpa_yaml`: Verify HPA manifest targets API deployment

## Artifacts

| File | Description |
|------|-------------|
| `Dockerfile.portal` | Streamlit portal container image |
| `k8s/portal-deployment.yaml` | Portal Deployment manifest |
| `k8s/portal-service.yaml` | Portal Service manifest |
| `k8s/ingress.yaml` | Ingress for external access |
| `k8s/api-hpa.yaml` | HPA for API autoscaling |
| `k8s/configmap.yaml` | Updated with GNN default |
| `serving/config.py` | Updated default model_type |
| `docs/gke_deployment.md` | GKE setup documentation |
| `docs/reports/34_gcp_deployment.md` | Step report |

## Success Criteria

- [ ] Portal Dockerfile builds and runs locally
- [ ] Portal and API pods start in kind cluster
- [ ] Ingress routes traffic correctly (portal at `/`, API at `/api`)
- [ ] ConfigMap and serving config default to GNN
- [ ] HPA configured for API pods
- [ ] GKE deployment documented
- [ ] All existing tests pass
- [ ] `ruff check .` passes
- [ ] Report documents local and GKE deployment

## When to Move On

- Both containers build and run in kind
- Ingress serves the portal externally (via port-forward at minimum)
- GKE deployment guide is complete and actionable

## Budget

20 minutes. Primarily Dockerfile creation and manifest writing; no model training.
