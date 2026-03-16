# Step 35: GKE Migration (Future Extension)

## Objective

Migrate the PC-SAFT portal and API from Cloud Run to a GKE (Google Kubernetes Engine) cluster for production-grade deployment with Kubernetes manifests, Ingress, HPA, and managed certificates. This step builds on the Cloud Run deployment from Step 34 and reuses the existing `k8s/` manifests from Step 06.

## Motivation

Cloud Run (Step 34) is the right starting point for a research tool: zero idle cost, no cluster management, automatic HTTPS. However, there are scenarios where GKE becomes necessary:

- **Cold-start latency**: Cloud Run's scale-to-zero means ~10-30s cold starts for model loading. GKE keeps pods warm.
- **Persistent volumes**: Cloud Run is stateless. If the submission CSV or model artifacts need persistent storage, GKE with PVCs is required.
- **GPU inference**: If a future model (e.g., ChemBERTa) benefits from GPU, GKE supports GPU node pools.
- **Service mesh / sidecars**: Complex networking, mTLS between services, or sidecar containers require Kubernetes.
- **Multi-service orchestration**: If the retrain pipeline (Step 07) or other services need to run alongside the API.

**If none of these apply, stay on Cloud Run.** This step may never be needed for the current project scope.

## Dependencies

- Step 34 completed: Cloud Run deployment validated and working
- `Dockerfile` exists for the API (from Step 06)
- `Dockerfile.portal` exists (from Step 34)
- `k8s/` manifests exist (namespace, configmap, deployment, service, PVC from Step 06)
- GCP project `project-fd592eb9-74e2-455d-959` with billing enabled

## Implementation Guide

### 35.1 Update ConfigMap to use GNN

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

Only set `gnn` if Step 31's promotion criteria were met.

### 35.2 Add Portal k8s manifests

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

### 35.3 Add Ingress for external access

Deployment contract:

1. **Public portal** at `/`
2. **Internal API** via Kubernetes service DNS (`http://pcsaft-api:80`) for the portal container
3. **Optional external API** only if a programmatic public endpoint is explicitly required

This matches the Streamlit architecture where `portal/api_client.py` makes server-side requests.

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
```

If you also want a public API, choose one explicit option:

1. **Separate hostnames**: `mlchem.andylitalo.com` for the portal, `api.mlchem.andylitalo.com` for FastAPI
2. **Ingress rewrite**: rewrite `/api/*` to `/*` at the ingress layer
3. **App prefix**: add a FastAPI root path so the service serves under `/api`

Option 1 is simplest for GKE.

### 35.4 Add HorizontalPodAutoscaler for the API

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

### 35.5 Local validation with kind

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

Verify that the portal can reach the API through the Kubernetes service DNS.

### 35.6 GKE cluster setup

```bash
PROJECT_ID=project-fd592eb9-74e2-455d-959
REGION=us-central1
CLUSTER_NAME=pcsaft-cluster

# Create an Autopilot cluster (simplest, pay-per-pod)
gcloud container clusters create-auto $CLUSTER_NAME \
  --project $PROJECT_ID \
  --region $REGION

# Or a Standard cluster for more control
gcloud container clusters create $CLUSTER_NAME \
  --project $PROJECT_ID \
  --region $REGION \
  --machine-type e2-standard-2 \
  --num-nodes 2 \
  --enable-autoscaling --min-nodes 1 --max-nodes 3

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME \
  --project $PROJECT_ID --region $REGION
```

### 35.7 Push images and deploy to GKE

```bash
REGISTRY=$REGION-docker.pkg.dev/$PROJECT_ID/pcsaft

# Tag and push (reuse Artifact Registry from Step 34)
docker tag pcsaft-api:latest $REGISTRY/api:v1
docker tag pcsaft-portal:latest $REGISTRY/portal:v1
docker push $REGISTRY/api:v1
docker push $REGISTRY/portal:v1

# Update manifests to use registry images
# (replace `image: pcsaft-api:latest` with `image: $REGISTRY/api:v1`)

# Apply all manifests
kubectl apply -f k8s/
```

### 35.8 TLS and custom domain on GKE

Create a managed certificate:

```yaml
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: pcsaft-cert
  namespace: pcsaft
spec:
  domains:
    - mlchem.andylitalo.com
```

Reserve a static IP and update DNS:

```bash
gcloud compute addresses create pcsaft-ip --global
IP=$(gcloud compute addresses describe pcsaft-ip --global --format='value(address)')
echo "Add an A record for mlchem.andylitalo.com -> $IP in Squarespace DNS"
```

In Squarespace Domains (domains.squarespace.com > andylitalo.com > DNS > Custom Records):
- **Type**: A
- **Host**: `mlchem`
- **Data**: the IP from the command above

Update the Ingress to use the managed certificate and static IP:

```yaml
metadata:
  annotations:
    networking.gke.io/managed-certificates: pcsaft-cert
    kubernetes.io/ingress.global-static-ip-name: pcsaft-ip
```

### 35.9 Monitoring

Enable GKE monitoring and logging:

```bash
gcloud container clusters update $CLUSTER_NAME \
  --project $PROJECT_ID --region $REGION \
  --enable-managed-prometheus \
  --logging=SYSTEM,WORKLOAD \
  --monitoring=SYSTEM,WORKLOAD
```

## Tests

Add to `tests/test_k8s_manifests.py`:

1. `test_portal_deployment_yaml`: Verify portal deployment manifest is valid YAML with required fields
2. `test_portal_service_yaml`: Verify portal service targets correct port
3. `test_ingress_yaml`: Verify ingress exposes portal at `/`
4. `test_configmap_model_type`: Verify configmap sets `PCSAFT_MODEL_TYPE`
5. `test_hpa_yaml`: Verify HPA targets API deployment
6. `test_portal_env_uses_internal_api`: Verify `PCSAFT_API_URL` points at `http://pcsaft-api:80`

## Artifacts

| File | Description |
|------|-------------|
| `k8s/portal-deployment.yaml` | Portal Deployment manifest |
| `k8s/portal-service.yaml` | Portal Service manifest |
| `k8s/ingress.yaml` | Ingress for external access |
| `k8s/api-hpa.yaml` | HPA for API autoscaling |
| `k8s/configmap.yaml` | Updated with GNN default |
| `docs/reports/35_gke_migration.md` | Step report |

## Success Criteria

- [ ] Portal and API pods start in kind cluster
- [ ] Portal can reach API through Kubernetes service DNS
- [ ] Ingress routes portal traffic correctly at `/`
- [ ] HPA configured for API pods
- [ ] GKE cluster created and services deployed
- [ ] Custom domain with managed TLS certificate working
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- Both containers run in kind
- Portal completes at least one prediction request against the API in kind
- GKE deployment working with public access
- Custom domain mapped and TLS provisioned

## Budget

30 minutes. Primarily manifest writing and cluster setup; no model training.
