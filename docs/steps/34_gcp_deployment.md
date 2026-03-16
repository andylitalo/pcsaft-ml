# Step 34: Cloud Run Deployment

## Objective

Deploy the Streamlit portal and FastAPI API to Google Cloud Run so the tool is accessible at a public HTTPS URL. Cloud Run scales to zero when idle, keeping costs near $0/month for a low-traffic research tool. This is the fastest path to a hosted PC-SAFT parameter estimator.

## Motivation

The primary deliverable of this project is a **hosted web UI** where researchers can predict PC-SAFT parameters, search for similar molecules, and screen candidates. The current infrastructure covers only the FastAPI API (Dockerfile, k8s manifests for local kind clusters). Missing pieces:

1. **Portal Dockerfile**: The Streamlit portal has no container image
2. **Cloud deployment**: No mechanism to serve the tool at a public URL
3. **Default model**: `serving/config.py` still uses `PCSAFT_MODEL_TYPE=rf` instead of GNN

Cloud Run is the right deployment target for prototyping because:
- **Scale-to-zero**: No cost when idle (the $300 free credits will last a very long time)
- **No cluster management**: No kubectl, Ingress controllers, or node pools
- **Automatic HTTPS**: TLS is handled by Cloud Run, including for custom domains
- **One-command deploys**: `gcloud run deploy` replaces the entire k8s manifest stack

GKE migration is available as Step 35 if production-grade Kubernetes is needed later.

## GCP Configuration

| Setting | Value |
|---------|-------|
| Project ID | `project-fd592eb9-74e2-455d-959` |
| Region | `us-central1` |
| Custom domain (deferred) | `mlchem.andylitalo.com` |
| Initial URL | Auto-assigned `*.run.app` |
| Billing | $300 free credits linked to project |

## Dependencies

- Step 31 completed: GNN retrained and validated (for setting GNN as default; can deploy with RF if Step 31 is not yet done)
- Step 33 completed: `/similar` endpoint exists in the API (optional; can deploy without it)
- `Dockerfile` exists for the API (from Step 06)
- `portal/app.py` exists with prediction, screening, and dashboard tabs
- `gcloud` CLI installed and authenticated

## Implementation Guide

### 34.1 Prerequisites: authenticate and enable APIs

```bash
PROJECT_ID=project-fd592eb9-74e2-455d-959
REGION=us-central1

gcloud auth login
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

### 34.2 Create Portal Dockerfile

Create `Dockerfile.portal` in the project root:

```dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir --prefix=/install ".[portal]"

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY portal/ portal/
COPY serving/ serving/
COPY screening/ screening/
COPY pcsaft_predict/ pcsaft_predict/
COPY model/ model/
COPY data/pcsaft_novel_predictions_v1.csv data/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "portal/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

Key decisions:
- Multi-stage build matching the API Dockerfile pattern (`python:3.11-slim`)
- Includes predictions CSV for local-fallback similarity search
- Includes `pcsaft_predict/` and model weights so portal can predict locally if API is unreachable
- Uses `pip install ".[portal]"` rather than uv (uv is a dev tool, not needed at runtime)

### 34.3 Create docker-compose.yaml for local validation

Create `docker-compose.yaml` in the project root:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      PCSAFT_MODEL_TYPE: "${PCSAFT_MODEL_TYPE:-rf}"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  portal:
    build:
      context: .
      dockerfile: Dockerfile.portal
    ports:
      - "8501:8501"
    environment:
      PCSAFT_API_URL: "http://api:8000"
    depends_on:
      api:
        condition: service_healthy
```

Test locally before deploying:

```bash
docker compose build
docker compose up -d

# Verify both services are healthy
docker compose ps
curl http://localhost:8000/health
# Open http://localhost:8501 in browser

# Clean up
docker compose down
```

### 34.4 Create Artifact Registry repository

```bash
gcloud artifacts repositories create pcsaft \
  --repository-format=docker \
  --location=$REGION \
  --project=$PROJECT_ID \
  --description="PC-SAFT parameter prediction containers"

# Configure Docker to authenticate with Artifact Registry
gcloud auth configure-docker $REGION-docker.pkg.dev
```

### 34.5 Build and push images

```bash
REGISTRY=$REGION-docker.pkg.dev/$PROJECT_ID/pcsaft

# Build and tag
docker build -t $REGISTRY/api:v1 -f Dockerfile .
docker build -t $REGISTRY/portal:v1 -f Dockerfile.portal .

# Push
docker push $REGISTRY/api:v1
docker push $REGISTRY/portal:v1
```

### 34.6 Deploy API to Cloud Run

```bash
gcloud run deploy pcsaft-api \
  --image $REGISTRY/api:v1 \
  --project $PROJECT_ID \
  --region $REGION \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8000 \
  --allow-unauthenticated \
  --set-env-vars "PCSAFT_MODEL_TYPE=${PCSAFT_MODEL_TYPE:-rf}"
```

`--min-instances 0` means scale-to-zero: free when idle, but cold starts take ~10-30s for model loading. If cold-start latency is unacceptable later, set `--min-instances 1` (~$15-25/month).

Record the API service URL from the output:

```bash
API_URL=$(gcloud run services describe pcsaft-api \
  --project $PROJECT_ID --region $REGION \
  --format='value(status.url)')
echo "API deployed at: $API_URL"
```

Verify:

```bash
curl $API_URL/health
```

### 34.7 Deploy Portal to Cloud Run

```bash
gcloud run deploy pcsaft-portal \
  --image $REGISTRY/portal:v1 \
  --project $PROJECT_ID \
  --region $REGION \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --port 8501 \
  --allow-unauthenticated \
  --set-env-vars "PCSAFT_API_URL=$API_URL"
```

Record the portal URL:

```bash
PORTAL_URL=$(gcloud run services describe pcsaft-portal \
  --project $PROJECT_ID --region $REGION \
  --format='value(status.url)')
echo "Portal deployed at: $PORTAL_URL"
```

Verify by opening `$PORTAL_URL` in a browser and completing a prediction.

### 34.8 Conditional GNN default

Only set `PCSAFT_MODEL_TYPE=gnn` in the Cloud Run deploy commands if Step 31's promotion criteria were met. If Step 31 is not yet complete or the GNN did not pass the fluorinated/robustness checks, deploy with `rf` for now.

When switching later:

```bash
gcloud run services update pcsaft-api \
  --project $PROJECT_ID --region $REGION \
  --set-env-vars PCSAFT_MODEL_TYPE=gnn
```

Also update `serving/config.py` default to match:

```python
model_type: str = "gnn"  # Changed from "rf" after Step 31 promotion
```

### 34.9 Custom domain mapping (deferred)

When ready to map `mlchem.andylitalo.com` to the portal:

**Step 1: Verify domain ownership with Google**

```bash
gcloud domains verify andylitalo.com
```

This opens a browser to the Google Search Console verification flow. Follow the instructions (typically adding a TXT record to DNS).

**Step 2: Create the domain mapping**

```bash
gcloud beta run domain-mappings create \
  --service pcsaft-portal \
  --domain mlchem.andylitalo.com \
  --region $REGION \
  --project $PROJECT_ID
```

**Step 3: Add DNS record in Squarespace Domains**

1. Go to [domains.squarespace.com](https://domains.squarespace.com)
2. Select `andylitalo.com`
3. Go to **DNS** > **Custom Records**
4. Add a new record:
   - **Type**: CNAME
   - **Host**: `mlchem`
   - **Data**: `ghs.googlehosted.com.`
   - **TTL**: default (3600)
5. Save

**Step 4: Wait and verify**

DNS propagation takes minutes to hours. TLS certificate provisioning may take up to 24 hours (usually much faster).

```bash
# Check mapping status
gcloud beta run domain-mappings describe \
  --domain mlchem.andylitalo.com \
  --region $REGION \
  --project $PROJECT_ID

# Verify once propagated
curl -I https://mlchem.andylitalo.com
```

Until the domain is mapped, the portal is reachable at the auto-assigned `https://pcsaft-portal-*.run.app` URL.

### 34.10 Redeployment workflow

For subsequent deploys after code changes:

```bash
# Increment version tag
VERSION=v2

# Rebuild, push, deploy API
docker build -t $REGISTRY/api:$VERSION -f Dockerfile .
docker push $REGISTRY/api:$VERSION
gcloud run deploy pcsaft-api \
  --image $REGISTRY/api:$VERSION \
  --project $PROJECT_ID --region $REGION

# Rebuild, push, deploy Portal
docker build -t $REGISTRY/portal:$VERSION -f Dockerfile.portal .
docker push $REGISTRY/portal:$VERSION
gcloud run deploy pcsaft-portal \
  --image $REGISTRY/portal:$VERSION \
  --project $PROJECT_ID --region $REGION
```

## Tests

Create or update `tests/test_deployment.py`:

1. `test_portal_dockerfile_exists`: Verify `Dockerfile.portal` exists
2. `test_docker_compose_exists`: Verify `docker-compose.yaml` exists and is valid YAML
3. `test_api_dockerfile_exists`: Verify `Dockerfile` exists
4. `test_portal_dockerfile_exposes_8501`: Verify EXPOSE 8501 in Dockerfile.portal
5. `test_api_dockerfile_exposes_8000`: Verify EXPOSE 8000 in Dockerfile

## Artifacts

| File | Description |
|------|-------------|
| `Dockerfile.portal` | Streamlit portal container image |
| `docker-compose.yaml` | Local two-service development stack |
| `serving/config.py` | Updated default model_type (if Step 31 supports it) |
| `docs/reports/34_cloud_run_deployment.md` | Step report |

## Success Criteria

- [ ] Portal Dockerfile builds successfully
- [ ] `docker compose up` starts both API and portal locally
- [ ] Portal can reach the API and complete a prediction request via docker-compose
- [ ] API deployed to Cloud Run and `/health` returns 200
- [ ] Portal deployed to Cloud Run and accessible at the `.run.app` URL
- [ ] A prediction request completes end-to-end through the hosted portal
- [ ] ConfigMap and serving config default to GNN only if Step 31 justified that promotion
- [ ] Custom domain mapping instructions documented (execution deferred)
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] `ruff check .` passes
- [ ] Report documents deployment URLs, cost estimate, and any issues encountered

## When to Move On

- Both containers build and run locally via docker-compose
- Portal and API are deployed to Cloud Run and reachable at public URLs
- At least one end-to-end prediction succeeds through the hosted portal
- Custom domain instructions are documented for later use

## Budget

20 minutes for Dockerfile creation, docker-compose, and documentation. Actual Cloud Run deployment is a manual step using the commands provided.
