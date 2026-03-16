# Results Report: Cloud Run Deployment

## Overview

Step 34 creates the deployment artifacts required to deploy the PC-SAFT parameter prediction tool to Google Cloud Run. This step provides the infrastructure-as-code needed for a production-ready, publicly accessible deployment while avoiding the complexity of Kubernetes cluster management. The implementation includes a portal Dockerfile, docker-compose for local validation, and comprehensive deployment tests.

## Deliverables

All deployment artifacts created and tested:

1. **Dockerfile.portal** (multi-stage build for Streamlit portal)
2. **docker-compose.yaml** (two-service local development stack)
3. **pyproject.toml** (updated dev dependencies with pyyaml)
4. **tests/test_deployment.py** (9 tests validating deployment structure)
5. **docs/reports/34_cloud_run_deployment.md** (this report)

## Dockerfile.portal Architecture

### Multi-Stage Build Strategy

The portal Dockerfile follows the same two-stage pattern as the API Dockerfile to optimize image size:

**Stage 1 (builder)**:
- Base: `python:3.11-slim`
- Installs portal dependencies using `pip install ".[portal]"` from `pyproject.toml`
- Uses `--no-cache-dir` and `--prefix=/install` for clean dependency isolation

**Stage 2 (runtime)**:
- Base: `python:3.11-slim` (fresh image)
- Installs system dependencies: `libxrender1`, `libxext6` (required by matplotlib/plotting libraries)
- Copies installed Python packages from builder stage
- Copies application code:
  - `portal/` (Streamlit app)
  - `serving/` (schemas and API client)
  - `screening/` (similarity search logic)
  - `pcsaft_predict/` (prediction module for local fallback)
  - `model/` (model weights for local fallback)
  - `data/pcsaft_novel_predictions_v1.csv` (precomputed predictions for similarity search)
- Exposes port 8501 (Streamlit default)
- HEALTHCHECK hits `/_stcore/health` every 30s (Streamlit built-in health endpoint)
- CMD runs `streamlit run portal/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true`

### Local Fallback Design

The portal includes both `pcsaft_predict/` and model weights, enabling predictions even if the API is unreachable. This resilience is important for Cloud Run deployments where cold starts may temporarily make the API unavailable.

The portal also includes the precomputed predictions CSV (`data/pcsaft_novel_predictions_v1.csv`) for local similarity search, avoiding hard dependencies on the API's `/similar` endpoint.

## Docker Compose Architecture

The `docker-compose.yaml` file enables local validation of both services before deploying to Cloud Run:

### Service Configuration

**API service**:
- Builds from `Dockerfile`
- Exposes port 8000
- Environment variable: `PCSAFT_MODEL_TYPE` (defaults to `rf` if not set)
- Health check using `curl -f http://localhost:8000/health`

**Portal service**:
- Builds from `Dockerfile.portal`
- Exposes port 8501
- Environment variable: `PCSAFT_API_URL=http://api:8000` (Docker internal networking)
- `depends_on: api` with `condition: service_healthy` ensures API is ready before portal starts

### Local Validation Workflow

```bash
# Build both images
docker compose build

# Start services (portal waits for API health check)
docker compose up -d

# Verify
docker compose ps
curl http://localhost:8000/health
# Open http://localhost:8501 in browser

# Clean up
docker compose down
```

This workflow validates the entire two-service stack locally before Cloud Run deployment, catching issues like missing dependencies, incorrect port configuration, or API connectivity problems.

## Testing

Created `tests/test_deployment.py` with 9 tests covering all deployment artifacts:

### File Existence
- `test_portal_dockerfile_exists`: Verifies `Dockerfile.portal` exists
- `test_api_dockerfile_exists`: Verifies `Dockerfile` exists
- `test_docker_compose_exists`: Verifies `docker-compose.yaml` exists and is valid YAML

### Dockerfile Configuration
- `test_portal_dockerfile_exposes_8501`: Verifies `EXPOSE 8501` in portal Dockerfile
- `test_api_dockerfile_exposes_8000`: Verifies `EXPOSE 8000` in API Dockerfile
- `test_portal_dockerfile_healthcheck`: Verifies HEALTHCHECK using `_stcore/health`
- `test_api_dockerfile_healthcheck`: Verifies HEALTHCHECK on `/health`

### Docker Compose Configuration
- `test_docker_compose_api_health_depends`: Verifies portal depends on API with health condition
- `test_docker_compose_environment_vars`: Verifies required environment variables are set

All 9 tests pass, confirming the deployment structure is correct.

## Cloud Run Deployment Instructions

The step guide provides complete Cloud Run deployment commands (section 34.4-34.7). The actual deployment is a manual step performed by the user:

### Prerequisites
- GCP project ID: `project-fd592eb9-74e2-455d-959`
- Region: `us-central1`
- `gcloud` CLI installed and authenticated
- Cloud Run and Artifact Registry APIs enabled

### Deployment Workflow
1. Create Artifact Registry repository for Docker images
2. Build and push both images (`api:v1` and `portal:v1`)
3. Deploy API to Cloud Run (2Gi memory, 1 CPU, scale-to-zero)
4. Deploy portal to Cloud Run (1Gi memory, 1 CPU, scale-to-zero with API URL)
5. Verify both services at auto-assigned `*.run.app` URLs

### Scale-to-Zero Cost Model
- `--min-instances 0`: Services scale to zero when idle (free)
- Cold start penalty: ~10-30s for model loading on first request after idle period
- Estimated monthly cost: $0-5/month with $300 free credits (will last years at low traffic)
- Optional `--min-instances 1` keeps one instance warm (~$15-25/month) if cold starts are unacceptable

### Custom Domain Mapping
The step guide includes instructions for mapping `mlchem.andylitalo.com` to the portal (section 34.9). This is deferred for now but documented for future use:
1. Verify domain ownership via Google Search Console
2. Create domain mapping in Cloud Run
3. Add CNAME record in Squarespace DNS: `mlchem` → `ghs.googlehosted.com.`
4. Wait for DNS propagation and TLS certificate provisioning

## Model Type Configuration

The deployment uses `PCSAFT_MODEL_TYPE=rf` as the default, matching the current `serving/config.py` setting. The step guide allows switching to `gnn` if Step 31 (GNN retraining) met the promotion criteria.

To switch model types later:
```bash
gcloud run services update pcsaft-api \
  --project $PROJECT_ID --region $REGION \
  --set-env-vars PCSAFT_MODEL_TYPE=gnn
```

The `serving/config.py` default should be updated to match the Cloud Run environment variable for consistency.

## Key Findings

### What Works Well

1. **Multi-stage builds**: Portal Dockerfile mirrors the API pattern, keeping images lean
2. **Local validation**: Docker Compose enables full-stack testing before cloud deployment
3. **Service health dependency**: Portal waits for API to be healthy before starting, avoiding startup race conditions
4. **Local fallback**: Portal can predict and search locally if API is unreachable, improving resilience
5. **Scale-to-zero economics**: Cloud Run's pricing model makes this deployment essentially free at low traffic

### Design Decisions

1. **Portal includes model weights**: Adds ~50-100MB to the image but enables local prediction fallback, critical for Cloud Run where cold starts may temporarily make the API unavailable

2. **Portal includes predictions CSV**: The 1.5MB CSV of precomputed predictions enables local similarity search, avoiding hard API dependencies

3. **RF default**: Keeping `PCSAFT_MODEL_TYPE=rf` matches the current serving config and avoids premature promotion of the GNN model before Step 31 validation

4. **Docker Compose for validation**: Not strictly required for Cloud Run deployment, but provides a critical validation step for the two-service architecture

### Limitations & Future Work

1. **No actual deployment**: This step creates the artifacts and documents the commands, but deployment is manual. The next step (Step 35, if pursued) would automate deployments via CI/CD.

2. **No monitoring/logging**: Cloud Run provides built-in logging, but no custom metrics or alerting are configured. Production would add Cloud Monitoring dashboards for request latency, error rates, and cold start frequency.

3. **No rate limiting**: The API has no request throttling. Production would add Cloud Armor or API Gateway for DDoS protection.

4. **No authentication**: `--allow-unauthenticated` makes the services public. Research tools often need this, but production might add IAM-based access control.

5. **Single region**: Deployed only in `us-central1`. Multi-region deployment would improve global latency but increase complexity.

### Deviations from Step Guide

None. The implementation follows the step guide exactly, including:
- Multi-stage Dockerfile pattern
- Docker Compose service dependency with health check
- All required tests
- Documented but deferred custom domain mapping

## Readiness Check

- [x] Dockerfile.portal exists and uses multi-stage build
- [x] Dockerfile.portal exposes port 8501
- [x] Dockerfile.portal includes HEALTHCHECK on `_stcore/health`
- [x] docker-compose.yaml exists with api and portal services
- [x] Portal depends on API with `service_healthy` condition
- [x] Environment variables configured (PCSAFT_MODEL_TYPE, PCSAFT_API_URL)
- [x] PyYAML added to dev dependencies for tests
- [x] All 9 deployment tests pass
- [x] No ruff errors in new test file
- [x] Cloud Run deployment commands documented in step guide

**Ready for manual Cloud Run deployment using the commands in `docs/steps/34_gcp_deployment.md`.**

## Cloud Run Deployment Checklist (Manual Step)

When ready to deploy, follow this sequence:

1. Authenticate and set GCP project:
   ```bash
   gcloud auth login
   gcloud config set project project-fd592eb9-74e2-455d-959
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com
   ```

2. Create Artifact Registry repository:
   ```bash
   gcloud artifacts repositories create pcsaft \
     --repository-format=docker \
     --location=us-central1 \
     --description="PC-SAFT parameter prediction containers"
   gcloud auth configure-docker us-central1-docker.pkg.dev
   ```

3. Build and push images:
   ```bash
   REGISTRY=us-central1-docker.pkg.dev/project-fd592eb9-74e2-455d-959/pcsaft
   docker build -t $REGISTRY/api:v1 -f Dockerfile .
   docker build -t $REGISTRY/portal:v1 -f Dockerfile.portal .
   docker push $REGISTRY/api:v1
   docker push $REGISTRY/portal:v1
   ```

4. Deploy API:
   ```bash
   gcloud run deploy pcsaft-api \
     --image $REGISTRY/api:v1 \
     --region us-central1 \
     --memory 2Gi --cpu 1 \
     --min-instances 0 --max-instances 3 \
     --port 8000 \
     --allow-unauthenticated \
     --set-env-vars PCSAFT_MODEL_TYPE=rf
   ```

5. Deploy portal (after capturing API URL):
   ```bash
   API_URL=$(gcloud run services describe pcsaft-api \
     --region us-central1 \
     --format='value(status.url)')

   gcloud run deploy pcsaft-portal \
     --image $REGISTRY/portal:v1 \
     --region us-central1 \
     --memory 1Gi --cpu 1 \
     --min-instances 0 --max-instances 2 \
     --port 8501 \
     --allow-unauthenticated \
     --set-env-vars PCSAFT_API_URL=$API_URL
   ```

6. Verify deployment:
   ```bash
   PORTAL_URL=$(gcloud run services describe pcsaft-portal \
     --region us-central1 \
     --format='value(status.url)')
   echo "Portal available at: $PORTAL_URL"
   ```

## Next Steps

With Cloud Run deployment artifacts complete, the tool is ready for public hosting. Potential follow-on work:

- **Step 35 (optional)**: GKE migration if Kubernetes features (node affinity, custom networking, GPU access) are needed
- **CI/CD**: Automate build/deploy on git push via Cloud Build or GitHub Actions
- **Monitoring**: Add Cloud Monitoring dashboards for request metrics and error tracking
- **Custom domain**: Execute the domain mapping commands when ready to switch from `*.run.app` to `mlchem.andylitalo.com`

The infrastructure-as-code approach makes all of these enhancements straightforward to add incrementally.
