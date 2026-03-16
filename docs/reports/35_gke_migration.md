# Results Report: GKE Migration Manifests (Future Extension)

## Overview

Step 35 creates Kubernetes manifests for a potential future migration from Cloud Run (Step 34) to Google Kubernetes Engine (GKE). This step produces no model training or evaluation — it is purely infrastructure manifest work.

**Important**: This step may never be needed for the current project scope. Cloud Run (Step 34) is the production deployment target. GKE migration is only warranted if cold-start latency, persistent volumes, GPU inference, or multi-service orchestration become requirements.

## Artifacts Created

| File | Description |
|------|-------------|
| `k8s/configmap.yaml` | Updated: `PCSAFT_MODEL_TYPE` changed from `rf` to `gnn` (justified by Step 31) |
| `k8s/portal-deployment.yaml` | Portal Deployment: 1 replica, port 8501, health probes, resource limits |
| `k8s/portal-service.yaml` | Portal ClusterIP Service: port 80 → targetPort 8501 |
| `k8s/ingress.yaml` | Ingress: routes `/` to portal service |
| `k8s/api-hpa.yaml` | HPA: API autoscaling, min 1 → max 5, target 70% CPU |
| `tests/test_k8s_manifests.py` | 6 tests validating all new manifests |

## Architecture

```
Internet → Ingress (/) → pcsaft-portal Service (:80) → Portal Pod (:8501)
                                                           ↓
                                              pcsaft-api Service (:80) → API Pod (:8000)
```

- **Public portal** at `/` via Ingress
- **Internal API** via Kubernetes service DNS (`http://pcsaft-api:80`)
- Portal communicates with API server-side (not browser-side)
- HPA scales API pods based on CPU utilization

## ConfigMap Change

Changed `PCSAFT_MODEL_TYPE` from `rf` to `gnn` based on Step 31 results:
- GNN R² = 0.86/0.73/0.91 on fluorinated molecules (product chemistry)
- GNN R² = 0.79/0.77/0.73 on pooled test set
- RF fails on unified dataset (R² = -0.44/0.30/0.24)

## When to Use GKE Over Cloud Run

| Scenario | Cloud Run | GKE |
|----------|-----------|-----|
| Low-traffic research tool | Best (scale-to-zero) | Overkill |
| Cold-start latency matters | ~10-30s cold starts | Pods stay warm |
| Persistent volumes needed | Not supported | PVCs available |
| GPU inference (ChemBERTa) | Limited | GPU node pools |
| Multi-service mesh | No sidecars | Full Kubernetes |

## Key Findings

- All manifests follow the patterns established in Step 06
- Portal deployment includes liveness/readiness probes on `/_stcore/health`
- Resource limits prevent noisy-neighbor issues in shared clusters
- HPA provides automatic scaling for bursty prediction workloads

## Deviations

None. Implementation follows the step guide exactly.

## Readiness Check

- [x] Portal and API manifests created with proper namespace, labels, probes
- [x] Ingress routes portal traffic correctly at `/`
- [x] HPA configured for API pods (min 1, max 5, 70% CPU target)
- [x] ConfigMap updated to GNN default
- [x] All new tests pass (6/6)
- [x] Existing k8s tests pass with updated configmap assertion
- [x] `ruff check .` passes
