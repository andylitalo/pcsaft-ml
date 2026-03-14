# CLAUDE.md — Project Context & Conventions

## Project Context

ML-driven blowing agent screening via PC-SAFT parameter prediction. The codebase currently has a Random Forest baseline trained on RDKit 2D descriptors. Eight progressive steps (defined in `docs/steps/01–08`) upgrade this into a production-grade MLOps system.

## Orchestration

For step execution, parallelization, agent coordination, and context management, read `PLAN.md`.
This file provides project context and conventions only.

## Quick Reference: The 8 Steps

| Step | Name | Key Deliverable | New Dependencies |
|------|------|----------------|-----------------|
| 01 | Morgan Fingerprints | `build_features()` in descriptors.py, RF comparison table | — |
| 02 | PyTorch Multi-Task NN | `model/nn/` package, trained NN, RF vs NN comparison | torch |
| 03 | Evaluation Harness | Registry pattern, parity/residual/learning-curve plots | — |
| 04 | ChemBERTa Fine-Tuning | `model/hf/` package, 3-way model comparison | transformers, datasets |
| 05 | FastAPI Serving | `serving/` package, REST API with /predict and /submit-data | fastapi, uvicorn |
| 06 | Docker & Kubernetes | Dockerfile, `k8s/` manifests, kind deployment | — (infra only) |
| 07 | Kubeflow Retrain Pipeline | `pipeline/` package, automated retrain DAG | kfp |
| 08 | Streamlit Portal | `portal/` package, researcher-facing web UI | streamlit |

## Directory Conventions

| Artifact | Location |
|----------|----------|
| Step guides | `docs/steps/NN_<name>.md` |
| Step reports | `docs/reports/NN_<name>.md` |
| Step figures | `figures/<step-name>/` |
| New model modules | `model/nn/`, `model/hf/` (as specified per step) |
| Serving code | `serving/` |
| Pipeline code | `pipeline/` |
| Portal code | `portal/` |
| K8s manifests | `k8s/` |
| Saved models/artifacts | `model/saved/` |
| Orchestrator state | `state.yaml`, `docs/orchestrator_log.md` |

## Commit Conventions

Each step gets one commit (or a small number if the step is large). Message format:

```
step NN: <short description>

<1-2 sentence summary of what changed and key metrics>
```

Example:
```
step 01: add Morgan fingerprint feature module

Combined 2048-bit Morgan FP with cleaned RDKit descriptors (~2,150 dims).
RF R² on ε/k improved from 0.27 to 0.31 on Esper test set.
```

## Dependency Management

New dependencies are added to `pyproject.toml` under `[project.optional-dependencies]` by phase:

```toml
[project.optional-dependencies]
dev = ["pytest", "ruff"]
nn = ["torch>=2.0,<3.0"]
hf = ["torch>=2.0,<3.0", "transformers>=4.30", "datasets>=2.14"]
serve = ["fastapi>=0.100", "uvicorn>=0.23", "pydantic-settings>=2.0"]
pipeline = ["kfp>=2.0"]
portal = ["streamlit>=1.28"]
```

This project uses **uv** for dependency management. Install what you need per step:

```bash
uv sync --extra dev --extra nn   # example for Step 02
```

## What the Reports Should Contain

Reports must follow the same format as the existing baseline reports (`docs/results_esper.md`, `docs/results_fallback.md`). Specifically:

### Required Sections

1. **Title**: `# Results Report: <Step Name> (<key context>)` — e.g., `# Results Report: Morgan Fingerprints (Esper Dataset, 1,801 molecules)`
2. **Model Performance**: Opening paragraph describing what was trained/changed, then a metrics table:

```
| Parameter | MAE | RMSE | R² (test) | Train samples | Test samples |
|-----------|-----|------|-----------|---------------|--------------|
| m (segments) | ... | ... | ... | ... | ... |
| σ (Å) | ... | ... | ... | ... | ... |
| ε/k (K) | ... | ... | ... | ... | ... |
```

Follow the table with a paragraph interpreting the results — what improved, what didn't, and why. Call out limitations honestly.

3. **Comparison to Baseline/Previous Step**: A side-by-side table showing the delta from the prior step (or RF baseline for Step 01). This is the core narrative of the project.
4. **Key Findings**: Bullet points or short paragraphs on surprises, what worked, what didn't.
5. **Figures**: Reference figure paths (e.g., `See figures/01_morgan_fingerprints/ for parity plots from this run.`).
6. **Deviations**: Anything that diverged from the step guide and why. Omit this section if there were none.
7. **Readiness Check**: Confirm the "when to move on" criteria from the step guide are met, as a checklist.
