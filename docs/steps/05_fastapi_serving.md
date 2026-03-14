# Step 5: FastAPI Model Serving

## Purpose

Wrap the best-performing model (from Steps 2–4) in a REST API so it can serve predictions over HTTP. This transforms the project from "a script I run locally" into "a service that other systems and users can call," which is the prerequisite for everything in Steps 6–8.

The API also introduces a `/submit-data` endpoint where researchers can contribute experimental PC-SAFT measurements. This is the data ingestion point for the online learning pipeline built in Step 7.

## What It Adds Over Step 4

| After Step 4 | After This Step |
|--------------|----------------|
| Predictions require running a Python script | Predictions via HTTP request (language-agnostic) |
| No way for external users to interact with the model | REST API with documented endpoints |
| No path for new experimental data to enter the system | `/submit-data` endpoint stores researcher contributions |
| No health monitoring | `/health` endpoint for liveness/readiness probes (K8s needs this) |

## Skills Demonstrated

- **FastAPI**: Async Python web framework, dependency injection, middleware
- **Pydantic**: Request/response schema validation with type safety
- **REST API design**: Resource naming, HTTP methods, status codes, error handling
- **Model serving patterns**: Model loading at startup, singleton pattern, warm-up prediction

## Implementation Guide

### 1. Project structure

```
serving/
    __init__.py
    app.py              # FastAPI application
    schemas.py          # Pydantic request/response models
    model_loader.py     # Load and cache the model at startup
    config.py           # Configuration (model path, DB URL, etc.)
```

### 2. Pydantic schemas (`serving/schemas.py`)

```python
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    smiles: list[str] = Field(
        ..., min_length=1, max_length=100,
        description="List of SMILES strings to predict PC-SAFT parameters for",
        json_schema_extra={"examples": [["C1CCCC1", "CC(=O)O"]]},
    )

class UncertaintyEstimate(BaseModel):
    m_std: float = Field(description="Std dev of segment number prediction")
    sigma_std: float = Field(description="Std dev of segment diameter prediction")
    epsilon_k_std: float = Field(description="Std dev of dispersion energy prediction")

class MoleculePrediction(BaseModel):
    smiles: str
    m: float = Field(description="Segment number")
    sigma: float = Field(description="Segment diameter (Å)")
    epsilon_k: float = Field(description="Dispersion energy (K)")
    uncertainty: UncertaintyEstimate | None = Field(None, description="Prediction uncertainty from ensemble/MC Dropout")
    in_domain: bool = Field(True, description="Whether molecule is within the model's applicability domain")
    is_associating: bool = Field(False, description="Whether molecule has association sites (OH, NH, COOH) -- 3-param PC-SAFT may be insufficient")
    valid: bool = Field(description="Whether the SMILES was parseable")

class PredictionResponse(BaseModel):
    predictions: list[MoleculePrediction]
    model_name: str
    model_version: str

class DataSubmission(BaseModel):
    smiles: str
    m: float = Field(gt=0, description="Measured segment number")
    sigma: float = Field(gt=0, description="Measured segment diameter (Å)")
    epsilon_k: float = Field(gt=0, description="Measured dispersion energy (K)")
    source: str = Field(description="Publication DOI or lab identifier")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    version: str
```

### 3. Model loader (`serving/model_loader.py`)

Load the model once at startup and cache it as a singleton. Support loading any registered model type:

```python
import torch
from model.registry import MODELS

class ModelServer:
    def __init__(self, model_type: str, model_path: str):
        self.model = MODELS[model_type]()
        self.model.load(model_path)
        self.model_type = model_type

        # Warm-up prediction to catch errors at startup, not at first request
        self._warmup()

    def predict(self, smiles_list: list[str]) -> list[dict]:
        return self.model.predict(smiles_list)

    def _warmup(self):
        self.predict(["C"])  # methane
```

### 4. FastAPI application (`serving/app.py`)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from serving.schemas import *
from serving.model_loader import ModelServer
from serving.config import settings

model_server: ModelServer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_server
    model_server = ModelServer(settings.model_type, settings.model_path)
    yield
    model_server = None

app = FastAPI(
    title="PC-SAFT Parameter Prediction API",
    description="Predict PC-SAFT parameters (m, σ, ε/k) from molecular SMILES",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model_server is not None,
        model_name=model_server.model_type,
        version="0.1.0",
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    results = model_server.predict(request.smiles)
    return PredictionResponse(
        predictions=results,
        model_name=model_server.model_type,
        model_version="0.1.0",
    )

@app.post("/submit-data", status_code=201)
async def submit_data(submission: DataSubmission):
    # For now, append to a CSV. Step 7 replaces this with a proper DB.
    ...
    return {"status": "accepted", "smiles": submission.smiles}
```

### 5. Configuration (`serving/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_type: str = "nn"
    model_path: str = "model/saved/nn_pcsaft.pt"
    submissions_path: str = "serving/submissions.csv"
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_prefix = "PCSAFT_"

settings = Settings()
```

This lets you override settings via environment variables (`PCSAFT_MODEL_TYPE=chemberta`), which is how Kubernetes ConfigMaps inject configuration.

### 6. Running locally

```bash
pip install fastapi uvicorn pydantic-settings
uvicorn serving.app:app --reload --host 0.0.0.0 --port 8000
```

Then test:

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["C1CCCC1", "CC(F)=CF"]}'

# Submit experimental data
curl -X POST http://localhost:8000/submit-data \
  -H "Content-Type: application/json" \
  -d '{"smiles": "C1CCCC1", "m": 2.3655, "sigma": 3.7114, "epsilon_k": 288.84, "source": "Gross2001"}'
```

### 7. Interactive docs

FastAPI auto-generates OpenAPI docs at `http://localhost:8000/docs`. This is a free, interactive API explorer that's great for demos.

### 8. Error handling

```python
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal prediction error", "type": type(exc).__name__},
    )
```

Validate SMILES before prediction and return 422 with a clear message for invalid inputs.

## Evaluation & Success Criteria

### Functional checks

- [ ] `GET /health` returns 200 with model info
- [ ] `POST /predict` with valid SMILES returns predictions with correct schema
- [ ] `POST /predict` with invalid SMILES returns `valid: false` for those molecules (not a 500)
- [ ] `POST /predict` with empty list returns 422 validation error
- [ ] `POST /submit-data` with valid data returns 201
- [ ] `POST /submit-data` with negative σ returns 422
- [ ] Predictions match the offline model output (numerical equivalence check)

### Performance checks

- [ ] Single-molecule prediction responds in < 500ms
- [ ] Batch of 100 molecules responds in < 5 seconds
- [ ] Server starts up in < 30 seconds (including model loading)

### What success looks like

- A colleague can predict PC-SAFT parameters without installing Python, RDKit, or any ML libraries -- just `curl`
- The API docs at `/docs` are self-explanatory
- Environment variable configuration works (you can switch models without code changes)

### When to move to Step 6

You're ready for Step 6 when:

1. All functional checks pass
2. The API runs stably for 10+ minutes under repeated requests without memory leaks
3. Configuration is externalized (env vars, not hardcoded paths)
4. You have a clear mental model of what needs to be in the Docker image vs. what gets injected at runtime
