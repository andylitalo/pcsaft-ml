"""FastAPI application for PC-SAFT parameter prediction."""

import csv
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from serving.config import settings
from serving.schemas import (
    DataSubmission,
    HealthResponse,
    MoleculePrediction,
    PredictionRequest,
    PredictionResponse,
    UncertaintyEstimate,
)

logger = logging.getLogger(__name__)

_model_server = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup and release at shutdown."""
    global _model_server
    from serving.model_loader import ModelServer

    logger.info("Loading model: %s", settings.model_type)
    _model_server = ModelServer(settings.model_type)
    logger.info("Model loaded successfully")
    yield
    _model_server = None
    logger.info("Model server shut down")


app = FastAPI(
    title="PC-SAFT Parameter Prediction API",
    description="Predict PC-SAFT parameters (m, sigma, epsilon/k) from molecular SMILES",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for liveness/readiness probes."""
    return HealthResponse(
        status="healthy" if _model_server is not None else "unhealthy",
        model_loaded=_model_server is not None,
        model_name=_model_server.model_type if _model_server else "none",
        version="0.1.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict PC-SAFT parameters for a batch of SMILES strings."""
    raw_preds = _model_server.predict(request.smiles)
    predictions = []
    for p in raw_preds:
        uncertainty = None
        if p["uncertainty"] is not None:
            uncertainty = UncertaintyEstimate(**p["uncertainty"])
        predictions.append(
            MoleculePrediction(
                smiles=p["smiles"],
                m=p["m"],
                sigma=p["sigma"],
                epsilon_k=p["epsilon_k"],
                uncertainty=uncertainty,
                in_domain=p["in_domain"],
                is_associating=p["is_associating"],
                valid=p["valid"],
            )
        )
    return PredictionResponse(
        predictions=predictions,
        model_name=_model_server.model_type,
        model_version="0.1.0",
    )


@app.post("/submit-data", status_code=201)
async def submit_data(submission: DataSubmission):
    """Accept experimental PC-SAFT data from researchers.

    Appends to a local CSV file. Step 7 replaces this with a proper database.
    """
    csv_path = Path(settings.submissions_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["smiles", "m", "sigma", "epsilon_k", "source"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(submission.model_dump())

    logger.info("Submission accepted: %s from %s", submission.smiles, submission.source)
    return {"status": "accepted", "smiles": submission.smiles}


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Catch-all handler so internal errors return JSON, not HTML tracebacks."""
    logger.exception("Unhandled exception in %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal prediction error", "type": type(exc).__name__},
    )
