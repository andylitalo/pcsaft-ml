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
    ReferenceMolecule,
    UncertaintyEstimate,
)

logger = logging.getLogger(__name__)

_model_server = None

# Curated reference molecules with known PC-SAFT parameters from Esper dataset
REFERENCE_MOLECULES = [
    # Alkanes
    ReferenceMolecule(
        name="Propane", smiles="CCC", m=1.98602, sigma=3.6244, epsilon_k=209.08586
    ),
    ReferenceMolecule(
        name="n-Butane", smiles="CCCC", m=2.31121, sigma=3.71557, epsilon_k=224.06583
    ),
    ReferenceMolecule(
        name="Isobutane", smiles="CC(C)C", m=2.23943, sigma=3.77033, epsilon_k=217.60377
    ),
    ReferenceMolecule(
        name="n-Pentane", smiles="CCCCC", m=2.74869, sigma=3.72936, epsilon_k=228.73279
    ),
    ReferenceMolecule(
        name="n-Hexane", smiles="CCCCCC", m=3.06506, sigma=3.79083, epsilon_k=236.46956
    ),
    ReferenceMolecule(
        name="n-Heptane", smiles="CCCCCCC", m=3.49412, sigma=3.79257, epsilon_k=238.11279
    ),
    ReferenceMolecule(
        name="n-Octane", smiles="CCCCCCCC", m=3.86069, sigma=3.81486, epsilon_k=241.43398
    ),
    ReferenceMolecule(
        name="n-Decane", smiles="CCCCCCCCCC", m=4.6267, sigma=3.84109, epsilon_k=244.92287
    ),
    # Cycloalkanes
    ReferenceMolecule(
        name="Cyclopentane", smiles="C1CCCC1", m=2.25774, sigma=3.75308, epsilon_k=273.50029
    ),
    ReferenceMolecule(
        name="Cyclohexane", smiles="C1CCCCC1", m=2.50027, sigma=3.85128, epsilon_k=280.36899
    ),
    ReferenceMolecule(
        name="Methylcyclohexane",
        smiles="CC1CCCCC1",
        m=2.66045,
        sigma=3.9956,
        epsilon_k=282.31227,
    ),
    # Aromatics
    ReferenceMolecule(
        name="Benzene", smiles="c1ccccc1", m=2.51627, sigma=3.61064, epsilon_k=284.11055
    ),
    ReferenceMolecule(
        name="Toluene", smiles="Cc1ccccc1", m=2.78719, sigma=3.72273, epsilon_k=287.51747
    ),
    ReferenceMolecule(
        name="Ethylbenzene",
        smiles="CCc1ccccc1",
        m=3.0887,
        sigma=3.781,
        epsilon_k=287.08422,
    ),
    # Fluorocarbons (refrigerants)
    ReferenceMolecule(
        name="R-32 (difluoromethane)",
        smiles="FCF",
        m=2.45035,
        sigma=2.80959,
        epsilon_k=162.36264,
    ),
    ReferenceMolecule(
        name="R-134a (1,1,1,2-tetrafluoroethane)",
        smiles="FC(F)C(F)(F)F",
        m=3.07064,
        sigma=3.13545,
        epsilon_k=154.7744,
    ),
    ReferenceMolecule(
        name="R-152a (1,1-difluoroethane)",
        smiles="CC(F)F",
        m=2.57721,
        sigma=3.15564,
        epsilon_k=180.47936,
    ),
    ReferenceMolecule(
        name="HFO-1234yf (2,3,3,3-tetrafluoropropene)",
        smiles="FC(=C(F)F)C(F)(F)F",
        m=3.40869,
        sigma=3.22719,
        epsilon_k=159.26613,
    ),
    # Ethers
    ReferenceMolecule(
        name="Diethyl ether", smiles="CCOCC", m=2.97883, sigma=3.50108, epsilon_k=219.43339
    ),
    ReferenceMolecule(
        name="MTBE (methyl tert-butyl ether)",
        smiles="COC(C)(C)C",
        m=2.94782,
        sigma=3.70393,
        epsilon_k=233.07331,
    ),
    # Ketones
    ReferenceMolecule(
        name="Acetone", smiles="CC(C)=O", m=2.78618, sigma=3.24884, epsilon_k=231.11777
    ),
    ReferenceMolecule(
        name="MEK (2-butanone)", smiles="CCC(C)=O", m=2.98582, sigma=3.39936, epsilon_k=235.58738
    ),
    # Alcohols
    ReferenceMolecule(
        name="Methanol", smiles="CO", m=2.25965, sigma=2.83016, epsilon_k=183.58634
    ),
    ReferenceMolecule(
        name="Ethanol", smiles="CCO", m=2.8866, sigma=2.95772, epsilon_k=187.26028
    ),
]


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


@app.get("/reference-molecules", response_model=list[ReferenceMolecule])
async def list_reference_molecules():
    """List curated reference molecules with known PC-SAFT parameters.

    Returns a library of 20+ common molecules from the Esper dataset,
    covering alkanes, cycloalkanes, aromatics, fluorocarbons, ethers, ketones, and alcohols.
    """
    return REFERENCE_MOLECULES


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
