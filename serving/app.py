"""FastAPI application for PC-SAFT parameter prediction."""

import csv
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from serving.config import settings
from serving.schemas import (
    DataSubmission,
    HealthResponse,
    MoleculePrediction,
    PredictionRequest,
    PredictionResponse,
    ReferenceMolecule,
    SimilarityRequest,
    SimilarityResponse,
    SimilarMolecule,
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


def precompute_fingerprints(smiles_list: list[str]) -> list:
    """Pre-compute Morgan fingerprints for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of RDKit ExplicitBitVect objects (or None for invalid SMILES)
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fps


def bulk_tanimoto(query_smiles: str, db_fps: list) -> list[float]:
    """Compute Tanimoto similarity between query and pre-computed DB fingerprints.

    Args:
        query_smiles: Query SMILES string
        db_fps: List of pre-computed Morgan fingerprints

    Returns:
        List of Tanimoto similarities (0.0 for invalid SMILES)
    """
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return [0.0] * len(db_fps)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
    valid_fps = [fp for fp in db_fps if fp is not None]
    if not valid_fps:
        return [0.0] * len(db_fps)
    bulk_sims = DataStructs.BulkTanimotoSimilarity(query_fp, valid_fps)
    results = []
    bulk_idx = 0
    for fp in db_fps:
        if fp is None:
            results.append(0.0)
        else:
            results.append(bulk_sims[bulk_idx])
            bulk_idx += 1
    return results


def parameter_distance(query_params: dict, db_params: pd.DataFrame) -> np.ndarray:
    """Compute normalized Euclidean distance in (m, sigma, epsilon_k) space.

    Uses relative normalization (divide by reference value) to handle different scales.
    Weights: m=1.0, sigma=1.0, epsilon_k=1.0 (equal weighting after normalization).

    Args:
        query_params: Dict with keys m, sigma, epsilon_k
        db_params: DataFrame with columns m, sigma, epsilon_k

    Returns:
        Array of distances
    """
    q_m = query_params["m"]
    q_sigma = query_params["sigma"]
    q_eps = query_params["epsilon_k"]

    db_m = db_params["m"].values
    db_sigma = db_params["sigma"].values
    db_eps = db_params["epsilon_k"].values

    # Relative distance (normalized by reference values)
    dm = ((db_m - q_m) / q_m) ** 2
    ds = ((db_sigma - q_sigma) / q_sigma) ** 2
    de = ((db_eps - q_eps) / q_eps) ** 2

    return np.sqrt(dm + ds + de)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup and release at shutdown."""
    global _model_server
    from serving.model_loader import ModelServer

    logger.info("Loading model: %s", settings.model_type)
    _model_server = ModelServer(settings.model_type)
    logger.info("Model loaded successfully")

    # Load search corpora
    predictions_csv = Path("data/pcsaft_novel_predictions_v1.csv")
    logger.info("Loading search corpora...")

    # Load novel predictions (skip comment lines)
    app.state.novel_predictions_db = pd.read_csv(predictions_csv, comment="#")
    app.state.novel_predictions_version = "pcsaft_novel_predictions_v1"

    # Load reference molecules
    app.state.reference_db = pd.DataFrame([r.model_dump() for r in REFERENCE_MOLECULES])

    # Pre-compute fingerprints for fast similarity search
    logger.info("Pre-computing Morgan fingerprints for novel predictions...")
    app.state.novel_fps = precompute_fingerprints(
        app.state.novel_predictions_db["smiles"].tolist()
    )

    logger.info("Pre-computing Morgan fingerprints for reference molecules...")
    app.state.reference_fps = precompute_fingerprints(
        app.state.reference_db["smiles"].tolist()
    )

    logger.info(
        f"Search corpora loaded: {len(app.state.novel_predictions_db)} novel + "
        f"{len(app.state.reference_db)} reference molecules"
    )

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
                tanimoto_nn=p.get("tanimoto_nn"),
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


@app.post("/similar", response_model=SimilarityResponse)
async def find_similar(request: SimilarityRequest):
    """Find similar molecules by PC-SAFT parameters or structural similarity.

    At least one of `smiles` or `(m, sigma, epsilon_k)` must be provided.
    If SMILES is given without parameters, parameters are predicted first.
    """
    # Validate input
    has_smiles = request.smiles is not None
    has_params = all(
        [request.m is not None, request.sigma is not None, request.epsilon_k is not None]
    )

    if not has_smiles and not has_params:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'smiles' or all three parameters (m, sigma, epsilon_k)",
        )

    # Select corpus
    if request.corpus == "reference":
        db = app.state.reference_db.copy()
        db_fps = app.state.reference_fps
        corpus_version = "reference_molecules_v1"
    elif request.corpus == "novel":
        db = app.state.novel_predictions_db.copy()
        db_fps = app.state.novel_fps
        corpus_version = app.state.novel_predictions_version
    else:  # "all"
        db = pd.concat(
            [app.state.reference_db, app.state.novel_predictions_db],
            ignore_index=True,
        )
        db_fps = app.state.reference_fps + app.state.novel_fps
        corpus_version = f"reference_molecules_v1+{app.state.novel_predictions_version}"

    # Add corpus label to distinguish reference vs novel in results
    if request.corpus == "all":
        corpus_labels = (
            ["reference"] * len(app.state.reference_db)
            + ["novel"] * len(app.state.novel_predictions_db)
        )
        db["corpus"] = corpus_labels
    else:
        db["corpus"] = request.corpus

    # Resolve query parameters
    if has_smiles and not has_params:
        # Predict parameters from SMILES
        pred_result = _model_server.predict([request.smiles])[0]
        if not pred_result["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid SMILES: {request.smiles}")
        query_params = {
            "m": pred_result["m"],
            "sigma": pred_result["sigma"],
            "epsilon_k": pred_result["epsilon_k"],
        }
    elif has_params:
        query_params = {"m": request.m, "sigma": request.sigma, "epsilon_k": request.epsilon_k}
    else:
        # has_smiles but also has_params (both provided)
        query_params = {"m": request.m, "sigma": request.sigma, "epsilon_k": request.epsilon_k}

    # Compute distances/similarities
    if request.metric in ("parameter", "both"):
        db["parameter_distance"] = parameter_distance(query_params, db)

    if request.metric in ("tanimoto", "both"):
        if not has_smiles:
            raise HTTPException(
                status_code=400,
                detail="Tanimoto similarity requires 'smiles' to be provided",
            )
        db["tanimoto_similarity"] = bulk_tanimoto(request.smiles, db_fps)

    # Sort and return top-k
    if request.metric == "tanimoto":
        results = db.sort_values("tanimoto_similarity", ascending=False)
    elif request.metric == "both":
        # Lexicographic sort: parameter distance first, then tanimoto similarity
        results = db.sort_values(
            ["parameter_distance", "tanimoto_similarity"],
            ascending=[True, False],
        )
    else:
        results = db.sort_values("parameter_distance", ascending=True)

    top_k = results.head(request.k)

    # Convert to response model
    neighbors = []
    for _, row in top_k.iterrows():
        param_dist = (
            float(row["parameter_distance"]) if "parameter_distance" in row else None
        )
        tanimoto_sim = (
            float(row["tanimoto_similarity"]) if "tanimoto_similarity" in row else None
        )
        bp_k = float(row["boiling_point_K"]) if pd.notna(row.get("boiling_point_K")) else None
        neighbors.append(
            SimilarMolecule(
                smiles=row["smiles"],
                corpus=row["corpus"],
                m=float(row["m"]),
                sigma=float(row["sigma"]),
                epsilon_k=float(row["epsilon_k"]),
                parameter_distance=param_dist,
                tanimoto_similarity=tanimoto_sim,
                mol_class=row["mol_class"] if pd.notna(row.get("mol_class")) else None,
                boiling_point_K=bp_k,
            )
        )

    return SimilarityResponse(
        query_smiles=request.smiles,
        query_params=query_params if query_params else None,
        corpus=request.corpus,
        corpus_version=corpus_version,
        neighbors=neighbors,
        metric=request.metric,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Catch-all handler so internal errors return JSON, not HTML tracebacks."""
    logger.exception("Unhandled exception in %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal prediction error", "type": type(exc).__name__},
    )
