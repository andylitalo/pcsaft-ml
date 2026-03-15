"""Pydantic request/response schemas for the PC-SAFT prediction API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Batch prediction request: one or more SMILES strings."""

    smiles: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of SMILES strings to predict PC-SAFT parameters for",
        json_schema_extra={"examples": [["C1CCCC1", "CC(=O)O"]]},
    )


class UncertaintyEstimate(BaseModel):
    """Per-molecule prediction uncertainty (ensemble std)."""

    m_std: float = Field(description="Std dev of segment number prediction")
    sigma_std: float = Field(description="Std dev of segment diameter prediction")
    epsilon_k_std: float = Field(description="Std dev of dispersion energy prediction")


class MoleculePrediction(BaseModel):
    """Prediction result for a single molecule."""

    smiles: str
    m: float = Field(description="Segment number")
    sigma: float = Field(description="Segment diameter (Angstrom)")
    epsilon_k: float = Field(description="Dispersion energy (K)")
    uncertainty: UncertaintyEstimate | None = Field(
        None,
        description="Prediction uncertainty from ensemble / MC Dropout",
    )
    in_domain: bool = Field(
        True, description="Whether molecule is within the applicability domain"
    )
    tanimoto_nn: float | None = Field(
        None, description="Max Tanimoto similarity to training set (0-1)"
    )
    is_associating: bool = Field(
        False,
        description=(
            "Whether molecule has association sites (OH, NH, COOH) -- "
            "3-param PC-SAFT may be insufficient"
        ),
    )
    valid: bool = Field(description="Whether the SMILES was parseable and prediction succeeded")


class PredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[MoleculePrediction]
    model_name: str
    model_version: str


class DataSubmission(BaseModel):
    """Researcher-submitted experimental PC-SAFT measurement."""

    smiles: str
    m: float = Field(gt=0, description="Measured segment number")
    sigma: float = Field(gt=0, description="Measured segment diameter (Angstrom)")
    epsilon_k: float = Field(gt=0, description="Measured dispersion energy (K)")
    source: str = Field(description="Publication DOI or lab identifier")


class ReferenceMolecule(BaseModel):
    """Reference molecule with known PC-SAFT parameters."""

    name: str
    smiles: str
    m: float
    sigma: float
    epsilon_k: float
    source: str = "Esper"


class HealthResponse(BaseModel):
    """Health-check response for liveness/readiness probes."""

    status: str
    model_loaded: bool
    model_name: str
    version: str
