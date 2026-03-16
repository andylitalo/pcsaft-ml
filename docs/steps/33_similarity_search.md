# Step 33: Similarity Search Endpoint and Portal UI

## Objective

Add a `POST /similar` endpoint to the FastAPI serving layer and a corresponding "Similar Molecules" UI section in the Streamlit portal. Given a query molecule (by SMILES or PC-SAFT parameter vector), return the k nearest neighbors from an explicitly versioned search corpus. The implementation must distinguish between:

1. **Reference / known molecules**: curated molecules with known provenance
2. **Novel prediction library**: the 4,663-molecule model-generated screening set

This distinction is required so the portal helps users explore nearby chemistry without implying that model-generated neighbors are experimental validation.

## Motivation

The hosted UI should support two primary workflows:
1. **Predict**: Enter a SMILES, get PC-SAFT parameters (already implemented)
2. **Search**: Find molecules in the prediction database that are structurally or thermodynamically similar to a query

Similarity search enables researchers to:
- Discover alternatives to a known refrigerant by finding structurally similar candidates with better properties
- Contextualize predictions by comparing to nearby molecules with similar parameters
- Explore the prediction database without scanning thousands of rows manually

It should **not** imply that neighbors from the model-generated library are experimentally validated analogs.

## Dependencies

- `serving/app.py` exists with `/predict` endpoint
- `serving/schemas.py` exists with Pydantic models
- `portal/app.py` exists with prediction tab
- `portal/components/predictor.py` exists
- `data/pcsaft_novel_predictions_v1.csv` exists (4,663 molecules)
- `serving/app.py` already exposes curated reference molecules at `/reference-molecules`
- Soft dependency on Step 31: ideally the predictions database uses GNN predictions, but RF predictions work for the search infrastructure

## Implementation Guide

### 33.1 Add `POST /similar` endpoint to `serving/app.py`

**Request schema** (add to `serving/schemas.py`):

```python
class SimilarityRequest(BaseModel):
    smiles: str | None = None       # Query by structure
    m: float | None = None          # Query by parameter vector
    sigma: float | None = None
    epsilon_k: float | None = None
    k: int = 10                     # Number of neighbors
    corpus: str = "all"             # "reference" or "novel" or "all"
    metric: str = "parameter"       # "parameter" or "tanimoto" or "both"
```

At least one of `smiles` or `(m, sigma, epsilon_k)` must be provided. If `smiles` is given and `metric` includes `"tanimoto"`, compute Morgan fingerprint similarity. If parameters are given (or derived from SMILES via the prediction model), compute normalized Euclidean distance in (m, σ, ε/k) space.

**Response schema**:

```python
class SimilarMolecule(BaseModel):
    smiles: str
    corpus: str
    m: float
    sigma: float
    epsilon_k: float
    parameter_distance: float | None = None
    tanimoto_similarity: float | None = None
    mol_class: str | None = None
    boiling_point_K: float | None = None

class SimilarityResponse(BaseModel):
    query_smiles: str | None
    query_params: dict | None
    corpus: str
    corpus_version: str | None
    neighbors: list[SimilarMolecule]
    metric: str
```

### 33.2 Load search corpora at startup

In `serving/app.py`, load the search corpora at application startup:

The existing `serving/app.py` already uses the `lifespan` context manager pattern (not the deprecated `@app.on_event("startup")`). Add the corpus loading to the existing lifespan handler:

```python
from pathlib import Path
import pandas as pd

PREDICTIONS_CSV = Path("data/pcsaft_novel_predictions_v1.csv")

# Inside the existing lifespan context manager:
async def lifespan(app: FastAPI):
    # ... existing model loading ...
    app.state.novel_predictions_db = pd.read_csv(PREDICTIONS_CSV, comment="#")
    app.state.reference_db = pd.DataFrame([
        r.model_dump() for r in REFERENCE_MOLECULES
    ])
    app.state.novel_predictions_version = "pcsaft_novel_predictions_v1"
    app.state.novel_fps = precompute_fingerprints(
        app.state.novel_predictions_db["smiles"].tolist()
    )
    app.state.reference_fps = precompute_fingerprints(
        app.state.reference_db["smiles"].tolist()
    )
    yield
    # ... existing cleanup ...
```

Skip lines starting with `#` (header comments in the CSV).

The novel library version must be surfaced in the response so the UI can show which search index is being used. When Step 31 changes the default model or regenerates the prediction library, regenerate and version the search corpus as part of the same release rather than silently searching stale predictions.

### 33.3 Implement distance metrics

**Parameter-space distance**: Normalized Euclidean distance using a single shared normalization rule. Do not say "same as `hfo_parameter_distance`" unless both places literally call the same helper. Either:

1. move the distance logic into a shared utility used by both screening and serving, or
2. document the exact normalization and weights in both places and keep them consistent

Example:

```python
import numpy as np

def parameter_distance(query_params, db_params, weights=(1.0, 1.0, 3.0)):
    """Normalized Euclidean distance in (m, sigma, epsilon_k) space."""
    q = np.array([query_params["m"], query_params["sigma"], query_params["epsilon_k"]])
    db = db_params[["m", "sigma", "epsilon_k"]].values
    # Normalize by column std
    std = db.std(axis=0)
    std[std == 0] = 1.0
    diff = (db - q) / std
    w = np.array(weights)
    return np.sqrt((diff ** 2 * w).sum(axis=1))
```

**Tanimoto similarity**: Morgan fingerprint (radius 2, 2048 bits) between query and each database molecule. Use RDKit's `DataStructs.BulkTanimotoSimilarity` for efficiency:

```python
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def precompute_fingerprints(smiles_list: list[str]) -> list:
    """Pre-compute Morgan fingerprints for a list of SMILES."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fps

def bulk_tanimoto(query_smiles: str, db_fps: list) -> list[float]:
    """Compute Tanimoto similarity between query and pre-computed DB fingerprints."""
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
```

Pre-compute Morgan fingerprints for all searchable molecules at startup and store them in `app.state`, separated by corpus. `BulkTanimotoSimilarity` is vectorized in C++ and significantly faster than per-molecule `TanimotoSimilarity` calls.

### 33.4 Endpoint implementation

```python
@app.post("/similar", response_model=SimilarityResponse)
async def find_similar(request: SimilarityRequest):
    if request.corpus == "reference":
        db = app.state.reference_db
        corpus_version = "reference_molecules_v1"
    elif request.corpus == "novel":
        db = app.state.novel_predictions_db
        corpus_version = app.state.novel_predictions_version
    else:
        db = pd.concat(
            [app.state.reference_db, app.state.novel_predictions_db],
            ignore_index=True,
        )
        corpus_version = f"reference_molecules_v1+{app.state.novel_predictions_version}"

    # Resolve query parameters
    if request.smiles:
        # Predict params from SMILES if not provided
        if request.m is None:
            predicted = model_server.predict(request.smiles)
            query_params = {"m": predicted.m, "sigma": predicted.sigma, "epsilon_k": predicted.epsilon_k}
        else:
            query_params = {"m": request.m, "sigma": request.sigma, "epsilon_k": request.epsilon_k}
    else:
        query_params = {"m": request.m, "sigma": request.sigma, "epsilon_k": request.epsilon_k}

    # Compute distances/similarities
    results = db.copy()
    if request.metric in ("parameter", "both"):
        results["parameter_distance"] = parameter_distance(query_params, db)
    if request.metric in ("tanimoto", "both") and request.smiles:
        # Use pre-computed fingerprints from app.state for performance
        results["tanimoto_similarity"] = bulk_tanimoto(request.smiles, db_fps)

    # Sort and return top-k
    if request.metric == "tanimoto":
        results = results.sort_values("tanimoto_similarity", ascending=False)
    elif request.metric == "both":
        # Define and document one deterministic blended ranking rule.
        # Example: rank by parameter distance first, then break ties by
        # descending tanimoto similarity.
        results = results.sort_values(
            ["parameter_distance", "tanimoto_similarity"],
            ascending=[True, False],
        )
    else:
        results = results.sort_values("parameter_distance", ascending=True)

    top_k = results.head(request.k)
    # Convert to response model ...
```

For `metric="both"`, avoid ambiguous behavior. Either define a blended score or define a deterministic lexicographic sort rule and document it in both the API docstring and the report.

### 33.5 Portal UI: Similar Molecules section

In `portal/components/predictor.py`, after the prediction result is displayed, add a "Similar Molecules" expandable section:

```python
with st.expander("Similar Molecules in Prediction Database"):
    k = st.slider("Number of neighbors", 5, 50, 10)
    metric = st.selectbox("Similarity metric", ["parameter", "tanimoto", "both"])
    corpus = st.selectbox(
        "Search corpus",
        ["reference", "novel", "all"],
        help="Reference = curated known molecules; Novel = model-generated screening library.",
    )

    if st.button("Find Similar"):
        response = api_client.find_similar(
            smiles=smiles_input,
            k=k,
            metric=metric,
            corpus=corpus,
        )
        st.caption(f"Corpus: {response.corpus} | Version: {response.corpus_version}")
        st.dataframe(response.neighbors)
```

Add a `find_similar()` method to `portal/api_client.py` that calls `POST /similar`.

If running in local fallback mode (no API server), implement the similarity search directly using the same corpus separation and distance functions so local demos do not diverge from deployed behavior.

### 33.6 Tests

Create tests in `tests/test_similarity.py`:

1. `test_similar_by_parameter`: Query with known PC-SAFT parameters, verify closest neighbor is structurally reasonable
2. `test_similar_by_smiles`: Query with a SMILES in the selected corpus, verify it returns itself as the closest match (distance ~0)
3. `test_similar_tanimoto`: Query with known SMILES, verify Tanimoto similarity is between 0 and 1
4. `test_similar_k_limit`: Verify exactly k neighbors returned
5. `test_similar_invalid_smiles`: Query with invalid SMILES, verify graceful error
6. `test_similar_parameter_only`: Query with parameters only (no SMILES), verify parameter distance works
7. `test_similar_corpus_selection`: Verify `reference`, `novel`, and `all` return the expected source set
8. `test_similar_response_versioning`: Verify `corpus_version` is returned and changes when the novel library version changes

## Artifacts

| File | Description |
|------|-------------|
| `serving/app.py` | Updated with `/similar` endpoint |
| `serving/schemas.py` | New request/response models |
| `portal/components/predictor.py` | Similar molecules UI section |
| `portal/api_client.py` | `find_similar()` method |
| `tests/test_similarity.py` | Endpoint and logic tests |
| `docs/reports/33_similarity_search.md` | Step report |

## Success Criteria

- [ ] `POST /similar` endpoint returns correct results for known test molecules
- [ ] Parameter-space and Tanimoto similarity both work
- [ ] Search results clearly distinguish reference molecules from model-generated novel molecules
- [ ] Search responses include corpus/version metadata
- [ ] Portal shows similarity results inline after prediction
- [ ] Local fallback mode works without API server
- [ ] All tests pass
- [ ] `ruff check .` passes
- [ ] Report documents corpus provenance, metric definitions, and example results

## When to Move On

- `/similar` endpoint returns correct, ranked results
- Portal displays similarity results with provenance and corpus version
- Tests cover both distance metrics and edge cases

## Budget

20 minutes. Primarily endpoint wiring and UI integration; no model training.
