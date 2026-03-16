# Step 33: Similarity Search Endpoint and Portal UI

## Objective

Add a `POST /similar` endpoint to the FastAPI serving layer and a corresponding "Similar Molecules" UI section in the Streamlit portal. Given a query molecule (by SMILES or PC-SAFT parameter vector), return the k nearest neighbors from the 4,663-molecule prediction database.

## Motivation

The hosted UI should support two primary workflows:
1. **Predict**: Enter a SMILES, get PC-SAFT parameters (already implemented)
2. **Search**: Find molecules in the prediction database that are structurally or thermodynamically similar to a query

Similarity search enables researchers to:
- Discover alternatives to a known refrigerant by finding structurally similar candidates with better properties
- Validate predictions by comparing to nearby molecules with similar parameters
- Explore the prediction database without scanning thousands of rows manually

## Dependencies

- `serving/app.py` exists with `/predict` endpoint
- `serving/schemas.py` exists with Pydantic models
- `portal/app.py` exists with prediction tab
- `portal/components/predictor.py` exists
- `data/pcsaft_novel_predictions_v1.csv` exists (4,663 molecules)
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
    metric: str = "parameter"       # "parameter" or "tanimoto" or "both"
```

At least one of `smiles` or `(m, sigma, epsilon_k)` must be provided. If `smiles` is given and `metric` includes `"tanimoto"`, compute Morgan fingerprint similarity. If parameters are given (or derived from SMILES via the prediction model), compute normalized Euclidean distance in (m, σ, ε/k) space.

**Response schema**:

```python
class SimilarMolecule(BaseModel):
    smiles: str
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
    neighbors: list[SimilarMolecule]
    metric: str
```

### 33.2 Load prediction database at startup

In `serving/app.py`, load the predictions CSV at application startup:

```python
from pathlib import Path
import pandas as pd

PREDICTIONS_CSV = Path("data/pcsaft_novel_predictions_v1.csv")

@app.on_event("startup")
async def load_predictions_db():
    app.state.predictions_db = pd.read_csv(
        PREDICTIONS_CSV, comment="#"
    )
```

Skip lines starting with `#` (header comments in the CSV).

### 33.3 Implement distance metrics

**Parameter-space distance**: Normalized Euclidean distance using the same normalization as `hfo_parameter_distance` in `screening/hfo_screening.py`:

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

def bulk_tanimoto(query_smiles, db_smiles_list):
    """Compute Tanimoto similarity between query and all DB molecules."""
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return [0.0] * len(db_smiles_list)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
    similarities = []
    for smi in db_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            similarities.append(0.0)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        similarities.append(DataStructs.TanimotoSimilarity(query_fp, fp))
    return similarities
```

For performance, pre-compute Morgan fingerprints for all 4,663 database molecules at startup and store them in `app.state`.

### 33.4 Endpoint implementation

```python
@app.post("/similar", response_model=SimilarityResponse)
async def find_similar(request: SimilarityRequest):
    db = app.state.predictions_db

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
        results["tanimoto_similarity"] = bulk_tanimoto(request.smiles, db["smiles"].tolist())

    # Sort and return top-k
    if request.metric == "tanimoto":
        results = results.sort_values("tanimoto_similarity", ascending=False)
    else:
        results = results.sort_values("parameter_distance", ascending=True)

    top_k = results.head(request.k)
    # Convert to response model ...
```

### 33.5 Portal UI: Similar Molecules section

In `portal/components/predictor.py`, after the prediction result is displayed, add a "Similar Molecules" expandable section:

```python
with st.expander("Similar Molecules in Prediction Database"):
    k = st.slider("Number of neighbors", 5, 50, 10)
    metric = st.selectbox("Similarity metric", ["parameter", "tanimoto", "both"])

    if st.button("Find Similar"):
        response = api_client.find_similar(
            smiles=smiles_input,
            k=k,
            metric=metric
        )
        st.dataframe(response.neighbors)
```

Add a `find_similar()` method to `portal/api_client.py` that calls `POST /similar`.

If running in local fallback mode (no API server), implement the similarity search directly using the CSV and the same distance functions.

### 33.6 Tests

Create tests in `tests/test_similarity.py`:

1. `test_similar_by_parameter`: Query with known PC-SAFT parameters, verify closest neighbor is structurally reasonable
2. `test_similar_by_smiles`: Query with a SMILES in the database, verify it returns itself as the closest match (distance ~0)
3. `test_similar_tanimoto`: Query with known SMILES, verify Tanimoto similarity is between 0 and 1
4. `test_similar_k_limit`: Verify exactly k neighbors returned
5. `test_similar_invalid_smiles`: Query with invalid SMILES, verify graceful error
6. `test_similar_parameter_only`: Query with parameters only (no SMILES), verify parameter distance works

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
- [ ] Portal shows similarity results inline after prediction
- [ ] Local fallback mode works without API server
- [ ] All tests pass
- [ ] `ruff check .` passes
- [ ] Report documents endpoint usage and example results

## When to Move On

- `/similar` endpoint returns correct, ranked results
- Portal displays similarity table after prediction
- Tests cover both distance metrics and edge cases

## Budget

20 minutes. Primarily endpoint wiring and UI integration; no model training.
