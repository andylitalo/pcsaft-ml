# Results Report: Similarity Search Endpoint and Portal UI

**Date**: 2026-03-16
**Step**: 33
**Objective**: Add `POST /similar` endpoint to FastAPI serving layer and corresponding Streamlit portal UI for finding similar molecules in the prediction database

## Implementation Summary

Added a complete similarity search system that allows users to find molecules with similar PC-SAFT parameters or chemical structure from two distinct corpora:

1. **Reference molecules** (n=24): Curated molecules with known PC-SAFT parameters from the Esper dataset
2. **Novel predictions** (n=4,663): Model-generated screening library from `data/pcsaft_novel_predictions_v1.csv`

The implementation distinguishes between these corpora to avoid implying that model-generated neighbors are experimentally validated.

## Key Features

### 1. FastAPI Endpoint: `POST /similar`

**Request schema**:
- `smiles` (optional): Query molecule by SMILES string
- `m`, `sigma`, `epsilon_k` (optional): Query by PC-SAFT parameter vector
- `k` (default 10): Number of neighbors to return
- `corpus` (default "all"): Search corpus ("reference", "novel", or "all")
- `metric` (default "parameter"): Distance metric ("parameter", "tanimoto", or "both")

At least one of `smiles` or `(m, sigma, epsilon_k)` must be provided. If SMILES is given without parameters, the model predicts parameters first.

**Response schema**:
- `query_smiles`: Query SMILES (if provided)
- `query_params`: Query PC-SAFT parameters (dict with m, sigma, epsilon_k)
- `corpus`: Search corpus used
- `corpus_version`: Versioned identifier for the search corpus (e.g., "pcsaft_novel_predictions_v1")
- `neighbors`: List of similar molecules with:
  - `smiles`, `corpus`, `m`, `sigma`, `epsilon_k`
  - `parameter_distance` (if metric includes "parameter")
  - `tanimoto_similarity` (if metric includes "tanimoto")
  - `mol_class`, `boiling_point_K` (if available)
- `metric`: Distance metric used

### 2. Distance Metrics

**Parameter-space distance**: Normalized Euclidean distance in (m, σ, ε/k) space using relative normalization (divide by reference value):

```
d = sqrt(((m - q_m) / q_m)^2 + ((σ - q_σ) / q_σ)^2 + ((ε/k - q_ε) / q_ε)^2)
```

This approach handles different parameter scales without requiring global statistics. Equal weights (1.0, 1.0, 1.0) after normalization.

**Tanimoto similarity**: Morgan fingerprint (radius 2, 2048 bits) similarity using RDKit's `BulkTanimotoSimilarity` for vectorized computation. All fingerprints pre-computed at startup for fast search.

**Combined metric ("both")**: Lexicographic sort by parameter distance (ascending), then tanimoto similarity (descending) to break ties.

### 3. Corpus Versioning

Each search response includes `corpus_version` to track which prediction library is being searched. This ensures that when the model is updated (Step 31+), the search corpus is also versioned and regenerated rather than silently searching stale predictions.

Current versions:
- Reference: `reference_molecules_v1`
- Novel: `pcsaft_novel_predictions_v1`
- All: `reference_molecules_v1+pcsaft_novel_predictions_v1`

### 4. Portal UI Integration

Added "Find Similar Molecules" expander in `portal/components/predictor.py` after each prediction result:

- Slider for k (5-50 neighbors)
- Selectbox for metric (parameter, tanimoto, both)
- Selectbox for corpus (all, reference, novel)
- Results displayed as DataFrame with SMILES, corpus, parameters, and distances
- Caption shows corpus version for transparency

### 5. Local Fallback Mode

Implemented local similarity search in `portal/api_client.py` for when no API server is running. Uses the same corpus separation and distance logic as the API endpoint to ensure consistent behavior between local and deployed modes.

## Test Coverage

All 11 tests pass in `tests/test_similarity.py`:

| Test | Purpose |
|------|---------|
| `test_similar_by_parameter` | Query with known PC-SAFT parameters, verify closest neighbor is reasonable |
| `test_similar_by_smiles` | Query with a SMILES, verify it returns itself as closest match (distance ~0) |
| `test_similar_tanimoto` | Verify Tanimoto similarity is in valid range [0, 1] |
| `test_similar_k_limit` | Verify at most k neighbors returned |
| `test_similar_invalid_smiles` | Graceful error handling for invalid SMILES |
| `test_similar_parameter_only` | Query with parameters only (no SMILES) works correctly |
| `test_similar_corpus_selection` | Verify reference, novel, and all return expected source sets |
| `test_similar_response_versioning` | Verify `corpus_version` is returned and correct |
| `test_similar_both_metric` | Combined metric returns both parameter_distance and tanimoto_similarity |
| `test_similar_missing_inputs` | Error when neither smiles nor parameters provided |
| `test_similar_tanimoto_without_smiles` | Error when using tanimoto without SMILES |

## Example Queries

### Query by SMILES (Cyclopentane)

```bash
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "C1CCCC1",
    "k": 5,
    "corpus": "all",
    "metric": "parameter"
  }'
```

Returns neighbors with similar PC-SAFT parameters, e.g., cyclohexane, methylcyclohexane.

### Query by Parameters

```bash
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{
    "m": 2.26,
    "sigma": 3.75,
    "epsilon_k": 273.0,
    "k": 5,
    "corpus": "reference",
    "metric": "parameter"
  }'
```

Returns reference molecules with similar parameters without requiring a SMILES string.

### Query with Tanimoto Similarity

```bash
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "c1ccccc1",
    "k": 5,
    "corpus": "all",
    "metric": "tanimoto"
  }'
```

Returns structurally similar aromatics (toluene, ethylbenzene, etc.).

## Corpus Provenance

The endpoint clearly distinguishes between:

1. **Reference molecules**: Curated set with experimental PC-SAFT parameters from Esper et al. (2018). These are known-good molecules used for training and validation.

2. **Novel predictions**: Model-generated screening library (4,663 molecules). Parameters are PREDICTED, not measured. Intended for screening and prioritization, NOT direct engineering use without validation.

The `corpus` field in each neighbor response ensures users can distinguish model predictions from experimental data. The UI help text reinforces this distinction.

## Performance Optimizations

1. **Pre-computed fingerprints**: All Morgan fingerprints computed at startup and stored in `app.state`, avoiding repeated computation during search
2. **BulkTanimotoSimilarity**: Vectorized C++ implementation for fast fingerprint comparison
3. **Relative normalization**: Parameter distance uses relative normalization (divide by reference value) rather than global statistics, avoiding need to recompute normalization on every search

## Deviations from Step Guide

None. Implementation follows the specification exactly, including:
- Corpus separation (reference vs novel)
- Explicit versioning (`corpus_version` in response)
- Local fallback mode in portal
- Distance metric definitions documented in code
- All requested tests

## Readiness Check

- [x] `POST /similar` endpoint returns correct results for known test molecules
- [x] Parameter-space and Tanimoto similarity both work
- [x] Search results clearly distinguish reference molecules from model-generated novel molecules
- [x] Search responses include corpus/version metadata
- [x] Portal shows similarity results inline after prediction
- [x] Local fallback mode works without API server
- [x] All tests pass (11/11)
- [x] `ruff check .` passes
- [x] Report documents corpus provenance, metric definitions, and example results

## Next Steps

Step 33 is complete. The similarity search infrastructure is ready for:

1. **Step 34+**: Integration with screening workflows to find alternatives to known refrigerants
2. **Model updates**: When Step 31 GNN predictions replace RF predictions, regenerate `data/pcsaft_novel_predictions_v1.csv` and version it as `v2` to track which model generated the search corpus
3. **Performance**: If search becomes slow at scale, consider adding approximate nearest neighbor indexing (e.g., FAISS for fingerprints, KD-tree for parameters)

## Key Learnings

1. **Corpus versioning is critical**: Explicitly versioning the search corpus (via `corpus_version`) ensures that when the underlying model changes, users know which predictions they're searching. This prevents silently searching stale predictions after model updates.

2. **Relative normalization simplifies parameter distance**: Using `(x - ref) / ref` rather than global statistics makes the distance metric independent of dataset statistics and easier to reason about.

3. **Pre-computing fingerprints at startup**: Adds ~5 seconds to startup time but makes searches instant (ms rather than seconds for 4,663 molecules).

4. **Distinguishing reference from novel**: Critical for user trust. The UI and API make it clear that novel predictions are model-generated, not experimental data.
