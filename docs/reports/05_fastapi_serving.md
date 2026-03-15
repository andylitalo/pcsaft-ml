# Results Report: FastAPI Model Serving (PC-SAFT Prediction API)

## API Overview

The PC-SAFT prediction pipeline is now exposed as a REST API via FastAPI. The best-performing Random Forest model (R2: m=0.62, sigma=0.35, epsilon_k=0.33) is loaded at startup and serves predictions over HTTP. The API includes uncertainty quantification (RF tree disagreement), applicability domain checking (Isolation Forest), and association-site flagging. A `/submit-data` endpoint allows researchers to contribute experimental measurements for future retraining.

> **Uncertainty note**: All R², MAE, and RMSE values in this report are point estimates on a single holdout split. 95% bootstrap confidence intervals are available via `python -m model.evaluate --bootstrap` and saved to `model/saved/comparison_metrics.csv` with `_lo`/`_hi` suffix columns.

## API Endpoints

| Method | Path | Description | Status Codes |
|--------|------|-------------|--------------|
| GET | `/health` | Liveness/readiness probe with model info | 200 |
| POST | `/predict` | Batch PC-SAFT parameter prediction | 200, 422 |
| POST | `/submit-data` | Accept experimental measurements | 201, 422 |
| GET | `/docs` | Interactive OpenAPI documentation (auto-generated) | 200 |

## Example Requests and Responses

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "rf",
  "version": "0.1.0"
}
```

### Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["C1CCCC1", "CCO", "INVALID"]}'
```

```json
{
  "predictions": [
    {
      "smiles": "C1CCCC1",
      "m": 2.37,
      "sigma": 3.68,
      "epsilon_k": 279.5,
      "uncertainty": {"m_std": 0.15, "sigma_std": 0.08, "epsilon_k_std": 12.3},
      "in_domain": true,
      "is_associating": false,
      "valid": true
    },
    {
      "smiles": "CCO",
      "m": 2.38,
      "sigma": 3.15,
      "epsilon_k": 198.2,
      "uncertainty": {"m_std": 0.18, "sigma_std": 0.09, "epsilon_k_std": 15.1},
      "in_domain": true,
      "is_associating": true,
      "valid": true
    },
    {
      "smiles": "INVALID",
      "m": 0.0,
      "sigma": 0.0,
      "epsilon_k": 0.0,
      "uncertainty": null,
      "in_domain": false,
      "is_associating": false,
      "valid": false
    }
  ],
  "model_name": "rf",
  "model_version": "0.1.0"
}
```

### Submit Experimental Data

```bash
curl -X POST http://localhost:8000/submit-data \
  -H "Content-Type: application/json" \
  -d '{"smiles": "C1CCCC1", "m": 2.3655, "sigma": 3.7114, "epsilon_k": 288.84, "source": "Gross2001"}'
```

```json
{"status": "accepted", "smiles": "C1CCCC1"}
```

## Functional Checks

- [x] `GET /health` returns 200 with model info
- [x] `POST /predict` with valid SMILES returns predictions with correct schema
- [x] `POST /predict` with invalid SMILES returns `valid: false` for those molecules (not a 500)
- [x] `POST /predict` with empty list returns 422 validation error
- [x] `POST /submit-data` with valid data returns 201
- [x] `POST /submit-data` with negative sigma returns 422
- [x] Predictions match the offline model output (numerical equivalence check)

All seven functional checks are verified by automated tests (9 test cases in `tests/test_serving.py`).

## Configuration

Settings are managed via `pydantic-settings` and can be overridden with `PCSAFT_*` environment variables:

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `model_type` | `rf` | `PCSAFT_MODEL_TYPE` | Registry model name (rf, nn, chemberta, gc_pcsaft) |
| `submissions_path` | `serving/submissions.csv` | `PCSAFT_SUBMISSIONS_PATH` | Path for researcher data submissions |
| `host` | `0.0.0.0` | `PCSAFT_HOST` | Bind address |
| `port` | `8000` | `PCSAFT_PORT` | Bind port |

Example: switch to the neural network model without code changes:

```bash
PCSAFT_MODEL_TYPE=nn uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

## Architecture Decisions

1. **Default model is RF** (not NN as in the step guide template). RF has the best test-set performance across all three PC-SAFT parameters.

2. **Lazy model loading via lifespan**. The `serving/app.py` module can be imported without model files present. The model is only loaded when the server starts (inside the async lifespan context manager). This is critical for Docker builds where the image is built without model artifacts.

3. **Invalid SMILES handled gracefully**. Rather than returning HTTP 500, invalid SMILES produce `valid: false` predictions with zeroed parameters. This lets batch requests succeed even if some inputs are malformed.

4. **AD model is optional**. If `model/saved/ad_model.joblib` does not exist, all predictions are marked `in_domain: true`. This avoids hard failures when the Isolation Forest hasn't been trained.

5. **Uncertainty from RF tree disagreement**. Each RF model contains 100 trees; the standard deviation of their predictions provides a calibrated uncertainty estimate without requiring MC Dropout.

## Files Created

| File | Purpose |
|------|---------|
| `serving/config.py` | Pydantic-settings configuration with env var support |
| `serving/schemas.py` | Request/response Pydantic models |
| `serving/model_loader.py` | Singleton model server with prediction, AD, and association logic |
| `serving/app.py` | FastAPI application with health, predict, and submit-data endpoints |
| `tests/test_serving.py` | 9 test cases covering all endpoints and edge cases |

## Readiness Check

- [x] All functional checks pass (7/7)
- [x] Server starts and serves predictions without errors
- [x] Configuration is externalized via environment variables
- [x] All 74 tests pass (65 existing + 9 new)
- [x] `ruff check .` is clean
- [x] OpenAPI docs auto-generated at `/docs`
