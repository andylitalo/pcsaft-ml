# PC-SAFT Parameter Prediction Portal

Streamlit-based web interface for predicting PC-SAFT equation-of-state parameters from molecular SMILES strings.

## Quick Start

### Local Development

```bash
# Terminal 1: Start the FastAPI backend
uvicorn serving.app:app --reload

# Terminal 2: Start the Streamlit portal
streamlit run portal/app.py
```

The portal will open in your browser at http://localhost:8501

### Custom API URL

```bash
# Connect to a remote API
PCSAFT_API_URL=http://api.example.com:8000 streamlit run portal/app.py
```

## Features

### 1. Predict Parameters
- Enter SMILES strings (one per line) or upload a CSV file
- View 2D molecular structures rendered via RDKit
- Compare predictions to cyclopentane reference (m=2.37, σ=3.71 Å, ε/k=289 K)
- Export results as CSV

### 2. Submit Experimental Data
- Contribute measured PC-SAFT parameters to improve the model
- Validated input fields with range checks
- Data is stored and used in the next training cycle

### 3. Model Dashboard
- Current model performance metrics (R², MAE)
- System health status
- Training data statistics

## Architecture

```
portal/
├── app.py                    # Main Streamlit application
├── api_client.py             # FastAPI client wrapper
└── components/
    ├── predictor.py          # Prediction page UI
    ├── submitter.py          # Data submission form
    ├── dashboard.py          # Model metrics dashboard
    └── molecule.py           # Molecule rendering utilities
```

## API Integration

The portal communicates with the FastAPI service via three endpoints:

- `GET /health` - System health and model status
- `POST /predict` - Batch parameter prediction
- `POST /submit-data` - Experimental data contribution

When the API is unreachable, the portal displays cached baseline metrics and helpful error messages.

## Deployment

### Docker

```bash
# Build portal image
docker build -t pcsaft-portal:latest -f Dockerfile.portal .

# Run with API connection
docker run -p 8501:8501 \
  -e PCSAFT_API_URL=http://pcsaft-api:8000 \
  pcsaft-portal:latest
```

### Kubernetes

See `k8s/portal-deployment.yaml` for manifest. Deploy alongside the API service:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/portal-deployment.yaml
```

The portal pod reads `PCSAFT_API_URL` from a ConfigMap to reach the in-cluster service.

## Testing

```bash
# Run portal-specific tests
pytest tests/test_portal.py -v

# Run full test suite
pytest tests/ -v
```

All tests mock HTTP requests to avoid requiring a live API server.

## Configuration

The portal reads these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PCSAFT_API_URL` | `http://localhost:8000` | FastAPI service URL |

## Troubleshooting

### "Cannot connect to API server"

The portal cannot reach the FastAPI service. Verify:

1. API is running: `curl http://localhost:8000/health`
2. API URL is correct: `echo $PCSAFT_API_URL`
3. Network connectivity (especially in Kubernetes)

When the API is down, the dashboard shows cached baseline metrics.

### Invalid SMILES rendering

Invalid SMILES strings render as gray placeholder images (240, 240, 240). Check that your SMILES is RDKit-parseable:

```python
from rdkit import Chem
mol = Chem.MolFromSmiles("YOUR_SMILES")
assert mol is not None
```

### Slow CSV uploads

Large CSV files (1000+ molecules) may take several seconds. The portal processes batches synchronously. For production use, submit large batches directly to the API and poll for results.

## License

This is a demonstration project for interview purposes.
