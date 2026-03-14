# Step 8: Streamlit Academic Portal

## Purpose

Build a researcher-facing web portal that makes the entire system accessible to domain scientists who don't write code. The portal provides a visual interface for predicting PC-SAFT parameters, exploring results, and contributing experimental data that feeds the online learning pipeline.

This is the user-facing capstone: it ties together the model (Steps 1–4), the API (Step 5), the deployment (Step 6), and the retrain pipeline (Step 7) into a single, usable product.

## What It Adds Over Step 7

| After Step 7 | After This Step |
|--------------|----------------|
| API requires `curl` or code to use | Visual web interface for non-programmers |
| No molecule visualization | 2D structure rendering in the browser |
| Raw JSON prediction output | Formatted results with comparison to cyclopentane |
| Data submission is a raw POST request | Guided form with validation and feedback |
| No visibility into model performance | Dashboard showing model metrics and training history |

## Skills Demonstrated

- **Streamlit**: Rapid prototyping of data-driven web applications
- **UX for scientific tools**: Making ML outputs interpretable for domain experts
- **End-to-end integration**: Frontend → API → model → pipeline
- **Data visualization**: Interactive plots, molecule rendering, similarity gauges

## Implementation Guide

### 1. Project structure

```
portal/
    __init__.py
    app.py              # Main Streamlit application
    api_client.py       # Wrapper around the FastAPI endpoints
    components/
        predictor.py    # Prediction input/output UI
        submitter.py    # Data submission form
        dashboard.py    # Model metrics dashboard
        molecule.py     # Molecule rendering utilities
```

### 2. Main application (`portal/app.py`)

```python
import streamlit as st

st.set_page_config(
    page_title="PC-SAFT Parameter Predictor",
    page_icon="🧪",
    layout="wide",
)

st.title("PC-SAFT Parameter Prediction Portal")

tab_predict, tab_submit, tab_dashboard = st.tabs([
    "Predict Parameters",
    "Submit Experimental Data",
    "Model Dashboard",
])
```

### 3. Prediction page (`portal/components/predictor.py`)

Key features:

- **SMILES input**: Text box for one or more SMILES strings, one per line
- **Molecule rendering**: Display 2D structure using RDKit's `Draw.MolToImage`
- **Results table**: Predicted m, σ, ε/k with comparison to cyclopentane
- **Similarity gauge**: Visual indicator of how close the candidate is to cyclopentane
- **Batch upload**: CSV file upload for bulk screening

```python
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io

def render_prediction_page(api_client):
    smiles_input = st.text_area(
        "Enter SMILES (one per line)",
        value="C1CCCC1\nCC(F)=CF",
        height=100,
    )

    if st.button("Predict", type="primary"):
        smiles_list = [s.strip() for s in smiles_input.strip().split("\n") if s.strip()]
        results = api_client.predict(smiles_list)

        for mol_result in results:
            col1, col2 = st.columns([1, 2])

            with col1:
                mol = Chem.MolFromSmiles(mol_result["smiles"])
                if mol:
                    img = Draw.MolToImage(mol, size=(250, 250))
                    st.image(img)

            with col2:
                st.markdown(f"**SMILES**: `{mol_result['smiles']}`")

                # Show predictions with comparison to cyclopentane
                cyc = {"m": 2.3655, "sigma": 3.7114, "epsilon_k": 288.84}
                for param in ["m", "sigma", "epsilon_k"]:
                    pred = mol_result[param]
                    ref = cyc[param]
                    pct_diff = abs(pred - ref) / ref * 100
                    st.metric(
                        label=param,
                        value=f"{pred:.4f}",
                        delta=f"{pct_diff:.1f}% from cyclopentane",
                        delta_color="inverse",
                    )
```

### 4. Data submission page (`portal/components/submitter.py`)

```python
def render_submission_page(api_client):
    st.markdown("""
    ### Contribute Experimental Data

    If you have experimentally measured PC-SAFT parameters, submit them here
    to help improve the model. Your data will be validated and incorporated
    into the next training cycle.
    """)

    with st.form("submission_form"):
        smiles = st.text_input("SMILES")
        col1, col2, col3 = st.columns(3)
        m = col1.number_input("m (segments)", min_value=0.5, max_value=10.0, step=0.01)
        sigma = col2.number_input("σ (Å)", min_value=2.0, max_value=6.0, step=0.01)
        epsilon_k = col3.number_input("ε/k (K)", min_value=50.0, max_value=600.0, step=0.1)
        source = st.text_input("Source (DOI or lab ID)")

        submitted = st.form_submit_button("Submit")
        if submitted:
            result = api_client.submit_data(smiles, m, sigma, epsilon_k, source)
            if result["status"] == "accepted":
                st.success("Data submitted successfully! It will be included in the next training cycle.")
            else:
                st.error(f"Submission rejected: {result.get('detail', 'Unknown error')}")
```

### 5. Dashboard page (`portal/components/dashboard.py`)

Show the current model's performance and history:

- **Current model metrics**: R², MAE, RMSE per target (fetched from a `/model-info` endpoint)
- **Training history**: Plot of R² over successive model versions (if model registry tracks this)
- **Data stats**: Number of training molecules, number of submitted data points, next retrain threshold
- **Recent submissions**: Table of the last 20 submitted data points and their validation status

```python
def render_dashboard(api_client):
    info = api_client.get_model_info()

    st.subheader("Current Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² (m)", f"{info['r2_m']:.3f}")
    col2.metric("R² (σ)", f"{info['r2_sigma']:.3f}")
    col3.metric("R² (ε/k)", f"{info['r2_epsilon_k']:.3f}")

    st.subheader("Training Data")
    st.metric("Total training molecules", info["n_training"])
    st.metric("Pending submissions", info["n_pending"])
    st.progress(info["n_pending"] / info["retrain_threshold"])
    st.caption(f"Retrain triggers at {info['retrain_threshold']} new data points")
```

### 6. API client (`portal/api_client.py`)

```python
import requests

class PCSAFTClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def predict(self, smiles_list: list[str]) -> list[dict]:
        resp = requests.post(f"{self.base_url}/predict", json={"smiles": smiles_list})
        resp.raise_for_status()
        return resp.json()["predictions"]

    def submit_data(self, smiles, m, sigma, epsilon_k, source):
        resp = requests.post(f"{self.base_url}/submit-data", json={
            "smiles": smiles, "m": m, "sigma": sigma,
            "epsilon_k": epsilon_k, "source": source,
        })
        return resp.json()

    def get_model_info(self):
        resp = requests.get(f"{self.base_url}/model-info")
        return resp.json()
```

### 7. Running the portal

```bash
pip install streamlit

# Locally (API must be running)
streamlit run portal/app.py

# With custom API URL
PCSAFT_API_URL=http://pcsaft-api.pcsaft.svc.cluster.local streamlit run portal/app.py
```

### 8. Deploy as a second K8s pod

Add a Streamlit Deployment to `k8s/`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pcsaft-portal
  namespace: pcsaft
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pcsaft-portal
  template:
    spec:
      containers:
        - name: portal
          image: pcsaft-portal:latest
          ports:
            - containerPort: 8501
          env:
            - name: PCSAFT_API_URL
              value: "http://pcsaft-api.pcsaft.svc.cluster.local"
```

### 9. Molecule drawing without RDKit (lighter alternative)

If you want to keep the portal image small (no RDKit), use the PubChem PUG REST API to fetch 2D depictions:

```python
def get_molecule_image(smiles: str) -> str:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/PNG"
    return url  # Use as st.image(url)
```

## Evaluation & Success Criteria

### Functional checks

- [ ] Prediction page: enter SMILES → see molecule image + predicted parameters
- [ ] Prediction page: batch CSV upload works for 50+ molecules
- [ ] Submission page: valid data is accepted and confirmed
- [ ] Submission page: invalid data (bad SMILES, out-of-range values) shows a clear error
- [ ] Dashboard: shows current model metrics
- [ ] Dashboard: progress bar toward next retrain threshold updates

### Usability checks

- [ ] A chemist with no Python knowledge can use the portal after 30 seconds of orientation
- [ ] The comparison to cyclopentane is immediately understandable
- [ ] Error messages are helpful, not stack traces
- [ ] Page loads in < 3 seconds

### End-to-end integration test

Run through the full cycle:

1. Open portal, predict PC-SAFT parameters for a known molecule
2. Verify predictions match the offline model output
3. Submit experimental data for 10+ molecules through the submission form
4. Trigger the Kubeflow retrain pipeline
5. After pipeline completes, verify the dashboard shows updated metrics
6. Predict again and observe (slightly) different predictions if the model was promoted

### What success looks like

- The portal is something you'd be proud to demo in an interview
- You can walk through the full story: "A researcher visits this page, gets predictions for their candidate molecules, and if they have experimental data, they contribute it back. The system validates their submission, retrains when enough data accumulates, and the predictions get better over time."
- The code is clean enough that an interviewer can follow it

### The Complete Picture

At the end of Step 8, you have:

```
Researcher → Streamlit portal → FastAPI → PyTorch NN / ChemBERTa → PC-SAFT predictions
                ↓ (submit data)
            FastAPI → submissions store → Kubeflow pipeline → retrain → promote → serving update
```

This is a legitimate MLOps system that demonstrates every skill on your list:
- **PyTorch**: Multi-task NN architecture and training (Steps 1–3)
- **HuggingFace**: ChemBERTa fine-tuning and Hub publishing (Step 4)
- **Kubernetes**: Container orchestration and deployment (Step 6)
- **Kubeflow**: ML pipeline automation and model management (Step 7)
- **Full-stack ML engineering**: API design, data validation, online learning, UX (Steps 5, 7, 8)
