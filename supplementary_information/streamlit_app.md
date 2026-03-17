# Supplementary Information: Streamlit Portal

## Overview

The Streamlit portal provides a researcher-facing web interface for predicting PC-SAFT equation-of-state parameters, screening molecular candidates, and contributing experimental data. It integrates the prediction model (Random Forest, GNN, or ChemBERTa), FastAPI backend, and retrain pipeline into a single web application accessible to domain scientists without requiring programming.

---

## Predict Parameters

The **Predict Parameters** tab enables single-molecule or batch prediction of PC-SAFT parameters (m, sigma, epsilon/k).

**Input:**
- Single SMILES string or multiple SMILES (one per line)
- Optionally, upload a CSV file containing a `smiles` column for batch prediction

**Output:**
- 2D molecular structure rendered via RDKit
- Predicted m (segment number), σ (segment diameter in Å), and ε/k (dispersion energy in K)
- Uncertainty estimates from tree-level variance when using the Random Forest model
- Parameter comparison against a selectable reference molecule (default: cyclopentane), with percent difference displayed
- Applicability domain score: Tanimoto similarity to the nearest training molecule (0–1 scale; ≥0.4 in-domain, 0.3–0.4 warning, &lt;0.3 out-of-domain)
- Association flag: molecules containing OH, NH, NH2, or COOH groups are flagged with a warning that 3-parameter PC-SAFT may be insufficient and the association variant should be considered

---

## Screen Candidates

The **Screen Candidates** tab supports batch screening of molecular libraries against application-specific criteria.

**Input:**
- CSV upload with a `smiles` column

**Filters (customizable):**
- Parameter ranges for m, sigma, and epsilon/k
- Molecular weight (MW) range
- Applicability domain threshold (require in-domain only)
- Tanimoto cutoff (minimum similarity to training set)

**Ranking:**
- Weighted Euclidean distance in normalized parameter space to a reference molecule
- Alternative ranking by Tanimoto (confidence proxy) or epsilon/k (solvation power proxy for solvents)

**Built-in presets:**
- **Blowing Agents**: Cyclopentane-like profiles; m 1.8–3.5, sigma 3.0–4.5 Å, epsilon/k 200–400 K; soft preferences for low GWP, low toxicity
- **Refrigerants**: HFC/HFO-like; reference R-134a; m 1.5–3.0, sigma 2.5–4.0 Å; soft preference for fluorinated
- **Industrial Solvents**: Green solvent analogs; reference ethyl acetate; m 1.5–4.0, sigma 3.0–5.0 Å; ranking by epsilon/k
- **General Exploration**: No application-specific filters; rank by Tanimoto (prediction confidence proxy)

Each preset includes caution text that thresholds are illustrative and not universally validated; thermodynamic and safety properties must be confirmed experimentally.

---

## Dashboard

The **Model Dashboard** tab displays model status and performance metrics.

**Content:**
- API health status
- Model type and load status
- R² scores for m, sigma, and epsilon/k (test set)
- MAE per parameter (m, σ in Å, ε/k in K)
- Total training set size
- Brief notes on parameter prediction difficulty and how to improve performance (e.g., submitting experimental data for retraining)

---

## Deployment

**Cloud Run:**
- Deployed to Google Cloud Run for scale-to-zero (no cost when idle)
- Uses `Dockerfile.portal`; builds the portal with FastAPI serving layer
- Public HTTPS URL via auto-assigned `*.run.app` domain
- Environment variable `PCSAFT_API_URL` points the portal to the backend

**Local:**
- Run via `streamlit run portal/app.py` from the project root
- If `PCSAFT_API_URL` is unset or empty, the portal falls back to the local `pcsaft_predict` library (no API required)
- Default API URL when configured: `http://localhost:8000`

---

## Generalizable Design

The portal is not specific to blowing agents. It can screen for thermodynamic analogs of any organic molecule:

1. **Configurable reference molecule**: Any molecule with known or predicted PC-SAFT parameters can be used as the comparison target; the API exposes a set of reference molecules, and users can add custom references.
2. **Application presets**: Presets for blowing agents, refrigerants, solvents, and general exploration demonstrate the same workflow applied across domains.
3. **Transferability**: The ML + PC-SAFT approach generalizes to broader molecular property screening tasks (refrigerant alternatives, green solvents, solubility screening) once the model is trained on appropriate data.

---

## Related Files

| Component       | Location                        |
|----------------|----------------------------------|
| Main application | `portal/app.py`                |
| Predictor       | `portal/components/predictor.py` |
| Screener        | `portal/components/screener.py` |
| Presets         | `portal/presets.py`             |
| Dashboard       | `portal/components/dashboard.py` |
| API client      | `portal/api_client.py`          |
| Backend API     | `serving/`                      |
