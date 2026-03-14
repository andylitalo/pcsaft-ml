# Screening Guide

## Pipeline Overview

```
[1/4] Generate HFO candidates (stereoisomer enumeration)
[2/4] Filter 1: SA Score ≤ 4.5 (synthesizability)
[3/4] Filter 2: PubChem patent check (freedom to operate)
[4/4] Filter 3: PC-SAFT prediction + cyclopentane similarity ranking
→ Output: screening/results/ranked_candidates.csv
```

## Stage 1: Candidate Generation

### HFO Scaffolds
Base structures are fluorinated propenes and butenes—the HFO-1234 family and analogs. These are chosen because:
- Low GWP (global warming potential) < 10
- Non-ozone-depleting (no chlorine/bromine)
- Appropriate boiling points for foam blowing

### Stereoisomer Enumeration
Uses `rdkit.Chem.EnumerateStereoisomers` to generate all E/Z isomers of fluorinated alkenes, since geometric isomers can have significantly different physical properties.

### Fluorine Variant Generation
For each scaffold, attempts to add/remove fluorine atoms at available C–H positions to explore the chemical space around known HFOs.

## Stage 2: SA Score Filter

Uses RDKit's SA Score (Ertl & Schuffenhauer, 2009):
- Scale: 1 (easy to synthesize) to 10 (very difficult)
- Default threshold: 4.5
- Most HFOs score 2–4 (relatively simple structures)

## Stage 3: Patent Check

Queries PubChem PUG REST API:
```
GET /rest/pug/compound/smiles/{SMILES}/xrefs/PatentID/JSON
```

- **404 response**: Compound not in PubChem → likely novel, passes filter
- **200 with patents**: Known compound with patents → filtered out
- **200 without patents**: Known but unpatented → passes
- **Rate limiting**: 0.3s delay between requests

Use `--skip-patents` for offline testing.

## Stage 4: PC-SAFT Ranking

### Distance Metric
Normalized Euclidean distance to cyclopentane:

```
d = sqrt( ((m - m_ref)/m_ref)² + ((σ - σ_ref)/σ_ref)² + ((ε - ε_ref)/ε_ref)² )
```

Normalization ensures each parameter contributes equally regardless of scale.

### Cyclopentane Reference
- m = 2.3655, σ = 3.7114 Å, ε/k = 288.84 K

### Closeness Criteria (rules of thumb, subject to change)
- **σ**: < 2% relative difference is considered "close" to cyclopentane
- **ε/k**: < 5% relative difference is considered "close" to cyclopentane

These thresholds reflect empirical experience with foaming behavior. Candidates meeting both criteria are expected to exhibit similar phase behavior during foam rise and curing.

### Output
`screening/results/ranked_candidates.csv` with columns:
- `smiles`: Canonical SMILES
- `sa_score`: Synthetic accessibility score
- `m`, `sigma`, `epsilon_k`: Predicted PC-SAFT parameters
- `distance`: Normalized distance to cyclopentane (lower = better)

## Usage

```bash
# Full pipeline (online, includes patent check)
python -m screening.run_screening

# Offline mode (skip patent API calls)
python -m screening.run_screening --skip-patents

# Custom SA threshold
python -m screening.run_screening --sa-threshold 3.5
```
