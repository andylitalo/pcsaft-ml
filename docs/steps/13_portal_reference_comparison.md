# Step 13: Portal Reference Molecule Comparison

## Purpose

The Phase 1 portal hardcodes cyclopentane (m=2.37, σ=3.71 Å, ε/k=289 K) as the comparison reference. A researcher studying a different application — foam insulation with a different density requirement, refrigeration, or aerosol propellants — may need to compare against pentane, isobutane, or a custom set of parameters. This step adds two levels of flexibility:

- **Level 1**: sidebar parameter inputs with cyclopentane defaults (custom numeric inputs)
- **Level 2**: dropdown of curated reference molecules from the Esper dataset with known parameters

An OOD warning banner is also added to surface the `in_domain` flag prominently in the UI.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| Cyclopentane hardcoded in predictor | User-adjustable reference (Level 1) |
| No molecule library in portal | Searchable dropdown of 20–30 common molecules (Level 2) |
| `in_domain` field returned by API but not shown | OOD warning banner in portal |

## Dependencies

- Requires: FastAPI serving layer (`serving/app.py`, `serving/schemas.py`)
- Requires: Streamlit portal (`portal/components/predictor.py`, `portal/api_client.py`)
- Does not require: new ML models or data

## Implementation Guide

### 13.1 Add `ReferenceMolecule` Schema (`serving/schemas.py`)

```python
class ReferenceMolecule(BaseModel):
    name: str
    smiles: str
    m: float
    sigma: float
    epsilon_k: float
    source: str = "Esper"
```

### 13.2 Add `GET /reference-molecules` Endpoint (`serving/app.py`)

Curated list of 20–30 common molecules with known PC-SAFT parameters (hardcoded from Esper dataset; no DB required):

```python
REFERENCE_MOLECULES = [
    ReferenceMolecule(name="Cyclopentane", smiles="C1CCCC1",
                      m=2.3655, sigma=3.7114, epsilon_k=288.84),
    ReferenceMolecule(name="n-Pentane", smiles="CCCCC",
                      m=2.6896, sigma=3.7729, epsilon_k=231.20),
    ReferenceMolecule(name="Isobutane", smiles="CC(C)C",
                      m=2.2616, sigma=3.7574, epsilon_k=216.53),
    ReferenceMolecule(name="Propane", smiles="CCC",
                      m=2.0020, sigma=3.6184, epsilon_k=208.11),
    ReferenceMolecule(name="n-Butane", smiles="CCCC",
                      m=2.3316, sigma=3.7086, epsilon_k=222.88),
    ReferenceMolecule(name="Cyclohexane", smiles="C1CCCCC1",
                      m=2.5303, sigma=3.8499, epsilon_k=278.11),
    ReferenceMolecule(name="Benzene", smiles="c1ccccc1",
                      m=2.4653, sigma=3.6478, epsilon_k=287.35),
    ReferenceMolecule(name="Toluene", smiles="Cc1ccccc1",
                      m=2.8149, sigma=3.7169, epsilon_k=285.69),
    # Add 12–22 more from Esper spanning HFOs, ethers, ketones
]

@app.get("/reference-molecules", response_model=list[ReferenceMolecule])
async def list_reference_molecules():
    return REFERENCE_MOLECULES
```

Select the full list of 20–30 from the Esper dataset to cover a range of molecular families: alkanes, cycloalkanes, aromatics, fluorocarbons, ethers, ketones, and at least 3–5 HFOs.

### 13.3 Add `get_reference_molecules()` to `portal/api_client.py`

```python
def get_reference_molecules(self) -> list[dict]:
    try:
        r = self._session.get(f"{self.base_url}/reference-molecules", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []
```

### 13.4 Update `portal/components/predictor.py`

**Level 2 — Reference molecule selectbox** (above the SMILES input):

```python
ref_molecules = api_client.get_reference_molecules()
ref_names = ["Custom..."] + [m["name"] for m in ref_molecules]
selected = st.selectbox("Compare against:", ref_names, index=1)

if selected == "Custom...":
    with st.expander("Custom reference parameters"):
        ref_m = st.number_input("m (segments)", value=2.37, step=0.01)
        ref_sigma = st.number_input("σ (Å)", value=3.71, step=0.01)
        ref_eps = st.number_input("ε/k (K)", value=289.0, step=1.0)
else:
    mol = next(m for m in ref_molecules if m["name"] == selected)
    ref_m, ref_sigma, ref_eps = mol["m"], mol["sigma"], mol["epsilon_k"]
    st.caption(f"SMILES: `{mol['smiles']}` | Source: {mol['source']}")
```

**OOD warning banner** (after receiving prediction):

```python
if result.get("in_domain") is False:
    st.warning(
        "**Out-of-domain**: This molecule is outside the training distribution. "
        "Predictions may be unreliable. Treat results with extra caution.",
        icon="⚠️"
    )
```

### 13.5 Update Distance Calculation

Replace hardcoded cyclopentane constants with the `ref_m, ref_sigma, ref_eps` variables in the distance display logic.

## Acceptance Criteria (When to Move On)

- [ ] `GET /reference-molecules` returns ≥ 20 molecules with correct schema
- [ ] Portal selectbox populates from API; selecting a molecule auto-fills reference parameters
- [ ] Custom parameter inputs available when "Custom..." is selected; defaults to cyclopentane
- [ ] OOD warning banner appears for out-of-domain predictions
- [ ] `ReferenceMolecule` schema in `serving/schemas.py` and tests covering the endpoint
- [ ] Report written at `docs/reports/13_portal_reference_comparison.md`
- [ ] Add ≥ 5 tests (endpoint schema, portal component, `get_reference_molecules`)
- [ ] All existing tests pass
- [ ] Commit: `step 13: portal reference molecule comparison (Level 1+2)`
