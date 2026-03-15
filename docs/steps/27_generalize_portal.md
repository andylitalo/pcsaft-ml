# Step 27: Generalize Portal for Multi-Domain Use

## Objective

Extend the Streamlit portal from a blowing-agent-specific screening tool into a
general-purpose PC-SAFT prediction interface that serves multiple application
domains (blowing agents, refrigerants, solvents, custom). Add a batch screening
mode and a direct-to-library fallback that eliminates the hard dependency on the
FastAPI server for casual use.

Important framing: the domain presets in this step are **UX heuristics**, not
validated scientific workflows. The portal should help users explore predictions
conservatively, not imply that the project has fully validated each domain-specific
screening recipe.

## Motivation

The portal (Step 08) was built around a single use case: compare candidates to
cyclopentane for blowing agent screening. Steps 24-25 already shifted the reference
to HFO-1336mzz(Z), but the portal doesn't reflect this. For open-source release,
the portal needs to serve the broader audience identified in the Phase 2 executive
summary:

- **Thermodynamic modelers**: want to predict PC-SAFT params for arbitrary molecules.
  Don't care about blowing agents. Need raw parameters, uncertainty, and AD flags.
- **Refrigerant engineers**: want to screen fluorinated candidates against R-134a
  or R-410A, not cyclopentane.
- **Solvent designers**: want to compare candidates by Henry's constant in a
  specific solvent, not by VP proximity.
- **Academic researchers**: want to benchmark models, submit experimental data,
  and export predictions for further analysis.

The portal already has reference molecule selection via the API
(`/reference-molecules` endpoint with 20+ molecules), but the UX doesn't surface
this flexibility. The application presets make the tool immediately useful for
each domain without requiring users to know which reference molecule to pick.

That convenience comes with scientific risk: preset thresholds and ranking
metrics should be presented as illustrative defaults, not as universally valid
domain recommendations.

## Dependencies

- Step 08 (Streamlit portal): `portal/` package exists with predictor, submitter, dashboard
- Step 26 (`pcsaft-predict` package): needed for direct-to-library fallback mode
- Step 24 (`screening/hfo_screening.py`): boiling point computation for the screening tab
- Step 25 (HFO-centric results): informs the blowing agent preset defaults

## Implementation Guide

### 27.1 Add Application Presets

Create `portal/presets.py` with preset configurations:

```python
PRESETS = {
    "blowing_agents": {
        "name": "Blowing Agent Screening",
        "description": "Screen fluorinated olefins for PU/PIR foam blowing agents",
        "default_reference": "HFO-1336mzz(Z)",
        "filters": {
            "boiling_point_range_K": (288, 323),
            "requires_cc_bond": True,
            "max_chlorine": 0,
            "min_fluorine": 2,
            "max_sa_score": 4.5,
        },
        "ranking_metric": "hfo_distance",
        "soft_preferences": ["henrys_constant"],
    },
    "refrigerants": {
        "name": "Refrigerant Screening",
        "description": "Screen candidates for HVAC refrigerant applications",
        "default_reference": "R-134a (1,1,1,2-tetrafluoroethane)",
        "filters": {
            "boiling_point_range_K": (220, 270),
            "max_sa_score": 4.0,
        },
        "ranking_metric": "parameter_distance",
        "soft_preferences": ["boiling_point"],
    },
    "solvents": {
        "name": "Solvent Design",
        "description": "Predict PC-SAFT parameters for solvent selection",
        "default_reference": "n-Hexane",
        "filters": {},
        "ranking_metric": "parameter_distance",
        "soft_preferences": ["henrys_constant"],
    },
    "general": {
        "name": "General Prediction",
        "description": "Predict PC-SAFT parameters for any molecule",
        "default_reference": None,
        "filters": {},
        "ranking_metric": None,
        "soft_preferences": [],
    },
}
```

Each preset should also carry a short caution string shown in the UI, for example:
"Illustrative heuristic for exploratory screening; thresholds are not universally
validated for this domain."

### 27.2 Restructure portal tabs

Update `portal/app.py` to use 4 tabs instead of 3:

1. **Predict** (existing, enhanced): Single-molecule or small-batch prediction
   with reference comparison. Add preset selector at the top that auto-selects
   the reference molecule and adjusts the UI context.

2. **Screen** (new): Batch screening mode for CSV uploads.
   - Upload CSV with SMILES column
   - Select preset (sets filters) or customize filters manually
   - Run predictions + filtering + ranking
   - Display screening funnel (bar chart of candidates at each filter stage)
   - Display ranked results table with sortable columns
   - Download enriched CSV with all predictions, UQ, AD, filter status, rank
   - Always show a visible disclaimer that rankings are exploratory unless the
     user has independently validated the chosen workflow for their domain

3. **Submit Data** (existing, unchanged): Experimental data submission.

4. **Dashboard** (existing, enhanced): Add model selector dropdown, display
   per-model metrics, show training data size per dataset.

### 27.3 Add direct-to-library fallback

Currently, the portal requires the FastAPI server to be running. For casual users
who just `pip install`ed the package, this is a barrier.

Add to `portal/api_client.py`:

```python
class PCSAFTClient:
    def __init__(self, api_url=None):
        self._api_url = api_url
        self._local_model = None

    def predict(self, smiles_list):
        if self._api_url:
            return self._predict_via_api(smiles_list)
        return self._predict_local(smiles_list)

    def _predict_local(self, smiles_list):
        if self._local_model is None:
            from pcsaft_predict import predict_with_uncertainty
            self._local_model = predict_with_uncertainty
        df = self._local_model(smiles_list)
        # Convert DataFrame to the same dict format as the API response
        ...
```

When `PCSAFT_API_URL` is not set, the portal uses the `pcsaft_predict` package
directly. This makes `streamlit run portal/app.py` work out of the box after
`pip install pcsaft-predict`.

### 27.4 Surface uncertainty and AD prominently

Currently uncertainty is buried in an expander and Tanimoto is shown as a small
text badge. For the open-source version:

- Show uncertainty as ± ranges directly in the metric display (not in expanders)
- Add a color-coded AD status bar: green (Tanimoto ≥ 0.4), yellow (0.3-0.4),
  red (< 0.3) — always visible, not collapsible
- In batch mode, add a "Confidence" column that combines UQ and AD into a single
  traffic-light indicator
- When predictions are low-similarity / high-uncertainty, prefer explicit text
  such as "out-of-domain hypothesis" over reassuring product language

### 27.5 Add boiling point to predictions (optional column)

When `teqp` is installed, add a "Compute thermodynamic properties" checkbox that
runs `compute_boiling_point()` and `compute_properties()` for each predicted
molecule. Display boiling point in the results table.

This is gated behind `teqp` availability — the portal still works without it,
just without thermo properties.

### 27.6 Tests

Add to `tests/test_portal.py` or create `tests/test_portal_presets.py`:

1. `test_presets_all_have_required_keys`: every preset has name, description,
   default_reference, filters, ranking_metric
2. `test_general_preset_has_no_filters`: general preset's filters dict is empty
3. `test_local_fallback_predict`: mock `pcsaft_predict.predict_with_uncertainty`,
   verify `PCSAFTClient(api_url=None).predict(["CCO"])` returns correct format
4. `test_batch_screening_smoke`: create a small DataFrame, run through screening
   logic, verify funnel counts are correct
5. `test_preset_reference_molecule_exists`: each preset's default_reference is in
   the REFERENCE_MOLECULES list (or is None)

## Key Outputs

| Artifact | Path |
|----------|------|
| Presets module | `portal/presets.py` |
| Updated portal app | `portal/app.py` (modified) |
| Updated API client | `portal/api_client.py` (modified) |
| Updated predictor | `portal/components/predictor.py` (modified) |
| New screening component | `portal/components/screener.py` |
| Tests | `tests/test_portal_presets.py` |
| Report | `docs/reports/27_generalize_portal.md` |

## Acceptance Criteria (When to Move On)

- [ ] Portal has 4 tabs: Predict, Screen, Submit Data, Dashboard
- [ ] Application preset selector works in both Predict and Screen tabs
- [ ] Selecting "Blowing Agent Screening" preset sets HFO-1336mzz(Z) as reference
- [ ] Selecting "Refrigerant Screening" preset sets R-134a as reference
- [ ] Selecting "General Prediction" shows raw params without reference comparison
- [ ] Batch screening mode: upload CSV → filter funnel → ranked results → download CSV
- [ ] Direct-to-library fallback: portal works without FastAPI server when `pcsaft_predict` is installed
- [ ] Uncertainty shown as ± values directly in metric display (not in expanders)
- [ ] AD status shown as color-coded bar (green/yellow/red) for every prediction
- [ ] Each preset is clearly labeled as an illustrative heuristic rather than a universally validated domain workflow
- [ ] Batch screening UI includes a visible disclaimer that rankings are exploratory
- [ ] Tests pass (≥ 5 new tests)
- [ ] Report at `docs/reports/27_generalize_portal.md`
- [ ] Existing portal tests still pass
