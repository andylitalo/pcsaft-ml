# Results Report: Generalize Portal for Multiple Application Domains

## Summary

Extended the Streamlit portal to support multiple application domains beyond blowing agents. The portal now provides:

1. **Domain presets**: Blowing agents, refrigerants, solvents, and general exploration
2. **Batch screening**: Upload CSV of candidates, apply domain-specific filters, rank by application-appropriate metrics
3. **Local fallback mode**: Direct `pcsaft_predict` library integration when API is unavailable
4. **Improved UX**: Uncertainty displayed inline with predictions, AD status as color-coded bars

The portal is now a multi-purpose screening tool suitable for different chemical engineering applications.

## New Components

### Presets System (`portal/presets.py`)

Four application domain presets, each with:
- **Name and description**: User-facing documentation
- **Default reference molecule**: Domain-appropriate comparison target
- **Filter criteria**: Parameter ranges, MW limits, AD requirements
- **Ranking metric**: Application-specific scoring (distance to reference, Tanimoto, dispersion energy)
- **Soft preferences**: Heuristic guidance (low GWP, low toxicity, biodegradable)
- **Caution disclaimer**: Explicit warning about heuristic nature of filters

| Preset | m range | σ range (Å) | ε/k range (K) | Max MW | Min Tanimoto | Ranking Metric |
|--------|---------|-------------|---------------|--------|--------------|----------------|
| Blowing Agents | 1.8–3.5 | 3.0–4.5 | 200–400 | 200 | 0.3 | Distance to reference |
| Refrigerants | 1.5–3.0 | 2.5–4.0 | 150–350 | 150 | 0.25 | Distance to reference |
| Solvents | 1.5–4.0 | 3.0–5.0 | 150–450 | 250 | 0.2 | ε/k (solvation proxy) |
| General | No range limits | No range limits | No range limits | No limit | 0.0 | Tanimoto (confidence) |

All presets except general require in-domain molecules (Tanimoto-based AD).

### Batch Screener (`portal/components/screener.py`)

New tab in the portal for high-throughput screening:

1. **Preset selection**: Choose application domain or customize filters
2. **CSV upload**: Batch predict hundreds of molecules
3. **Filtering pipeline**: Apply parameter ranges, MW, AD, Tanimoto cutoffs
4. **Ranking**: Sort by distance to reference, Tanimoto, or individual parameters
5. **Screening funnel**: Visualize input → valid → passed filters
6. **Results table**: Top 10 candidates with scores
7. **Download**: Enriched CSV with predictions, filter reasons, rank scores

The screener includes visible disclaimers about the heuristic nature of thresholds and the need for experimental validation.

### Local Fallback Mode (`portal/api_client.py`)

API client now supports `base_url=None` for direct library usage:

```python
# API mode (default)
client = PCSAFTClient(base_url="http://localhost:8000")

# Local mode (no API server required)
client = PCSAFTClient(base_url=None)
result = client.predict(["CCO", "c1ccccc1"])
```

In local mode:
- Predictions use `pcsaft_predict.predict_with_uncertainty()` directly
- Reference molecules are hardcoded (Cyclopentane, R-134a, Ethyl acetate, Benzene, Toluene)
- Uncertainty estimates always included
- No dependency on FastAPI server

This enables standalone use of the portal for offline exploration.

### Improved Predictor UX (`portal/components/predictor.py`)

1. **Uncertainty inline**: Display predictions as `2.3655 ± 0.0450` directly in metrics (not collapsed in expander)
2. **AD status bar**: Color-coded banner (green/yellow/red) based on Tanimoto score
   - Green (≥0.4): In-domain, reliable
   - Yellow (0.3–0.4): Moderate, use caution
   - Red (<0.3): Out-of-domain, high uncertainty

## Test Coverage

New test suite in `tests/test_portal_presets.py` with 14 tests:

- **5 preset tests**: Required keys, filter ranges, general preset structure
- **2 local fallback tests**: Prediction with mocked library, reference molecules
- **3 batch screening tests**: Filtering, ranking by distance to reference, ranking by Tanimoto
- **2 preset reference tests**: Default references exist in API and local mode
- **2 API client tests**: Local vs API mode initialization

All tests pass (34 total including existing portal tests).

## User Workflow Examples

### Example 1: Blowing Agent Screening

1. Navigate to "Screen Candidates" tab
2. Select "Blowing Agents" preset (default)
3. Upload `candidates.csv` with 500 molecules
4. See screening funnel: 500 → 485 valid → 127 passed filters
5. Review top 10 candidates ranked by similarity to Cyclopentane
6. Download enriched CSV with predictions and filter reasons
7. Validate top candidates with thermodynamic simulations (Step 09)

### Example 2: Refrigerant Exploration (Local Mode)

1. Set `export PCSAFT_API_URL=""` to enable local mode
2. Start portal: `streamlit run portal/app.py`
3. Select "Refrigerants" preset (R-134a reference)
4. Upload CSV of HFC/HFO candidates
5. Customize filters: Relax min_tanimoto from 0.25 to 0.15 for broader exploration
6. Rank by distance to R-134a
7. Download results for GWP and toxicity analysis

### Example 3: Solvent Discovery with Custom Filters

1. Select "Industrial Solvents" preset (Ethyl acetate reference)
2. Expand "View/Customize Filters"
3. Adjust ε/k range to 300–450 K for strong solvation
4. Increase max_mw to 300 for bulkier solvents
5. Ranking metric: ε/k (higher is better for solvation)
6. Review top candidates with high dispersion energy
7. Note caution disclaimer about experimental validation

## Key Findings

### Preset Design Philosophy

Each preset balances two competing goals:
1. **Restrictive enough** to filter out clearly unsuitable candidates
2. **Permissive enough** to allow exploratory discovery

The "general" preset provides an escape hatch for unrestricted exploration.

### Ranking Metrics

Three ranking strategies implemented:

1. **Distance to reference** (blowing agents, refrigerants): Euclidean distance in normalized parameter space (m, σ, ε/k)
   - Assumes target molecule has known good properties
   - Useful when searching for drop-in replacements

2. **Individual parameter** (solvents): Rank by ε/k alone
   - Useful when one property dominates application performance
   - Example: High ε/k correlates with solvation power

3. **Tanimoto similarity** (general): Rank by max similarity to training set
   - Proxy for prediction confidence
   - Useful for exploratory screening when target profile is unknown

### Local vs API Mode Trade-offs

| Mode | Pros | Cons |
|------|------|------|
| API | Centralized model, versioning, monitoring | Requires server, network latency |
| Local | Standalone, offline, fast | No model versioning, larger memory footprint |

Local mode is ideal for exploratory analysis and demos. API mode is required for production MLOps (Step 07 Kubeflow pipelines).

### Caution Disclaimers

Every preset includes an explicit disclaimer:

> "Illustrative heuristic for exploratory screening; thresholds are not universally validated for [domain]. Always validate [critical properties] before experimental use."

This addresses a key risk: Users might treat ML predictions as ground truth without experimental validation. The disclaimer is visible at the top of the screener UI and in downloadable CSVs.

## Comparison to Previous Portal (Step 08)

| Feature | Step 08 (Blowing Agents Only) | Step 27 (Generalized) |
|---------|-------------------------------|------------------------|
| Application domains | 1 (blowing agents) | 4 (blowing agents, refrigerants, solvents, general) |
| Screening mode | Single-molecule predict | Batch CSV upload + filtering |
| Reference molecules | Cyclopentane only | 5+ references, domain-specific defaults |
| Ranking | Manual comparison | Automated ranking with multiple metrics |
| Uncertainty display | Collapsed expander | Inline with predictions (2.36 ± 0.05) |
| AD status | Text warning | Color-coded bar (green/yellow/red) |
| API dependency | Required | Optional (local fallback mode) |
| Disclaimers | None | Explicit caution on every preset |

## Deviations from Step Guide

None. All deliverables implemented as specified:
- ✅ `portal/presets.py` with 4 domain presets
- ✅ `portal/components/screener.py` with batch screening UI
- ✅ `portal/app.py` updated to 4 tabs
- ✅ `portal/api_client.py` with local fallback (base_url=None)
- ✅ `portal/components/predictor.py` with inline uncertainty and AD bar
- ✅ `tests/test_portal_presets.py` with 14 tests (5+ as required)
- ✅ Visible disclaimers on all presets

## Readiness Check

- [x] Four domain presets implemented with required keys
- [x] General preset has no filters (min_tanimoto=0.0, require_in_domain=False)
- [x] Batch screener tab functional with upload, filter, rank, download
- [x] Local fallback mode works (PCSAFTClient(base_url=None))
- [x] Uncertainty displayed inline (not in expanders)
- [x] AD status as color-coded bar (green/yellow/red)
- [x] 14 tests pass (presets, local mode, screening, references)
- [x] Caution disclaimers visible on all presets
- [x] All existing portal tests still pass (backward compatibility)

## Next Steps

The portal is now a general-purpose PC-SAFT screening tool. Future enhancements could include:

1. **Custom preset builder**: Allow users to save/load custom filter configurations
2. **Multi-objective ranking**: Pareto frontier for trade-offs (e.g., low GWP vs high ε/k)
3. **Thermodynamic validation integration**: Auto-trigger Step 09 teqp validation for top candidates
4. **Export to experimental workflow**: Generate lab notebook templates for top hits
5. **Preset versioning**: Track filter evolution as domain knowledge improves

The foundation for domain-agnostic molecular screening is now in place.
