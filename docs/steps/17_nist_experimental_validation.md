# Step 17: NIST WebBook Experimental Validation

## Purpose

All thermodynamic validation in the pipeline is computational (EOS closure: predicted PC-SAFT parameters → teqp → VP/density). This step adds experimental validation by comparing teqp-computed vapor pressure and liquid density against NIST WebBook experimental measurements for a held-out set. This answers: "do the predicted parameters produce accurate thermodynamic properties?" rather than "are the predicted parameters close to the reference parameters?"

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| EOS closure testing only | Property-level validation against experiment |
| No external validation of the EOS pipeline | NIST WebBook comparison for 50–100 compounds |
| VP accuracy unknown for ML predictions | MARE(VP) and MARE(density) on held-out experimental data |

## Skills Demonstrated

- **External validation**: Comparing model output against independent experimental data
- **Data acquisition**: Programmatic extraction from NIST WebBook
- **Property-level evaluation**: Moving beyond parameter metrics to thermodynamic property metrics

## Dependencies

- Requires: Step 09 (`model/thermodynamic.py`, `compute_properties()`)
- Requires: Network access to NIST WebBook
- New package: None (uses `requests` and `BeautifulSoup` or regex parsing)

## Implementation Guide

### 17.1 NIST WebBook Data Extraction

NIST WebBook (webbook.nist.gov) provides experimental thermodynamic data but has no modern REST API. Programmatic access via URL construction:

```
https://webbook.nist.gov/cgi/cbook.cgi?ID=C287923&Mask=4&Type=ANTOINE&Plot=on
```

Where `C287923` is the CAS number for cyclopentane and `Mask=4` requests vapor pressure data.

**Approach:**
1. Select 50–100 molecules from the Esper test set that have known CAS numbers
2. Query NIST WebBook for experimental VP at 298.15 K (or interpolate Antoine parameters)
3. Query for liquid density at 298.15 K if available
4. Compare against `compute_properties(m_pred, sigma_pred, epsilon_k_pred, T=298.15)`

**Rate limiting:** Add 1-second delays between requests. NIST has no published rate limits but heavy scraping may be blocked.

### 17.2 Build Validation Dataset

Create `model/data/nist_validation.csv` with columns: smiles, cas_number, vp_exp_Pa, rho_exp_mol_m3, T_K, source_url.

### 17.3 Compute Validation Metrics

For each molecule in the validation set:
1. Predict PC-SAFT parameters using the RF model
2. Compute VP and density using `compute_properties()`
3. Compare to experimental values

Report:
- MARE(VP): mean absolute relative error in vapor pressure
- MARE(density): mean absolute relative error in liquid density
- Fraction of molecules where VP prediction is within ±20% of experiment
- Comparison: VP from Esper parameters vs VP from ML-predicted parameters vs experiment

### 17.4 SPT-PCSAFT Parameter Comparison (Optional Extension)

Winter et al. (2023) provide predicted PC-SAFT parameters for 13,645 components. If their parameter CSV is accessible (arxiv ancillary file `PC-SAFT-Parameter.csv`):
1. Download their predicted parameters for the same 50–100 validation molecules
2. Compare VP from our predictions vs their predictions vs experiment
3. This provides a direct benchmark against the literature SOTA

## Acceptance Criteria

- [ ] NIST validation dataset: ≥ 50 molecules with experimental VP at 298 K
- [ ] MARE(VP) computed for RF predictions
- [ ] Comparison figure: predicted VP vs experimental VP (parity plot)
- [ ] If accessible: SPT-PCSAFT parameter comparison included
- [ ] Report at `docs/reports/17_nist_experimental_validation.md`
