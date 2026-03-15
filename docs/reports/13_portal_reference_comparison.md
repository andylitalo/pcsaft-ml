# Results Report: Portal Reference Molecule Comparison (Step 13)

## Summary

This step extends the Streamlit portal to support flexible reference molecule comparison, replacing the hardcoded cyclopentane reference with two levels of user control:

1. **Level 1**: Custom numeric inputs for reference parameters (m, σ, ε/k)
2. **Level 2**: Dropdown selector with 24 curated reference molecules from the Esper dataset

Additionally, an out-of-domain (OOD) warning banner is now prominently displayed when predictions fall outside the model's applicability domain.

## Implementation

### Backend Changes (FastAPI)

**1. Added `ReferenceMolecule` schema** (`serving/schemas.py`)
- New Pydantic model with fields: name, smiles, m, sigma, epsilon_k, source
- Default source set to "Esper"

**2. Added `GET /reference-molecules` endpoint** (`serving/app.py`)
- Returns a hardcoded list of 24 reference molecules spanning:
  - **Alkanes** (8): propane, n-butane, isobutane, n-pentane, n-hexane, n-heptane, n-octane, n-decane
  - **Cycloalkanes** (3): cyclopentane, cyclohexane, methylcyclohexane
  - **Aromatics** (3): benzene, toluene, ethylbenzene
  - **Fluorocarbons** (4): R-32, R-134a, R-152a, HFO-1234yf
  - **Ethers** (2): diethyl ether, MTBE
  - **Ketones** (2): acetone, MEK
  - **Alcohols** (2): methanol, ethanol
- All parameters sourced from the Esper dataset (`model/data/esper_pcsaft.csv`)

### Frontend Changes (Streamlit Portal)

**3. Added `get_reference_molecules()` to API client** (`portal/api_client.py`)
- Fetches reference molecules from the new endpoint
- Returns empty list on error (graceful degradation)

**4. Updated predictor component** (`portal/components/predictor.py`)
- Removed hardcoded `CYCLOPENTANE_REF` constant
- Added reference molecule selectbox before SMILES input
- Default selection: Cyclopentane (index 1 after "Custom...")
- "Custom..." option reveals number inputs for m, σ, ε/k with cyclopentane defaults
- SMILES and source displayed below selectbox for selected molecule
- Updated OOD warning banner to be more prominent:
  - Uses `st.warning()` with icon for better visibility
  - Appears immediately after SMILES display
  - Clear messaging: "Out-of-domain: This molecule is outside the training distribution..."
- Distance calculation now uses selected reference parameters (ref_m, ref_sigma, ref_eps)
- Metric delta text updated to show reference molecule name (e.g., "5.1% diff from Cyclopentane")

### Testing

Added 7 new tests in `tests/test_reference_molecules.py`:

1. `test_reference_molecules_endpoint_returns_list` — Verifies endpoint returns ≥20 molecules
2. `test_reference_molecules_schema` — Validates all required fields present and correct types
3. `test_reference_molecules_includes_cyclopentane` — Confirms cyclopentane present with correct parameters
4. `test_reference_molecules_diverse_categories` — Checks presence of alkanes, cycloalkanes, aromatics, fluorocarbons, ethers, ketones
5. `test_api_client_get_reference_molecules` — Tests API client method with mocked response
6. `test_api_client_get_reference_molecules_error_handling` — Verifies graceful error handling (empty list on failure)
7. `test_reference_molecule_schema_validation` — Tests Pydantic schema validation and default source

Updated existing tests in `tests/test_portal.py`:
- Removed `test_predictor_module_imports` check for removed `CYCLOPENTANE_REF` constant
- Renamed `test_cyclopentane_comparison_logic` → `test_reference_comparison_logic` (tests generic calculation)

All 183 tests pass.

## Key Findings

### User Experience Improvements

**Flexibility**: Researchers can now compare against any reference molecule in the Esper library, not just cyclopentane. For example:
- Foam blowing agent researchers can compare to pentane or isobutane
- Refrigeration researchers can compare to R-134a or HFO-1234yf
- Aerosol propellant researchers can compare to MTBE or diethyl ether

**Visibility**: The OOD warning banner is now impossible to miss, addressing the previous issue where the `in_domain` flag was only available in the API response but not surfaced in the UI.

**Fallback**: If the API is unreachable, the portal defaults to cyclopentane parameters (2.3655, 3.7114, 288.84) and displays a warning, ensuring the tool remains usable even during API downtime.

### Data Quality

All 24 reference molecules have parameters directly from the Esper dataset. Example validation:
- Cyclopentane: m=2.25774, σ=3.75308 Å, ε/k=273.50029 K (SMILES: C1CCCC1)
- R-134a: m=3.07064, σ=3.13545 Å, ε/k=154.7744 K (SMILES: FC(F)C(F)(F)F)
- Acetone: m=2.78618, σ=3.24884 Å, ε/k=231.11777 K (SMILES: CC(C)=O)

These values differ slightly from literature-hardcoded cyclopentane (m=2.3655 vs 2.25774) due to different fitting procedures, but are internally consistent within the Esper dataset.

### Coverage

The reference library covers:
- Molecular weight range: 32 Da (methanol) to 142 Da (n-decane)
- Epsilon/k range: 154.8 K (R-134a) to 287.5 K (toluene)
- Sigma range: 2.81 Å (R-32) to 3.99 Å (methylcyclohexane)
- Applications: blowing agents, refrigerants, solvents, fuels

### Limitations

**No persistence**: Reference selection is not saved across sessions. A future enhancement could store user preferences in browser local storage or a user profile.

**Hardcoded list**: The reference library is static. Step 15 or later could add a `/reference-molecules?search=...` query parameter or allow users to add custom reference molecules from the submission history.

**No validation in custom mode**: When "Custom..." is selected, there is no validation that the input parameters are physically reasonable. A researcher could enter m=0.01 or ε/k=10,000 K without a warning.

## Acceptance Criteria

All acceptance criteria from the step guide are met:

- ✅ `GET /reference-molecules` returns 24 molecules with correct schema
- ✅ Portal selectbox populates from API; selecting a molecule auto-fills reference parameters
- ✅ Custom parameter inputs available when "Custom..." is selected; defaults to cyclopentane (m=2.3655, σ=3.7114, ε/k=288.84)
- ✅ OOD warning banner appears for out-of-domain predictions (tested with `in_domain=False`)
- ✅ `ReferenceMolecule` schema in `serving/schemas.py` and tests covering the endpoint
- ✅ Report written at `docs/reports/13_portal_reference_comparison.md`
- ✅ 7 new tests added (all passing)
- ✅ All 183 existing tests pass
- ✅ Ruff clean on modified files (serving/, portal/, tests/)

## Next Steps

This step is complete and ready for Step 14 (Portal Session History) or Step 15 (Portal Batch Export), which will add session management and CSV export functionality to the portal.

Potential future enhancements:
- Add search/filter to reference molecule dropdown
- Allow users to save custom reference molecules
- Add parameter range validation in custom mode
- Persist reference selection across sessions
- Add a "Compare to multiple references" mode for side-by-side comparison
