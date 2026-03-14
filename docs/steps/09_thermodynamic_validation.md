# Step 9: Thermodynamic Validation (EOS Closure)

## Purpose

Steps 01–04 predict PC-SAFT *parameters* (m, σ, ε/k) and rank candidates by their Euclidean distance to cyclopentane in parameter space. But parameter proximity is only a proxy for what actually matters: **thermodynamic property proximity**. Two molecules can have similar (m, σ, ε/k) vectors yet differ meaningfully in vapor pressure or liquid density if the sensitivity landscape is nonlinear.

This step closes the loop by plugging predicted parameters into a PC-SAFT equation of state solver, computing real thermodynamic properties (vapor pressure, liquid density, and optionally enthalpy of vaporization), and comparing those to cyclopentane's measured values. The result is a **property-space ranking** that either validates or refines the parameter-space screening from earlier steps.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| Candidates ranked by parameter-space distance only | Property-space validation: vapor pressure, liquid density at reference T |
| No check that parameter proximity → property proximity | Correlation analysis between parameter distance and property distance |
| Screening metric is a proxy | Direct thermodynamic comparison against cyclopentane |

## Skills Demonstrated

- **Equation of state**: Using PC-SAFT EOS to compute thermodynamic properties from molecular parameters
- **Scientific validation**: Closing the loop between ML predictions and physical observables
- **Critical analysis**: Evaluating whether the screening proxy (parameter distance) actually correlates with the target property similarity

## Dependencies

- **Requires**: At least one completed screening run with predicted PC-SAFT parameters (from Step 01 or Step 04)
- **Does not require**: Steps 05–08 (infra). Can run entirely offline.
- **Python package**: `teqp` (NIST, lightweight, CPU-only, ~4μs per calculation)

## Implementation Guide

### 9.1 Install teqp

Add `teqp` to `pyproject.toml` under a new optional dep group:

```toml
[project.optional-dependencies]
thermo = ["teqp>=0.22"]
```

Install: `uv sync --extra dev --extra thermo`

### 9.2 Create `model/thermodynamic.py`

This module wraps teqp to compute properties from PC-SAFT parameters.

```python
"""Thermodynamic property computation from PC-SAFT parameters via teqp."""

import teqp
import numpy as np

# Cyclopentane reference parameters (Gross & Sadowski 2001)
CYCLOPENTANE = {"m": 2.3655, "sigma": 3.7114, "epsilon_k": 265.83}
# Reference conditions for comparison
T_REF = 298.15  # K (25 °C)
T_BOIL_REF = 322.4  # K (cyclopentane normal boiling point)


def make_pcsaft_model(m, sigma, epsilon_k):
    """Create a teqp PCSAFT model from parameters."""
    c = teqp.SAFTCoeffs()
    c.m = float(m)
    c.sigma_Angstrom = float(sigma)
    c.epsilon_over_k = float(epsilon_k)
    return teqp.PCSAFTEOS([c])


def compute_properties(m, sigma, epsilon_k, T=T_REF):
    """Compute vapor pressure and liquid density at temperature T.

    Returns dict with keys: vapor_pressure_Pa, liquid_density_mol_m3.
    Returns None values if the calculation fails (e.g., above critical T).
    """
    ...


def validate_candidates(candidates_df, reference=CYCLOPENTANE):
    """Add thermodynamic property columns to a screening results DataFrame.

    Expects columns: m_pred, sigma_pred, epsilon_k_pred (or m, sigma, epsilon_k).
    Adds columns: vp_Pa, rho_liq, vp_ratio_to_ref, rho_ratio_to_ref, property_distance.
    """
    ...
```

Key design decisions:
- Use `teqp.PCSAFTEOS` (non-associating, matches our 3-parameter model)
- Compute properties at T_REF = 298.15 K (standard conditions) and optionally at T_BOIL of cyclopentane
- Wrap in try/except: some parameter combinations may be unphysical or supercritical at T_REF
- Return `NaN` for failed calculations rather than crashing

### 9.3 Create `scripts/run_thermo_validation.py`

A standalone script that:

1. Loads the screening results CSV (from Step 01 or Step 04's expanded screening)
2. Computes cyclopentane's reference properties from its known PC-SAFT parameters
3. For each candidate: computes vapor pressure and liquid density from predicted parameters
4. Calculates a **property-space distance** (normalized difference in vapor pressure and liquid density vs. cyclopentane)
5. Compares the property-space ranking to the parameter-space ranking
6. Outputs:
   - `model/saved/thermo_validation.csv` with property columns appended
   - Rank correlation (Spearman ρ) between parameter-distance rank and property-distance rank
   - Scatter plot: parameter distance vs. property distance

```bash
uv run python scripts/run_thermo_validation.py \
    --screening-results screening/results/screening_results.csv \
    --output model/saved/thermo_validation.csv
```

### 9.4 Generate Figures

Save to `figures/09_thermodynamic_validation/`:

1. **Parameter distance vs. property distance scatter**: Does parameter proximity predict property proximity? Color points by AD flag if available.
2. **Vapor pressure parity plot**: predicted VP vs. cyclopentane reference VP, annotated with top-10 candidates from both rankings.
3. **Rank comparison**: side-by-side bar chart showing how the top-20 candidates from parameter-space ranking shift when re-ranked by property-space distance.
4. **Sensitivity analysis** (optional): How much does each parameter (m, σ, ε/k) contribute to vapor pressure sensitivity at the reference T? This contextualizes the distance metric weights from Step 01 section 1C.

### 9.5 Analysis Questions to Answer in the Report

The report should address:

1. **Does parameter distance correlate with property distance?** (Spearman ρ > 0.8 = good proxy; < 0.5 = metric needs rethinking)
2. **Do any top-ranked candidates (by parameter distance) drop out when validated thermodynamically?** If so, why?
3. **Are the distance metric weights (3:1:1 for ε/k:σ:m) justified by the sensitivity analysis?** Does ε/k actually dominate vapor pressure as assumed?
4. **How many candidates fail the EOS calculation entirely?** (supercritical, unphysical parameters) — this is an indirect quality check on the ML predictions.

## Evaluation Criteria

| Criterion | Target |
|-----------|--------|
| `model/thermodynamic.py` exists and is tested | Required |
| Validation script runs on screening results without error | Required |
| Spearman correlation between parameter rank and property rank computed | Required |
| At least 3 figures in `figures/09_thermodynamic_validation/` | Required |
| Report addresses all 4 analysis questions | Required |
| Optional: sensitivity analysis of parameter → VP contribution | Stretch |

## When to Move On

This is the final step. The project is complete when:

- [ ] Thermodynamic validation report is written and approved
- [ ] Correlation between parameter-space and property-space rankings is documented
- [ ] Figures are generated
- [ ] All 9 step reports exist in `docs/reports/`
- [ ] Key finding (does the proxy work?) is stated clearly for interview discussion
