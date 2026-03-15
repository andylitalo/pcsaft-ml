# Step 12: Multi-Criteria Candidate Verification

## Purpose

Step 09 established that parameter-space ranking is a weak proxy for property-space ranking (Spearman ρ = 0.372). Phase 1 identified two difluorocyclobutane isomers as the only candidates passing all five screening criteria at 298.15 K. This step:

1. Formalizes the 5-criterion audit as a reproducible script
2. Extends the EOS validation to three temperatures (273 K, 298 K, 323 K) to check stability of the ranking across the blowing agent operating range
3. Writes a clean `verified_candidates.csv` with full pass/fail evidence for every candidate

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| Candidates ranked at 298 K only | Multi-temperature (273/298/323 K) property comparison |
| Top candidates identified informally | Formal 5-criterion pass/fail table |
| No single authoritative candidate file | `model/saved/verified_candidates.csv` |

## Five Criteria

1. **In-domain (AD)**: `in_domain == True` from Isolation Forest (column already in `thermo_validation.csv`)
2. **Parameter proximity**: parameter-space distance to cyclopentane ≤ top-20 threshold
3. **EOS convergence**: `vapor_pressure_Pa` is not NaN at all three temperatures
4. **Vapor pressure proximity**: VP ratio to cyclopentane within [0.5, 1.5] at all temperatures
5. **Synthetic accessibility**: SA score ≤ 4.5 (column already in `thermo_validation.csv`)

## Dependencies

- Requires: `model/saved/thermo_validation.csv` (exists from Step 09)
- Requires: `model/thermodynamic.py` with `compute_properties(m, sigma, epsilon_k, T)` (exists)
- Requires: `teqp` installed (`uv sync --extra dev --extra thermo`)

## Implementation Guide

### 12.1 Create `scripts/verify_candidates.py`

```python
"""Multi-criteria candidate verification with multi-temperature EOS."""
import pandas as pd
from model.thermodynamic import compute_properties, CYCLOPENTANE

TEMPERATURES = [273.15, 298.15, 323.15]  # K
PARAM_TOP_N = 20  # top-N by parameter distance
VP_RATIO_BOUNDS = (0.5, 1.5)
SA_THRESHOLD = 4.5

def run_verification(thermo_csv: str, output_csv: str):
    df = pd.read_csv(thermo_csv)

    # Compute reference VP at each temperature
    ref_vp = {}
    for T in TEMPERATURES:
        props = compute_properties(**CYCLOPENTANE, T=T)
        ref_vp[T] = props["vapor_pressure_Pa"]

    # Compute candidate properties at all temperatures
    results = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        for T in TEMPERATURES:
            props = compute_properties(row["m"], row["sigma"], row["epsilon_k"], T=T)
            rec[f"vp_Pa_{int(T)}K"] = props["vapor_pressure_Pa"]
            rec[f"rho_{int(T)}K"] = props["liquid_density_mol_m3"]
            if not pd.isna(props["vapor_pressure_Pa"]):
                rec[f"vp_ratio_{int(T)}K"] = props["vapor_pressure_Pa"] / ref_vp[T]
            else:
                rec[f"vp_ratio_{int(T)}K"] = float("nan")
        results.append(rec)

    df_out = pd.DataFrame(results)

    # Apply criteria
    param_threshold = df_out["param_distance"].nsmallest(PARAM_TOP_N).max()

    df_out["criterion_ad"] = df_out.get("in_domain", True)
    df_out["criterion_param"] = df_out["param_distance"] <= param_threshold
    df_out["criterion_eos"] = df_out[[f"vp_Pa_{int(T)}K" for T in TEMPERATURES]].notna().all(axis=1)
    df_out["criterion_vp"] = df_out[[f"vp_ratio_{int(T)}K" for T in TEMPERATURES]].apply(
        lambda row: all(VP_RATIO_BOUNDS[0] <= v <= VP_RATIO_BOUNDS[1]
                        for v in row if not pd.isna(v)), axis=1
    )
    df_out["criterion_sa"] = df_out["sa_score"] <= SA_THRESHOLD
    df_out["all_criteria"] = (
        df_out["criterion_ad"] & df_out["criterion_param"] &
        df_out["criterion_eos"] & df_out["criterion_vp"] & df_out["criterion_sa"]
    )

    df_out.to_csv(output_csv, index=False)
    return df_out[df_out["all_criteria"]]


if __name__ == "__main__":
    verified = run_verification(
        "model/saved/thermo_validation.csv",
        "model/saved/verified_candidates.csv"
    )
    print(f"\n{len(verified)} candidates pass all 5 criteria:\n")
    print(verified[["smiles", "sa_score", "param_distance",
                     "vp_ratio_298K", "vp_ratio_273K", "vp_ratio_323K"]].to_string())
```

### 12.2 Run and Inspect Output

```bash
python scripts/verify_candidates.py
```

Review `model/saved/verified_candidates.csv`. For any verified candidates, check:
- Do property ratios stay within bounds at 273 K and 323 K, or do they drift?
- Are there additional candidates that pass at 298 K but fail at the temperature extremes?

### 12.3 Generate Figures

- `figures/12_candidates/multi_temp_vp_ratio.png`: VP ratio vs temperature (273/298/323 K) for all candidates that pass criteria 1–4, with a shaded band at [0.5, 1.5]. Highlight verified candidates.
- `figures/12_candidates/criteria_breakdown.png`: Stacked bar chart showing how many candidates fail at each criterion (funnel visualization).
- `figures/12_candidates/parity_verified.png`: Predicted vs cyclopentane reference for m, σ, ε/k for verified candidates only; include error bars from RF uncertainty.

## Acceptance Criteria (When to Move On)

- [ ] `scripts/verify_candidates.py` runs without error
- [ ] `model/saved/verified_candidates.csv` written with all 5 criterion columns + multi-T EOS columns
- [ ] Verified candidates identified and documented; compare to Phase 1 result (difluorocyclobutane isomers expected)
- [ ] Three figures saved to `figures/12_candidates/`
- [ ] Report written at `docs/reports/12_candidate_verification.md` with full criteria table
- [ ] Add ≥ 5 tests in `tests/test_verify_candidates.py`
- [ ] All existing tests pass
- [ ] Commit: `step 12: multi-criteria candidate verification audit`
