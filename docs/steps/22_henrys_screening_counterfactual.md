# Step 22: Henry's Constant Batch Screening & Robustness Analysis

## Objective

Persist the Henry's constant methodology from Step 21 into reusable code, batch-compute
Henry's constants for all ~736 VP-passing candidates from Step 20, and produce the
VP-only vs Henry-screened counterfactual comparison that is the project's strongest
demonstration of ML value. Add solvent sensitivity and uncertainty analyses to establish
credibility of the standout candidates.

## Motivation

Step 21 computed Henry's constants for a handful of representative molecules and
established the key insight: parameter-space proximity (especially epsilon/k) controls
mixture behavior through the Boltzmann factor. But the Step 21 code was not persisted,
and only ~20 molecules were evaluated. This step scales the analysis to the full
VP-passing candidate pool and produces the data needed for the narrative report (Step 23).

The counterfactual comparison is critical: ~736 candidates pass VP proximity screening
(131 HFOs + 605 HCFOs), but many of those have Henry's constants orders of magnitude
above cyclopentane — they would be nearly insoluble in polyol despite matching
cyclopentane's volatility. Showing this collapse is the clearest way to demonstrate that
ML-predicted PC-SAFT parameters enable a chemistry-relevant screen that pure-property
matching misses.

## Dependencies

- Requires: `model/saved/thermo_validation_systematic.csv` (exists from Step 20)
- Requires: `screening/results/ranked_candidates_systematic.csv` (exists from Step 20)
- Requires: `model/thermodynamic.py` with `compute_properties()` (exists from Step 09)
- Requires: `teqp` installed (`uv sync --extra dev --extra thermo`)
- Requires: trained RF model for uncertainty extraction (`model/saved/rf_*.joblib`)

## Implementation Guide

### 22.1 Persist `compute_henrys_constant()` in `model/thermodynamic.py`

Implement the Henry's constant computation from Step 21's methodology. Add to
`model/thermodynamic.py`:

```python
HEXANE = {"m": 3.0576, "sigma": 3.7983, "epsilon_k": 236.77}
POLYOL_LIKE = {"m": 123.0, "sigma": 3.01, "epsilon_k": 228.5}

def compute_henrys_constant(solute_params, solvent_params=HEXANE, T=298.15, k_ij=0.0):
    """Henry's constant for solute at infinite dilution in solvent via PC-SAFT.

    Returns dict with keys: H_Pa, phi_inf, P_sat_solvent_Pa, ln_phi_inf
    Returns None if VLE or fugacity computation fails.

    Method:
    1. Pure solvent VLE -> rho_L, P_sat via pure_VLE_T
    2. Binary model [solute, solvent] at rho_vec = [eps*rho_L, (1-eps)*rho_L]
    3. phi_i_inf = exp(get_fugacity_coefficients(T, rho_vec)[0])
    4. H = phi_i_inf * P_sat
    """
```

Also add a batch function:

```python
def batch_henrys_constant(candidates_df, solvent_params=HEXANE, T=298.15, k_ij=0.0):
    """Compute Henry's constant for all candidates in a DataFrame.

    Expects columns: m, sigma, epsilon_k.
    Adds columns: H_Pa, H_ratio (relative to cyclopentane in same solvent).
    """
```

Validate by reproducing the Step 21 reference values:
- H(cyclopentane in hexane) should be ~51.68 kPa
- H ratios for commercial HFOs should match Step 21 tables

### 22.2 Batch Compute: VP-Passing Candidates + Commercial Agents

Load the VP-passing candidates from Step 20. These are:
- All candidates from `thermo_validation_systematic.csv` where VP ratio is in [0.5, 1.5]
  at 298 K (131 HFOs + 605 HCFOs + some Cl-olefins = ~736 total)
- Plus the 4 commercial agents: HFO-1234ze, HFO-1234yf, HCFO-1233zd, HFO-1336mzz

Run `batch_henrys_constant()` with hexane solvent for all. Save results to
`model/saved/henrys_constant_screening.csv` with columns:
- smiles, m, sigma, epsilon_k, param_distance, vp_ratio_298K
- H_Pa, H_ratio, log10_H_ratio
- epsilon_ij, epsilon_ij_ratio (cross-interaction energy vs cyclopentane-hexane)

### 22.3 VP-Only vs Henry-Screened Counterfactual

From the batch results, produce two shortlists:

**VP-only shortlist**: all candidates with VP ratio in [0.5, 1.5] (the ~736).

**Henry-screened shortlist**: candidates with H/H(ref) in [0.5, 2.0].

Report:
- How many VP-only candidates have H ratios > 10? > 100? > 1000?
- How many survive the Henry screen?
- Side-by-side table of top-10 by VP proximity vs top-10 by Henry proximity
- Summary statistics: median H ratio for HFOs vs HCFOs vs Cl-olefins

Save the counterfactual comparison to `model/saved/vp_vs_henrys_comparison.csv`.

### 22.4 Solvent Sensitivity Study (Polyol-Like Parameters)

Re-run `batch_henrys_constant()` with `solvent_params=POLYOL_LIKE` for the same
candidate pool. This is a robustness check, not a definitive polyol answer (real polyol
requires association parameters our 3-param PC-SAFT does not support).

The polyol-like solvent (m=123) is a polymer chain. teqp may have numerical
difficulties with extreme m asymmetry. Use this fallback chain if VLE fails for the
pure polyol-like solvent:

1. Try m=123 (full polyol-like parameters: m=123, sigma=3.01, epsilon/k=228.5)
2. If that fails, try m=45 (oligomer-length chain, same sigma and epsilon/k)
3. If that fails, try m=20 (short oligomer)
4. If that fails, try m=10 (last resort)
5. If all fail, report the limitation honestly and rely on hexane results

For whichever m value succeeds, report that value clearly and note the deviation from
the original polyol parameters. If multiple m values succeed, run the highest and
report whether the ranking changes across the m values that worked.

Compare:
- Spearman rho between hexane H ratios and polyol-like H ratios
- Which candidates change rank by > 5 positions?
- Attribute changes to m-dependent (combinatorial entropy) or sigma-dependent effects

Save to `model/saved/henrys_polyol_sensitivity.csv`.

### 22.5 Uncertainty and AD Credibility for Standout Candidates

For all candidates passing the Henry screen (H/H_ref in [0.5, 2.0], likely ~20-50
molecules):

**AD assessment**:
- Tanimoto similarity to nearest training molecule (load training SMILES, compute
  Morgan fingerprints, find max Tanimoto for each candidate)
- Isolation Forest score from `model/nn/ad.py` if available, otherwise train a new
  IsolationForest on the RF training features

**Prediction uncertainty**:
- Load the trained RF model. Extract per-tree predictions for each candidate. Compute
  std across trees for each of (m, sigma, epsilon_k).

**Sensitivity analysis**:
- For each standout candidate, recompute Henry's constant at:
  - epsilon_k + 1*std_epsilon_k and epsilon_k - 1*std_epsilon_k
  - k_ij = +0.05 and k_ij = -0.05
- Report H/H_ref at nominal, +1sigma, -1sigma, +k_ij, -k_ij

Save to `model/saved/standout_candidates_credibility.csv` with columns:
- smiles, m, sigma, epsilon_k
- std_m, std_sigma, std_epsilon_k
- tanimoto_max, ad_flag (in_domain / warning / ood)
- H_ratio_nominal, H_ratio_plus_1sigma, H_ratio_minus_1sigma
- H_ratio_kij_plus005, H_ratio_kij_minus005

### 22.6 Tests

Add tests in `tests/test_henrys_constant.py`:
- `test_henrys_cyclopentane_in_hexane`: H ≈ 51.68 kPa (within 5%)
- `test_henrys_ratio_commercial_hfo`: HFO-1234ze H ratio >> 10
- `test_henrys_ratio_verified_candidate`: Cl-olefin H ratio ≈ 1.0 (within 20%)
- `test_batch_henrys_returns_expected_columns`: DataFrame schema check
- `test_kij_sensitivity_direction`: positive k_ij should reduce H (stronger cross-interaction)

## Key Outputs

| Artifact | Path |
|----------|------|
| Henry's constant function | `model/thermodynamic.py::compute_henrys_constant()` |
| Batch Henry's results | `model/saved/henrys_constant_screening.csv` |
| VP vs Henry comparison | `model/saved/vp_vs_henrys_comparison.csv` |
| Polyol sensitivity | `model/saved/henrys_polyol_sensitivity.csv` |
| Standout credibility | `model/saved/standout_candidates_credibility.csv` |
| Tests | `tests/test_henrys_constant.py` |

## Acceptance Criteria (When to Move On)

- [ ] `compute_henrys_constant()` persisted in `model/thermodynamic.py`, reproduces Step 21 values
- [ ] Henry's constant computed for all ~736 VP-passing candidates + commercial agents
- [ ] VP-only vs Henry-screened comparison table produced with clear collapse in candidate count
- [ ] Polyol-like sensitivity study completed (or limitation documented if teqp fails at all m values)
- [ ] Uncertainty and AD credibility table for ~20-50 Henry-screened standout candidates
- [ ] epsilon/k +/- 1 sigma and k_ij sensitivity computed for standouts
- [ ] 5+ tests pass in `tests/test_henrys_constant.py`
- [ ] All existing tests pass
- [ ] Report written at `docs/reports/22_henrys_screening_counterfactual.md`
- [ ] Commit: `step 22: batch Henry's constant screening with counterfactual and robustness analysis`
