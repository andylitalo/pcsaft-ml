# Step 24: Boiling Point Module & Commercial HFO Validation

## Objective

Build a self-contained module (`screening/hfo_screening.py`) that computes normal
boiling points from PC-SAFT parameters and defines HFO-1336mzz(Z) as a reference
molecule. Validate predicted boiling points against experimental values for ~15
commercial fluorinated compounds, with the explicit goal of establishing whether
this computation is suitable for **exploratory screening**. This step does **not**
constitute full validation of the downstream HFO-screening workflow by itself.

## Motivation

The Phase 2 analysis (Steps 20-23) conclusively disproved the original hypothesis
that PC-SAFT parameter similarity to **cyclopentane** identifies viable blowing agent
replacements. Zero HFOs passed all screening criteria. Meanwhile, the industry's
commercial winner — HFO-1336mzz(Z) (Honeywell Solstice LBA) — ranks 4,318 out of
4,663 in our cyclopentane-distance ranking.

The pivot: instead of searching for cyclopentane drop-ins, search for novel molecules
in the **commercially-proven HFO property space**, clustered near HFO-1336mzz(Z).

This step adds the critical missing property: **normal boiling point**. A blowing
agent must vaporize during the foam rise (~15-50°C boiling range for rigid PU/PIR
foam). Boiling point was not part of the Phase 2 screening, but it is the single
most important practical filter — a molecule with the right boiling point will
vaporize at the right time during processing, and a molecule outside this range is
disqualified regardless of its other properties.

Boiling point is computable from PC-SAFT by solving for the temperature at which
the equation-of-state saturation pressure equals 1 atm (101,325 Pa). The existing
`model/thermodynamic.py::compute_properties()` already computes VP at arbitrary T;
we just need a root-finding wrapper.

**ML role**: The trained RF model predicts (m, σ, ε/k) from SMILES. Those
ML-predicted parameters feed into the teqp equation of state to compute boiling
points — without the ML model, we have no thermodynamic properties for the 4,663
novel candidates. The boiling-point module therefore enables **first-pass,
hypothesis-generating** screening, not high-confidence process design. Step 25
depends on this module.

## Dependencies

- `model/thermodynamic.py` must exist with `compute_properties(m, sigma, epsilon_k, T)`
- `model/data/fluorinated_pcsaft.csv` must exist (literature PC-SAFT for ~35 fluorinated compounds)
- `scipy` must be available (already in environment for `brentq`)
- `model/predict.py` must exist with `predict_pcsaft()` (RF inference)

## Implementation Guide

### 24.1 Create `screening/hfo_screening.py`

New module — do NOT modify any existing files. Import from existing modules.

#### Reference constants

```python
HFO_1336MZZ = {
    "name": "HFO-1336mzz(Z)",
    "smiles": "F/C(=C\\C(F)(F)F)C(F)(F)F",
    "m": 2.7894,
    "sigma": 3.4521,
    "epsilon_k": 178.44,
    "source": "Konnova 2014",
    "boiling_point_K": 306.55,  # 33.4°C experimental
}
```

Also define a dict of known experimental boiling points (in K) for validation:

```python
KNOWN_BOILING_POINTS = {
    "FCF": 221.5,                           # R-32, -51.7°C
    "FCC(F)(F)F": 247.1,                    # R-134a, -26.1°C
    "FC(F)C(F)(F)F": 225.1,                 # R-125, -48.1°C
    "CC(F)(F)F": 225.9,                     # R-143a, -47.2°C
    "CC(F)F": 249.1,                        # R-152a, -24.0°C
    "FC(F)C(F)(F)C(F)(F)F": 256.8,          # R-227ea, -16.3°C
    "FC(F)F": 191.1,                        # R-23, -82.1°C
    "C=C(F)C(F)(F)F": 243.7,               # HFO-1234yf, -29.5°C
    "F/C=C/C(F)(F)F": 254.2,               # HFO-1234ze(E), -19.0°C
    "F/C(=C\\C(F)(F)F)C(F)(F)F": 306.55,   # HFO-1336mzz(Z), 33.4°C
    "CF": 194.8,                            # fluoromethane, -78.4°C
    "CCF": 235.5,                           # fluoroethane, -37.7°C
    "FC(F)CC(F)(F)F": 288.5,               # R-245fa, 15.3°C
    "CC(F)C(F)C(F)(F)F": 313.3,            # R-365mfc, 40.2°C
    "FC(F)Cl": 232.3,                       # R-22, -40.8°C
}
```

These span -82°C to +40°C, covering the full range of blowing agent candidates.

#### `compute_boiling_point(m, sigma, epsilon_k)`

Use `scipy.optimize.brentq` to find T such that:

```
compute_properties(m, sigma, epsilon_k, T)["vapor_pressure_Pa"] - 101325.0 = 0
```

- Search bracket: [100, 600] K
- If VLE fails at a given T (returns NaN), the objective should return a large
  positive or negative value (not NaN) so brentq can continue searching.
- If brentq fails entirely (ValueError), return NaN.
- Return: float (boiling point in Kelvin), or NaN on failure.

Import `compute_properties` from `model.thermodynamic` — do not duplicate the
implementation.

#### `batch_boiling_points(candidates_df)`

Apply `compute_boiling_point` row-wise to a DataFrame with columns `m`, `sigma`,
`epsilon_k`. Add a `boiling_point_K` column. Print progress every 100 rows.

Return the DataFrame with the added column.

#### `hfo_parameter_distance(m, sigma, epsilon_k, reference=HFO_1336MZZ, weights=None)`

Weighted Euclidean distance in normalized PC-SAFT parameter space:

```
d = sqrt(w_m * ((m - ref_m)/ref_m)^2 + w_s * ((s - ref_s)/ref_s)^2 + w_e * ((e - ref_e)/ref_e)^2)
```

Default weights: `{"m": 1.0, "sigma": 1.0, "epsilon_k": 1.0}` (equal — we are no
longer fighting the ε/k gap as with the cyclopentane reference).

#### `apply_hfo_filters(df)`

Apply the hard filters for HFO-centric screening. Expects a DataFrame with columns:
`smiles`, `m`, `sigma`, `epsilon_k`, `boiling_point_K`, `sa_score`.

Filters (configurable via keyword arguments with these defaults):

1. **Boiling point**: 288 K ≤ T_b ≤ 323 K (15-50°C)
2. **Contains C=C bond**: `Chem.MolFromSmiles(smi).HasSubstructMatch(Chem.MolFromSmarts("C=C"))`
3. **No Cl atoms**: count of Cl = 0 (zero ODP)
4. **F count ≥ 2**: at least 2 fluorine atoms (flammability suppression)
5. **SA score ≤ 4.5**: synthesizability

Return the filtered DataFrame and a dict of per-filter pass counts (for the
screening funnel).

#### `validate_boiling_points()`

Standalone validation function:
1. Load `model/data/fluorinated_pcsaft.csv`
2. For each molecule with a known boiling point (from `KNOWN_BOILING_POINTS`), compute
   the PC-SAFT-predicted boiling point from the **literature** PC-SAFT parameters
3. Compare predicted vs experimental: MAE, RMSE, R², max error
4. Return a DataFrame with columns: `smiles`, `name`, `m`, `sigma`, `epsilon_k`,
   `T_b_experimental_K`, `T_b_predicted_K`, `error_K`

This validation uses **literature PC-SAFT parameters** (not RF predictions) to
isolate the accuracy of the boiling-point-from-PC-SAFT computation itself. If the
EOS gives poor boiling points even from fitted parameters, the whole downstream
screening is suspect. Treat this as a **module check**, not a complete external
validation of Step 25.

#### `validate_boiling_points_rf()`

Second validation: predict PC-SAFT parameters via the RF model for the same
molecules, then compute boiling points from the **RF-predicted** parameters.
This measures the end-to-end pipeline accuracy (SMILES → RF → PC-SAFT → T_b).

Compare to experimental boiling points and report MAE, RMSE, R².

#### Required error analysis

For both validation paths (`lit params -> T_b` and `RF params -> T_b`), report:

1. The 3 worst outliers by absolute error
2. Any EOS/root-finding failures (NaN or no bracket found)
3. A short qualitative subgroup analysis (for example: light HFCs, commercial
   HFOs, heavier blowing agents)

Do **not** present a single parity metric as sufficient on its own. The point of
this step is to understand where the calculation is trustworthy enough for
screening and where it is not.

### 24.2 Create `tests/test_hfo_screening.py`

Tests (keep minimal — one smoke test per function):

1. `test_compute_boiling_point_cyclopentane`: compute T_b for cyclopentane
   (m=2.3655, σ=3.7114, ε/k=288.84). Expected: ~322 K (49°C). Assert within ±10 K.

2. `test_compute_boiling_point_hfo1336mzz`: compute T_b from HFO-1336mzz(Z)
   literature params. Expected: ~307 K (33.4°C). Assert within ±10 K.

3. `test_hfo_parameter_distance_self`: distance of HFO-1336mzz(Z) to itself = 0.

4. `test_hfo_parameter_distance_cyclopentane`: distance of cyclopentane to
   HFO-1336mzz(Z) should be substantial (> 0.3).

5. `test_apply_hfo_filters_smoke`: create a small DataFrame with 5 rows (mix of
   passing and failing molecules), verify correct filtering.

6. `test_validate_boiling_points_runs`: call `validate_boiling_points()`, assert
   it returns a DataFrame with > 10 rows and all expected columns.

### 24.3 Create validation figure

Save to `figures/24_hfo_boiling_point_validation/`.

**Figure: `boiling_point_parity.png`**

Parity plot: T_b(predicted from literature PC-SAFT) vs T_b(experimental) for the
~15 commercial compounds. Perfect prediction is the diagonal. Label each point with
its compound name. Include R² and MAE in the legend.

Optionally overlay a second series: T_b(predicted from RF PC-SAFT) vs experimental,
to show the additional error from the ML prediction step.

### 24.4 Write report

Write `docs/reports/24_hfo_boiling_point_validation.md` with:

1. **Title**: `# Results Report: Boiling Point Module & HFO Reference Validation`
2. **Method**: How T_b is computed from PC-SAFT (bisection on VP(T) = 1 atm)
3. **Validation table**:

| Compound | T_b exp (K) | T_b PC-SAFT lit (K) | Error (K) | T_b RF pred (K) | Error (K) |
|----------|-------------|---------------------|-----------|-----------------|-----------|
| R-32     | 221.5       | ...                 | ...       | ...             | ...       |
| ...      | ...         | ...                 | ...       | ...             | ...       |

4. **Summary metrics**: MAE, RMSE, R² for both validation paths
5. **Figures**: reference `figures/24_hfo_boiling_point_validation/boiling_point_parity.png`
6. **Key findings**: Is the PC-SAFT boiling point computation accurate enough for
   screening? What is the additional error from RF prediction?
7. **Outliers and failure modes**: worst compounds, EOS failures, and any
   subclass-specific bias
8. **Scope statement**: explicitly state that this step validates a boiling-point
   computation module for exploratory screening; it does not validate the full
   HFO-centric workflow or industrial use
9. **Readiness check**: checklist (see below)

## Key Outputs

| Artifact | Path |
|----------|------|
| HFO screening module | `screening/hfo_screening.py` |
| Tests | `tests/test_hfo_screening.py` |
| Parity plot | `figures/24_hfo_boiling_point_validation/boiling_point_parity.png` |
| Report | `docs/reports/24_hfo_boiling_point_validation.md` |

## Acceptance Criteria (When to Move On)

- [ ] `screening/hfo_screening.py` exists with all 6 functions: `compute_boiling_point`, `batch_boiling_points`, `hfo_parameter_distance`, `apply_hfo_filters`, `validate_boiling_points`, `validate_boiling_points_rf`
- [ ] `HFO_1336MZZ` and `KNOWN_BOILING_POINTS` constants defined
- [ ] `tests/test_hfo_screening.py` has ≥ 5 tests, all passing
- [ ] Boiling point validation: T_b from literature PC-SAFT achieves MAE < 15 K and R² > 0.9 against experimental values
- [ ] Report identifies worst outliers and any EOS/root-finding failures for both validation paths
- [ ] Report includes a short subgroup analysis, not only aggregate parity metrics
- [ ] Report explicitly states that this step validates a computation module for exploratory screening, not the full downstream scientific claim set
- [ ] Parity plot saved to `figures/24_hfo_boiling_point_validation/`
- [ ] Report at `docs/reports/24_hfo_boiling_point_validation.md` with validation table and summary metrics
- [ ] All existing tests still pass
- [ ] No existing files modified — only new files created
