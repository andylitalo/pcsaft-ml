# Step 25: HFO-Centric Screening, Figures & Report

## Objective

Run the full blowing-agent screening pipeline using **HFO-1336mzz(Z)** as the
reference molecule instead of cyclopentane. Apply physically-motivated filters
(boiling point, GWP, ODP, flammability), rank by PC-SAFT proximity to HFO-1336mzz(Z),
generate analysis figures, and write the capstone report.

This step should be framed as **exploratory hypothesis ranking in an out-of-domain
chemical space**, not as validated discovery of industrially ready blowing-agent
replacements.

## Motivation

Steps 20-23 proved that filtering for cyclopentane-like PC-SAFT parameters is the
wrong question — the industry's commercial HFO blowing agents succeed not by matching
cyclopentane's thermodynamics, but by operating in a **different property region**
with adapted foam formulations (specialized silicone surfactants, redesigned polyol
blends, modified processing conditions).

This step asks a better exploratory question: **are there novel fluorinated
olefins in the commercially-proven HFO property space that are worth prioritizing
for follow-up?** The reference
is HFO-1336mzz(Z) (Honeywell Solstice LBA), the leading fourth-generation blowing
agent:

- **Boiling point**: 33.4°C (ideal for rigid PU/PIR foam processing)
- **GWP**: 2 (near-zero, from C=C double bond → short atmospheric lifetime)
- **ODP**: 0 (no chlorine)
- **Flammability**: non-flammable (ASHRAE A1, sufficient fluorination)
- **PC-SAFT**: m=2.7894, σ=3.4521 Å, ε/k=178.44 K

The screening criteria shift accordingly:

| Phase 2 (cyclopentane) | Phase 3 (HFO-centric) |
|------------------------|-----------------------|
| VP ratio to cyclopentane [0.5, 1.5] | Boiling point 15-50°C (288-323 K) |
| Henry's constant ratio [0.5, 2.0] (hard gate) | Henry's constant (soft preference, lower is better) |
| Parameter distance to cyclopentane (3:1:1 ε/k-weighted) | Parameter distance to HFO-1336mzz(Z) (1:1:1 equal) |
| Liquid density ratio to cyclopentane | Dropped (irrelevant for HFO formulations) |
| SA ≤ 4.5, C=C required, Cl allowed | SA ≤ 4.5, C=C required, **Cl = 0** (zero ODP), **F ≥ 2** |

**ML role**: The trained RF model predicts (m, σ, ε/k) for all 4,663 candidates
from their SMILES strings. These ML-predicted parameters feed into the teqp equation
of state to compute boiling points, vapor pressures, and Henry's constants. Without
the ML model, we would have thermodynamic properties for zero of the novel candidates.
The filtering criteria are rule-based, but they operate entirely on ML-generated
inputs.

Because many fluorinated candidates are structurally far from the training set,
the outputs of this step should be treated as **screening hypotheses** rather than
validated recommendations.

## Dependencies

- **Step 24** completed: `screening/hfo_screening.py` exists with `compute_boiling_point`,
  `batch_boiling_points`, `hfo_parameter_distance`, `apply_hfo_filters`
- `screening/generate.py::generate_systematic_candidates()` exists (from Step 20)
- `model/predict.py::predict_pcsaft()` exists (RF model trained)
- `model/thermodynamic.py::compute_henrys_constant()` exists (from Step 21)
- `model/data/fluorinated_pcsaft.csv` exists (commercial HFO reference data)

## Implementation Guide

### 25.1 Create `scripts/step25_hfo_centric_screening.py`

Main script that runs the full pipeline. Use argparse for configuration.

#### Pipeline stages:

Before Stage 1, print a warning banner to stdout and repeat it in the report:

> "This workflow ranks OOD hypotheses using ML-predicted PC-SAFT parameters and
> EOS-derived properties. It is suitable for prioritization and analysis, not for
> final material selection without experimental validation."

**Stage 1: Generate candidates**

Reuse `generate_systematic_candidates()` from `screening.generate` with the same
settings as Step 20: 16 alkene backbones, max_cl=1, max_mw=200. This produces the
same ~4,663 candidates (after SA filtering in Stage 3).

Note: even though the final filter requires Cl=0, generate with max_cl=1 to
maintain the full enumeration for comparison statistics (how many HCFOs were
excluded by the ODP filter).

**Stage 2: Predict PC-SAFT parameters**

Call `predict_pcsaft()` on all candidate SMILES. This invokes the trained RF model.
The output DataFrame has columns: `smiles`, `m`, `sigma`, `epsilon_k`.

Also load `model/data/fluorinated_pcsaft.csv` and compute boiling points for the
commercial reference molecules (for comparison in the report).

**Stage 3: Compute boiling points**

Call `batch_boiling_points()` from `screening.hfo_screening` on the predictions
DataFrame. This adds a `boiling_point_K` column.

Print summary statistics: median, mean, std, min, max boiling points. Print count
of NaN (EOS convergence failures).

**Stage 4: Compute SA scores**

For each SMILES, compute the SA score using `sascorer.calculateScore()`. Add
`sa_score` column.

**Stage 5: Apply HFO-centric filters**

Call `apply_hfo_filters()`. This returns the filtered DataFrame and per-filter
pass counts. Print the screening funnel:

```
Total candidates:          4,663
Boiling point [288, 323K]: ???
Contains C=C:              ???
Cl = 0 (zero ODP):         ???
F ≥ 2 (flame suppression): ???
SA ≤ 4.5:                  ???
All filters:               ???
```

**Stage 6: Rank by HFO proximity**

For all filter-passing candidates, compute `hfo_parameter_distance()` and sort
ascending (closest to HFO-1336mzz(Z) first). Add `hfo_distance` column.

**Stage 7: Compute Henry's constant for top candidates**

For the top 50 filter-passing candidates (or all if fewer than 50 pass), compute
Henry's constant in hexane at 298.15 K using `compute_henrys_constant()` from
`model.thermodynamic`. Add `H_Pa` and `H_ratio` columns (ratio vs cyclopentane).

Henry's constant is a **soft preference** — lower is better (reduces surfactant
requirements) but is not a hard gate. Commercial HFOs have H ratios of 35-10^7
relative to cyclopentane and still work.

Also compute Henry's constant for HFO-1336mzz(Z) itself (from literature params)
as the reference point for comparison.

**Stage 8: Save results**

Save to `screening/results/hfo_centric_ranked.csv` with columns:

```
smiles, m, sigma, epsilon_k, boiling_point_K, boiling_point_C, sa_score,
n_fluorine, n_chlorine, has_cc_double_bond, hfo_distance,
H_Pa, H_ratio, mol_class (hfo/hcfo/cl_olefin), backbone
```

Print the top 20 candidates to stdout as **top exploratory hypotheses**, not
"best candidates" or "recommended replacements."

### 25.2 Generate figures

Save all figures to `figures/25_hfo_centric_screening/`.

**Figure 1: `screening_funnel.png`**

Horizontal bar chart showing candidate count at each filter stage. Bars colored
by filter type. Annotate each bar with the count and percentage of total.
Show the funnel from 4,663 → ... → final passing count.

**Figure 2: `boiling_point_vs_hfo_distance.png`**

Scatter plot for ALL candidates (not just filter-passing). X-axis: HFO parameter
distance. Y-axis: predicted boiling point (°C). Color by number of fluorine atoms
(colorbar). Draw horizontal lines at 15°C and 50°C (boiling point acceptance
window). Mark HFO-1336mzz(Z) with a star. Mark the filter-passing candidates with
distinct markers.

This figure visualizes where candidates live in the (distance, boiling point) space
and whether the filter window captures a meaningful cluster near the reference.

**Figure 3: `parameter_space_comparison.png`**

Three-panel figure (1 row × 3 columns) showing the PC-SAFT parameter distributions:

- Panel 1: m histogram for all candidates vs filter-passing vs HFO-1336mzz(Z) reference line
- Panel 2: σ histogram, same layout
- Panel 3: ε/k histogram, same layout

This shows whether the filter-passing candidates cluster near HFO-1336mzz(Z) in
each parameter dimension independently.

**Figure 4: `henrys_soft_preference.png`**

For the top 50 candidates with computed Henry's constants: scatter of H_ratio
(log scale, y-axis) vs HFO distance (x-axis). Mark HFO-1336mzz(Z) reference with
a star. Draw a horizontal line at H_ratio for HFO-1336mzz(Z). Color by boiling
point. Annotate the 5 candidates with lowest Henry's constant.

This visualizes the soft-preference dimension: among filter-passing candidates,
which ones also have the best (lowest) Henry's constant?

**Figure 5: `structural_motif_analysis.png`**

Bar chart of the backbone distribution among filter-passing candidates. Group by
the alkene backbone from which each candidate was generated (ethene, propene,
1-butene, 2-butene, isobutylene, cyclopropene, cyclobutene, ...). Color by
average HFO distance within each backbone group.

This reveals which molecular scaffolds are most promising for HFO-like properties.

### 25.3 Write report

Write `docs/reports/25_hfo_centric_screening.md`.

**Follow the report template from `CLAUDE.md`** with these sections:

**1. Title**: `# Results Report: HFO-Centric Screening (4,663 candidates, HFO-1336mzz(Z) reference)`

**2. Motivation for Pivot** (2-3 paragraphs):

Why the cyclopentane-centric approach was the wrong question. Three key points:
- Zero HFOs pass cyclopentane-similarity screening (Steps 20-23)
- Commercial HFOs succeed via formulation redesign, not parameter matching
- Henry's constant is not a deal-breaker when emulsifiers/surfactants handle
  HFO-polyol incompatibility (industry practice: specialized silicone surfactants)
- Boiling point is the missing critical filter (determines vaporization timing)

**3. Method** (2 paragraphs + filter table):

Pipeline: SMILES → RF model → PC-SAFT (m, σ, ε/k) → teqp EOS → boiling point.
Then rule-based filters applied to ML-predicted properties.

Table of filter criteria with rationale (from the table in the Motivation section
above).

Explicitly state the ML role: the RF model is the engine that generates
thermodynamic properties for 4,663 novel molecules with no experimental data.
Also explicitly state the limitation: these are OOD, model-generated properties
used for prioritization, not validated replacement metrics.

**4. Screening Results** (funnel + tables):

Screening funnel (from Stage 5 output).

Top 20 exploratory hypotheses table:

| Rank | SMILES | Name | T_b (°C) | HFO dist | ε/k (K) | n_F | SA | H_ratio |
|------|--------|------|----------|----------|---------|-----|-----|---------|

Compare with commercial HFOs (HFO-1336mzz(Z), HFO-1234ze(E), R-245fa, R-365mfc):
where do they fall in the ranking? Do any filter-passing candidates have properties
intermediate between the commercial agents?

**5. Analysis: Structural Motifs Near HFO-1336mzz(Z)** (2-3 paragraphs):

Which backbones dominate among filter-passing candidates? What structural features
(chain length, branching, ring inclusion, fluorine placement) produce HFO-like
boiling points? Are there candidates with slightly higher ε/k than HFO-1336mzz(Z)
— meaning better inherent solubility in polyols — while retaining the right
boiling point? These would reduce surfactant requirements relative to current
commercial agents.

Keep the interpretation conservative: this section is about **patterns in a model-
generated hypothesis library**, not proof that these motifs will translate to real
formulation performance.

**6. Henry's Constant as Soft Preference** (1-2 paragraphs):

Among filter-passing candidates, which have the lowest Henry's constant? How does
this compare to HFO-1336mzz(Z)? Frame as: lower H means easier formulation
(less surfactant needed), but H is not a gate — current commercial agents have
H ratios of 35-10^7 vs cyclopentane and work fine.

**7. Figures**: reference all 5 figures by path.

**8. Limitations**:

- RF model ε/k R² = 0.33 on Esper test set; predictions are screening-quality
- Boiling point validation MAE from Step 24 (cite the number)
- 0% of candidates in-domain (Tanimoto ≥ 0.4); predictions are extrapolations
- 3-parameter PC-SAFT does not model association (real polyol has H-bonding)
- Flammability proxy (F count ≥ 2) is crude; ASHRAE testing required for real
  classification
- Thermal stability, toxicity, environmental fate are not screened
- Gas-phase thermal conductivity (determines foam k-factor) is not computed — it
  is a critical property for insulation foam and would be the next filter to add
- Solvent proxy: Henry's constants are computed in hexane (or another model
  solvent), not in a real polyol formulation
- `k_ij = 0` and EOS-derived properties from predicted parameters add a second
  layer of model error beyond the ML prediction itself
- Output ranking is for exploratory prioritization only, not industrial decision-making

**9. Readiness Check**: checklist (see below).

## Key Outputs

| Artifact | Path |
|----------|------|
| Screening script | `scripts/step25_hfo_centric_screening.py` |
| Ranked candidates CSV | `screening/results/hfo_centric_ranked.csv` |
| Screening funnel | `figures/25_hfo_centric_screening/screening_funnel.png` |
| Boiling point vs distance | `figures/25_hfo_centric_screening/boiling_point_vs_hfo_distance.png` |
| Parameter space comparison | `figures/25_hfo_centric_screening/parameter_space_comparison.png` |
| Henry's soft preference | `figures/25_hfo_centric_screening/henrys_soft_preference.png` |
| Structural motif analysis | `figures/25_hfo_centric_screening/structural_motif_analysis.png` |
| Report | `docs/reports/25_hfo_centric_screening.md` |

## Acceptance Criteria (When to Move On)

- [ ] `scripts/step25_hfo_centric_screening.py` runs end-to-end without errors
- [ ] Screening funnel printed and saved; candidate counts at each filter stage are plausible
- [ ] `screening/results/hfo_centric_ranked.csv` saved with all required columns
- [ ] All 5 figures saved to `figures/25_hfo_centric_screening/`
- [ ] Henry's constant computed for top candidates (or all filter-passing if < 50)
- [ ] Report at `docs/reports/25_hfo_centric_screening.md` with all 9 sections
- [ ] Report honestly states limitations (model quality, AD, missing properties)
- [ ] Report compares filter-passing candidates to commercial HFOs
- [ ] Report explicitly labels the output as exploratory OOD hypothesis ranking, not validated candidate discovery
- [ ] Top-ranked molecules are described as hypotheses for follow-up measurement, not recommendations
- [ ] All existing tests still pass
- [ ] No existing files modified — only new files created
