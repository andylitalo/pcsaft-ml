# Step 23: Novel Predictions Package, Narrative Report & Figures

## Objective

Package the ~4,612 novel PC-SAFT predictions with trust signals as a standalone
deliverable. Write the project's capstone narrative report and generate five key
figures that tell the ML-for-chemistry story. Optionally produce an active learning
shortlist of molecules prioritized for experimental characterization.

## Motivation

The computational results from Steps 20-22 contain a compelling story about ML's value
for chemistry, but it is scattered across multiple reports. This step consolidates
everything into a single interview-ready narrative and packages the novel predictions
as a reusable dataset.

The story has two co-equal leads:
1. **The rule, quantified at scale**: fluorination systematically reduces epsilon/k,
   making most HFOs/HCFOs thermodynamically incompatible with cyclopentane. ML
   quantified the boundary across 4,663 molecules.
2. **The exceptions ML found**: a narrow class of lightly-fluorinated strained rings
   has PC-SAFT parameters near cyclopentane despite containing fluorine. None have
   literature PC-SAFT parameters. A chemist screening 20 representative HFOs by hand
   would not have tested these strained-ring edge cases.

## Dependencies

- Requires: `model/saved/henrys_constant_screening.csv` (from Step 22)
- Requires: `model/saved/vp_vs_henrys_comparison.csv` (from Step 22)
- Requires: `model/saved/henrys_polyol_sensitivity.csv` (from Step 22)
- Requires: `model/saved/standout_candidates_credibility.csv` (from Step 22)
- Requires: `screening/results/ranked_candidates_systematic.csv` (from Step 20)
- Requires: `model/data/spt_pcsaft.csv` for overlap validation
- Requires: trained RF model for prediction extraction
- Requires: matplotlib for figure generation

## Implementation Guide

### 23.1 Package Novel PC-SAFT Predictions

Create `scripts/package_novel_predictions.py`.

Load all 4,663 candidates from `ranked_candidates_systematic.csv`. Cross-reference
against SPT-PCSAFT (`model/data/spt_pcsaft.csv`) by InChI or canonical SMILES to
identify the ~51 overlapping molecules and the ~4,612 novel ones.

For each molecule, assemble:
- **Identity**: SMILES, InChI (computed via RDKit), common name (if available from
  PubChem or predefined mapping)
- **Predicted PC-SAFT**: m, sigma, epsilon_k
- **Uncertainty**: std_m, std_sigma, std_epsilon_k (from RF tree variance)
- **AD**: tanimoto_max (to nearest training molecule), ad_flag (in_domain/warning/ood
  based on Tanimoto thresholds: >= 0.4 in-domain, [0.3, 0.4) warning, < 0.3 ood)
- **Screening**: param_distance, vp_ratio_298K, H_ratio (if computed in Step 22),
  log10_H_ratio
- **Practical**: MW, n_fluorine, n_chlorine, has_double_bond (C=C), sa_score
- **Molecule class**: hfo / hcfo / cl_olefin (based on atom content)

Save to `model/saved/novel_pcsaft_predictions.csv`.

### 23.2 Validate Against SPT-PCSAFT Overlap

For the ~51 overlapping molecules (present in both enumeration and SPT-PCSAFT):
- Compare RF-predicted (m, sigma, epsilon_k) vs SPT-PCSAFT fitted values
- Report MAE, RMSE, and R^2 for each parameter on this overlap set
- This is the credibility anchor: it shows how accurate the predictions are for
  molecules in the relevant chemical space (fluorinated olefins)

Include a brief table in the report:

| Parameter | N overlap | MAE | RMSE | R^2 |
|-----------|-----------|-----|------|-----|
| m         | ~51       | ... | ...  | ... |
| sigma     | ~51       | ... | ...  | ... |
| epsilon/k | ~51       | ... | ...  | ... |

### 23.3 Generate Figures

Save all figures to `figures/23_ml_chemistry_narrative/`.

**Figure 1: Visual causal chain — epsilon/k vs fluorine count**

A scatter plot of epsilon/k vs number of fluorine atoms for all 4,663 candidates,
colored by molecule class (HFO, HCFO, Cl-olefin). Overlay cyclopentane as a horizontal
reference line at epsilon/k = 288.84 K. Annotate the standout strained-ring candidates
and commercial HFOs. Show the epsilon/k gap visually. This communicates the general
rule (fluorination lowers epsilon/k) and the exceptions (strained rings near the line)
in a single image.

**Figure 2: VP-only vs Henry-screened comparison**

Two-panel figure:
- Left: histogram of H/H(ref) for all ~736 VP-passing candidates, log scale x-axis.
  Shade the [0.5, 2.0] acceptance region. Label how many fall inside vs outside.
- Right: bar chart showing candidate count at each screening tier: all 4,663 ->
  VP-passing 736 -> Henry-screened ~20-50 -> uncertainty-robust (subset).

**Figure 3: Exponential sensitivity curve**

Plot H/H(ref) vs epsilon_ij/epsilon_ij(ref) for all candidates with computed Henry's
constants. Overlay the theoretical Boltzmann curve (H/H_ref ~ exp(-delta_epsilon_ij/kT)).
Label commercial HFOs, verified Cl-olefins, and the standout strained-ring candidates.
This is the physics-driven figure that explains the exponential penalty.

**Figure 4: Uncertainty envelope**

For the ~20-50 standout candidates (from `standout_candidates_credibility.csv`):
Plot H/H(ref) with error bars spanning [H_ratio_minus_1sigma, H_ratio_plus_1sigma].
Sort candidates by nominal H ratio along the y-axis. Draw the [0.5, 2.0] acceptance
band as a vertical shaded region. Show which candidates are robust (error bars within
band) vs uncertain (error bars crossing the boundary). This motivates experimental
validation for the boundary cases.

**Figure 5: Pareto frontier**

Plot |H/H(ref) - 1| on the y-axis (proximity to ideal, lower is better) vs n_fluorine
on the x-axis (proxy for GWP benefit, higher is better). Lower-right is the
Pareto-optimal corner (close to cyclopentane AND highly fluorinated). Color by molecule
class. Label the Pareto-front candidates. This visualizes the fundamental trade-off:
environmental performance vs thermodynamic compatibility.

### 23.4 Write Narrative Report

Write `docs/reports/23_ml_chemistry_narrative.md` following the seven-point arc.

Required sections:

**Title**: `# Results Report: ML-Driven Blowing Agent Screening — What ML Found That Chemistry Alone Could Not`

**1. The Industrial Question** (1-2 paragraphs):
Cyclopentane is the dominant blowing agent in rigid PU foam but is flammable (GWP ~5).
Can ML-predicted PC-SAFT parameters identify lower-GWP fluorinated replacements with
similar thermodynamic behavior?

**2. The Search** (1 paragraph + table):
4,663 candidates from systematic enumeration (Step 20). 4,612 have no literature
PC-SAFT parameters. Predicted by RF model trained on Esper + ML-SAFT + fluorinated
data (~2,000 molecules).

**3. The Rule, Quantified at Scale** (2-3 paragraphs + reference Figure 1):
Zero HFOs pass all five criteria. Fluorination reduces epsilon/k. Boltzmann
amplification makes the penalty exponential. Include the epsilon_ij deficit table
from Step 21 (0% deficit -> H ratio 1, 22% deficit -> H ratio 10^7). This explains
why the PU foam industry does not use HFOs as direct cyclopentane replacements.

**4. The Exceptions ML Found** (2-3 paragraphs + candidate table):
The strained-ring exception class. List top ~10 candidates with predicted params,
H ratio, AD status, uncertainty. Highlight tetrafluoromethylenecyclopropane (4 F atoms,
epsilon/k = 249 K, H/H_ref = 0.96). Explain why ring strain and geometric packing
compensate for the fluorination-induced epsilon/k reduction. Note that none have
literature PC-SAFT parameters — they exist only as ML predictions. Acknowledge
practical limitations honestly (ring strain -> thermal instability, Cl -> ODP).

**5. VP-Only vs Henry-Screened Counterfactual** (1-2 paragraphs + reference Figure 2):
The collapse from ~736 to ~20-50 candidates. Why VP matching is misleading for
formulation-relevant screening. ML's exact role: predicted PC-SAFT parameters enable
both screens; the Henry's-constant computation itself is deterministic thermodynamics
via teqp.

**6. Trust Signals and Robustness** (2 paragraphs + reference Figure 4):
Uncertainty analysis: which candidates are robust within +/- 1 sigma, which could flip.
Polyol sensitivity result (Spearman rho from Step 22). k_ij sensitivity. SPT-PCSAFT
overlap validation MAE/RMSE from 23.2. Frame the candidates as hypotheses for
experimental validation, not proven replacements.

**7. Novel Data and Transferable Methodology** (1-2 paragraphs):
4,612 novel PC-SAFT predictions packaged with trust signals. Pipeline is
target-agnostic: change the reference molecule or candidate library and the same
workflow applies. Active learning shortlist (if computed) demonstrates ML closing the
loop between computation and lab work.

**Limitations** section (required, not optional — honesty is the credibility signal):
- epsilon/k R^2 = 0.33 on Esper test set; predictions are screening-quality, not
  reference values
- Only ~210 fluorinated compounds in training data (~10.7% of training set)
- 3-parameter PC-SAFT does not model association (real polyol has H-bonding)
- Hexane proxy for solvent; polyol sensitivity is a robustness check, not validation
- Strained-ring candidates may have stability, reactivity, or toxicity issues not
  captured by PC-SAFT screening
- AD fallback (100% in-domain) was used in Step 20 because ChemBERTa AD was
  unavailable; real AD rejection rate would be ~28%

**Figures** section: reference all 5 figure paths in `figures/23_ml_chemistry_narrative/`.

### 23.5 (Optional) Active Learning Shortlist

If Step 22 results permit, identify 5-10 molecules for experimental prioritization.

Criteria (all must hold):
- Henry-screened standout: H/H_ref in [0.5, 2.0] at nominal prediction
- High epsilon_k uncertainty: std_epsilon_k in top quartile of standouts
- Decision-boundary proximity: H ratio could cross 1.0 within +/- 1 sigma
- Synthetically accessible: SA score < 3.5
- In-domain or warning (not OOD) by AD

Frame as: "These are the molecules where one VLE experiment would most change the
screening outcome."

Save to `model/saved/active_learning_shortlist.csv`.

## Key Outputs

| Artifact | Path |
|----------|------|
| Novel predictions CSV | `model/saved/novel_pcsaft_predictions.csv` |
| Active learning shortlist | `model/saved/active_learning_shortlist.csv` (optional) |
| Figure 1: Causal chain | `figures/23_ml_chemistry_narrative/epsilon_k_vs_fluorines.png` |
| Figure 2: VP vs Henry | `figures/23_ml_chemistry_narrative/vp_vs_henrys_comparison.png` |
| Figure 3: Exponential curve | `figures/23_ml_chemistry_narrative/henrys_sensitivity.png` |
| Figure 4: Uncertainty | `figures/23_ml_chemistry_narrative/uncertainty_envelope.png` |
| Figure 5: Pareto | `figures/23_ml_chemistry_narrative/pareto_frontier.png` |
| Narrative report | `docs/reports/23_ml_chemistry_narrative.md` |

## Acceptance Criteria (When to Move On)

- [ ] `model/saved/novel_pcsaft_predictions.csv` contains ~4,612 rows with all required columns
- [ ] SPT-PCSAFT overlap validation table computed and included in report
- [ ] All 5 figures generated and saved to `figures/23_ml_chemistry_narrative/`
- [ ] Narrative report written at `docs/reports/23_ml_chemistry_narrative.md` with all 7 sections + Limitations
- [ ] Report honestly states limitations (epsilon/k R^2, training data coverage, 3-param limitation)
- [ ] (Optional) Active learning shortlist of 5-10 molecules saved
- [ ] All existing tests pass
- [ ] Commit: `step 23: narrative report, novel predictions package, and key figures`
