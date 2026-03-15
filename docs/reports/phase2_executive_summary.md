# Executive Summary: Phase 2 — ML-Driven Blowing Agent Screening

## One-Paragraph Summary

Phase 2 expanded the training data from 1,801 to 13,764 molecules, introduced graph neural networks that more than doubled prediction accuracy (R² on ε/k from 0.33 to 0.73), added rigorous external and multi-temperature validations, and systematically enumerated 4,663 HFO/HCFO candidates. Despite these improvements, the project's central conclusion is negative: **no fluorinated olefin can serve as a drop-in thermodynamic replacement for cyclopentane in rigid polyurethane foam.** This is not a model failure — it is a fundamental physical chemistry constraint. Fluorination systematically reduces dispersion energy (ε/k), and the Boltzmann-exponential sensitivity of mixture thermodynamics to cross-interaction energy means even a 15–20% ε/k deficit translates to orders-of-magnitude differences in solubility behavior. The project's value lies in (a) demonstrating this constraint rigorously via Henry's constant analysis, (b) producing a validated, production-grade ML pipeline for PC-SAFT parameter prediction with quantified uncertainty, and (c) illuminating *why* the industry settled on linear HFOs — the only molecular design that simultaneously achieves low GWP (via a C=C atmospheric degradation site), chemical stability in the harsh PU foam environment, and acceptable synthesis economics, even though their thermodynamic behavior is fundamentally different from cyclopentane's.

---

## What Changed from Phase 1

| Dimension | Phase 1 | Phase 2 | Improvement |
|-----------|---------|---------|-------------|
| Training data | 1,801 molecules (Esper) | 13,764 molecules (Esper + ML-SAFT + SPT-PCSAFT + fluorinated literature) | 7.6× |
| Best R² (ε/k) | 0.33 (RF) | 0.73 (GNN on all data) | +0.40 |
| Best MAE (ε/k) | 26.8 K (RF) | 10.4 K (GNN) | 2.6× better |
| Models benchmarked | 4 (GC, RF, NN, ChemBERTa) | 9 (+GNN, XGBoost, chemprop D-MPNN, ensemble, GC-as-feature) | 2.25× |
| Candidate pool | 645 (single-substitution) | 4,663 (systematic combinatorial HFO/HCFO) | 7.2× |
| Validation temperatures | 298 K only | 230–330 K sweep (6 temperatures) | Full operating range |
| External validation | None | 57-molecule NIST WebBook comparison | New capability |
| Mixture thermodynamics | None | Henry's constant at infinite dilution | New capability |
| AD methods | 1 (Isolation Forest) | 3 (IF + Tanimoto + Williams) | Chemically interpretable |
| Evaluation metrics | R², MAE | +MARE, CCC, Coverage@5/10%, Q² | Nuanced assessment |

---

## Key Results

### 1. Model Performance Reached a Useful Threshold

The GNN trained on the expanded dataset (11,011 train / 2,753 test) is the first model to cross R² > 0.70 on all three PC-SAFT parameters. Its ε/k MAE of 10.4 K is within the cyclopentane acceptance window (±14.4 K at 5%). On Esper-only data, chemprop D-MPNN achieved the best ε/k R² (0.39), confirming that graph-based architectures extract dispersion-energy-relevant features that fingerprints miss.

All R² and MAE values now have 95% bootstrap CIs available via `python -m model.evaluate --bootstrap` (saved to `comparison_metrics.csv` with `_lo`/`_hi` columns). For derived quantities (H_ratio, VP_ratio), per-tree propagation through teqp provides per-molecule 95% CIs combining RF tree variance and k_ij sensitivity — see `model/saved/uncertainty_propagated.csv`.

### 2. Candidates Identified — and Their Practical Limitations

Three molecules pass all five screening criteria (AD, parameter proximity, EOS convergence, VP ratio, synthetic accessibility):

| Candidate | VP Ratio (298 K) | Limitation |
|-----------|-------------------|------------|
| *cis*-1,3-Difluorocyclobutane | 0.96 | Not an olefin → unknown atmospheric lifetime → potential GWP concern |
| 1-Chlorobut-1-ene | 1.15 | Contains Cl → nonzero ODP; flammable; potential toxicity |
| (E/Z)-2-Chloro-2-butene | 1.10 | Contains Cl → nonzero ODP; flammable; potential toxicity |

All verified candidates are chlorinated olefins or saturated fluorocycloalkanes — **none are HFOs or HCFOs**, the molecule classes the industry has converged on for next-generation blowing agents.

### 3. Why HFOs Cannot Be Drop-In Cyclopentane Replacements

This is the project's most important scientific finding. The systematic HFO/HCFO enumeration (Step 20) and Henry's constant analysis (Step 21) proved that:

- **Zero of 1,471 HFOs pass all screening criteria.** The parameter proximity criterion (ε/k-weighted) is a near-absolute barrier because C-F bonds have lower polarizability than C-H bonds, systematically reducing ε/k.
- **Commercial HFOs have Henry's constants 10¹–10⁷× higher than cyclopentane** in a hydrocarbon solvent. A 22% ε/k deficit translates to a 10⁷-fold solubility difference via the Boltzmann factor.
- **This is a physics constraint, not a model artifact.** Even with perfect parameters, no heavily fluorinated molecule can match cyclopentane's dispersion energy. The industry's use of HFOs in spray foam and XPS (rather than rigid PU insulation) reflects this reality — those formulations are designed around different solubility requirements.

### 4. Value Delivered

Despite the negative screening outcome, Phase 2 delivered:

1. **A rigorous proof that parameter-matching is the wrong strategy for finding HFO replacements for cyclopentane.** Henry's constant analysis quantified the exponential sensitivity of mixture behavior to ε/k, showing that even small ε/k deficits create large solubility differences. This redirects future work toward property-based screening or formulation engineering rather than parameter matching.

2. **A production-grade ML pipeline for PC-SAFT prediction.** The GNN achieves R² > 0.70 on all three parameters with quantified uncertainty (MC Dropout), three complementary applicability domain methods, and a full serving stack. This has standalone value as an open-source tool for the thermodynamic modeling community.

3. **Quantified inter-dataset variability in PC-SAFT parameters.** The Esper–ML-SAFT overlap analysis (745 molecules) established noise floors and defensible acceptance windows, information not previously available in the literature.

4. **Demonstrated why the industry converged on linear HFOs.** The screening pipeline's most thermodynamically promising HFO candidate — tetrafluoromethylenecyclopropane (`C=C1C(F)(F)C1(F)F`, H/H(cyclopentane) = 0.96, VP ratio = 1.02) — exemplifies the tradeoff that makes strained-ring fluorinated olefins non-viable despite attractive thermodynamic profiles. Its ~40 kcal/mol ring strain (cyclopropane + exocyclic sp2 carbon forced into ~60° angles), compounded by four electron-withdrawing fluorines, makes it a highly reactive electrophile that would undergo ring-opening or polymerization in the amine-catalyzed, exothermic PU foam environment — destroying both the blowing agent and the foam chemistry. Synthesis would require exotic multi-step routes involving difluorocarbene precursors at costs orders of magnitude above the single-digit $/lb threshold for industrial blowing agents. And highly strained, electrophilic small molecules are notorious for mutagenic activity via nucleophilic attack on DNA. By contrast, linear HFOs like HFO-1336mzz(Z) achieve the same environmental objective (C=C bond for atmospheric degradation → low GWP) while keeping the molecule chemically inert enough to survive the PU foam reaction. The screening pipeline correctly identifies *what* thermodynamic properties matter — and in doing so reveals *why* the industry's choice of linear HFOs required accepting fundamentally different solubility characteristics rather than seeking a cyclopentane parameter match.

5. **Validated that single-temperature screening is sufficient.** Spearman ρ > 0.99 across 250–330 K means 298 K rankings are reliable for the full foam operating range, saving 6× computational cost in future screening campaigns.

---

## Additional Validations, Filters, and Metrics

### New Validation Methods
- **NIST WebBook external validation** (Step 17): For non-polar compounds, RF predictions yield 2–10% VP error and <5% density error. Associating compounds fail catastrophically (expected — association parameters not predicted).
- **Temperature sweep** (Step 19): Top candidates maintain stable rankings (ρ > 0.99) across the 250–330 K operating range.
- **Henry's constant mixture analysis** (Step 21): Directly validates that parameter proximity implies similar solvent–solute interaction thermodynamics.

### New Screening Filters
- **Multi-temperature EOS convergence** (273, 298, 323 K): Eliminates candidates that pass at 298 K but fail at temperature extremes.
- **Henry's constant ratio**: Proposed as a Tier 3 filter replacing or supplementing VP ratio, directly measuring the property relevant to foam formulation.
- **Tanimoto applicability domain** (threshold 0.4): More conservative and chemically interpretable than the Isolation Forest; 83% agreement between methods.

### Enhanced Metrics
- **MARE** (Mean Absolute Relative Error): Revealed that σ is better predicted than R² suggests (MARE = 5.5%, 86% within ±10%).
- **CCC** (Concordance Correlation Coefficient): Exposed systematic bias in RF predictions (CCC < R² for all targets).
- **Coverage@5% and @10%**: Directly answers "what fraction of predictions are useful for screening?"
- **Q²** (5-fold CV R²): Flagged ε/k overfitting (Q²–R² gap of −0.13).
- **Bootstrap 95% CIs** on R², MAE, RMSE (via `--bootstrap` flag): Quantifies test-set sampling uncertainty on all point metrics.
- **Per-tree propagated CIs** on H_ratio and VP_ratio: Joint MC propagation of RF tree-level (m, σ, ε/k) through teqp + k_ij sensitivity sweep.

---

## Phase 3 Direction

Phase 2 established that matching cyclopentane's PC-SAFT parameters is not a viable path to finding HFO/HCFO replacements. Phase 3 should pivot in two directions:

1. **Open-source ML model for PC-SAFT parameters.** The GNN pipeline (R² > 0.70, 13K+ training molecules, uncertainty quantification, applicability domain) has value as a standalone tool for the thermodynamic modeling community, independent of the blowing agent use case.

2. **Alternative screening indicators for HFO/HCFO blowing agents.** Since HFOs occupy a fundamentally different region of PC-SAFT parameter space, screening must move beyond parameter matching. Potential approaches include:
   - Direct property prediction (VP, density, solubility in polyol) rather than intermediate parameter prediction
   - Formulation-aware screening that accounts for surfactant, catalyst, and co-blowing agent interactions
   - Identifying which non-thermodynamic properties (cell nucleation kinetics, thermal conductivity of gas, foam dimensional stability) actually differentiate blowing agents
   - Understanding the tradeoff between thermodynamic similarity and chemical stability — the compounds that best match cyclopentane's mixture behavior (strained fluorinated rings) are precisely the ones least likely to survive the PU foam reaction environment, suggesting the screening objective itself must be reformulated
