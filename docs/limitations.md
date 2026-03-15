# Scientific Limitations

This document catalogues the known scientific limitations of the ML-driven PC-SAFT screening pipeline, their estimated impact, and mitigations where available. It is intended for internal review before any external presentation and should be updated as limitations are addressed.

## 1. PC-SAFT Parameter Variability (Target Noise)

### What it is

PC-SAFT parameters (m, σ, ε/k) are not physical observables — they are regression coefficients obtained by fitting the PC-SAFT equation of state to experimental VLE and/or liquid density data. The fitted values depend on:

- **Experimental data used**: which properties (VP, liquid density, speed of sound), at which temperatures and pressures, and from which laboratory
- **Fitting protocol**: objective function weights, temperature range, inclusion of mixture data
- **EOS variant**: original PC-SAFT (Gross & Sadowski 2001), PCP-SAFT (Esper et al. 2023), sPC-SAFT, GC-PC-SAFT — each yields different "optimal" parameters for the same molecule
- **Parameter compensation**: different (m, σ, ε/k) combinations can reproduce the same macroscopic properties within experimental uncertainty, so the parameters are not uniquely determined even with perfect data

### Measured magnitude (Step 10 inter-dataset variability analysis)

The Esper and ML-SAFT datasets share 745 molecules with independently fitted PCP-SAFT parameters. For the 567 **non-associating** overlap molecules (the relevant class for this project):

| Parameter | Median %Δ | Mean %Δ | P75 %Δ | P90 %Δ | P95 %Δ |
|-----------|-----------|---------|--------|--------|--------|
| m | 3.19% | 8.72% | 8.01% | 22.91% | 47.64% |
| σ | 1.16% | 3.75% | 2.78% | 8.22% | 22.35% |
| ε/k | 1.85% | 6.46% | 5.27% | 14.61% | 31.52% |

The median disagreement is small (1–3%), but the distribution has heavy tails. The previous tight acceptance windows (m ±5%, σ ±2%, ε/k ±5%) were exceeded by 49%, 47%, and 39% of non-associating overlap molecules, respectively — meaning they would have rejected molecules whose "true" parameters are consistent with the acceptance range under an equally valid alternative fit.

**Updated acceptance windows** (set at ~P85 of inter-method variability for non-associating compounds):

| Parameter | Old window | New window | VP implication |
|-----------|-----------|------------|---------------|
| m | ±5% | **±15%** | ~15% VP change (least sensitive) |
| σ | ±2% | **±4%** | ~20% VP change |
| ε/k | ±5% | **±10%** | ~50% VP change (most sensitive) |

These are the smallest windows defensible given inherent variability in PC-SAFT parameter estimation from experimental data. See `model/saved/interdataset_variability.csv` for the full overlap analysis.

### How the variability was estimated

1. **Cross-dataset comparison (completed)**: 745 molecules appearing in both Esper and ML-SAFT were matched by InChI. Absolute and relative differences computed for each parameter. Results split by associating/non-associating using the ML-SAFT `associating` flag. Full results in `docs/reports/10_mlsaft_integration.md`.
2. **Bootstrap refitting** (future): for key molecules (cyclopentane, top candidates), refit PC-SAFT parameters to subsets of experimental data and observe the resulting spread. Requires raw experimental data, e.g., from NIST TDE.
3. **Sensitivity propagation** (completed, Step 09): ±10% ε/k → ~50% VP change; ±2% σ → ~10% VP change.

### Impact on the project

Moderate. The variability affects absolute parameter comparisons but is less damaging for ranking. Within the Esper dataset (where all parameters are fitted with a uniform protocol), the noise is smaller than the cross-dataset numbers above. The cross-dataset noise floor is the relevant constraint when combining data sources or when claiming generalizability.

### Mitigation

- Acceptance windows widened to m ±15%, σ ±4%, ε/k ±10% (implemented in `screening/filters.py`)
- Parameter screening treated as coarse Tier 1 filter; Tier 2 (EOS) and Tier 3 (Henry's constant, see `docs/reports/21_henrys_constant_analysis.md`) validate shortlisted candidates
- RF confidence intervals provide per-prediction uncertainty

### References

- Gross & Sadowski (2001) *Ind. Eng. Chem. Res.* 40:1244 — parameter compensation (Section 5)
- Esper et al. (2023) — uniform refitting protocol, inter-source reconciliation
- Liang et al. (2012) *Ind. Eng. Chem. Res.* 51:14903 — VLE sensitivity to PC-SAFT parameters
- Felton et al. (2024) *Chem. Eng. J.* 489:151999 — ML-SAFT dataset used in this analysis

---

## 2. Noisy Structure–Property Relationship

### What it is

The mapping from molecular structure (SMILES) to PC-SAFT parameters is a QSPR (Quantitative Structure–Property Relationship) problem. Unlike predicting measurable observables (boiling point, LogP), predicting PC-SAFT parameters involves predicting model-dependent abstractions. The "ground truth" labels contain noise from the parameter variability described above, creating an irreducible noise floor that no model can overcome.

### Evidence

GC-PC-SAFT — a physics-based method specifically designed to predict PC-SAFT parameters from molecular group contributions — achieves R² = −1.16 on σ and −0.04 on ε/k on the Esper test set. Predicting the dataset mean is better than using the physics-based prediction. This indicates that even physically motivated additive models cannot capture the non-additive, data-quality-dependent variance in the targets.

The ML models (RF: R² = 0.35/0.33 for σ/ε/k) outperform GC-PC-SAFT by learning non-linear feature interactions, but remain far below the levels reported in the literature for directly observable properties. However, the enhanced metrics (Step 11) reveal that σ is better predicted than R² suggests: MARE = 5.5% and 70.6% of predictions within ±5%. The low R² is partly an artifact of σ's narrow natural range (~3.2–5.5 Å).

### Is the noise floor the binding constraint?

No. The empirically measured noise-floor R² from inter-dataset variability (Step 10, non-associating compounds):

| Parameter | Noise-floor max R² (cross-dataset) | Current RF R² | Gap |
|-----------|------------------------------------|---------------|-----|
| m | 0.912 | 0.62 | 0.29 |
| σ | 0.684 | 0.35 | 0.33 |
| ε/k | 0.764 | 0.33 | 0.43 |

Within a single consistently-fitted dataset like Esper, the achievable R² is even higher (the cross-dataset noise from different fitting protocols does not apply). Published within-dataset results support this:

- Habicht et al. (2023, *Fluid Phase Equilib.* 565:113657): AARD < 5% for non-associating PC-SAFT parameters using ECFP + deep NN, suggesting R² ≈ 0.85–0.93
- Winter et al. (2023, arXiv:2309.12404): SPT-PCSAFT trained on experimental data through the EOS; 4× improvement over GC methods; predicted parameters for 13,645 components
- Felton et al. (2024, *Chem. Eng. J.* 489:151999): ML-SAFT framework, 40% AAD in VP, 8% AAD in density

The dominant limitation is model capacity and training data, not target noise.

### ML model value over GC-PC-SAFT

Despite the GC method's negative R² on σ and ε/k, it is not completely uninformative — its CCC for m is 0.733 (reasonable correlation, high variance). The ML models provide genuine value:

1. **Non-linear interactions**: RF captures cross-feature interactions (ring strain × fluorine count, molecular weight × branching) that additive group contributions miss
2. **Learned compensation**: ML models implicitly learn the correlated nature of m/σ/ε/k, while GC treats each independently
3. **Better coverage at 10%**: RF achieves 74% coverage@10% for ε/k vs GC's 42%

**GC-PC-SAFT as an input feature**: GC predictions could be used as additional features in the RF/NN, providing physically motivated priors. Even though GC predictions are poor on their own, they encode group-additivity information that Morgan fingerprints don't capture directly. This is analogous to using physics-based features alongside learned representations — a well-established strategy in molecular property prediction (e.g., Faber et al. 2017, Yang et al. 2019). This has not yet been implemented but is a low-cost experiment for Phase E.

### Achievable R² with improvements

| Change | Expected ε/k R² | Effort |
|--------|-----------------|--------|
| Current RF baseline | 0.33 | — |
| + ML-SAFT data | 0.38–0.43 | Low |
| + XGBoost | 0.41–0.48 | Low |
| + GC-PC-SAFT as input feature | 0.36–0.45 | Low |
| + D-MPNN (chemprop) | 0.50–0.65 | Medium |
| + Ensemble (RF + GNN + ChemBERTa) | 0.55–0.70 | Medium |
| Literature SOTA (GNN, large dataset) | 0.70–0.85 | High |

### Success metric gate

Based on the noise-floor analysis and literature benchmarks, the following R² targets serve as gate-checking criteria for model development:

| Parameter | Minimum viable (screening) | Good (ranking) | Excellent (literature SOTA) | Noise ceiling (cross-dataset) |
|-----------|---------------------------|----------------|---------------------------|------------------------------|
| m | 0.55 | 0.70 | 0.85+ | 0.91 |
| σ | 0.40 | 0.55 | 0.70+ | 0.68 |
| ε/k | 0.40 | 0.55 | 0.70+ | 0.76 |

"Minimum viable" = sufficient for coarse screening with wide acceptance windows and Tier 2/3 validation. "Good" = sufficient for moderate-confidence ranking. "Excellent" = approaches published SOTA and the cross-dataset noise ceiling. Within-dataset ceiling is higher (~0.85–0.93).

---

## 3. Model Scope: Non-Associating Parameters Only

The models predict only the three non-associating PC-SAFT parameters (m, σ, ε/k). Associating compounds (those with hydrogen-bond donors/acceptors) additionally require two association parameters (κ^AB, ε^AB/k) that are not predicted. This was investigated and determined to be out of scope for this project.

**Why this remains acceptable and is not planned for extension:**
- Cyclopentane is non-associating, so the reference compound requires only three parameters
- Most candidate HFOs/HFCs are non-associating or weakly associating (C–F bonds are poor hydrogen bond donors)
- Only ~200–500 compounds in the open literature have experimentally fitted association parameters — far too few relative to our ~2,150-dimensional feature space. The gold-standard databases (DDB, DIPPR) are commercially licensed.
- With three parameters, the screening already fails to find a close cyclopentane match. Requiring simultaneous agreement on five parameters would make the search strictly harder, reducing the probability of finding viable candidates rather than improving it.
- The EOS backend (teqp) does not support association; switching to the `pcsaft` package or a custom PyTorch implementation would be required.
- The state-of-the-art approach (Winter et al. 2025, SPT-PCSAFT) trains end-to-end on commercially licensed experimental data with a differentiable PC-SAFT implementation — a fundamentally different architecture that cannot be reproduced with open data alone.

---

## 4. Applicability Domain (AD) and Out-of-Distribution (OOD) Predictions

### What AD and OOD mean

The **applicability domain** is the region of chemical space where the model has sufficient training examples to make reliable predictions. A molecule is **out-of-distribution** (OOD) when its structural features are sufficiently different from the training set that the model is extrapolating rather than interpolating.

### Why this matters

The training set (Esper, 1,801 molecules) is dominated by common industrial chemicals: alkanes, alcohols, aromatics, ethers. Novel blowing agent candidates — fluorinated cyclopropanes, cyclobutanes with multiple C–F bonds, unsaturated fluorinated rings — may have structural motifs that are rare or absent in training. The model has limited basis for predicting their parameters.

### Current findings

- 8.0% of test molecules (same distribution as training) are flagged OOD by Isolation Forest
- RF R² drops 0.15–0.31 on OOD molecules vs. in-domain molecules
- Novel HFO candidates will likely have higher OOD rates due to underrepresentation of fluorinated ring systems

### Current gaps

- Isolation Forest is not chemically interpretable (cannot explain *why* a molecule is OOD)
- No Tanimoto similarity AD (the standard QSAR approach)
- No Williams plot (leverage-based AD, standard in regulatory QSAR)
- AD warnings are not prominently surfaced in the portal

### Mitigation

Planned additions (see next steps): Tanimoto similarity threshold (max similarity to training set < 0.3–0.4 triggers warning), Williams plot in evaluation harness, prominent OOD warning banner in portal.

---

## 5. Underrepresentation of Fluorinated Ring Systems

The training data contains few fluorinated ring systems — the exact chemical class most relevant to novel blowing agent candidates. This is a structural data gap, not a model limitation.

### Potential data sources

| Source | Contents | Access |
|--------|----------|--------|
| ML-SAFT (Heid et al. 2023) | ~988 molecules, broader structural coverage | Open (Figshare), download scaffolded |
| Refrigerant PC-SAFT literature | Published parameters for HFCs/HFOs (R-134a, R-32, etc.) | Open (scattered across papers) |
| NIST TDE | Experimental VP/density for fluorinated compounds | Licensed (commercial) |
| NIST WebBook | Experimental data for ~70,000+ compounds | Open (web scraping, no bulk API) |
| DDB (Dortmund Data Bank) | Curated experimental data | Licensed (academic access possible) |
| FeOs parameter tables | Open-source SAFT parameters | Open (GitHub) |

### Honest assessment

The specific molecules most relevant to this project (novel fluorinated ring systems) are the ones with the least available data. Expanding the training set with ML-SAFT and refrigerant literature will help modestly, but a fundamental improvement requires either (a) experimental PC-SAFT parameter fitting for target compounds, or (b) enough structural coverage in training that the model can interpolate to fluorinated rings from nearby chemical space (e.g., fluorinated chains + unfluorinated rings → fluorinated rings via learned feature interactions).

---

## 6. Validation Against Experiment

### Current state

All thermodynamic validation in the pipeline is computational: predicted PC-SAFT parameters are propagated through the teqp equation of state to compute vapor pressure and liquid density, which are compared against the *same EOS applied to reference parameters*. This is EOS closure testing, not experimental validation.

### What is missing

A true external validation would compare `teqp(ML-predicted params) → VP/density` against experimental measurements from NIST WebBook or another independent source for a held-out set. This would answer the question "do the predicted parameters produce accurate thermodynamic properties?" rather than "are the predicted parameters close to the reference parameters?"

### NIST WebBook access

NIST WebBook (webbook.nist.gov) provides experimental thermodynamic data but has no modern REST API. Programmatic access requires URL-based queries with HTML/CSV parsing. This is feasible for validating ~50–100 compounds but not for bulk data retrieval. NIST TDE has more structured access but requires a license.

### Impact

Without experimental validation, the pipeline's output is a predicted ranking, not a validated recommendation. Any top candidate should be validated against experimental PC-SAFT fits before committing to synthesis.

---

## 7. Spearman ρ = 0.372: Parameter Distance as a Ranking Proxy

### What it measures

The Spearman rank correlation between parameter-space distance (weighted Euclidean in m/σ/ε/k space) and property-space distance (normalized Euclidean in VP/density space at 298 K). ρ = 0.372 is statistically significant (p = 1.3 × 10⁻¹⁸) but moderate — parameter ranking is a weak proxy for single-component property ranking.

### Important context

The property-space distance used in the Spearman calculation is based on **single-component** vapor pressure and liquid density at a **single temperature** (298 K). Blowing agent selection depends on **multi-component** properties (solubility in polyol, Henry's constant, activity coefficients) across an **operating temperature range** (230–330 K).

PC-SAFT's value for blowing agent screening is precisely its ability to predict **mixture properties** from pure-component parameters via combining rules (Lorentz–Berthelot: σ₁₂ = (σ₁+σ₂)/2, ε₁₂/k = √(ε₁/k · ε₂/k)). Two molecules with similar (m, σ, ε/k) will have similar cross-interaction parameters and therefore similar Henry's constants regardless of whether their single-component VP values match precisely.

The Step 21 Henry's constant analysis (see `docs/reports/21_henrys_constant_analysis.md`) directly validated this. Henry's constants at infinite dilution in n-hexane were computed using teqp binary PC-SAFT. The verified Cl-olefin candidates have H/H(cyclopentane) = 1.02–1.05 (near-identical mixture behavior), while commercial HFOs have H ratios of 10¹–10⁷ (categorically different). The exponential Boltzmann sensitivity to ε_ij means a 22% deficit in cross-interaction energy translates to 7 orders of magnitude in Henry's constant, confirming that ε/k-weighted parameter proximity is the correct screening metric for mixture behavior.

### Interpretation

- The low Spearman ρ is a **legitimate caveat** for parameter-distance-based screening — it should be disclosed in any presentation
- It is **not as damaging as it appears** for the actual use case, because the single-component VP comparison at 298 K is not the decision-relevant metric for blowing agents
- The right response is to use **wider acceptance windows**, combine parameter ranking with additional filters (SA score, EOS convergence, AD check), and always validate shortlisted candidates through Tier 2/3

### What it means for presentation

State honestly: "Parameter-space proximity is a moderate proxy for thermodynamic similarity (Spearman ρ = 0.37 against single-component VP). We use it as a coarse first filter, not a final ranking. Shortlisted candidates are validated through EOS computation and mixture property calculation before recommendation."

---

## 8. Confidence Intervals Are Wide Relative to Acceptance Windows

Step 11 established approximate 95% CIs from RF tree disagreement:

| Parameter | Mean CI half-width | Acceptance window | Ratio |
|-----------|-------------------|-------------------|-------|
| m | ±1.59 segments | ±0.118 (5%) | 13× |
| σ | ±0.466 Å | ±0.074 Å (2%) | 6× |
| ε/k | ±58.0 K | ±14.4 K (5%) | 4× |

Consequence: 0/645 candidates clear the `ci_high_conf` filter (entire CI within acceptance window). The model cannot guarantee that any candidate's parameters are within the cyclopentane-like region.

This is an honest reflection of current model accuracy, not a code bug. Narrower CIs require either better models (reducing prediction variance) or conformal prediction (providing statistically guaranteed intervals that redistribute width based on local confidence).

---

## 9. Literature Comparison: Heid et al. / Winter et al. / Habicht et al.

Three recent publications represent the state of the art for ML-based PC-SAFT parameter prediction. Understanding their approaches clarifies what performance is achievable and where our pipeline sits.

### Felton et al. (2024) — ML-SAFT

**Reference:** Felton, K.C. et al. "ML-SAFT: A machine learning framework for PCP-SAFT parameter prediction." *Chem. Eng. J.* 489:151999.

**What they did:**
- Created the largest published PCP-SAFT parameter dataset by automated large-scale regression from DDB (Dortmund Data Bank) experimental data
- Evaluated multiple ML architectures including Random Forest, feed-forward NN, and D-MPNN (directed message-passing neural network, via the `chemprop` package)
- Best model achieved 40% AAD in vapor pressure and 8% AAD in density predictions (property-level, not parameter-level metrics)
- Key innovation: automated regression pipeline that can generate training data at scale

**How we differ:**
- We use the Esper dataset (independently fitted); they created their own dataset from DDB
- Our best model is RF with combined Morgan+RDKit features; their best is likely D-MPNN
- They report property-level AAD (VP, density); we report parameter-level R²/MARE
- Their dataset is available and our data loader already supports it (`load_data("combined")`)

**What we could adopt:**
- The D-MPNN architecture via `chemprop` — this is the single highest-ROI model upgrade (see Phase E, Step 15)
- Their dataset, which we have now downloaded (870 molecules)
- Their property-level evaluation approach (validate via VP/density AAD rather than parameter R²)

### Winter et al. (2023) — SPT-PCSAFT

**Reference:** Winter, B. et al. "Understanding the language of molecules: Predicting pure component parameters for the PC-SAFT equation of state from SMILES." arXiv:2309.12404. Published in *Digital Discovery* (2025).

**What they did:**
- Built a SMILES-to-Properties Transformer (SPT) — an NLP model that reads SMILES tokens
- **Key innovation: trained directly on experimental VP and density data through the PC-SAFT EOS**, not on fitted parameters. The neural network architecture incorporates PC-SAFT as a differentiable layer, so gradients flow from the VP/density loss back to the predicted parameters.
- This sidesteps the parameter non-uniqueness problem entirely: the model finds whatever (m, σ, ε/k) combination best reproduces the experimental observables, without being penalized for picking a different parameter-space minimum than the training set
- Outperforms traditional GC methods by 4× in mean average percentage deviation
- Predicted parameters for 13,645 components (publicly available)
- Captures stereoisomer effects without special handling

**How we differ:**
- We train on fitted parameters (standard QSPR approach); they train on experimental observables through the EOS (physics-informed approach)
- Their approach is fundamentally more principled for this problem because it avoids the parameter noise floor
- However, their approach requires differentiable EOS implementation (complex engineering)

**What we could adopt:**
- Their predicted parameters for 13,645 components as additional training data or external validation
- Their evaluation methodology (VP/density AAD instead of parameter R²)
- Long-term: the physics-in-the-loop training approach (significant engineering effort)

### Habicht et al. (2023) — ECFP + Deep NN

**Reference:** Habicht, J. et al. "Predicting PC-SAFT pure-component parameters by machine learning using a molecular fingerprint as key input." *Fluid Phase Equilib.* 565:113657.

**What they did:**
- Deep neural network with extended-connectivity fingerprints (ECFP) at bit lengths 2^10, 2^12, 2^14
- Focused on non-associating molecules only
- Achieved AARD < 5% for PC-SAFT parameters (corresponding to R² ≈ 0.85–0.93)
- Validated by comparing calculated VP and density from predicted parameters against experimental data

**How we differ:**
- Similar approach (fingerprints → NN), but we use Morgan FP + RDKit descriptors → RF instead of ECFP → deep NN
- They achieve significantly better performance, likely due to deeper architecture and possibly different dataset curation
- They focus on non-associating only (matching our scope)

**What we could adopt:**
- Deeper NN architecture with ECFP (essentially what our Step 02 aimed for, but with more capacity)
- Their validation approach (predict → compute VP/density → compare to experiment)

### Summary: Where we stand

| Approach | This project (RF) | ML-SAFT (D-MPNN) | SPT-PCSAFT (Transformer) | Habicht (Deep NN) |
|----------|-------------------|-------------------|--------------------------|-------------------|
| Input | Morgan FP + RDKit | Molecular graph | SMILES tokens | ECFP |
| Training target | Fitted params | Fitted params | **Experimental VP/density** | Fitted params |
| ε/k accuracy | R² ≈ 0.33 | ~40% VP AAD | ~4× better than GC | AARD < 5% |
| Dataset size | 1,801 | ~870 | >1,000 | ~1,000 |
| Complexity | Low | Medium | High | Medium |

The key gap is model architecture (RF vs D-MPNN/Transformer) more than data. The D-MPNN via `chemprop` is the most practical upgrade path.

## 10. Temperature Range

All current validation is at 298.15 K (room temperature). Blowing agent performance depends on behavior across the operating temperature range of polyurethane foam production, typically 230–330 K. The EOS is temperature-dependent and the relative ranking of candidates may shift at different temperatures.

Extending validation to a temperature sweep is straightforward (the `compute_properties()` function accepts a temperature argument) but has not been done.

---

## Summary: Which Limitations Are Blocking vs. Caveats?

| # | Limitation | Blocking or Caveat? | Resolution Path |
|---|-----------|---------------------|-----------------|
| 1 | Parameter variability in targets | Caveat — windows widened to match measured variability | Completed: cross-dataset analysis (Step 10), windows at m±15%, σ±4%, ε/k±10% |
| 2 | Noisy structure–property relationship | Caveat — ML still provides value over GC | GC-as-feature experiment; better models partially overcome noise |
| 3 | Non-associating parameters only | Acceptable for this prototype | Add association parameters if needed later |
| 4 | AD/OOD detection gaps | Caveat — add Tanimoto + Williams plot | Implementation planned (Phase E) |
| 5 | Fluorinated ring underrepresentation | Data gap — addressable | Phase E: refrigerant literature, NIST, FeOs (Steps 16–18) |
| 6 | No experimental validation | Caveat — should be done before synthesis | Phase E: NIST WebBook comparison (Step 17) |
| 7 | Spearman ρ = 0.372 | Caveat — less damaging than it appears for mixture properties | Wider windows, Tier 2/3 validation |
| 8 | Wide CIs | Improvable — conformal prediction, better models | Conformal prediction highest-ROI fix (Phase D) |
| 9 | Literature gap to SOTA (R² 0.33 vs 0.85) | Improvable — not at noise floor | D-MPNN (chemprop), ensemble, more data (Phase E) |
| 10 | Single temperature validation | Caveat — extend to temperature sweep | Straightforward implementation |

### R² Success Metric Gates

| Parameter | Current | Minimum viable | Good | Literature SOTA | Cross-dataset noise ceiling |
|-----------|---------|----------------|------|-----------------|---------------------------|
| m | 0.62 | 0.55 | 0.70 | 0.85+ | 0.91 |
| σ | 0.35 | 0.40 | 0.55 | 0.70+ | 0.68 |
| ε/k | 0.33 | 0.40 | 0.55 | 0.70+ | 0.76 |

Current m is already above "minimum viable." σ and ε/k require model upgrades (D-MPNN, ensemble) to reach minimum viable. The cross-dataset noise ceiling for σ (0.68) is the binding constraint for within-dataset performance on σ — reaching R² > 0.68 for σ on held-out data from the same dataset as training is possible but approaching the physical limit.
