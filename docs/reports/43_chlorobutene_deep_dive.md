# Results Report: Chlorobutene Thermodynamic Near-Hit Deep-Dive

**Step 43** | Retrospective audit of the three chlorobutene isomers identified as the **only** molecules passing all five thermodynamic criteria in the systematic screen (Steps 20-21, 4,663 candidates). This step answers: **how confident should we be that chlorobutene is a real thermodynamic near-hit rather than an artifact of model uncertainty?**

The three target molecules are:

| Molecule | SMILES | CAS |
|----------|--------|-----|
| 1-chlorobut-1-ene | `C=C(Cl)CC` | 4894-61-5 |
| (Z)-2-chloro-2-butene | `C/C=C(/C)Cl` | 2211-68-9 |
| (E)-2-chloro-2-butene | `C/C=C(\C)Cl` | 2211-67-8 |

---

## Assessment Setup

### Evidence Hierarchy

Evidence is interpreted in strict priority order:

1. **External data** (experimental boiling points)
2. **Local chemistry benchmark** (Esper chlorinated alkene neighbors)
3. **Applicability domain and RF uncertainty** (Tanimoto + tree CIs)
4. **RF vs SPT disagreement** (supporting diagnostic)

### Data Sources

- **Esper dataset** (1,801 molecules): training set with 5 chlorinated alkene neighbors
- **SPT-PCSAFT** (13,643 molecules): independent ML predictions from Winter et al. (2023)
- **CRC Handbook**: experimental boiling points for all 3 target molecules
- **RF model**: 100-tree Random Forest trained on RDKit descriptors + Morgan fingerprints
- **NIST validation** (Step 17): halogenated-compound failure-mode context

---

## 1. Literature Evidence: Boiling Point Comparison

Experimental boiling points from the CRC Handbook provide the strongest available external adjudication between RF and SPT parameter sets.

| Molecule | T_b (exp) | T_b (RF) | RF error | T_b (SPT) | SPT error |
|----------|-----------|----------|----------|-----------|-----------|
| 1-chlorobut-1-ene | 337 K | 344.0 K | **+7.0 K** | 330.4 K | -6.6 K |
| (Z)-2-chloro-2-butene | 341 K | 344.9 K | **+3.9 K** | 329.9 K | -11.1 K |
| (E)-2-chloro-2-butene | 336 K | 344.9 K | **+8.9 K** | 329.0 K | -7.0 K |

**Interpretation**: RF over-predicts T_b by +4 to +9 K (mean: +6.6 K). SPT under-predicts by -7 to -11 K (mean: -8.2 K). Both methods produce boiling points within ~10 K of experiment, but RF is modestly closer on average. This is consistent with RF's epsilon_k being slightly high (higher epsilon_k leads to higher T_b) and SPT's epsilon_k being somewhat too low. Neither parameter set produces grossly incorrect thermodynamic behavior.

---

## 2. Nearest-Neighbor Esper Benchmark

Five structurally related chlorinated alkenes in the Esper training set provide the best local performance benchmark. RF predictions are compared against Esper ground truth, and SPT is benchmarked where available.

### RF Prediction Errors (RF - Esper)

| Neighbor | Esper m | RF err m | Esper sigma | RF err sigma | Esper eps/k | RF err eps/k |
|----------|---------|----------|-------------|--------------|-------------|--------------|
| Vinyl chloride | 1.938 | +0.070 | 3.469 | -0.114 | 239.9 | **+4.3** |
| Allyl chloride | 2.440 | +0.064 | 3.491 | -0.043 | 254.1 | **+1.0** |
| 2-Chloropropene | 1.913 | -0.061 | 3.820 | -0.149 | 273.2 | **-7.0** |
| 1,1-Dichloroethene | 2.039 | +0.033 | 3.676 | -0.050 | 275.9 | **-1.9** |
| 1,2-Dichloroethene | 2.212 | -0.007 | 3.560 | -0.048 | 285.9 | **-4.6** |
| **Mean** | | **+0.020** | | **-0.081** | | **-1.6** |

### SPT Prediction Errors (SPT - Esper)

| Neighbor | Esper eps/k | SPT eps/k | SPT err eps/k |
|----------|-------------|-----------|---------------|
| Vinyl chloride | 239.9 | 217.7 | **-22.2** |
| Allyl chloride | 254.1 | 241.9 | **-12.2** |
| 2-Chloropropene | 273.2 | 233.2 | **-40.0** |
| 1,1-Dichloroethene | 275.9 | 238.8 | **-37.1** |
| 1,2-Dichloroethene | 285.9 | (not in SPT) | --- |
| **Mean** | | | **-27.9** |

### Bias Direction

- **RF**: mean epsilon_k signed error of **-1.6 K** across 5 chlorinated alkene neighbors. There is no systematic over-prediction of epsilon_k for this chemical subclass. The errors are small and balanced (range: -7.0 to +4.3 K), indicating that the RF model captures the dispersion-energy regime of chlorinated alkenes reasonably well.

- **SPT**: mean epsilon_k signed error of **-27.9 K** across 4 available neighbors. SPT systematically and substantially under-predicts epsilon_k for chlorinated alkenes. This under-prediction bias explains the SPT epsilon_k values of ~244 K for the chlorobutene targets -- they are likely biased downward by ~28 K relative to the true Esper-equivalent values.

This is the most consequential finding of the analysis. The RF's epsilon_k prediction of ~268 K for the chlorobutene targets is *not* inflated by local bias. In fact, RF slightly under-predicts epsilon_k for this chemical class. The SPT values of ~244 K, while internally consistent with SPT's general behavior, are likely 20-30 K below the true values due to SPT's systematic under-prediction in the chlorinated-alkene regime.

See `figures/43_chlorobutene_deep_dive/neighbor_benchmark.png` for a visualization of the signed error pattern.

---

## 3. Tanimoto AD Status

| Target | Tanimoto NN | Nearest Esper Neighbor | AD Status |
|--------|-------------|------------------------|-----------|
| 1-chlorobut-1-ene | **0.500** | `C=C(C)CC` (3-methylbut-1-ene) | in_domain |
| (Z)-2-chloro-2-butene | **0.455** | `CC=C(C)C` (2-methylbut-2-ene) | in_domain |
| (E)-2-chloro-2-butene | **0.455** | `CC=C(C)C` (2-methylbut-2-ene) | in_domain |

All three targets pass the in-domain threshold (Tanimoto >= 0.4). The nearest Esper neighbors are C4 branched alkenes, not the chlorinated alkenes from Section 2 -- indicating that the Morgan fingerprint similarity is driven primarily by the carbon skeleton rather than the chlorine substituent.

This means the targets are within the model's general applicability domain, but the chlorine-specific prediction performance depends on the local chlorinated-alkene benchmark from Section 2, where RF performed well.

See `figures/43_chlorobutene_deep_dive/ad_similarity_heatmap.png` for the pairwise similarity matrix.

---

## 4. Parameter-Level Uncertainty

RF predictions with 95% CI (z = 1.96, tree-variance intervals):

| Target | m (95% CI) | sigma (95% CI) | eps/k (95% CI) |
|--------|------------|----------------|----------------|
| 1-chlorobut-1-ene | 2.60 (2.05, 3.14) | 3.704 (3.474, 3.934) | 267.8 (214.4, 321.3) |
| (Z)-2-chloro-2-butene | 2.63 (2.17, 3.09) | 3.660 (3.371, 3.949) | 267.1 (218.2, 315.9) |
| (E)-2-chloro-2-butene | 2.63 (2.17, 3.09) | 3.660 (3.371, 3.949) | 267.1 (218.2, 315.9) |

**Caveats**:
- The epsilon_k intervals are wide (~107 K span) and have empirical coverage of only 90.3% at the nominal 95% level (from Step 39 calibration).
- CI overlap with SPT values (~244 K) should be treated as suggestive, not decisive. The SPT values fall within the RF CI, but this is expected given the CI width.
- The epsilon_k interval is the most consequential because Henry's constant is exponentially sensitive to cross-interaction energy.

The Z and E isomers of 2-chloro-2-butene have identical RF predictions, as expected -- the E/Z stereochemistry is not captured by Morgan fingerprints or RDKit 2D descriptors.

See `figures/43_chlorobutene_deep_dive/parameter_ci_comparison.png`.

---

## 5. EOS-Propagated Uncertainty

Per-tree joint (m, sigma, epsilon_k) predictions were propagated through teqp (100 trees, 5 k_ij values, 3 molecules).

### VP Ratio (VP / VP_cyclopentane)

| Target | Mean | 95% CI |
|--------|------|--------|
| 1-chlorobut-1-ene | 2.76 | [0.20, 11.2] |
| (Z)-2-chloro-2-butene | 4.47 | [0.23, 55.0] |
| (E)-2-chloro-2-butene | 4.47 | [0.23, 55.0] |

### H/H_ref (Henry's Constant Ratio)

| Target | Tree-only mean | Tree-only 95% CI | Combined mean | Combined 95% CI |
|--------|----------------|-------------------|---------------|-----------------|
| 1-chlorobut-1-ene | 7.3e9 | [0.43, 2770] | 2.4e20 | [0.21, 96147] |
| (Z)-2-chloro-2-butene | 985 | [0.45, 29.3] | 6.7e8 | [0.20, 646] |
| (E)-2-chloro-2-butene | 985 | [0.45, 29.3] | 6.7e8 | [0.20, 646] |

**Interpretation**: The propagated H/H_ref distributions are enormously right-skewed -- a few outlier trees with low epsilon_k produce very high H/H_ref (exponential amplification), which inflates the mean to physically meaningless values. The more informative statistics are the **percentile bounds**:

- The **lower 2.5th percentile** of H/H_ref is 0.2-0.4, meaning that some tree predictions support an H/H_ref below 1 (stronger near-hit than the point estimate suggests).
- The **upper 97.5th percentile** spans 30-96,000, meaning that other tree predictions place these molecules far outside the acceptance band.
- The **point estimate** H/H_ref = 1.02-1.05 (from Section 6) falls well within the distribution but cannot be claimed with high confidence.

The qualitative conclusion is that the near-hit designation is **unstable under uncertainty propagation**. The RF point estimate places the chlorobutenes inside the acceptance band [0.5, 2.0], but the propagated 95% CI extends far outside it. This does not prove the near-hit is an artifact -- it means the data cannot distinguish between a true near-hit and a coincidental parameter alignment.

See `figures/43_chlorobutene_deep_dive/henry_ratio_ci.png`.

---

## 6. RF vs SPT-PCSAFT Diagnostic

| Parameter | 1-chlorobut-1-ene ||| (Z)-2-chloro-2-butene ||| (E)-2-chloro-2-butene |||
|-----------|------|------|-------|------|------|-------|------|------|-------|
| | RF | SPT | Delta | RF | SPT | Delta | RF | SPT | Delta |
| m | 2.60 | 2.83 | -0.23 | 2.63 | 2.83 | -0.20 | 2.63 | 2.84 | -0.21 |
| sigma (A) | 3.704 | 3.605 | +0.099 | 3.660 | 3.483 | +0.177 | 3.660 | 3.489 | +0.171 |
| eps/k (K) | 267.8 | 243.7 | **+24.1** | 267.1 | 246.1 | **+21.0** | 267.1 | 244.6 | **+22.5** |
| H/H_ref | 1.05 | 1.78 | | 1.02 | 1.84 | | 1.02 | 1.91 | |

**Key diagnostic**: The ~22-24 K disagreement in epsilon_k produces qualitatively different Henry's verdicts:
- At RF epsilon_k (~268 K): H/H_ref = 1.02-1.05 (within acceptance band, strong near-hit)
- At SPT epsilon_k (~244 K): H/H_ref = 1.78-1.91 (within acceptance band, but marginal)

However, Section 2 established that SPT systematically under-predicts epsilon_k by ~28 K for chlorinated alkenes, while RF has essentially no bias (-1.6 K). This means the SPT epsilon_k values are likely too low, and the RF values are more trustworthy for this chemical class.

Importantly, even the SPT-implied H/H_ref of 1.78-1.91 is **still within the [0.5, 2.0] acceptance band**. Both parameter sets agree that the chlorobutenes are thermodynamically similar to cyclopentane in a mixture context -- they disagree only on the degree of similarity.

---

## 7. Chlorinated-Compound Bias Discussion

### NIST Validation Context

Step 17 found that dichloromethane (CH2Cl2) had a vapor pressure error of 1,858%, marking it as a severe failure mode. The report concluded that "polar compounds are problematic." However, the chlorobutene targets differ from dichloromethane in important ways:

- **Dichloromethane** is a small, symmetric, non-unsaturated dichloride with a very strong dipole (1.6 D). Its polarity is dominated by the two C-Cl bonds in close proximity.
- **The chlorobutene targets** are C4 vinylic monochlorides with a single C-Cl bond on an sp2 carbon. Their dipole moment is ~2.2 D (from SPT data), which is higher than dichloromethane but distributed differently due to the larger molecular frame and the vinylic C=C bond.

### Local Benchmark Evidence

The five Esper chlorinated alkene neighbors (Section 2) include three vinylic monochlorides (vinyl chloride, allyl chloride, 2-chloropropene) and two dichloroethenes. The RF errors on these molecules are moderate:

- Vinylic monochloride mean epsilon_k error: **-0.6 K** (vinyl chloride: +4.3, allyl chloride: +1.0, 2-chloropropene: -7.0)
- Dichloroethene mean epsilon_k error: **-3.2 K** (1,1-DCE: -1.9, 1,2-DCE: -4.6)

Both groups show small errors, indicating that the RF model handles chlorinated alkenes (both mono- and di-substituted) far better than the fully polar, non-unsaturated dichloromethane. The vinylic C-Cl bond appears to be well-represented in the training-set feature space.

### Classification

The chlorobutene regime is classified as a **tolerable case**: moderate polarity with the key structural features (vinylic C-Cl, C4 skeleton) well-represented among Esper training molecules. The dichloromethane failure mode does not generalize to vinylic monochlorides.

---

## 8. Comparison to Prior Claims

### Step 20 (Systematic Enumeration)
- **Claim**: Three chlorobutene isomers pass all 5 thermodynamic criteria.
- **This step**: Confirmed. The RF point estimates and SPT parameters both place these molecules within the acceptance bands for all screening criteria.

### Step 21 (Henry's Constant Analysis)
- **Claim**: H/H_ref ~ 1.02-1.05 for the chlorobutene targets.
- **This step**: The RF point-estimate H/H_ref of 1.02-1.05 is confirmed. However, the propagated 95% CI is too wide to claim this with confidence. SPT-implied H/H_ref of 1.78-1.91 also passes the [0.5, 2.0] band but at the marginal edge.

### Step 23 (Narrative Conclusion)
- **Claim**: Chlorobutenes are practically non-viable (ODP, toxicity, reactivity, regulatory barriers) regardless of thermodynamic assessment.
- **This step**: Fully consistent. This analysis neither strengthens nor weakens the practical non-viability argument. It addresses only the scientific question of thermodynamic signal robustness.

---

## Key Findings

- **RF has no systematic epsilon_k bias for chlorinated alkenes** (mean error: -1.6 K across 5 Esper neighbors). The epsilon_k prediction of ~268 K for chlorobutenes is not inflated.

- **SPT systematically under-predicts epsilon_k by ~28 K** for chlorinated alkenes. The SPT value of ~244 K is likely biased low, not a "true" alternative estimate.

- **Both parameter sets place H/H_ref within [0.5, 2.0]**: RF gives H/H_ref = 1.02-1.05 (strong near-hit); SPT gives H/H_ref = 1.78-1.91 (marginal near-hit). The near-hit conclusion is robust to the RF-vs-SPT parameter disagreement.

- **RF boiling points are closer to experiment** (+6.6 K mean error vs -8.2 K for SPT), providing external validation that the RF parameter set is more physically accurate for this chemical class.

- **All three targets are in the Tanimoto applicability domain** (Tanimoto >= 0.45).

- **EOS-propagated uncertainty is enormous**: H/H_ref 95% CIs span 4-5 orders of magnitude due to exponential sensitivity to epsilon_k. The near-hit conclusion cannot be stated with high statistical confidence from the RF uncertainty analysis alone.

- **Z and E isomers of 2-chloro-2-butene are indistinguishable** to the RF model (identical predictions), because E/Z stereochemistry is not encoded in Morgan fingerprints or RDKit 2D descriptors.

---

## Conclusion: Ambiguous Near-Hit

The chlorobutene near-hit signal is classified as an **ambiguous near-hit**: the weight of evidence leans toward a genuine thermodynamic proximity to cyclopentane, but the data cannot exclude an artifact interpretation with high confidence.

**Evidence for a genuine near-hit**:
1. RF has no local epsilon_k bias for chlorinated alkenes (strongest evidence)
2. RF boiling points are within +4-9 K of experiment
3. All targets are in the Tanimoto applicability domain
4. Even SPT-implied H/H_ref (1.78-1.91) falls within the acceptance band

**Evidence for ambiguity**:
1. EOS-propagated 95% CIs span orders of magnitude -- the point estimate is not statistically distinguishable from a non-near-hit at the 95% level
2. The epsilon_k CI width (~107 K) is much larger than the RF-SPT disagreement (~22-24 K)
3. E/Z stereochemistry is invisible to the model, so one "near-hit" is actually two distinct molecules with identical predictions

**The overall assessment**: the thermodynamic near-hit is more likely real than artifactual, based on the evidence hierarchy (external data and local benchmarks favor RF). However, the propagated uncertainty is too wide to claim this with the confidence needed for an engineering recommendation. This is a moot point: as established in Step 23, the chlorobutenes are practically non-viable regardless (ODP > 0, toxic, reactive vinyl halides, regulatory barriers under the Montreal Protocol). The scientific question is answered; the engineering question was never open.

---

## Figures

All figures are saved to `figures/43_chlorobutene_deep_dive/`:

1. **`neighbor_benchmark.png`**: Bar chart of RF prediction error (RF - Esper) for each parameter across 5 chlorinated-alkene Esper neighbors. Shows no systematic epsilon_k over-prediction.

2. **`parameter_ci_comparison.png`**: Dot plot of RF point estimates with 95% CI, SPT overlays, and Esper neighbor parameter ranges for all three target molecules (3 panels: m, sigma, epsilon_k).

3. **`henry_ratio_ci.png`**: Forest plot of H/H_ref for each target: tree-only CI and combined (tree + k_ij) CI, with acceptance band [0.5, 2.0] shaded and SPT-implied H/H_ref as separate markers.

4. **`ad_similarity_heatmap.png`**: Tanimoto similarity matrix between the 3 target molecules and the 5 nearest Esper chlorinated-alkene neighbors.

---

## Limitations

- **No experimental PC-SAFT parameters exist** for the chlorobutene targets. The boiling-point comparison is an indirect adjudication.
- **E/Z stereochemistry is not encoded** in Morgan fingerprints or RDKit 2D descriptors. The model treats the Z and E isomers of 2-chloro-2-butene identically.
- **The epsilon_k CI is under-calibrated** (90.3% empirical coverage at 95% nominal level), so the reported intervals may be slightly too narrow.
- **SPT-PCSAFT is a single alternative**, not ground truth. Its systematic epsilon_k under-prediction for chlorinated alkenes (-27.9 K mean) indicates it has its own biases.
- **The Henry's constant computation uses n-hexane as model solvent**, not a real polyol. Relative comparisons (H_candidate / H_cyclopentane) are more robust than absolute values, but the solvent cancellation is approximate.
- **The 100-tree RF ensemble** produces a discrete distribution of predictions. The percentile-based CIs depend on the tails of this distribution, where individual outlier trees can dominate.

---

## Readiness Check

- [x] Step is clearly framed as **Step 43**, with no numbering conflict
- [x] Chlorobutene is treated as a **thermodynamic near-hit under audit**, not a practical candidate
- [x] Literature boiling point comparison is performed (RF vs SPT vs experimental)
- [x] Nearest-neighbor chlorinated-alkene benchmarking completed against 5 Esper molecules
- [x] RF bias direction for epsilon_k determined: **-1.6 K mean** (slight under-prediction, no systematic over-prediction)
- [x] Tanimoto similarity and nearest-neighbor identity reported for all three targets
- [x] Parameter-level CIs computed with epsilon_k coverage caveat noted
- [x] EOS-propagated H/H_ref and VP_ratio intervals computed and interpreted cautiously
- [x] RF vs SPT disagreement discussed as supporting diagnostic evidence only
- [x] Chlorinated-compound bias contextualized against NIST validation findings
- [x] Final report assigns outcome: **ambiguous near-hit**
- [x] Report does not reopen the project's main conclusion (no practical drop-in exists)
- [x] All figures saved to `figures/43_chlorobutene_deep_dive/`
- [x] `model/saved/chlorobutene_deep_dive.csv` contains all computed data
