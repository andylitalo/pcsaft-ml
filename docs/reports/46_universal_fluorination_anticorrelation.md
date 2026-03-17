# Results Report: Cross-Class Fluorination Context (Esper Dataset, 1,734 carbon-containing molecules)

**Step 46** | Tests whether the fluorination-lowers-ε/k pattern observed in the halogenated olefin screen (Steps 20–23) also appears across a broader set of carbon-containing Esper molecules. This is **supporting context** for the project narrative, not a proof of a universal physical law.

---

## 1. Motivation

Steps 20–23 established the fluorination–ε/k anti-correlation within the enumerated halogenated olefin search space. A natural follow-up: **"Is this just a peculiarity of halogenated olefins, or does the same trend appear more broadly in experimental PC-SAFT data?"**

The Esper dataset contains 1,801 experimentally fitted PC-SAFT parameters spanning diverse organic chemistries. After filtering to carbon-containing molecular species (see §2), it provides a descriptive comparison set for testing whether the pattern holds across coarse functional-group classes.

---

## 2. Filtering and Definitions

### Filtering

- **Input**: 1,801 Esper molecules
- **Filter**: require at least one carbon atom (excludes elemental entries, inorganic halides, noble gases, diatomic molecules)
- **Excluded**: 67 non-carbon species (e.g., HF, SF₆, Br₂, PBr₃, SiH₄, N₂O, etc.)
- **Retained**: **1,734 carbon-containing molecules**

### Fluorination metric

The primary axis is **fluorine mass fraction** = (mass of F atoms) / (molecular weight), consistent with the ASHRAE 34 heuristic used in Step 41:

| F mass fraction | ASHRAE heuristic | Interpretation |
|---|---|---|
| ≥ 0.65 | A1-like | Non-flammable heuristic |
| 0.50–0.65 | A2L-like | Mildly flammable heuristic |
| 0.30–0.50 | A2 | Moderately flammable |
| < 0.30 | A3 | Flammable |

These are structural heuristics, not regulatory classifications. Actual ASHRAE 34 classification requires flash-point testing, burning-velocity measurement, and heat-of-combustion data.

### Class labels

Molecules are assigned coarse functional-class labels by SMARTS/atom-presence matching. These are **descriptive bins for visualization**, not native Esper annotations or authoritative chemical ontology.

---

## 3. Class Distribution

| Class | N | Mean ε/k (K) |
|---|---|---|
| Oxygenated (no halogen) | 608 | 265.1 |
| Aliphatic HC | 338 | 258.3 |
| Nitrogen-containing | 264 | 292.0 |
| Chloro/bromo | 139 | 288.3 |
| Fluorinated (no O) | 102 | 204.0 |
| Aromatic HC | 101 | 302.1 |
| Sulfur-containing | 92 | 305.7 |
| Organosilicon | 47 | 233.7 |
| Fluoroether/fluoro-oxy | 43 | 191.0 |

The fluorinated classes (fluorinated no-O and fluoroether/fluoro-oxy) have notably lower mean ε/k than their non-fluorinated counterparts.

---

## 4. Fluorination Gradient

| Fluorine mass fraction | ASHRAE heuristic | N | Mean ε/k (K) | Median | Max |
|---|---|---|---|---|---|
| No fluorine | A3 | 1,579 | 274.0 | 266.6 | 550.0 |
| Low (<0.30) | A3 | 22 | 273.0 | 256.8 | 550.0 |
| Moderate (0.30–0.50) | A2 | 31 | 222.8 | 215.6 | 341.3 |
| Elevated (0.50–0.65) | A2L | 35 | 196.1 | 189.4 | 318.7 |
| High (≥0.65) | A1 | 67 | 176.3 | 177.3 | 259.4 |

Mean ε/k drops from 274.0 K (unfluorinated) to 176.3 K (A1-like fluorination) — a 98 K deficit. The drop is monotonic across the fluorine mass fraction bins. The A1 bin (≥0.65, n=67) has a maximum ε/k of 259.4 K, still 29 K below cyclopentane.

---

## 5. Class-Pair Comparison

| Class pair | N (base) | Mean ε/k (base) | N (fluor.) | Mean ε/k (fluor.) | Δ ε/k |
|---|---|---|---|---|---|
| Aliphatic HC → Fluorinated | 338 | 258.3 | 102 | 204.0 | **−54 K** |
| Oxygenated → Fluoro-oxygenated | 608 | 265.1 | 43 | 191.0 | **−74 K** |

Both pairs show a substantial ε/k reduction (54–74 K) when fluorine is introduced. These are descriptive comparisons — the base and fluorinated groups are not strict molecular pairs, and the comparison mixes molecules of varying size and complexity. The direction and magnitude of the effect are consistent with the olefin-specific finding from Steps 20–23.

See `figures/46_universal_anticorrelation/class_comparison_bars.png`.

---

## 6. Quadrant Counts

Using ε/k ≥ 269 K (within 20 K of cyclopentane) and fluorine mass fraction ≥ 0.50 (A2L heuristic) as boundaries:

| Region | N |
|---|---|
| High ε/k, flammable side (F mass frac < 0.50) | 764 |
| High ε/k, A2L+ (F mass frac ≥ 0.50) | **1** |
| High ε/k, A1 (F mass frac ≥ 0.65) | **0** |
| Low ε/k, A2L+ | 101 |
| Low ε/k, flammable side | 868 |

See `figures/46_universal_anticorrelation/epsk_vs_fluorination.png`.

---

## 7. Manual Triage: The One A2L+ Hit

A single carbon-containing Esper molecule has both ε/k ≥ 269 K and F mass frac ≥ 0.50:

| Property | Value |
|---|---|
| SMILES | `OCC(F)(F)C(F)(F)F` |
| Name | 2,2,3,3,3-pentafluoro-1-propanol |
| ε/k | 318.7 K |
| F mass fraction | 0.63 |
| ASHRAE heuristic | A2L |
| MW | 150.0 Da |
| Class | fluoroether/fluoro-oxy |

**Relevance assessment: not a meaningful counterexample.**

1. **Associating compound.** This is a primary alcohol (–OH group). The Esper 3-parameter PC-SAFT fit conflates hydrogen-bond association energy with dispersion energy into a single ε/k value. For non-associating compounds (the scope of this project's models), ε/k represents pure dispersion. For associating compounds, ε/k is artificially inflated because the fit compensates for the missing association parameters (ε^AB, κ^AB). This molecule's high ε/k is at least partially an artifact of the parameterization, not evidence of high dispersion energy.

2. **Not a viable blowing agent.** Alcohols are reactive with isocyanate (forming urethane linkages), hygroscopic, and have elevated boiling points due to hydrogen bonding. T_b for this compound is ~80°C, well above the 15–50°C blowing-agent window. The molecule would react out of the PU foam system rather than acting as a physical blowing agent.

3. **Not in the project's screening scope.** The project screens non-associating halogenated olefins with C=C bonds (for low GWP). This molecule has no double bond, is associating, and would not pass the project's structural filters.

**Conclusion:** The lone hit in the A2L+ / high-ε/k region is an off-scope associating compound whose ε/k is inflated by hydrogen-bond effects in the 3-parameter PC-SAFT model. No candidate-relevant carbon-containing molecule in the Esper dataset combines cyclopentane-like ε/k with A2L-level or higher fluorination.

---

## 8. Candidate-Like Subset

Restricting to carbon-containing molecules in a blowing-agent-plausible molecular-weight range (50–200 Da):

- **1,389 molecules** in this range
- **651** with ε/k ≥ 270 K
- **1** with ε/k ≥ 270 K AND F mass frac ≥ 0.50 (the pentafluoropropanol above)
- **0** with ε/k ≥ 270 K AND F mass frac ≥ 0.65

---

## 9. Key Findings

1. **The fluorination–ε/k pattern observed in the olefin screen is consistent with the broader Esper corpus.** Across 1,734 carbon-containing molecules, mean ε/k drops monotonically from 274 K (unfluorinated) to 176 K (A1-like fluorination). This drop appears in both the aliphatic HC → fluorinated and oxygenated → fluoro-oxygenated class pairs.

2. **The high-ε/k, high-fluorination region is nearly empty.** Of 1,734 molecules, only 1 has both ε/k ≥ 269 K and A2L-level fluorination — an associating alcohol whose ε/k is inflated by hydrogen-bonding effects. Zero molecules reach the A1 boundary.

3. **This is supporting context, not a universal proof.** The Esper dataset is a convenience sample of 1,734 organic molecules, not an exhaustive enumeration of chemical space. It does not contain every possible molecular architecture (e.g., fluorinated cage compounds, MOFs). The finding is that the project's core narrative — fluorination systematically reduces ε/k — is consistent with the broader experimental record, not that exceptions are physically impossible.

---

## 10. Figures

All figures saved to `figures/46_universal_anticorrelation/`:

1. **`epsk_vs_fluorination.png`**: Scatter plot of ε/k vs fluorine mass fraction for 1,734 carbon-containing Esper molecules. Colored by coarse functional class. Vertical lines at A2L (0.50) and A1 (0.65) heuristic boundaries. Horizontal dashed line at cyclopentane ε/k. Subgroup counts shown in legend.
2. **`class_comparison_bars.png`**: Bar chart of mean ε/k for three class pairs (aliphatic HC vs fluorinated, aromatic vs fluoroaromatic, oxygenated vs fluoro-oxygenated). Subgroup counts on each bar.

---

## 11. Limitations

- **Esper is a convenience sample, not a designed experiment.** The class distribution is unbalanced (608 oxygenated vs. 43 fluoro-oxy). Cross-class comparisons are descriptive, not controlled.
- **Class labels are inferred from SMARTS/atom presence**, not native Esper annotations. Some edge-case molecules may be misclassified.
- **F mass fraction is a rough flammability proxy.** Actual flammability depends on flash point, LEL/UEL, heat of combustion, and molecular geometry. The ASHRAE heuristic is a structural screening tool, not a regulatory classification.
- **Associating compounds** (alcohols, acids, amines) have ε/k values that conflate dispersion and association energy in 3-parameter PC-SAFT. These molecules are not directly comparable to the non-associating compounds in the project's screening scope.
- **The analysis covers existing Esper entries only.** Novel molecular architectures not represented in the dataset cannot be ruled out by this analysis.

---

## 12. Readiness Check

- [x] Filtered to 1,734 carbon-containing molecules; 67 non-carbon species excluded and documented
- [x] Primary fluorination axis is fluorine mass fraction (Step 41 ASHRAE heuristic)
- [x] Every high-ε/k, high-fluorination molecule listed and manually triaged (1 hit: pentafluoropropanol, off-scope associating compound)
- [x] Class-pair comparisons report subgroup counts
- [x] Conclusion framed as supporting context for the project narrative, not a universal law
- [x] All figures saved to `figures/46_universal_anticorrelation/`
- [x] Classified data saved to `model/saved/step46_esper_classified.csv`
