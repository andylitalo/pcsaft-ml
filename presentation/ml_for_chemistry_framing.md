# ML for Accelerating Chemical R&D

## Framing and Narrative

This document frames the project as a talk on how ML can accelerate chemical research and development, organized around the inverse design problem. It synthesizes the analysis, findings, honest caveats, and retrospective lessons from the full project arc.

---

## The Big Picture

Chemical manufacturing begins with knowledge of chemical properties. Measuring these properties is expensive in time and money. Computation (ab initio, MD, COSMO-RS) speeds this up but remains costly per molecule. Once a dataset is representative enough, ML can interpolate properties for a large corpus of molecules — fast enough to screen thousands of candidates for desired properties. While the data available are often not sufficient, and ML-predicted datasets can introduce systematic bias (as this project demonstrates), carefully validated ML models trained on experimental data can meaningfully accelerate the candidate identification stage of chemical R&D.

This project demonstrates one instantiation: predicting PC-SAFT equation-of-state parameters from molecular structure to screen halogenated olefin candidates as potential substitutes for cyclopentane in polyurethane foam. The lessons about data quality, model selection, validation methodology, and physics-based embeddings apply broadly to ML-driven molecular property prediction.

---

## Act 1 — The Inverse Design Problem

**"Given a set of desired thermophysical properties, which molecules in chemical space satisfy them?"**

This is the inverse design problem. The traditional pipeline:

1. **Measure properties** (experimental) — slow, expensive, gold-standard accuracy
2. **Compute properties** (ab initio, group-contribution) — faster but still costly or low-fidelity
3. **Predict properties** (ML) — fast enough to screen thousands, but requires training data and careful validation

Each step trades accuracy for throughput. The question this project investigates: **at what point is ML accurate enough to be a useful screening filter, and how do we know when to trust it?**

### The Motivating Application

Polyurethane foam insulation uses cyclopentane as a blowing agent — it has ideal thermophysical properties (moderate boiling point, good solubility in polyol, compatible with foam processing). But cyclopentane is flammable. The industry has shifted toward hydrofluoroolefins (HFOs) for low global warming potential, but HFOs have fundamentally different solubility characteristics and require completely reformulated foam systems.

This project asks a clean scientific question:

**Are there halogenated olefins with thermodynamic properties similar to cyclopentane's? If so, what are they — and what prevents them from being practical replacements?**

The candidate space — halogenated olefins — is not an arbitrary choice but a consequence of intersecting hard constraints. Non-flammability requires halogenation (specifically fluorination). Ultra-low GWP requires an atmospheric degradation site (C=C double bond). Zero ODP excludes chlorine. The correct boiling point range limits molecular weight to C2–C6. Every other organic molecule class (ethers, ketones, siloxanes, amines) fails at least one of these constraints — either they remain flammable, or they lack a low-GWP atmospheric degradation pathway. This is why the industry converged on HFOs, and why the ML-driven screen focuses on halogenated olefins. Cross-class analysis of the Esper experimental dataset (Act 4) provides supporting evidence that the same fluorination–ε/k pattern holds more broadly, though it is not an exhaustive proof across all of organic chemistry.

This separates the thermodynamic question (answerable by ML) from the practical constraints (ODP, toxicity, flammability, thermal stability) that ultimately determine industrial viability.

### Why Predict Parameters, Not Properties Directly?

The project predicts three PC-SAFT equation-of-state parameters (m, σ, ε/k) rather than predicting boiling point or solubility directly:

- **Physical consistency**: predictions satisfy thermodynamic identities
- **Mixture behavior**: parameters connect to solubility via Lorentz-Berthelot combining rules — this is what enabled the Henry's constant analysis that quantitatively closed the scientific question
- **Multi-level validation**: parameter accuracy, property accuracy, and screening utility can each be assessed independently
- **Uncertainty propagation**: errors in predicted parameters can be propagated through known equations to derived quantities (vapor pressure, density, Henry's constant)

A direct black-box property predictor would miss the exponential sensitivity of mixture thermodynamics to dispersion energy — the core physics that explains why fluorinated molecules cannot match cyclopentane's behavior.

---

## Act 2 — Building the ML Pipeline

### What Was Already Available?

Before training our own models, we surveyed existing resources for ML-predicted PC-SAFT parameters. The most promising was SPT-PCSAFT (Winter et al. 2025): not a general-purpose predictive model, but a large published dataset of 13,646 molecules with ML-predicted PC-SAFT parameters — an order of magnitude more coverage than any experimentally fitted collection. If these predictions were accurate, we could skip model training entirely and go straight to screening.

They weren't. An inter-dataset variability analysis on 745 molecules shared between the experimental Esper dataset and the independently fitted ML-SAFT dataset (Felton et al. 2024) established the noise floor: median parameter disagreement of 1–3%, but heavy tails — P95 reached 31% for ε/k and 48% for m (Step 10 report). When we compared SPT-PCSAFT predictions against experimentally fitted Esper parameters for fluorinated compounds specifically, the disagreement was systematic, not random: SPT-PCSAFT predicts ε/k approximately 75 K higher than experimental fits for fluorinated molecules (Step 31 source-stratified analysis). For context, the entire screening filter window for boiling point is only 35 K wide. A 75 K bias in the parameter that dominates boiling point prediction renders the dataset unusable for our deployment domain.

The definitive confirmation came from external validation against 15 fluorinated refrigerants with independently published, literature-optimized PC-SAFT parameters: a GNN trained on 87% SPT-PCSAFT data achieved a boiling point MAE of 133.7 K — 16.4× worse than a simple Random Forest trained on only 1,801 experimentally fitted molecules (8.2 K MAE). The model that appeared excellent by aggregate metrics (R²(ε/k) = 0.73) was catastrophically wrong where it mattered (Step 38 validation report; see Act 3 for the full data quality story).

### The Dataset We Used

This meant training our own models. The experimental dataset — 1,801 molecules with experimentally fitted PC-SAFT parameters from Esper et al. — is small by ML standards. The project ultimately favored it despite its small size, accepting lower aggregate R² in exchange for accuracy on the deployment domain. This is a deliberate trade-off: smaller and clean over larger and biased.

### Three Model Classes Tell the Story

Rather than the full nine-architecture comparison the project executed, the narrative is cleanest with three model classes that represent a progression in how domain knowledge enters the model:

**GC-PC-SAFT (group-contribution, no ML):** Hand-crafted additive rules mapping functional groups to parameter increments. Achieves R²(m) = 0.40 but R²(ε/k) = -0.04 — worse than predicting the mean for the parameter that matters most. The additive assumption breaks down for the non-additive, many-body interactions that determine dispersion energy.

**Random Forest (domain knowledge encoded in features):** 170 RDKit descriptors (molecular weight, topological polar surface area, ring counts, electronegativity proxies) plus 2,048-bit Morgan fingerprints provide strong inductive bias. R²(m) = 0.62, R²(ε/k) = 0.33. In the low-data regime (~1,800 molecules), these hand-crafted features encoding chemical knowledge outperform learned representations because the RF doesn't need to learn what "electronegativity" means from examples.

**Graph Neural Network (domain knowledge learned from data):** Message-passing on molecular graphs, where each atom's (or bond's) representation is updated based on its bonded neighbors. Two architectures were tested: GINEConv (atom-level message-passing) and chemprop's D-MPNN (directed bond-level message-passing). On the same ~1,900 experimental molecules, GNN R²(ε/k) = 0.41 (GINEConv) and 0.39 (chemprop) — a modest but consistent improvement over RF. With 7× more training data (including SPT-PCSAFT), GNN R²(ε/k) = 0.73.

### The GNN–Physics Connection

The GNN's architectural advantage on ε/k has a physical explanation. PC-SAFT models molecules as chains of spherical segments where dispersion energy depends on how segments interact with their neighbors — the local chemical environment. GNN message-passing mirrors this: after 4 layers, each node encodes a 4-hop neighborhood, capturing the local context that determines segment properties. This is conceptually a data-driven generalization of the group-contribution method — learned rather than hand-crafted group contributions, with the ability to capture non-additive interactions between groups.

On the same experimental data, both GNN (GINEConv) and chemprop (D-MPNN) consistently outperformed fingerprint-based models on ε/k (0.41 and 0.39 vs. RF's 0.33). The advantage is real but modest at this data scale.

### The Data–Architecture Interaction

The key finding: **data quantity dominated architecture choice.** On the same ~1,900 molecules, architectural differences between RF, XGBoost, chemprop, and GNN produced ±0.08 R² differences. With 7× more training data, the GNN leapt by +0.40 R² on ε/k. The crossover point — where GNN starts to consistently outperform RF — is roughly 5,000–10,000 molecules. Below that, invest in features (domain expertise). Above that, invest in learned representations.

But the 7× larger dataset carried systematic bias that poisoned the model for the actual deployment domain (see Act 3).

---

## Act 3 — The Data Quality Story

This is the project's most rigorous and broadly applicable contribution.

### The Setup

The SPT-PCSAFT dataset (13,646 ML-predicted parameters) was added to the 1,801 experimental Esper parameters. The GNN trained on this unified corpus achieved R²(ε/k) = 0.73 — crossing the useful threshold for the first time. On the fluorinated subset of the test set, the GNN achieved R²(ε/k) = 0.91. This appeared to be a breakthrough: the GNN was promoted as the production model for fluorinated compound screening.

### The Collapse

External validation against 15 fluorinated refrigerants with independently published, literature-optimized PC-SAFT parameters told a different story:


| Metric            | GNN (unified, 13,764 molecules) | RF (Esper-only, 1,801 molecules) | RF Advantage |
| ----------------- | ------------------------------- | -------------------------------- | ------------ |
| ε/k MAE           | 77.6 K                          | 14.3 K                           | 5.4×         |
| Boiling point MAE | 133.7 K                         | 8.2 K                            | **16.4×**    |
| ε/k R²            | -25.08                          | -0.47                            | —            |


The model that appeared excellent by aggregate metrics was **16× worse on the deployment domain**. The boiling point MAE of 133.7 K renders the GNN useless for screening — the entire filter window (288–323 K) is only 35 K wide.

### The Diagnosis

**Source-stratified evaluation** revealed the root cause. The GNN achieved R² = 0.90 on SPT-PCSAFT test data but only R² = 0.36 on Esper test data. The two datasets disagree systematically: SPT-PCSAFT predicts ε/k ~75 K higher than Esper for fluorinated molecules. The GNN, trained on 87% SPT data, learned to reproduce SPT's smooth but systematically biased distribution.

**Retraining the GNN on Esper-only data** (Step 38c) still failed — boiling point MAE = 142 K. This isolated the cause: it is not only SPT contamination but also an architectural limitation. With only ~210 fluorinated molecules in the 1,801-molecule Esper corpus, the GNN's 195K parameters cannot learn meaningful representations for this chemical class. RF succeeds because its hand-crafted RDKit descriptors encode electronegativity, polarizability, and other chemical knowledge that the GNN must learn from data it doesn't have.

**Inter-dataset variability analysis** (745 molecules shared between Esper and ML-SAFT) quantified the noise floor: median parameter disagreement of 1–3%, but heavy tails up to P95 = 48%. This is the first published quantification of inter-method variability in PC-SAFT parameter fitting and has implications for the entire thermodynamic modeling community.

### The Lesson

**Data quality > data quantity > model architecture.** A small, clean, experimentally fitted dataset produced a model that was 16× more accurate on the deployment domain than a 7× larger dataset dominated by ML-predicted labels. The "better" model by aggregate metrics was catastrophically worse where it mattered.

The meta-lesson: **in-distribution test-set performance can dramatically overstate real-world utility.** R² = 0.91 on the in-distribution fluorinated test set vs. R² = -25 on the external validation set — for the same model. Only external validation against independently published experimental data distinguished the two. This is not a hypothetical risk; it's a quantified example from this project.

### Uncertainty Quantification — Choosing the Right Metric

Three UQ methods were applied:

- **RF tree variance** (100-tree ensemble): measures ensemble disagreement. Globally conservative (80% empirical coverage at 1σ), but overestimates prediction error by up to 6× for well-represented chemistry because the ensemble mean converges much more tightly than individual trees.
- **MC Dropout (GNN)**: stochastic forward passes for per-prediction uncertainty. **7.6× miscalibrated** on the fluorinated validation set: 0–6.7% of errors fell within 1σ (expected: 68%). The model was confidently wrong.
- **Propagated uncertainty**: RF tree-level joint parameter predictions run through the EOS to derived quantities. Produces physically meaningless intervals (H ratio CIs spanning orders of magnitude) because outlier trees with low ε/k get exponentially amplified by the Boltzmann factor.

The key retrospective insight: **tree-variance intervals answer the wrong question for in-domain predictions.** They measure "how much do 100 trees disagree?" rather than "how far is the ensemble mean from the true value?" For the chlorobutene candidates, local benchmark calibration — comparing RF predictions against 5 structurally analogous chlorinated alkenes in the training set — provided a direct answer: ε/k RMSE of 4.3 K, vs. tree std of 27 K. Propagating the locally calibrated error (±9 K) instead of the tree variance (±53 K) through the EOS produces H ratio intervals of [0.8, 1.5] instead of [0.4, 2,770].

The broader lesson: **UQ must be validated externally and calibrated locally.** MC Dropout was 7.6× miscalibrated OOD. Tree variance was ~6× conservative for well-represented chemistry. Neither raw UQ method produced intervals that matched actual prediction errors. The most informative uncertainty estimate came from the simplest source: comparing predictions to known values for structurally similar molecules.

---

## Act 4 — The Scientific Finding

### The Thermodynamic Question, Answered

**Are there halogenated olefins with thermodynamic properties similar to cyclopentane's?**

Yes — but only chlorinated ones.

Systematic enumeration of all F/Cl-substituted C2–C6 olefins (10,700 candidates from 16 alkene backbones) followed by ML-predicted PC-SAFT parameters, EOS-computed boiling points, and a 7-filter screening cascade identified three candidates passing all criteria:


| Candidate             | ε/k (K) [local 95%] | VP ratio, 298 K | H/H(cyclopentane) [local cal.] | Class     |
| --------------------- | -------------------- | --------------- | ------------------------------ | --------- |
| 1-chlorobut-1-ene     | 267.8 [259, 276]     | 1.15            | 1.05 [~0.8, ~1.5]             | Cl-olefin |
| (Z)-2-chloro-2-butene | 267.1 [259, 276]     | 1.10            | 1.02 [~0.8, ~1.5]             | Cl-olefin |
| (E)-2-chloro-2-butene | 267.1 [259, 276]     | 1.10            | 1.02 [~0.8, ~1.5]             | Cl-olefin |

*Locally calibrated 95% intervals: ε/k ± 2 × local benchmark RMSE (4.3 K), derived from RF prediction errors on 5 structurally analogous chlorinated alkenes in the Esper training set (signed errors: -7.0 to +4.3 K, RMSE 4.3 K). H ratio range estimated from Boltzmann sensitivity at corresponding ε_ij deficits. Raw tree-variance 95% CIs are much wider (ε/k: [214, 321]; H ratio: [0.4, 2,770]) because they measure ensemble disagreement among 100 individual trees, not the expected prediction error of the ensemble mean — the tree std overestimates actual error by ~6× for this well-represented chemical class. Source: Step 43 chlorobutene deep-dive.*


All three are simple chlorinated butenes. Their predicted PC-SAFT parameters are physically reasonable, the VP ratios are stable across the 273–323 K operating range, and the Henry's constant ratios confirm near-identical dissolution thermodynamics to cyclopentane. The molecules are structurally simple and in-domain for the RF model (passed Isolation Forest AD). Confidence in these predictions is moderate — the highest of any predictions in the project — though limited by R²(ε/k) = 0.33 and the absence of direct experimental validation of the PC-SAFT parameters for these specific molecules. See the confidence assessment section below.

**Zero of 1,471 pure HFOs pass all screening criteria.** The bottleneck is ε/k: fluorination systematically reduces dispersion energy below cyclopentane's value, and no molecular design in the fluorinated olefin space can overcome this fundamental physical chemistry constraint.

### Why Fluorination Destroys the Match — The Boltzmann Sensitivity

PC-SAFT predicts mixture behavior from pure-component parameters. The cross-interaction energy between a blowing agent (solute) and the polyol matrix (solvent) follows from combining rules:

```
ε_ij/k = √(ε_i/k · ε_j/k)
```

Henry's constant — the quantity that directly determines dissolution behavior — depends on the residual chemical potential through a Boltzmann factor:

```
H ∝ exp(−ε_ij / kT)
```

This creates extreme, nonlinear sensitivity to the cross-interaction energy:


| ε_ij deficit      | H/H(cyclopentane) | Physical meaning                          |
| ----------------- | ----------------- | ----------------------------------------- |
| 4% (chlorobutene) | 1.05              | Near-identical dissolution behavior       |
| 7%                | ~1.3              | Detectable but compensable in formulation |
| 16% (HCFO-1233zd) | 61                | Two orders of magnitude different         |
| 22% (HFO-1234ze)  | 29,000,000        | Seven orders of magnitude different       |


The verified chlorinated candidates sit at 4% deficit (H ratio ~1.05). Every HFO sits at 16–22% deficit (H ratio 60 to 10^7). There is no gentle gradient — there is a **cliff**, and the cliff falls exactly where the practical constraints force you.

**Are these Henry's constant values surprising?** Not qualitatively — any physical chemist familiar with fluorocarbons would expect HFOs to be less soluble than hydrocarbons in polyol-like solvents. What the analysis adds is not qualitative surprise but **quantitative precision and exhaustive closure**: we computed H for the entire enumerated chemical space, identified the exact boundary between "matches cyclopentane" and "doesn't match" (around 7–10% ε_ij deficit), and proved that every molecule with enough fluorines to satisfy GWP/flammability constraints falls on the wrong side of that cliff. Without the systematic ML-driven screen, one could always argue "maybe the right fluorinated molecule hasn't been tested yet." The screen closes that argument.

### The Practical Elimination

The chlorinated candidates that match cyclopentane's thermodynamics are eliminated by non-thermodynamic constraints:

- **Ozone depletion potential (ODP)**: Chlorinated olefins release Cl radicals upon atmospheric degradation — the problem the Montreal Protocol was designed to solve
- **Toxicity**: Short-chain chlorinated alkenes are mutagenic and hepatotoxic (cf. vinyl chloride)
- **Reactivity**: Vinylic C-Cl bonds are susceptible to nucleophilic substitution by amine catalysts in the foam reaction
- **Regulatory**: Chlorinated VOCs face restrictions under REACH, EPA TSCA, and national chemical inventories

### Cross-Class Fluorination Context (Step 46)

The halogenated olefin screen proved the negative result for 10,700 candidates. A natural objection: **is this just a peculiarity of the olefin backbone?**

The Esper training dataset provides supporting context. Filtering to 1,734 carbon-containing molecules with **experimental** PC-SAFT parameters, and using fluorine mass fraction (consistent with the Step 41 ASHRAE heuristic), the same fluorination–ε/k pattern appears:

| F mass fraction | ASHRAE heuristic | N | Mean ε/k (K) |
|---|---|---|---|
| No fluorine | A3 | 1,579 | 274.0 |
| 0.50–0.65 | A2L | 35 | 196.1 |
| ≥ 0.65 | A1 (non-flammable) | 67 | 176.3 |

The pattern is consistent across class pairs — aliphatic hydrocarbons vs. fluorinated (Δε/k = −54 K, n=338 vs. n=102), oxygenated compounds vs. fluoro-oxygenated (Δε/k = −74 K, n=608 vs. n=43).

Of 1,734 carbon-containing Esper molecules, only **one** has both cyclopentane-like ε/k (≥269 K) and A2L-level fluorination: 2,2,3,3,3-pentafluoropropanol — an associating alcohol whose high ε/k is partially an artifact of the 3-parameter PC-SAFT model conflating H-bond association with dispersion energy. **Zero** reach the A1 (non-flammable) boundary with cyclopentane-like ε/k.

See `figures/46_universal_anticorrelation/epsk_vs_fluorination.png`.

This is **supporting context, not a universal proof.** The Esper dataset is a convenience sample of 1,734 organic molecules, not an exhaustive enumeration of chemical space. What it shows is that the project's core narrative — fluorination systematically reduces ε/k — is consistent with the broader experimental record across multiple functional-group classes.

### The Industry Implication

Thermodynamic similarity to cyclopentane and practical viability are **anti-correlated** in halogenated olefin space, and the broader Esper data is consistent with this pattern extending across other organic molecule classes. The molecular features that preserve cyclopentane-like dispersion energy (polarizable C-H bonds, minimal halogenation) are the features that make a molecule flammable. The features that satisfy non-flammability constraints (heavy fluorination) reduce ε/k below cyclopentane's value.

Commercial HFO-1336mzz(Z) — the industry's actual HFO blowing agent — ranks 4,318 out of 4,663 by parameter similarity to cyclopentane, with H/H_ref = 35. It succeeds not by matching cyclopentane's properties but because foam formulations were completely redesigned around its own thermodynamics. **The industry's reformulation approach is not a compromise — it is the only viable path**, because the thermodynamic constraint and the non-flammability constraint are fundamentally opposed. ML-driven exhaustive screening proves this for the halogenated olefin space; experimental cross-class data is consistent with the same conclusion holding more broadly.

### Confidence in the Positive Result (Chlorinated Butenes)

The chlorinated butene predictions are the most credible in the project, but honest assessment requires noting both strengths and limitations:

**Strengths:**

- Structurally simple (C4H7Cl), likely in-domain (passed Isolation Forest AD)
- Parameters are physically reasonable for a C4 chloroolefin
- VP ratio stable across 273–323 K (1.14–1.16), EOS converges at all temperatures
- H/H_ref = 1.05 is quantitatively consistent with the 4% ε_ij deficit predicted by Boltzmann sensitivity
- Self-consistent across multiple independently derived quantities
- The model's known systematic error for polar compounds (underprediction of ε/k due to missing dipole term) would make the real thermodynamic match *better*, not worse

**Limitations:**

- R²(ε/k) = 0.33 on the test set — the model explains only one-third of variance even for in-domain molecules
- No experimental validation of this specific molecule's PC-SAFT parameters (checking published literature for chlorobutene PC-SAFT parameters would be the single most impactful validation)
- 3-parameter PC-SAFT does not explicitly model polarity; the NIST validation showed 1,858% VP error for dichloromethane (strong dipole)
- Tanimoto similarity is 0.46–0.50 (in-domain), but nearest Esper neighbors are C4 branched alkenes, not chlorinated compounds — the chlorine-specific prediction performance depends on the 5 chlorinated-alkene Esper neighbors where RF has mean ε/k error of −1.6 K (Step 43)

**Uncertainty — local calibration vs. tree variance:**

The raw RF tree-variance 95% CIs for ε/k span ~107 K and, when propagated through the Boltzmann factor, yield H ratio intervals spanning orders of magnitude ([0.43, 2,770] for 1-chlorobut-1-ene). However, these intervals measure ensemble disagreement among individual trees, not the expected prediction error of the ensemble mean. For chlorinated alkenes specifically, the RF ensemble mean achieves RMSE of only 4.3 K on 5 structurally analogous Esper training molecules (signed errors: −7.0 to +4.3 K) — the tree std overestimates actual error by ~6×. Using the local benchmark RMSE as a calibrated prediction error (ε/k ± 9 K at 95%), the propagated H ratio range is approximately [0.8, 1.5] — firmly within the [0.5, 2.0] acceptance band. The near-hit designation is stable under locally calibrated uncertainty, though the local benchmark is based on only 5 molecules.

**Assessment:** The pipeline behaves correctly — it identifies a structurally and thermodynamically reasonable match that is in-domain, survives locally calibrated uncertainty propagation, and gets eliminated for the right (non-ML) reasons. This constitutes a credible positive control even without experimental confirmation of the exact parameter values.

### Note on the Tetrafluoromethylenecyclopropane "Exception"

Reports from this project highlighted tetrafluoromethylenecyclopropane (`C=C1C(F)(F)C1(F)F`) as a standout candidate — 4 fluorines yet H/H_ref = 0.96. This appears to be a molecule that "shouldn't work" based on the ε/k rule but does due to strained-ring packing effects.

**We have low confidence in this prediction.** Tanimoto similarity = 0.273 (well below the 0.4 in-domain threshold). The RF has no relevant training examples for strained 3-membered fluorinated rings with exocyclic double bonds. The per-tree ε/k uncertainty (±12 K at 1σ), propagated through the Boltzmann factor from an ε/k of 249.3 K, gives an H ratio 95% CI spanning roughly from ~1 to ~10^4. The point estimate of 0.96 is consistent with the data, but so is "essentially insoluble." The structural explanation ("ring strain preserves packing efficiency") is a post-hoc rationalization fitted to a noisy point estimate.

This molecule is more useful as an **illustration of the limits of ML-driven screening** — showing that OOD predictions carry enormous uncertainty that gets exponentially amplified by the physics — than as a real finding.

---

## Act 5 — Lessons for ML in Chemistry

### 1. External validation is non-negotiable

R² = 0.91 on the in-distribution fluorinated test set. R² = -25 on the external validation set. Same model. Only external validation against independently published experimental data distinguished them. Internal hold-out metrics, even stratified by chemistry, can mask systematic dataset bias.

### 2. Data quality trumps data quantity and model architecture

A 1,801-molecule experimental dataset produced a model 16× more accurate on the deployment domain than a 13,764-molecule dataset dominated by ML-predicted labels. On the same data, architectural differences between RF, XGBoost, chemprop, and GNN were ±0.08 R². The hierarchy is clear: data quality > data quantity > model architecture.

### 3. UQ must be validated externally and calibrated locally

MC Dropout uncertainty underestimated true errors by 7.6× on out-of-distribution fluorinated compounds (0–6.7% coverage at 1σ vs. expected 68%). RF tree variance overestimated true errors by ~6× for well-represented chemistry (tree std 27 K vs. local RMSE 4.3 K for chlorinated alkenes). Both raw UQ methods produced misleading intervals — one dangerously tight, one uselessly wide. The most informative uncertainty came from local calibration: comparing predictions to known values for structurally similar molecules. UQ methods should be validated against external data and, where possible, calibrated against local benchmarks.

### 4. Applicability domain awareness determines whether predictions are findings or hypotheses

0% of fluorinated screening candidates were in-domain (Tanimoto ≥ 0.4). This doesn't mean the predictions are useless — but it means they are hypotheses for experimental validation, not definitive answers. The chlorinated butene candidates were in-domain; the fluorinated candidates were not. Presenting both with equal confidence would be dishonest.

### 5. Predict parameters of physics-based models for interpretability and mixture behavior

Predicting PC-SAFT parameters rather than boiling point directly enabled: (a) the Henry's constant analysis that quantitatively closed the scientific question, (b) uncertainty propagation through known equations, (c) multi-level validation (parameters → properties → screening utility → mixture behavior), and (d) physically consistent predictions satisfying thermodynamic identities. A direct property predictor would have missed the exponential Boltzmann sensitivity that is the core physics insight.

### 6. GNNs generalize the group-contribution method — when data suffice

GNN message-passing — whether atom-level (GINEConv) or bond-level (chemprop's D-MPNN) — mirrors the physics of how molecular segments interact with their neighbors, conceptually generalizing additive group-contribution rules with learned, non-additive contributions. This advantage is real (consistent +0.08 R² on ε/k over fingerprint-based models on the same data, for both GNN variants) but requires sufficient training data. Below ~5,000 molecules, invest in feature engineering (domain expertise); above that, invest in learned representations.

### 7. ML enables exhaustive screening that closes open questions

The most valuable scientific contribution was a negative result: no fluorinated olefin can be a thermodynamic drop-in replacement for cyclopentane. ML made this conclusion possible by predicting properties for the entire enumerated chemical space. Without ML, "maybe the right molecule hasn't been tested yet" would remain a defensible position. Cross-class analysis of the Esper experimental dataset (Step 46) provides supporting context that the same fluorination–ε/k pattern holds beyond olefins, though it is a descriptive finding from a convenience sample rather than an exhaustive proof.

---

## Suggested Talk Structure

### Act 1: The Inverse Design Problem (3–5 min)

- Chemical R&D starts with properties → measurement is expensive → ML accelerates screening
- The pipeline: enumerate candidates → predict properties → screen → validate
- This project: are there halogenated olefins with thermodynamic properties similar to cyclopentane?
- Why halogenated olefins specifically: the intersection of non-flammability (fluorination), low GWP (C=C bond), zero ODP (no Cl), and correct volatility forces you into this space — other molecule classes fail at least one hard constraint
- Why predict PC-SAFT parameters, not properties directly: mixture behavior, physical consistency, uncertainty propagation

### Act 2: Building the ML Pipeline (5–7 min)

- First looked at what was available: SPT-PCSAFT (13,646 ML-predicted parameters) — promising scale, but systematic ~75 K ε/k bias for fluorinated compounds (validated via inter-dataset analysis and external refrigerant set)
- So we trained our own models on 1,801 experimentally fitted PC-SAFT parameters (small but clean)
- Three model classes: GC (additive rules) → RF (engineered features) → GNN (GINEConv, chemprop D-MPNN — learned representations)
- Both GNN architectures outperform fingerprints on ε/k because message-passing mirrors the physics
- But RF wins in the low-data regime — domain knowledge in features vs. learned from data

### Act 3: The Data Quality Story (5–7 min)

- A large ML-predicted dataset boosted aggregate R² from 0.33 to 0.73
- Source-stratified evaluation revealed systematic bias (~75 K ε/k offset for fluorinated molecules)
- External validation: the "best" model was 16× worse on the deployment domain
- Lesson: data quality > data quantity > model architecture
- UQ was 7.6× miscalibrated OOD (GNN MC Dropout) and ~6× overconservative in-domain (RF tree variance) — local calibration against structurally similar molecules gives the most informative intervals

### Act 4: The Scientific Finding (5–7 min)

- The thermodynamic question: chlorinated butenes match cyclopentane (H/H_ref ≈ 1.05)
- The practical elimination: those candidates fail on ODP, toxicity, regulatory
- No HFO crosses the threshold — Boltzmann sensitivity creates a cliff, not a slope
- **Cross-class context** (supporting figure): ε/k vs fluorine mass fraction for 1,734 carbon-containing Esper molecules, colored by class. The A2L+/high-ε/k region is nearly empty (1 off-scope associating alcohol). A1/high-ε/k region is empty. This is experimental data, consistent with the olefin-specific finding.
- The industry reformulates because it's the only viable path: thermodynamic similarity and non-flammability are opposed in halogenated olefin space, and the broader experimental record is consistent with the same pattern across other organic classes

### Act 5: Lessons for ML in Chemistry (3–5 min)

- External validation is non-negotiable
- Applicability domain determines whether predictions are findings or hypotheses
- Predict parameters of physics-based models for interpretability and mixture behavior
- The pipeline is transferable to any molecular property prediction + screening task

---

## What I'd Do Differently (Retrospective)

### Start with the external validation set

Curate the 15-compound fluorinated refrigerant set on day one. Use it as the deployment-relevant benchmark throughout. Every model and dataset choice gets tested against it before proceeding. This would have caught the SPT-PCSAFT bias months earlier.

### Design the dataset comparison as the central experiment

Same architecture, same hyperparameters, three training sets (experimental-only, unified, experimental + targeted augmentation), all evaluated against the same external validation set. This isolates data source from data quantity from data targeting.

### Include a direct property prediction baseline

Train models to predict boiling point directly from structure and compare accuracy to the PC-SAFT parameter pipeline. This would sharpen the "why predict parameters?" argument from assertion to evidence.

### Front-load the Henry's constant analysis

Computing Henry's constant right after the first screening pass (instead of at Step 21) would have immediately established ε/k as the critical parameter and provided the physics motivation for parameter prediction.

### Fewer architectures, deeper analysis

Three models (GC, RF, GNN) tell the story. Nine models dilute the narrative. Save the full comparison for a paper appendix.

### Frame uncertainty as a first-class deliverable from the start

Every prediction with confidence intervals from Step 1, including local calibration against structurally similar training molecules — not just raw model variance. The MC Dropout miscalibration would have been caught when the GNN was first trained, and the tree-variance overconservatism would have been identified earlier by comparing tree std to actual prediction errors on held-out chemical subclasses.

### Validate the positive result

Look up whether 1-chlorobut-1-ene or closely related molecules (1-chlorobutane, vinyl chloride, allyl chloride) have published PC-SAFT parameters. This single check would either confirm or refute the pipeline's most credible prediction and could have been done in an afternoon.