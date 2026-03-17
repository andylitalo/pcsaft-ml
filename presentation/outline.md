---

## A 20-minute presentation for an AI-for-Chemistry startup showcasing a scientifically careful ML-for-chemistry story. The talk should emphasize what was validated, clearly mark what remains hypothesis-level, and keep supplementary links aligned with the strongest supporting evidence for Q&A.

# Title: Machine Learning Estimates Thermodynamic Properties of Molecules to Accelerate Search for Suitable Alternatives for a Polyurethane Blowing Agent

## Preamble

**Moonshot**: Find a polyurethane blowing agent that is cheap, safe, performant, and compliant — a decades-long R&D target that keeps moving as the regulatory landscape tightens.

This project started from an unanswered question from my PhD: not "can ML find a miracle molecule?", but "how fast can a validated ML + physics workflow close a hard chemical-screening question?"

# The Inverse Design Problem

**"Given a set of desired thermophysical properties, which molecules in chemical space satisfy them?"** Each approach to answering this -- experiment, simulation, ML -- trades accuracy for throughput. Only ML meets the speed requirement: this project asks whether ML is accurate enough to close a specific screening question credibly.

## Context: PU foams and blowing agents

- Rigid closed-cell PU foams are the workhorse commodity insulation (buildings, refrigerators, aerospace). The blowing agent directly controls thermal performance: gas-phase MW drives conductive heat transfer; nucleation behavior drives radiative heat transfer via cell structure.
- Cyclopentane is one of the best blowing agents thermodynamically -- good volatility, good matrix compatibility, low cost, zero ODP. But it is highly flammable (ASHRAE A3), a VOC, and has GWP ~5.
- Industry moved to HFOs/HCFOs for safety and regulatory compliance, but these are not drop-in replacements -- they require completely reformulated foam systems. Was reformulation inevitable, or are there other candidate molecules?

This talk focuses on the thermodynamic side of that question: which molecules have the right volatility and matrix compatibility (e.g., how well the gas dissolves in the polymer), and can ML answer that screening question credibly enough to save experimental effort?

## How can we explore other alternatives to cyclopentane?

To make that answerable, I turned it into an ML-assisted thermodynamic screening problem.

### Assumptions

- We are not predicting foam performance directly
- We are using thermodynamic similarity to cyclopentane as a scientifically grounded proxy for drop-in behavior
- The core thermodynamic representation is the 3-parameter, non-associating PC-SAFT model: `m`, `sigma`, and `epsilon/k`
- This is appropriate for non-associating and moderately polar molecules, but not for strongly associating chemistry; that scope limit matters throughout the talk ([primer](supplementary_information/pcsaft_primer.md), [why association is out of scope](supplementary_information/association_parameters_out_of_scope.md))

That gives the actual scientific goals.

### Scientific Questions (Goals)

Primary question:

*Within the screened candidate space, which molecules have PC-SAFT parameters close to cyclopentane's?*

Secondary questions:

- *Do they also have a suitable boiling point?*
- *Do they also have similar dissolution thermodynamics?*

We approximate dissolution similarity using Henry's constant in `n`-hexane at 298 K. `n`-Hexane is not a real polyol, so this is a ranking proxy rather than a literal process model, but it lets us test whether parameter similarity translates into similar mixture behavior ([why n-hexane?](supplementary_information/hexane_solvent_choice.md)).

# Proposed Solution: ML prediction of PC-SAFT parameters to filter against cyclopentane's

## Let's first consider all possible approaches to answering our scientific questions:

In order of increasing throughput and levels of abstraction:

1. Experimental measurement of thermodynamic properties and fitting of parameters (e.g., VLE data, vapor pressure curves, liquid density, speed of sound) (~days/molecule)
2. Quantum simulations (COSMO-RS, molecular dynamics, ab initio DFT) (~hours/molecule)
3. Phenomenological/empirical models (coarse-grained models, ML) (~seconds/molecule)

To screen thousands of candidates in a weekend, I need option 3. The question becomes whether the model is accurate enough, honest enough about uncertainty, and validated on the chemistry I care about.

## Baseline: What models are available?

### SPT-PCSAFT has issues

SPT-PCSAFT (Winter et al., *Digital Discovery* 2025, 4, 1142-1157; DOI: 10.1039/D4DD00077C) published a dataset of 13,646 ML-predicted PC-SAFT parameters trained on a SMILES transformer.

- Model weights are not published; only the predicted parameter dataset is available (CC-BY-NC-SA 4.0)
- More importantly, it is not reliable as ground truth for this project's deployment domain: fluorinated compounds show a large source-dependent shift relative to experimentally fitted parameters ([details](supplementary_information/spt_pcsaft_issues.md))

## Train Our Own Model

### On what dataset?

- Unfortunately, most datasets of PC-SAFT parameters are proprietary (e.g., Dortmund Data Bank / DDB, DIPPR)
- SPT-PCSAFT is large, but it is itself ML-predicted; training on it risks learning its bias rather than experimental chemistry
- The clean experimental anchor is Esper: 1,801 molecules with experimentally fitted parameters. The project ultimately favored this small clean dataset over the 7x larger pooled alternative, accepting lower aggregate R^2 in exchange for accuracy on the deployment domain

### With what features?

I tested three representation styles:

- RDKit 2D descriptors: ~200 topological and electronic descriptors (MW, TPSA, ring counts, electronegativity proxies), cleaned to ~170 after removing zero-variance and highly correlated features ([details](supplementary_information/feature_engineering.md))
- Morgan Fingerprints: 2048-bit circular fingerprints at radius 2 (ECFP4-equivalent), encoding substructure presence/absence ([details](supplementary_information/feature_engineering.md))
- 3D Conformer features: encode spatial arrangement of atoms ([details](supplementary_information/feature_engineering.md))

We excluded 3D conformer features because conformer generation is computationally expensive and adds pipeline complexity; they are an optimization for later work.

SMILES-based representations (e.g., fine-tuned ChemBERTa) were also tested, but on this dataset they were worse than feature-engineered models: lower accuracy on `epsilon/k`, slower inference, and less interpretable applicability-domain checks ([details](supplementary_information/smiles_chemberta_issues.md))

### With what architecture?

The problem is tabular regression on ~2,200 molecular descriptors with ~1,800 training samples. That regime strongly constrains model choice: tree ensembles are the established strong baseline for QSAR on engineered features (Svetnik et al. 2003; confirmed by Grinsztajn et al. 2022 showing tree models match or beat deep learning on medium-sized tabular data). GNNs are the natural learned-representation challenger because their message-passing mirrors PC-SAFT's own chain-of-segments physics — each layer aggregates local atomic context, analogous to how PC-SAFT computes segment-level dispersion energy. This gives a **feature-engineering baseline vs. learned-representation challenger** comparison, which is the standard experimental design in ML for molecular properties ([model selection rationale](supplementary_information/model_comparison.md)).

1. Group-contribution: no ML, pure hand-crafted additivity. When there is no relevant training data at all, this is the best available option -- but the additive assumption limits it severely on `epsilon/k` (R^2 = -0.04).
2. Random Forest on RDKit + Morgan features: chemistry encoded in the feature space. RF has been the default QSAR baseline since Svetnik et al. (2003); bagging + feature subsampling provides natural regularization for high-dimensional, low-sample data, and its no-extrapolation property is conservative but appropriate for screening.
3. GNNs: chemistry learned from the molecular graph itself. Message-passing can capture non-additive interactions between functional groups that fingerprints miss ([architecture details](supplementary_information/gnn_architectures.md)).

XGBoost was also tested as a second tree ensemble (boosting vs. RF's bagging). The two tied on `epsilon/k` (R^2 = 0.33 vs 0.33), confirming that the bottleneck is the feature representation, not the tree optimization strategy ([why XGBoost ties RF here](supplementary_information/model_comparison.md)).

This is where the "Bitter Lesson" appears in a constrained form: with enough clean data, the GNN should eventually win (estimated crossover around 5,000-10,000 molecules, extrapolated from two data points), and on pooled datasets it does look stronger. But the deployment decision in this project was driven by external validation, not by the prettiest aggregate metric ([model comparison](supplementary_information/model_comparison.md), [Bitter Lesson note](supplementary_information/bitter_lesson_chemistry.md))

### Metrics of Success

A good model will have

- Low MAE: interpretable in physical units (K for epsilon/k, Angstroms for sigma)
- R^2: fraction of variance explained; the baseline noise floor comes from inter-dataset variability (~1-3% median, heavy tails to 30%+)
- We also tracked MARE (scale-invariant) and CCC (penalizes systematic bias)

([Metric definitions and rationale](supplementary_information/metrics_rationale.md))

### Evaluation

On the Esper test set (n = 361), RF (RDKit descriptors + Morgan fingerprints, ~2,200 features) achieved `R^2 = 0.62 / 0.35 / 0.33` on `m / sigma / epsilon/k`. GNN variants slightly improved `epsilon/k` on the same data, and the unified-dataset GNN reached `R^2(epsilon/k) = 0.73`. But that higher aggregate score did not survive external validation on fluorinated compounds.

That became the key model-selection result:

- RF trained on 1,801 experimentally fitted molecules gave boiling-point MAE = 8.2 K on 15 external fluorinated refrigerants
- The GNN trained on 13,764 pooled molecules gave boiling-point MAE = 133.7 K on the same set
- So RF, not GNN, was chosen for screening because it was far more accurate on the chemistry that mattered ([fluorinated validation](supplementary_information/fluorinated_validation.md))
- Uncertainty told the same story: the GNN's MC Dropout intervals were 7.6x miscalibrated on the fluorinated validation set (0-6.7% of errors within 1-sigma vs expected 68%). The model was not just wrong -- it was confidently wrong.


| Model               | Training data      | epsilon/k R^2 | Fluorinated BP MAE (K) |
| ------------------- | ------------------ | ------------- | ---------------------- |
| GC-PC-SAFT (no ML)  | hand-crafted rules | -0.04         | --                     |
| RF (RDKit + Morgan) | 1,801 experimental | 0.33          | **8.2**                |
| GNN (Esper-only)    | 1,801 experimental | 0.41          | 142.3                  |
| GNN (unified)       | 13,764 pooled      | 0.73          | 133.7                  |


The table focuses on epsilon/k because it is the parameter that dominates the screening question (see "Why Parameter Differences Matter" below). GNN outperforms RF on aggregate test-set R^2, but both GNN variants fail catastrophically on the external fluorinated validation (16x worse than RF).

([Full 8-model comparison](supplementary_information/model_comparison.md)) See also: [R^2 heatmap](figures/15_improved_models/r2_heatmap.png) and [external fluorinated validation](figures/38_gnn_fluorinated_validation/boiling_point_parity.png)

### The Broader ML Lesson

- In-distribution test metrics can be badly misleading
- Larger training sets are not automatically better if they are label-biased
- Raw uncertainty estimates (tree variance, MC Dropout) can be badly miscalibrated in both directions; local calibration against structurally similar known molecules gave the most useful intervals
- The hierarchy is clear: **data quality > data quantity > model architecture**, and deployment-domain validation matters more than benchmark R^2

## Filter candidate molecules

Filters applied in cascade ([methodology details](supplementary_information/screening_methodology.md)):

- Enumerate 10,700 F/Cl-substituted C2-C6 olefins
- Use the 1,471 pure-HFO subset for the regulation-compliant search
- Rank by weighted PC-SAFT parameter proximity to cyclopentane
- Filter by boiling point, vapor-pressure similarity, Henry's-constant similarity, EOS convergence, and practical chemistry constraints

**Results, carefully stated**:

- In the pure-HFO subset, no molecule met the full drop-in thermodynamic criteria
- In the broader F/Cl-substituted olefin space, three chlorinated butenes emerged as the best thermodynamic near-hits
- Those chlorinated butenes are model-based near-hits, not experimentally confirmed winners
- For industry context: HFO-1336mzz(Z), the commercial HFO blowing agent, ranks 4,318 out of 4,663 by parameter similarity to cyclopentane (H/H_ref = 35). It works not by matching cyclopentane but because foam systems were completely redesigned around it.


| Candidate             | epsilon/k (K) [local 95% CI] | VP ratio (298 K) | H/H_cyclopentane [local cal.] |
| --------------------- | ---------------------------- | ---------------- | ----------------------------- |
| 1-chlorobut-1-ene     | 267.8 [259, 276]             | 1.15             | 1.05 [~0.8, ~1.5]             |
| (Z)-2-chloro-2-butene | 267.1 [259, 276]             | 1.10             | 1.02 [~0.8, ~1.5]             |
| (E)-2-chloro-2-butene | 267.1 [259, 276]             | 1.10             | 1.02 [~0.8, ~1.5]             |


All three are chlorinated butenes. Zero of 1,471 pure HFOs passed the strict drop-in screen. The CIs above are locally calibrated: RF predictions compared against 5 structurally analogous chlorinated alkenes in the Esper training set give epsilon/k RMSE = 4.3 K, vs raw tree-variance std of 27 K (~6x overconservative). Propagating the locally calibrated error through the EOS gives physically meaningful H ratio intervals, whereas raw tree-variance propagation produces intervals spanning orders of magnitude. ([chlorobutene deep-dive](figures/43_chlorobutene_deep_dive/neighbor_benchmark.png), [Henry's ratio CIs](figures/43_chlorobutene_deep_dive/henry_ratio_ci.png))

### Scientific Rationale

Chlorinated alkenes preserve more cyclopentane-like dispersion energy than heavily fluorinated olefins do, while still landing in a usable boiling-point regime. In the screened olefin space, increasing fluorination systematically pulls `epsilon/k` downward and drives candidates away from cyclopentane-like mixture behavior. Broader experimental context from the Esper dataset is consistent with the same trend, but should be presented as supporting evidence rather than universal proof. See [experimental cross-class context](figures/46_universal_anticorrelation/epsk_vs_fluorination.png) and [screened-candidate trend](figures/22_ml_chemistry_narrative/epsilon_k_vs_fluorines.png).

### Why Parameter Differences Matter So Much

The important physics is not just that `epsilon/k` shifts. It is that the shift gets amplified in mixture thermodynamics:

- `epsilon_ij / k = sqrt((epsilon_i/k)(epsilon_j/k))`
- Henry's constant depends exponentially on the interaction energy (simplified; actual values below computed through the full PC-SAFT EOS via teqp)

Quantitatively, this creates a cliff, not a slope (values from Step 22/43 EOS calculations):


| epsilon_ij deficit | H/H(cyclopentane) | Physical meaning                          |
| ------------------ | ----------------- | ----------------------------------------- |
| 4% (chlorobutene)  | 1.05              | Near-identical dissolution behavior       |
| 7%                 | ~1.3              | Detectable but compensable in formulation |
| 16% (HCFO-1233zd)  | 61                | Two orders of magnitude different         |
| 22% (HFO-1234ze)   | 29,000,000        | Seven orders of magnitude different       |


There is no gentle gradient -- there is a cliff, and the cliff falls exactly where the practical constraints force you. Every HFO sits at 16-22% deficit. This is the strongest scientific payoff of predicting PC-SAFT parameters instead of only predicting boiling point directly: a direct property predictor would miss this exponential sensitivity entirely. ([why parameters, not properties](supplementary_information/approach_validity.md), [Henry sensitivity figure](figures/22_ml_chemistry_narrative/henrys_sensitivity.png))

### The answer

The molecules that are thermodynamically similar to cyclopentane all have severe practical disqualifiers (ODP, toxicity, regulatory). You can't maintain cyclopentane's thermodynamic convenience while staying in compliance — so the real question is finding the right tradeoff *within* compliance, not preserving past properties. ML-driven screening closed that question in a weekend of compute instead of years of bench experiments — though validating the ML models themselves required weeks of careful iteration.

### Limitations: What molecules is this not appropriate for?

- Fluorinated molecules remain the highest-risk predictions. Even when RF outperformed GNN badly on the external fluorinated validation set, those predictions are still screening-quality, not engineering-quality, and should be treated as prioritization hypotheses rather than design data. Uncertainty estimates for OOD fluorinated predictions should not be trusted at face value -- MC Dropout was 7.6x overconfident; tree variance was ~6x conservative for in-domain chemistry. ([details](supplementary_information/fluorinated_validation.md))
- Strongly polar or associating molecules are out of scope for the 3-parameter model; molecules with OH, NH, or COOH groups should not be over-interpreted ([why association is out of scope](supplementary_information/association_parameters_out_of_scope.md))
- The chlorobutene near-hits are the most credible predictions in the pipeline: locally benchmarked, in-domain, and now cross-checked against experimentally fitted parameters for C4 chloroalkane structural analogs (1-chlorobutane, 2-chlorobutane) and the C3 vinylic analog (2-chloropropene). The near-hit designation is robust under all plausible ε/k scenarios. Direct experimental PC-SAFT parameters for the chlorobutene isomers themselves remain unavailable. ([details](supplementary_information/structural_analog_validation.md))

# Conclusion: No drop-in fluorinated candidate found, but a useful and validated workflow

## Feasible Candidates

The best thermodynamic near-hits are the three chlorinated butenes, but they are eliminated by non-thermodynamic constraints:

- **Ozone depletion potential (ODP)**: chlorinated olefins release Cl radicals upon atmospheric degradation -- the problem the Montreal Protocol was designed to solve
- **Toxicity**: short-chain chlorinated alkenes raise major toxicological and regulatory concerns
- **Reactivity**: vinyl halides are poor fits for the foam process environment
- **Regulatory**: chlorinated VOCs face restrictions under REACH, EPA TSCA, and national chemical inventories
- **Availability**: not commercially manufactured at scale

## The Chlorobutenes as Positive Control

The chlorobutenes are not a practical discovery — they fail on ODP, toxicity, and regulation. But they serve as a **positive control for the pipeline**: the workflow surfaced thermodynamically plausible near-hits that are in-domain, locally benchmarked, and self-consistent across multiple independently derived quantities. They were then eliminated for the right (non-ML) reasons. This is what a well-functioning screening pipeline should do.

### Structural Analog Validation

A post-hoc check against published PC-SAFT parameters for closely related molecules further supports the near-hit designation. 1-chlorobutane (the saturated C4 analog) has experimentally fitted ε/k = 256.9 K in the Esper dataset — providing a conservative lower bound for the chlorobutene prediction of ~268 K. The ~11 K gap is physically consistent with the vinylic Cl effect observed in the C3 pair: 1-chloropropane (ε/k = 253.3) → 2-chloropropene (ε/k = 273.2, Δ = +20 K). Even using 1-chlorobutane's parameters directly, the cross-interaction deficit vs cyclopentane is only ~6%, giving H/H_ref ≈ 1.15 — well within the [0.5, 2.0] acceptance band. The near-hit is robust across the full plausible range of ε/k. No experimental PC-SAFT parameters exist for the chlorobutene isomers themselves. ([structural analog validation details](supplementary_information/structural_analog_validation.md))

## Valid Approach with Reasonable Accuracy

Evidence of the approach's validity ([full report](supplementary_information/approach_validity.md)):

1. **Multi-level validation**: parameter-level, property-level, and screening-level checks all point in the same direction
2. **Data quality > data quantity > model architecture**: the experimentally anchored RF was 16x more accurate than the higher-R^2 pooled-data GNN on the deployment domain
3. **Physical consistency matters**: predicting PC-SAFT parameters, rather than a single property, enabled the mixture-thermodynamics analysis that quantitatively closed the question
4. **The negative result is scoped but meaningful**: within the enumerated halogenated-olefin space, no pure HFO behaved like a drop-in cyclopentane replacement
5. **Structural analog cross-check**: experimentally fitted parameters for 1-chlorobutane (ε/k = 256.9 K) and 2-chloropropene (ε/k = 273.2 K) bracket the RF chlorobutene prediction (~268 K), and all three scenarios confirm the near-hit within the Henry's acceptance band ([details](supplementary_information/structural_analog_validation.md))

## Future Work

- Repeat the workflow around today's commercial blowing agents rather than cyclopentane, using a larger experimentally anchored fluorinated dataset
- Generalize the workflow to other non-associating screening problems where the reference molecule, property windows, and applicability domain are clear ([Streamlit app note](supplementary_information/streamlit_app.md))

## Takeaways

1. **External validation is non-negotiable.** R^2 = 0.73 on the aggregate test set; boiling-point MAE = 133.7 K on the deployment domain. Only external validation against independently published experimental data distinguished the two.
2. **Predict parameters of physics-based models, not properties directly.** The exponential Boltzmann sensitivity — the cliff, not a slope — is the core physics insight. A direct property predictor would have missed it entirely.
3. **In the low-data regime, feature engineering beats learned representations.** With ~1,800 molecules, hand-crafted chemical descriptors outperform GNNs because the RF doesn't need to learn what electronegativity means from examples. With enough clean data (estimated >5,000-10,000 molecules), learned representations should eventually win — [this has already happened elsewhere in chemistry](supplementary_information/bitter_lesson_chemistry.md).
4. **ML enables exhaustive screening that closes open questions.** The most valuable outcome was a negative result: no fluorinated olefin can be a thermodynamic drop-in for cyclopentane. Without ML, "maybe the right molecule hasn't been tested yet" would remain a defensible position.

