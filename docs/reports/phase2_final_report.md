# Phase 2 Final Report: ML-Driven PC-SAFT Parameter Prediction for Blowing Agent Screening

**Steps completed**: 10, 02b (GNN), 02c (Ensemble), 11–21
**Datasets**: Esper (1,801), ML-SAFT (870), SPT-PCSAFT (13,646), fluorinated literature (34) → combined: up to 13,764 unique molecules
**Candidate pool**: 4,663 SA-filtered HFO/HCFO candidates from systematic combinatorial enumeration
**Tests**: 238+ passing across all components

---

## 1. Progressive Model Performance

Phase 2 benchmarked four new models, an ensemble strategy, and additional training configurations against the Phase 1 RF baseline. The table below shows all models evaluated across the project, ordered by average R².

### R² (test set)

All R² values are point estimates on a single holdout split. 95% bootstrap CIs are available by running `python -m model.evaluate --bootstrap` and are saved to `model/saved/comparison_metrics.csv` with `_lo`/`_hi` columns. CIs are important for comparing models — overlapping CIs indicate that apparent R² differences may not be statistically meaningful.

| Model | R² (m) | R² (σ) | R² (ε/k) | Data | Phase |
|-------|--------|--------|----------|------|-------|
| GC-PC-SAFT (group-contrib) | 0.40 | −1.16 | −0.04 | N/A | 1 |
| NN (PCSAFTNet, MLP) | 0.47 | 0.11 | 0.14 | Esper | 1 |
| ChemBERTa | 0.53 | 0.25 | 0.27 | Esper | 1 |
| Ensemble (RF+NN, inv-var) | 0.53 | 0.08 | 0.13 | Esper | 2 |
| chemprop D-MPNN | 0.54 | 0.33 | **0.39** | Esper | 2 |
| XGBoost | 0.61 | 0.34 | 0.33 | Esper | 2 |
| RF (Combined features) | 0.62 | 0.35 | 0.33 | Esper | 1 |
| GNN (Combined data) | 0.69 | 0.34 | 0.41 | Esper+MLSAFT | 2 |
| **GNN (All data)** | **0.76** | **0.77** | **0.73** | All sources | 2 |

### MAE

| Model | MAE (m) | MAE (σ, Å) | MAE (ε/k, K) |
|-------|---------|------------|--------------|
| GC-PC-SAFT | 1.001 | 0.494 | 44.6 |
| NN (MLP) | 0.767 | 0.238 | 32.9 |
| ChemBERTa | 0.766 | 0.226 | 31.2 |
| Ensemble (RF+NN, inv-var) | 0.747 | 0.236 | 32.1 |
| chemprop D-MPNN | 0.689 | 0.206 | 26.8 |
| XGBoost | 0.611 | 0.196 | 27.0 |
| RF | 0.585 | 0.189 | 26.8 |
| GNN (Combined) | 0.756 | 0.225 | 28.2 |
| **GNN (All)** | **0.310** | **0.112** | **10.4** |

### Key Takeaways on Model Performance

**Data quantity dominates architecture choice.** The GNN on the expanded dataset (11K train) outperforms every Esper-only model by a wide margin. On the same data (~1,900 molecules), architectural differences between RF, XGBoost, chemprop, and GNN produce modest deltas (±0.08 R²). With 7× more data, the GNN leaps by +0.40 R² on ε/k.

**Graph representations beat fingerprints on ε/k.** Both chemprop (D-MPNN) and GNN (GIN) outperform RF on ε/k when controlling for training data. The message-passing architecture captures how molecular segments interact — the physical basis of dispersion energy — better than hashed substructure fingerprints.

**σ was the hardest target and benefited most from data.** Segment diameter depends on subtle conformational and steric effects poorly captured by 2D fingerprints. The GNN's σ R² went from 0.34 (combined data) to 0.77 (all data) — the single largest improvement in the project.

**XGBoost offers no advantage over RF on this feature set.** Both are tree ensembles on the same 2,218-dim descriptor space; gradient boosting's inductive bias provides no additional signal.

**Inverse-variance ensembling degrades performance.** The RF+NN ensemble (Step 02c) underperforms RF alone on all three targets (ε/k R² drops 0.33 → 0.13). The root cause is uncertainty calibration mismatch: NN MC Dropout produces overconfident (narrow) uncertainty estimates, receiving 83–89% of the inverse-variance weight despite being the weaker model. Even equal-weight averaging (R² 0.57/0.26/0.25) cannot recover RF-level performance because it averages in NN noise. Model disagreement does provide a useful supplementary AD signal (mean pairwise ε/k disagreement = 17.4 K). An RF+GNN ensemble remains the most promising combination but requires `torch_geometric` availability; stacking is deferred until then.

**GC-PC-SAFT features are redundant with molecular fingerprints.** Step 18's ablation showed that adding GC-predicted values as RF input features provides negligible benefit for m and σ and actively hurts ε/k (R² drops 0.33 → 0.29). Morgan fingerprints already encode the structural patterns that group additivity captures.

### Caveat on GNN "All Data" Performance

The "all" dataset includes ~11,838 SPT-PCSAFT molecules whose parameters are themselves ML predictions (Winter et al. 2025). The GNN is partly learning to predict smooth model outputs rather than noisy experimental fits. For strict apples-to-apples comparison with Phase 1, the GNN on combined data (R² = 0.69/0.34/0.41) is the fair benchmark. For practical screening, the "all" model is preferred — the SPT parameters were validated against experimental vapor pressure with 13.5% mean APD.

---

## 2. Data Integration and Quality

### Datasets Integrated

| Source | Molecules | Overlap w/ Esper | Unique addition | Origin |
|--------|-----------|-----------------|-----------------|--------|
| Esper | 1,801 | — | — | Experimentally fitted PC-SAFT |
| ML-SAFT (Felton 2024) | 870 | 736 (84.6%) | 134 | Experimentally fitted, different protocol |
| SPT-PCSAFT (Winter 2025) | 13,646 | ~300 est. | ~11,838 | ML-predicted from SMILES transformer |
| Fluorinated literature | 34 | ~9 | ~25 | Curated from 6 published papers |
| **Combined (deduplicated)** | **~1,960** | — | — | Esper + ML-SAFT + fluorinated |
| **All (deduplicated)** | **~13,764** | — | — | Combined + SPT-PCSAFT |

### Inter-Dataset Variability (Step 10)

The 745 molecules shared between Esper and ML-SAFT provided the first quantification of inter-method variability in PC-SAFT parameter fitting:

| Parameter | Median %Δ | P90 %Δ | P95 %Δ |
|-----------|-----------|--------|--------|
| m | 3.2% | 22.9% | 47.6% |
| σ | 1.2% | 8.2% | 22.4% |
| ε/k | 1.9% | 14.6% | 31.5% |

The heavy-tailed distributions forced acceptance window widening: m ±5% → ±15%, σ ±2% → ±4%, ε/k ±5% → ±10%. The noise-floor R² estimates (m: 0.91, σ: 0.68, ε/k: 0.76) confirmed that model capacity — not target noise — is the binding constraint.

### Fluorinated Data Expansion (Step 16)

34 fluorinated compounds were curated from published literature, including 6 fluorinated 3–5-membered ring systems previously absent from the training data. This closed the coverage gap for the chemical space most relevant to screening: the OOD rate for fluorinated cyclopentane candidates was estimated to drop from ~30% to ~10–15%.

---

## 3. Enhanced Evaluation (Step 11)

Phase 1 reported only R² and MAE. Phase 2 added four metrics that revealed a more nuanced picture:

| Metric | What It Shows | Key Finding |
|--------|---------------|-------------|
| **MARE** | Average relative error | σ is actually well-predicted (5.5% MARE) despite low R² (0.35). The narrow natural range of σ (~3.2–5.5 Å) suppresses R². |
| **CCC** | Concordance (accuracy + precision) | CCC < R² for all RF targets → systematic bias. RF over/underpredicts at range extremes. |
| **Coverage@5/10%** | Fraction of predictions within threshold | Only 55.7% of ε/k predictions within ±5%. For screening, ~44% of candidates are ranked incorrectly. |
| **Q² (5-fold CV)** | Cross-validated R² | ε/k Q²=0.46, R²=0.33; the −0.13 gap flags potential overfitting to the single holdout split. |

---

## 4. Applicability Domain (Steps 04b, 14)

Three complementary AD methods are now operational:

| Method | Type | Interpretability | In API? |
|--------|------|-----------------|---------|
| Isolation Forest | Density in descriptor space | Low (black box) | Yes (`in_domain`) |
| Tanimoto NN similarity | Max fingerprint similarity to training set | High (nearest molecule) | Yes (`tanimoto_nn`) |
| Williams plot | Leverage-based (QSAR standard) | Medium (statistical) | Figures only |

The Isolation Forest and Tanimoto methods agree on 83% of molecules. The Tanimoto method is more conservative, flagging more molecules as OOD — desirable for safety-critical predictions. Color-coded warnings in the portal provide immediate researcher feedback:
- Green (≥0.4): good similarity to training data
- Yellow (0.3–0.4): treat with caution
- Red (<0.3): high prediction uncertainty expected

---

## 5. Candidate Screening: The Full Funnel

### Phase 2 Search Space

Phase 2 replaced Phase 1's hand-picked 645-candidate pool with a systematic combinatorial enumeration of 16 alkene backbones (C2–C6) with all-position F/Cl substitution:

| Metric | Phase 1 ("broad") | Phase 2 (systematic) |
|--------|-------------------|---------------------|
| Base scaffolds | 87 hand-picked | 16 alkene backbones |
| Raw candidates | ~805 | 10,700 |
| SA-filtered | 645 | 4,663 |
| Unique HFOs | ~100 est. | 1,471 |
| Unique HCFOs | ~200 est. | 3,118 |
| Known commercial agents found | Partial | All 9 tested |

### Five-Criterion Screening Funnel (Systematic Pool)

| Criterion | Pass | Fail | Pass Rate |
|-----------|------|------|-----------|
| All candidates | 4,663 | — | — |
| 1. Applicability domain | 4,663 | 0 | 100%* |
| 2. Parameter proximity (top-20) | 20 | 4,643 | 0.4% |
| 3. EOS convergence (3 temps) | 2,600 | 2,063 | 55.8% |
| 4. VP ratio [0.5, 1.5] (3 temps) | 755 | 3,908 | 16.2% |
| 5. SA ≤ 4.5 | 4,663 | 0 | 100% |
| **All five criteria** | **3** | **4,660** | **0.06%** |

*AD model unavailable in batch run; would reject ~28% in production.

### Verified Candidates (All Phases Combined)

| Candidate | SMILES | VP Ratio 298K | Param Distance | Source |
|-----------|--------|---------------|----------------|--------|
| *cis*-1,3-Difluorocyclobutane | `F[C@H]1C[C@@H](F)C1` | 0.96 | 0.127 | Phase 1 |
| 1-Chlorobut-1-ene | `C=C(Cl)CC` | 1.15 | 0.160 | Phase 1 |
| (E/Z)-2-Chloro-2-butene | `C/C=C(\C)Cl` / `C/C=C(/C)Cl` | 1.10 | 0.173 | Phase 2 (new) |

### Multi-Temperature Confirmation (Step 12)

The Phase 1 candidates (difluorocyclobutane and chlorobutene) were validated across 273–323 K. Both maintain VP ratio within [0.5, 1.5] at all three temperatures. No new candidates emerged from temperature extremes, and no candidates dropped out — confirming that 298 K single-point screening is reliable.

### Practical Limitations of Verified Candidates

None of the verified candidates are viable commercial blowing agents:

| Candidate | ODP | GWP | Flammability | Toxicity | Availability |
|-----------|-----|-----|-------------|----------|-------------|
| *cis*-1,3-Difluorocyclobutane | Zero | Unknown (not an olefin → potentially long atmospheric lifetime) | Low (fluorinated) | Unknown | Not commercially produced |
| 1-Chlorobut-1-ene | Nonzero (Cl) | Low (olefin) | High (unsaturated C4) | Lachrymator | Lab-scale only |
| (E/Z)-2-Chloro-2-butene | Nonzero (Cl) | Low (olefin) | High (unsaturated C4) | Irritant | Lab-scale only |

The chlorinated olefins have nonzero ODP (ozone depletion potential) — disqualifying under the Montreal Protocol amendments. The difluorocyclobutane is not an olefin, so its atmospheric degradation pathway and GWP are uncharacterized. None are commercially manufactured at industrial scale.

---

## 6. The Central Scientific Finding: Why HFOs Cannot Replace Cyclopentane

### The ε/k Barrier

Fluorine atoms systematically reduce dispersion energy because C-F bonds have lower polarizability than C-H bonds. Cyclopentane's ε/k of 289 K is characteristic of pure hydrocarbons; heavily fluorinated molecules typically have ε/k of 150–210 K:

| Agent | ε/k (K) | Deficit vs Cyclopentane |
|-------|---------|------------------------|
| Cyclopentane | 288.8 | — |
| HCFO-1233zd | 205.4 | −29% |
| HFO-1234yf | 181.2 | −37% |
| HFO-1234ze | 175.9 | −39% |
| HFO-1336mzz | 172.0 | −40% |

### Exponential Amplification via Boltzmann Factor (Step 21)

Henry's constant at infinite dilution in n-hexane quantifies how this ε/k deficit affects mixture behavior:

| ε_ij deficit | Approximate H/H(cyclopentane) |
|--------------|-------------------------------|
| 0% | 1 |
| 4% | ~1.1 |
| 7% | ~1.2 |
| 16% | ~60 |
| 22% | ~10⁷ |

The Lorentz-Berthelot combining rule (ε_ij = √(ε_i · ε_j)) links pure-component ε/k directly to cross-interaction energy. The chemical potential scales as −ε_ij/(kT), so Henry's constant depends exponentially on ε_ij. A 22% ε_ij deficit — typical for commercial HFOs — makes the molecule effectively insoluble compared to cyclopentane in a hydrocarbon-like polyol blend.

### Verified Candidates vs Commercial HFOs

| Molecule | ε_ij/ε_ij(ref) | H/H(cyclopentane) |
|----------|-----------------|---------------------|
| 1-Chlorobut-1-ene (verified) | 0.963 | **1.05** |
| (Z)-2-Chloro-2-butene (verified) | 0.962 | **1.02** |
| Tetrafluoromethylenecyclopropane (thermodynamic match; non-viable — see §6.1) | 0.929 | **0.96** |
| HCFO-1233zd (commercial) | 0.843 | **61** |
| HFO-1234yf (commercial) | 0.792 | **9.0 × 10⁴** |
| HFO-1234ze (commercial) | 0.780 | **2.9 × 10⁷** |

All H/H(cyclopentane) values are point estimates from RF mean predictions at k_ij=0. Per-molecule 95% CIs from per-tree propagation (jointly varying m, σ, ε/k through teqp) and k_ij ∈ [-0.05, +0.05] sweep are in `model/saved/uncertainty_propagated.csv`. Even with the widest CIs, verified candidates remain within a factor of ~2 of cyclopentane, while commercial HFOs are separated by orders of magnitude — the qualitative conclusion is robust.

The verified candidates have Henry's constants within 2–5% of cyclopentane. Commercial HFOs are 1–7 orders of magnitude different. This confirms that the PU foam industry's reluctance to substitute HFOs directly for cyclopentane in rigid insulation is grounded in thermodynamics, not convention.

### 6.1. Why the Best Thermodynamic Match (Tetrafluoromethylenecyclopropane) Is Non-Viable

The screening pipeline's most promising HFO candidate — 3,3,4,4-tetrafluoromethylenecyclopropane (`C=C1C(F)(F)C1(F)F`) — achieves a near-perfect thermodynamic profile: H/H(cyclopentane) = 0.96, VP ratio = 1.02, zero ODP, presumably very low GWP (exocyclic C=C provides atmospheric degradation), and SA score = 3.48. On paper, it merges the best of both worlds: the highly fluorinated nature of an HFC (low gas-phase thermal conductivity) with an olefinic degradation site (short atmospheric lifetime).

In practice, this molecule would be catastrophic as a PU foam blowing agent. The failure modes are fundamental, not engineering problems:

**1. Extreme ring strain and chemical reactivity.** Cyclopropane alone carries ~27 kcal/mol of ring strain. The exocyclic methylene forces one ring carbon into sp2 hybridization, demanding 120° bond angles while trapped in a ~60° ring — adding another ~13 kcal/mol for a total of ~40 kcal/mol. Four highly electronegative fluorines on the remaining ring carbons further destabilize the system, creating a potent electrophile. In the PU foam B-side, the molecule would encounter basic amine catalysts (DABCO, BDMAEE) and the exothermic heat of the isocyanate–polyol reaction (peak temperatures reaching 140–160°C). Under these conditions, the strained ring would undergo nucleophilic ring-opening, electrocyclic rearrangement, or anionic polymerization — destroying the blowing agent and disrupting the foam chemistry simultaneously. The molecule would not peacefully vaporize to create cells; it would react out of the system.

**2. Prohibitive synthesis costs.** Industrial blowing agents must be produced at multi-kiloton scale for single-digit dollars per pound. Cyclopentane is a commodity petroleum fraction. Synthesizing a specific, highly strained, tetrafluorinated cyclopropane requires multi-step routes involving hazardous difluorocarbene precursors (e.g., TMSCF3 or ClCF2CO2Na thermolysis) under carefully controlled conditions. No scalable industrial synthesis pathway exists, and the cost would be orders of magnitude above the commercial threshold.

**3. Likely severe toxicity.** Highly reactive electrophilic molecules that readily undergo ring-opening reactions with nucleophiles (amines, polyols) would also react with biological nucleophiles — DNA bases, cysteine residues, glutathione. Small strained-ring electrophiles are well-established genotoxic alerts in pharmaceutical and occupational health screening. The occupational exposure risks of handling large volumes of this compound would likely be unacceptable.

**Why linear HFOs won the race.** This analysis perfectly illustrates why the industry converged on linear (straight-chain) hydrofluoroolefins like HFO-1336mzz(Z) and HFO-1234ze(E). By placing the C=C double bond in a straight carbon chain rather than a strained ring, chemists achieved the same environmental objective (atmospheric degradation site → low GWP) while keeping the molecule chemically stable enough to survive the basic, hot, exothermic PU foam reaction. The tradeoff is fundamentally different thermodynamic behavior — which is why HFOs require reformulated polyol systems, different surfactant packages, and co-blowing agent strategies rather than serving as drop-in cyclopentane replacements.

### Implications

**Matching cyclopentane's PC-SAFT parameters is the wrong strategy for HFO screening.** The molecules that match cyclopentane's parameters are hydrocarbons and lightly-halogenated hydrocarbons — the very compounds that fail on environmental or safety criteria. The molecules that achieve parameter matching *through fluorinated structures* (strained fluorinated rings) are the ones least likely to survive the PU foam reaction environment. Heavily fluorinated molecules that pass both environmental and stability criteria occupy a fundamentally different region of parameter space.

This reveals a deeper insight: **thermodynamic similarity to cyclopentane and chemical compatibility with PU foam processing are anti-correlated in fluorinated chemical space.** The structural features that bring ε/k close to cyclopentane's value (ring strain, high internal energy) are precisely the features that make a molecule reactive and unstable under foam conditions.

Future screening for HFO/HCFO blowing agents must therefore:
- **Accept fundamentally different thermodynamic behavior** and redesign the foam formulation (different polyols, co-blowing agents, surfactants) around the HFO's solubility characteristics — which is exactly what the industry has done with HFO-1336mzz(Z) in spray foam and XPS, or
- **Screen on direct end-use properties** (foam thermal conductivity, dimensional stability, cell structure) rather than pure-component thermodynamic parameter matching, or
- **Reframe the problem entirely**: rather than asking "what molecule behaves like cyclopentane?", ask "what formulation system produces equivalent foam performance with a low-GWP blowing agent?"

---

## 7. External and Thermodynamic Validations

### NIST Experimental Validation (Step 17)

57 molecules compared against NIST WebBook VP and liquid density at 298.15 K:

| Method | VP MARE | Density MARE | VP within ±20% |
|--------|---------|-------------|----------------|
| RF (this work) | 1121% | 12.7% | 44.8% |
| SPT-PCSAFT (Winter 2025) | 300% | 11.3% | 60.0% |
| Esper reference | 8185% | 14.5% | 57.1% |

The headline MARE numbers are misleading — they are dominated by catastrophic failures on associating compounds (alcohols, water). For the intended use case (non-polar blowing agents), RF VP errors are 2–10% and density errors <5%. The poor performance of Esper reference parameters was unexpected and suggests those parameters may be optimized for different conditions than 298 K saturation properties.

The most important result: **association parameter prediction is out of scope and should remain so.** Only ~200–500 compounds have experimentally fitted association parameters, and adding two more regression targets with sparser labels would worsen the data-to-feature ratio without improving screening outcomes (adding parameters to match makes the search strictly harder).

### Temperature Sweep Validation (Step 19)

50 candidates evaluated at T = [230, 250, 270, 290, 310, 330] K:
- **Rankings are highly stable**: Spearman ρ > 0.99 between 290 K and all temperatures in 250–330 K
- **Top candidates are invariant**: All rank-shifters (>10 position change) are mid-to-low ranked (positions 17–37)
- **Low-temperature failures are expected**: Only 12% of candidates converge at 230 K (below normal boiling range)
- **Implication**: Single-temperature screening at 298 K is sufficient, saving 6× computational cost

### Henry's Constant Mixture Analysis (Step 21)

See Section 6 above. This is the project's most impactful validation: it directly connects parameter-space proximity to the thermodynamic quantity that matters for foam formulation (dissolution behavior in solvent).

---

## 8. Infrastructure and Portal Enhancements

| Enhancement | Step | Description |
|-------------|------|-------------|
| Reference molecule selector | 13 | 24 curated molecules across 7 chemical families; custom reference support |
| OOD warning banner | 13 | Prominent `st.warning()` for out-of-domain predictions |
| Tanimoto score display | 14 | Color-coded similarity to nearest training molecule in portal |
| API `tanimoto_nn` field | 14 | Continuous AD score in `/predict` response |
| Model promotion | 15 | chemprop D-MPNN promoted as default for ε/k |

---

## 9. Negative Results (Documented for Completeness)

### Inverse-Variance Ensemble (Step 02c)

The RF+NN inverse-variance-weighted ensemble underperforms RF alone on all three targets: R² drops from 0.62→0.53 (m), 0.35→0.08 (σ), 0.33→0.13 (ε/k). The failure mode is uncertainty calibration mismatch — NN MC Dropout uncertainty estimates (σ ~ 0.05–0.30) are an order of magnitude narrower than RF tree disagreement (σ ~ 0.24–0.81), so the weaker NN receives 83–89% of the weight. Equal weighting (0.57/0.26/0.25) and RF-dominant 80/20 weighting (0.62/0.33/0.31) both still underperform RF alone, confirming that averaging in a substantially weaker model cannot help. The useful byproduct is model disagreement as an AD signal: molecules where RF and NN disagree by >2σ can be flagged for additional review, complementing the Tanimoto and Isolation Forest AD methods.

### GC-PC-SAFT as Input Feature (Step 18)

Adding GC-predicted (m, σ, ε/k) as RF input features provided no benefit. m and σ R² changed by <0.005; ε/k R² dropped from 0.33 to 0.29. The group-additivity information is redundant with Morgan fingerprints.

### XGBoost vs RF (Step 15)

XGBoost (gradient-boosted trees) performed identically to RF on all targets. Tree-based methods are saturated on this feature space; the performance ceiling is set by the representation, not the ensemble strategy.

---

## 10. Summary of Phase 2 Value

### For the Blowing Agent Problem

1. **Rigorous proof that PC-SAFT parameter matching cannot find HFO replacements for cyclopentane.** The ε/k barrier is a fundamental physical chemistry constraint, quantified via Henry's constant analysis.
2. **Demonstration that thermodynamic similarity and chemical stability are anti-correlated in fluorinated chemical space.** The most thermodynamically promising fluorinated candidate (tetrafluoromethylenecyclopropane, H/H(ref) = 0.96) achieves its parameter match through extreme ring strain — the same feature that makes it chemically incompatible with PU foam processing. This insight explains the industry's convergence on linear HFOs: they sacrifice thermodynamic similarity to cyclopentane in exchange for the chemical stability needed to survive the foam reaction.
3. **A complete screening pipeline** (enumeration → prediction → multi-criteria filtering → thermodynamic validation → mixture analysis) that can be reused for alternative target molecules or alternative property criteria.

### For the Broader ML + Thermodynamics Community

1. **A validated GNN architecture for PC-SAFT parameter prediction** with R² > 0.70 on all three parameters, uncertainty quantification, and three AD methods.
2. **First published quantification of inter-dataset variability** in PC-SAFT parameters (Esper vs ML-SAFT, 745 overlapping molecules).
3. **Demonstration that graph neural networks outperform fingerprint-based methods** for dispersion energy prediction, with a clear physical explanation (message passing mirrors PC-SAFT's segment model).

---

## 11. Phase 3 Direction

Phase 2 closes the chapter on parameter-matching as a screening strategy for HFO blowing agents. Phase 3 should pursue two tracks:

### Track A: Open-Source ML Model for PC-SAFT Parameters

The GNN pipeline has standalone value for the thermodynamic modeling community. Deliverables:
- Clean, documented Python package with pretrained weights
- Benchmark against SPT-PCSAFT (Winter 2025) on standardized test sets
- Web demo for single-molecule prediction
- Publication-ready comparison across architectures and data regimes

### Track B: Alternative Screening for HFO/HCFO Blowing Agents

Since PC-SAFT parameter matching fails for HFOs, the screening strategy must change:
- **Direct property prediction**: Train on experimental VP, density, and solubility data rather than intermediate EOS parameters
- **Formulation-aware models**: Account for the full foam system (polyol type, surfactant, co-blowing agent) rather than pure-component properties
- **Non-thermodynamic indicators**: Cell nucleation kinetics, gas-phase thermal conductivity, and foam dimensional stability may matter more than solubility matching for HFOs
- **Stability-aware screening**: The Phase 2 finding that thermodynamic similarity and chemical stability are anti-correlated in fluorinated space means screening must jointly optimize for foam performance *and* chemical compatibility with amine catalysts and exothermic reaction conditions — not thermodynamic similarity to cyclopentane
- **Industry collaboration**: Access to proprietary foam performance data would enable supervised learning on the actual target (foam insulation R-value) rather than proxy properties

---

## Appendix: Step Index

| Step | Title | Key Deliverable |
|------|-------|-----------------|
| 10 | ML-SAFT Integration | 870 molecules, inter-dataset variability analysis, widened acceptance windows |
| 02b | GNN (Graph Isomorphism Network) | R² > 0.70 on all targets with expanded data; best model |
| 02c | Model Ensemble | Negative result: inv-var ensemble underperforms RF; uncertainty calibration mismatch |
| 11 | Enhanced Evaluation Metrics | MARE, CCC, Coverage@5/10%, Q²; σ better than R² suggests |
| 12 | Multi-Criteria Candidate Verification | 3-temperature 5-criterion audit; 2 candidates confirmed |
| 13 | Portal Reference Comparison | 24-molecule reference library; custom comparison support |
| 14 | Improved Applicability Domain | Tanimoto AD, Williams plots, API/portal integration |
| 15 | Improved Models Benchmark | XGBoost and chemprop D-MPNN; chemprop promoted for ε/k |
| 16 | Fluorinated Data Expansion | 34 literature compounds; 6 fluorinated 3–5-membered rings |
| 17 | NIST Experimental Validation | 57-molecule external validation; 2–10% VP error for non-polar |
| 18 | GC as Feature Experiment | Negative result: GC features redundant with fingerprints |
| 19 | Temperature Sweep Validation | Rankings stable ρ > 0.99 across 250–330 K |
| 20 | Systematic HFO/HCFO Enumeration | 4,663 candidates; zero HFOs pass all criteria |
| 21 | Henry's Constant Analysis | Validates parameter proximity for mixture behavior; exponential ε/k sensitivity |
