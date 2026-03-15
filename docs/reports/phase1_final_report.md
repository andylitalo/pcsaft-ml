# Phase 1 Final Report: ML-Driven PC-SAFT Parameter Prediction for Blowing Agent Screening

**Dataset**: Esper PC-SAFT dataset, 1,801 molecules (1,440 train / 361 test, stratified 80/20 split on binned ε/k)
**Objective**: Predict three PC-SAFT parameters (m, σ, ε/k) from molecular structure, then use parameter-space proximity to cyclopentane (m=2.37, σ=3.71 Å, ε/k=289 K) to rank candidate blowing agents.
**Steps completed**: 01–09

---

## 1. Progressive Model Performance

Each step improved or contextualized the prediction of the three PC-SAFT parameters. The table below shows R² and MAE on the held-out test set (361 molecules) for every model, in the order they were developed.

### R² (test set)

| Model | R² (m) | R² (σ) | R² (ε/k) | Notes |
|-------|--------|--------|----------|-------|
| GC-PC-SAFT (domain baseline) | 0.40 | −1.16 | −0.04 | Group-additivity; fails on σ and ε/k |
| RF — RDKit only (Step 00 baseline) | 0.61 | 0.34 | 0.31 | 170 cleaned RDKit 2D descriptors |
| RF — Morgan only (Step 01) | 0.43 | 0.17 | 0.17 | 2,048-bit ECFP4; worse than RDKit alone |
| **RF — Combined (Step 01)** | **0.62** | **0.35** | **0.33** | 2,218 features; best overall |
| NN — PCSAFTNet (Step 02/03) | 0.47 | 0.11 | 0.14 | Small-data regime; over-parameterized |
| **ChemBERTa (Step 04)** | **0.53** | **0.25** | **0.27** | SMILES-native; strong second place |

### MAE

| Model | MAE (m) | MAE (σ, Å) | MAE (ε/k, K) |
|-------|---------|------------|--------------|
| GC-PC-SAFT | 1.001 | 0.494 | 44.6 |
| RF (Combined) | 0.585 | 0.189 | 26.8 |
| NN (PCSAFTNet) | 0.767 | 0.238 | 32.9 |
| ChemBERTa | 0.766 | 0.226 | 31.2 |

**Summary**: The Random Forest with combined Morgan+RDKit features is the best model on all three targets. ChemBERTa is a strong second, requiring no feature engineering. The NN is penalized by the small-data regime (~500K parameters, 1,440 samples). All ML models beat the group-contribution baseline; GC-PC-SAFT has negative R² on σ and ε/k.

---

## 2. Applicability Domain Analysis (Step 03)

The Isolation Forest AD model (contamination=0.05) flagged **8.0% of test molecules (29/361) as out-of-domain**. Performance drops substantially on OOD molecules:

| Model | Target | R² (in-domain) | R² (OOD) | Drop |
|-------|--------|----------------|----------|------|
| RF | m | 0.640 | 0.330 | −0.31 |
| RF | σ | 0.369 | 0.076 | −0.29 |
| RF | ε/k | 0.351 | 0.188 | −0.16 |
| ChemBERTa | m | 0.534 | 0.351 | −0.18 |
| ChemBERTa | ε/k | 0.303 | 0.015 | −0.29 |

The RF shows the clearest OOD degradation, validating the AD check as a meaningful deployment gate. RF tree-disagreement uncertainty is better calibrated than NN MC Dropout.

---

## 3. Infrastructure (Steps 05–08)

| Step | Deliverable | Status |
|------|-------------|--------|
| 05 | FastAPI REST API (`/predict`, `/submit-data`, `/health`) | Complete; 74 tests |
| 06 | Multi-stage Dockerfile + 5 Kubernetes manifests (pcsaft namespace, 2-replica deployment) | Complete; 88 tests |
| 07 | 6-component Kubeflow KFP pipeline: validate → merge → retrain → evaluate → compare → promote | Complete; 99 tests |
| 08 | Streamlit portal: Predict / Submit / Dashboard tabs; cyclopentane comparison UX | Complete; 119 tests |

All 128 project tests pass post-Step 09. The API returns `in_domain`, `uncertainty`, and `is_associating` per molecule. The portal degrades gracefully when the API is offline.

---

## 4. Thermodynamic Validation (Step 09)

The core assumption of the screening strategy — that proximity in PC-SAFT parameter space implies proximity in thermodynamic property space — was tested by running the teqp EOS on all 645 SA-filtered screening candidates at 298.15 K.

**Cyclopentane reference properties** (computed via teqp, PC-SAFT parameters from Gross & Sadowski 2001):
- Vapor pressure: 16.25 kPa
- Liquid density: 10,865.5 mol/m³

**EOS results**:
- 523/645 (81.1%) converged to physical VLE
- 122/645 (18.9%) failed — parameter combinations thermodynamically inconsistent

**Spearman ρ = 0.372** (p = 1.3 × 10⁻¹⁸) between parameter-space and property-space distances.

This is statistically significant but **moderate**. Practically:
- Only **2/20** top-parameter candidates remained in the top-20 when re-ranked by property distance
- The #1 parameter-space candidate (ClC1=CC1, chlorocyclopropene) **failed EOS entirely**
- Sensitivity analysis shows ε/k dominates vapor pressure (~50% VP change per 10% ε/k change); current 3:1:1 (ε/k:σ:m) weights may be too conservative — 5:2:1 suggested

---

## 5. Candidate Verdict

Combining all selection criteria:

| Criterion | Filter |
|-----------|--------|
| 1. Applicability domain | Isolation Forest in-domain |
| 2. Parameter proximity | Top-20 by weighted Euclidean distance to cyclopentane |
| 3. EOS convergence | teqp VLE calculation succeeds |
| 4. Property proximity | VP ratio 0.5–1.5× cyclopentane; density ratio 0.8–1.2× |
| 5. Synthetic accessibility | SA score ≤ 4.5 |

**Candidates passing all five criteria**:

| SMILES | Name | SA Score | Param Rank | Property Rank | VP ratio |
|--------|------|----------|------------|---------------|----------|
| `F[C@H]1C[C@@H](F)C1` | (1R,3S)-1,3-difluorocyclobutane | 3.30 | 10 | 1 | ~1.1× |
| `F[C@H]1C[C@H](F)C1` | (1R,3R)-1,3-difluorocyclobutane | 3.30 | 11 | 2 | ~1.1× |

These two difluorocyclobutane diastereomers are the **only candidates that survive all five filters simultaneously**. They rank 10th and 11th by parameter distance but 1st and 2nd by property distance. The top parameter-distance candidates (chlorocyclopropene, cyclobutene) have 2–58× higher vapor pressure than cyclopentane.

**Recommendation**: The difluorocyclobutane isomers are the best candidates for experimental follow-up. Their vapor pressures are close to cyclopentane (~1.1×), they are synthetically accessible (SA ≤ 3.5), and both EOS calculations converge. However, note that these properties are derived from ML-predicted parameters (RF model, R²(ε/k)=0.33), not experimental fits.

---

## 6. Scientific Limitations

These are limitations to disclose before any external presentation.

### 6.1 Model accuracy is moderate, not high

The best model (RF Combined) achieves R² = 0.62 on m, but only 0.35 and 0.33 on σ and ε/k. The latter two parameters dominate thermodynamic property predictions (sensitivity analysis: ε/k explains ~50% of VP variation). An R² of 0.33 on ε/k means the model explains only one-third of variance in the target most critical to vapor pressure.

A 95% confidence interval on R² = 0.33 (n=361 test samples) spans approximately ±0.08, so the true predictive power could range from 0.25 to 0.41.

### 6.2 PC-SAFT parameters are fit-dependent

The Esper training set aggregates parameters from multiple literature sources using different fitting protocols, different equation-of-state variants (PC-SAFT, PCP-SAFT, GC-PC-SAFT), and different experimental temperature/pressure ranges. This heterogeneity introduces label noise that is not captured in any uncertainty estimate. A molecule reported with ε/k = 250 K in one source and 270 K in another is indistinguishable from model error.

### 6.3 Parameter-space proximity ≠ property-space proximity

Spearman ρ = 0.372 between parameter rank and property rank. Only 2/20 top-parameter candidates survive re-ranking by property distance. The weighted Euclidean distance in (m, σ, ε/k) space is a loose proxy for thermodynamic similarity. **Ranking by parameter distance alone is unreliable for high-confidence screening**.

The current 3:1:1 (ε/k:σ:m) weighting is not strongly validated; sensitivity analysis suggests 5:2:1 would better reflect ε/k's dominance of vapor pressure.

### 6.4 Association parameters are not predicted

The models predict only the three non-associating parameters. Blowing agent candidates with H-bond donors or acceptors (hydroxy groups, NH, etc.) require εAB/k and κAB for accurate EOS. The portal currently flags `is_associating = True` for such molecules but cannot predict association parameters. These candidates are effectively unscreened.

### 6.5 All validation is computational, not experimental

Thermodynamic validation compares teqp(predicted params) to teqp(literature params for cyclopentane). No step validates ML-predicted parameters against experimental vapor pressure or density data for the screened candidates themselves. A true validation would compare `teqp(predicted params)` vs NIST WebBook experimental data for a held-out molecule set.

### 6.6 Training data is biased toward common industrials

The Esper dataset is dominated by simple alkanes, alcohols, ketones, and aromatics. Fluorinated ring systems — the chemical space most relevant to blowing agents — are underrepresented. This contributes to the 8% OOD rate on the test set and likely a higher OOD rate for novel HFO/HCFO candidates.

### 6.7 Screening covers 298 K only

Blowing agent performance spans approximately 230–330 K depending on application. All EOS validation was performed at 298.15 K. The candidate ranking may shift at operating temperatures.

### 6.8 No GWP/ODP or regulatory screening

Global warming potential (GWP) and ozone depletion potential (ODP) are not predicted. Regulatory compliance (F-gas regulations, REACH, SNAP) is not assessed. SA score is a proxy for synthetic accessibility only.

---

## 7. Recommendations for Phase 2

Based on the above limitations, the following improvements are prioritized:

| Priority | Action | Limitation Addressed |
|----------|--------|---------------------|
| 1 | Integrate ML-SAFT dataset (already scaffolded) | 6.1, 6.6 — more data, broader coverage |
| 2 | Add MARE, coverage@5%/10%, CCC, Q² metrics | 6.1 — more honest accuracy reporting |
| 3 | Multi-criteria candidate verification at 273/298/323 K | 6.7 — temperature range |
| 4 | Portal reference molecule comparison (Level 1+2) | Usability for non-cyclopentane targets |
| 5 | Tanimoto similarity AD + Williams plot | 6.6 — interpretable OOD flagging |
| 6 | XGBoost and chemprop (D-MPNN) benchmark | 6.1 — better models |

---

## 8. Figure Index

| Figure | Location | Description |
|--------|----------|-------------|
| RF parity (Morgan+RDKit) | `figures/01_morgan_fingerprints/parity_combined_rf.png` | 3-panel parity plot for combined RF |
| R² comparison (4 methods) | `figures/01_morgan_fingerprints/comparison_r2.png` | GC-SAFT vs RF variants |
| NN loss curves | `figures/02_pytorch_nn/nn_loss_curves.png` | Training/validation loss; best epoch 14 |
| RF vs NN R² | `figures/02_pytorch_nn/rf_vs_nn_r2_comparison.png` | Side-by-side bar chart |
| 3×3 parity grid | `figures/03_evaluation_harness/parity_comparison.png` | GC-SAFT / RF / NN, AD-colored |
| Residual distributions | `figures/03_evaluation_harness/residual_distributions.png` | Per model/target histograms |
| Uncertainty calibration | `figures/03_evaluation_harness/uncertainty_calibration.png` | RF vs NN calibration |
| 4-way R² comparison | `figures/04_chemberta/four_way_r2_comparison.png` | All models including ChemBERTa |
| Screening results | `figures/04_chemberta/screening_results.png` | Distance histogram, 645 candidates |
| Parameter vs property distance | `figures/09_thermodynamic_validation/param_vs_property_distance.png` | Spearman ρ = 0.372 scatter |
| Rank comparison | `figures/09_thermodynamic_validation/rank_comparison.png` | Bump chart: 18/20 top-param drop out |
| Sensitivity analysis | `figures/09_thermodynamic_validation/sensitivity_analysis.png` | ε/k dominates VP |
| VP comparison | `figures/09_thermodynamic_validation/vapor_pressure_comparison.png` | VP for top-20 candidates |
