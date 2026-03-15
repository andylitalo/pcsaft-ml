# Orchestrator Log

Each entry records what was completed, decisions made, and what comes next.
New orchestrator instances should read the last 2-3 entries for context.

---

## 2026-03-14: Step 01 — Morgan Fingerprints (Completed)

**What was done**: Implemented Morgan FP computation (`build_features()`), GC-PC-SAFT simplified baseline,
ML-SAFT dataset loader (download module created but data not fetched), InChI-based deduplication,
stratified splitting by epsilon_k quintiles, StandardScaler for RDKit features. Updated train/evaluate/predict
to use `build_features()`. Added 21 new tests (33 total). Generated parity plots and comparison bar chart.

**Key metrics**: RF(Combined) R² = m:0.62, σ:0.35, ε/k:0.33. Modest +1-2% over RDKit-only baseline.
All ML models beat GC-PC-SAFT domain baseline. Morgan-only RF underperformed RDKit-only (expected for trees).

**Decisions**: Used Esper-only data (ML-SAFT not downloaded). Stratified split changes baseline numbers
slightly from original report. feature_config.json + rdkit_names approach ensures inference consistency.

**Next action**: Start Step 02 (PyTorch Multi-Task NN). Install torch, create `model/nn/` package.
Gate 02 check passes. Branch: `step-02-pytorch-nn`.

---

## 2026-03-14: Step 02 — PyTorch Multi-Task NN (Completed)

**What was done**: Implemented PCSAFTNet (shared trunk 512→256→128, 3 task heads), full training loop
with AdamW/LR scheduling/early stopping, MC Dropout UQ (n_forward=30), RF tree-disagreement UQ,
Isolation Forest AD model. Added --model {rf,nn} flag to train.py. 7 new tests (40 total).

**Key metrics**: NN R² = m:0.42, σ:0.04, ε/k:0.09. Underperforms RF (0.62/0.35/0.33) as expected in
small-data regime (1,440 samples, ~500K params). Best epoch 14, early stopped at 34.

**Decisions**: NN result is documented honestly — expected outcome for small-data + high-dim features.
ChemBERTa (Step 04) with pre-trained embeddings should perform better.

**Next action**: Start Step 03 (Evaluation Harness). Create registry, unified evaluate, comparison figs.
Gate 03 check passes. Branch: `step-03-evaluation-harness`.

---

## 2026-03-14: Step 03 — Evaluation Harness (Completed)

**What was done**: Created model registry (`model/registry.py`) with `@register_model` decorator wrapping
GC-PC-SAFT, RF, NN behind common `load()`/`predict()`/`predict_with_uncertainty()` interface. Completely
rewrote `model/evaluate.py` for multi-model unified evaluation CLI. Generated 4 figures: 3×3 parity plots
with AD coloring, residual distributions, NN learning curves, uncertainty calibration. AD analysis shows
8% OOD, RF R² drops 0.15-0.31 for OOD molecules. 12 new tests (52 total).

**Key metrics**: RF remains best: R²=0.62/0.35/0.33. NN middle ground: R²=0.47/0.11/0.14. GC-PC-SAFT
baseline: R²=0.40/-1.16/-0.04. RF uncertainty better calibrated than NN MC Dropout.

**Decisions**: Registry extensible — ChemBERTa (Step 04) needs only `@register_model("chemberta")` class.
comparison_metrics.csv gitignored (generated artifact).

**Next action**: Start Step 04 (ChemBERTa Fine-Tuning). Install transformers/datasets, CPU proof-of-concept.
Gate 04 check passes. Branch: `step-04-chemberta`.

---

## 2026-03-14: Step 04 — ChemBERTa Fine-Tuning (Completed)

**What was done**: Fine-tuned ChemBERTa (seyonec/ChemBERTa-zinc-base-v1, 85M params) for PC-SAFT
regression. Created model/hf/ package with ChemBERTaForPCSAFT, PCSAFTSmilesDataset, training script.
Registered in model/registry.py. Added --model chemberta to train.py. Expanded screening/generate.py
with HCFOs, HFEs, unsaturated hydrocarbons, cyclic fluorinated scaffolds (805 candidates, 645 after SA).
Generated 4 figures, wrote 4-way comparison report. 13 new tests (65 total).

**Key metrics**: ChemBERTa R²=0.53/0.25/0.27 — strong 2nd behind RF (0.62/0.35/0.33), significantly
beats NN (0.47/0.11/0.14). Transfer learning from 77M SMILES helps despite only 1,440 fine-tuning samples.
Early stopped at epoch 8, ~2.5 min on CPU. Best model for deployment: RF.

**Decisions**: RF chosen as deployment model. ChemBERTa recommended for transfer-learning scenarios with
more data. AD limitations for ChemBERTa documented (Morgan FP space vs tokenizer space). Expanded screening
uses RF for ranking.

**Next action**: Start Step 05 (FastAPI Serving). Tier 2 auto-approve if tests+ruff+gate pass.
Gate 05 check passes. Branch: `step-05-fastapi`.

---

## 2026-03-14: Step 05 — FastAPI Serving (Completed, Auto-Approved)

**What was done**: Created REST API with /health, /predict, /submit-data endpoints. RF model loaded at
startup via model registry. Predictions include uncertainty (tree disagreement), AD check (Isolation Forest),
and association flagging. Pydantic schemas for request/response validation. Config via PCSAFT_* env vars.
9 new tests (74 total).

**Key features**: Invalid SMILES return valid=false (not 500). Batch predictions up to 100 molecules.
Submit-data endpoint appends to CSV for future retraining. OpenAPI docs at /docs.

**Tier 2 auto-approve**: 74/74 tests pass, ruff clean, gate 06 OK, report exists.

**Next action**: Start Step 06 (Docker & Kubernetes).

---

## 2026-03-14: Step 06 — Docker & Kubernetes (Completed, Auto-Approved)

**What was done**: Created multi-stage Dockerfile (python:3.11-slim, builder+runtime), .dockerignore, and 5
K8s manifests: namespace, configmap (RF default), deployment (2 replicas, liveness/readiness probes on
/health, resource limits 250m-1000m CPU / 512Mi-2Gi RAM), service (ClusterIP 80->8000), PVC (1Gi for
submissions). 14 new tests verify file structure and YAML content. No Docker build (macOS, no Docker).

**Tier 2 auto-approve**: 88/88 tests pass, ruff clean, gate 07 OK, report exists.

**Next action**: Start Step 07 (Kubeflow Retrain Pipeline).

---

## 2026-03-14: Step 07 — Kubeflow Retrain Pipeline (Completed, Auto-Approved)

**What was done**: Created 6-component KFP pipeline DAG: validate_data (SMILES + range checks),
merge_datasets (InChI dedup + versioned snapshots), retrain_model (RF on merged data),
evaluate_model (MAE/RMSE/R² on test set), compare_models (champion/challenger avg R²),
promote_model (conditional copy to production). Pipeline compiles to 22KB YAML. Cron-based
trigger checks for >= 10 new submissions. Self-contained components with all imports inside
function body for KFP container isolation. 11 new tests (99 total).

**Key features**: Champion/challenger promotion requires avg R² improvement > 0.005.
OutputPath(str) used instead of Input[Dataset]/Output[Model] for simpler testing.
InChI-based deduplication handles SMILES variations.

**Tier 2 auto-approve**: 99/99 tests pass, ruff clean, gate 08 OK, report exists.

**Next action**: Start Step 08 (Streamlit Portal).

---

## 2026-03-14: Step 08 — Streamlit Academic Portal (Completed, Auto-Approved)

**What was done**: Created 3-tab Streamlit portal: prediction page (SMILES input, RDKit 2D molecule
rendering, m/sigma/epsilon_k display with cyclopentane comparison via st.metric, batch CSV upload),
data submission form (SMILES + parameter ranges + source DOI, validation before submit), model
dashboard (health status, R² metrics, fallback data when API unavailable). PCSAFTClient wrapper
with graceful error handling. 20 new tests (119 total).

**Key features**: Molecule rendering via RDKit Draw.MolToImage. Cyclopentane comparison shows
% difference for each parameter. Dashboard falls back to hardcoded RF metrics when API is down.
Batch prediction supports CSV upload for bulk screening.

**Tier 2 auto-approve**: 119/119 tests pass, ruff clean, gate 09 OK, report exists.

**Next action**: Start Step 09 (Thermodynamic Validation). This is Tier 1 — requires human approval.

---

## 2026-03-14: Step 09 — Thermodynamic Validation (Completed, Approved)

**What was done**: Created `model/thermodynamic.py` with teqp-based EOS property computation
(vapor pressure, liquid density at 298.15 K). Ran validation on all 645 screening candidates.
Computed Spearman rank correlation between parameter-space and property-space distances.
Generated 4 figures (param vs property scatter, VP comparison, rank comparison bump chart,
sensitivity analysis). 9 new tests (128 total).

**Key metrics**: Spearman ρ = 0.372 (moderate correlation). EOS success rate 81.1% (523/645).
Only 2/20 top parameter-ranked candidates stay in top-20 after property re-ranking. Best
thermodynamic matches: difluorocyclobutane (VP ratio 0.96), fluoropentene (VP ratio 0.99).

**Decisions**: Parameter-space distance is a moderate but imperfect proxy. Production recommendation:
use parameter distance for cheap initial ranking, then validate top-50 with EOS before synthesis.
Sensitivity analysis suggests reweighting from 3:1:1 to 5:2:1 (ε/k:σ:m).

**This is the final step. All 9 steps complete.**

---

## 2026-03-14: Step 04B — ChemBERTa Applicability Domain (Completed, Approved)

**What was done**: Built ChemBERTa-specific AD detector using CLS embedding space (768-dim)
instead of descriptor-space fallback (2,150-dim). Created `model/hf/ad.py` with IsolationForest
on training CLS embeddings. Refactored `ChemBERTaForPCSAFT` to expose `encode()` method.
Updated registry (`embed()`, `predict_in_domain()`), evaluation (model-specific `_get_ad_labels()`),
and serving (branches on `model_type` for AD). 10 new tests (138 total). 3 figures generated.

**Key metrics**: In-domain m R² = 0.608 (vs 0.535 overall, +14% relative). 26.9% flagged OOD
(vs ~21% for descriptor-space AD). σ shows weak AD separation. ε/k OOD R² noisy (97 molecules).
RF and NN descriptor-space AD completely unchanged.

**Decisions**: IsolationForest(contamination=0.05) on CLS embeddings — matches existing convention.
Embedding-space AD is more architecturally consistent for transformer models. Higher OOD fraction
is expected since learned representations are more selective than handcrafted descriptors.

**Next action**: Branch step complete. All 9 steps + Step 04B finished.

---

## 2026-03-15: Phase 2 Begin — Steps 10, 11, 02B (Previously Completed)

**Context**: Phase 2 step guides appeared in docs/steps/ for steps 02b, 02c, 10-19.
Steps 10, 11, and 02b were already completed on the `phase2/experiments` branch
(5 commits ahead of main). Fixed 17 ruff lint errors from those steps.

**Step 10 (ML-SAFT Integration)**: Integrated SPT-PCSAFT dataset. RF retrain on
combined data confirmed Phase 1 metrics (R² m:0.6194, σ:0.3527, ε/k:0.3298).

**Step 11 (Enhanced Metrics)**: Added MARE, CCC, coverage@5%/10%, Q² metrics.
RF MARE(ε/k)=0.113, CCC(ε/k)=0.496, coverage@10%=74%.

**Step 02B (GNN)**: Graph neural network with PCSAFTGraphNet architecture.
GNN R² m:0.76, σ:0.77, ε/k:0.73 — major improvement over RF baseline.

---

## 2026-03-15: Batch 1 — Steps 12, 18, 19 (Completed)

Three agents ran in parallel on independent branches. All merged to phase2/experiments.

**Step 12 (Candidate Verification)**: 5-criterion screening audit. 2/645 candidates
pass all criteria (both difluorocyclobutane isomers), confirming Phase 1 verdict.
Multi-temperature EOS validation at 273/298/323 K. 10 new tests.

**Step 18 (GC-as-Feature)**: NEGATIVE RESULT. GC-PC-SAFT predictions as RF input
features provide no benefit (m: +0.004, σ: +0.005, ε/k: -0.037 R² delta). Baseline
Morgan+RDKit remains optimal. 4 new tests.

**Step 19 (Temperature Sweep)**: VP and density at 6 temperatures (230-330 K) for
top-50 candidates. Rank stability excellent (Spearman ρ > 0.99 for 250-330 K).
Validates single-temperature screening at 298 K. 9 new tests.

**Current test count**: 161 passing, ruff clean.

**Next action**: Launch Batch 2 — Steps 13 (Portal Reference) and 16 (Fluorinated Data).

---

## 2026-03-15: Batch 2 — Steps 13, 16 (Completed)

Two agents ran in parallel on independent branches. All merged to phase2/experiments.

**Step 13 (Portal Reference Comparison)**: Added GET /reference-molecules endpoint
with 24 curated molecules from Esper dataset. Portal selectbox replaces hardcoded
cyclopentane reference. OOD warning banner for out-of-domain predictions. 7 new tests.

**Step 16 (Fluorinated Data Expansion)**: Curated 34 fluorinated compounds (HFCs,
HFOs, PFCs, fluoroethers) with PC-SAFT parameters from literature. Updated load.py
with source='fluorinated' option. Analysis script for chemical diversity. 15 new tests.

**Current test count**: 183 passing, ruff clean.

**Next action**: Launch Batch 3 — Steps 14 (Improved AD) and 17 (NIST Validation).

---

## 2026-03-15: Batch 3 — Steps 14, 17 (Completed)

Two agents ran in parallel on independent branches. All merged to phase2/experiments.

**Step 14 (Improved Applicability Domain)**: TanimotoAD using Morgan FP nearest-neighbor
similarity (threshold 0.4). Williams plot leverage diagnostics for all 3 targets.
Updated serving API (tanimoto_nn field) and portal (color-coded AD warnings).
16 new tests, 4 figures.

**Step 17 (NIST Experimental Validation)**: Curated 57 NIST molecules with experimental
VP and density at 298.15 K. RF MARE(VP) = 2-10% for non-polar compounds, catastrophic
for associating (1000-25000%). MARE(density) = 12.7%. SPT comparison: MARE(VP) = 300%.
15 new tests, 4 figures.

**Current test count**: 214 passing, ruff clean.

**Next action**: Launch Batch 4 — Steps 15 (Improved Models) and 02c (Model Ensemble).

---
