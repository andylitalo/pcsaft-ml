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
