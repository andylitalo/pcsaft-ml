# Executive Summary: ML-Driven Blowing Agent Screening (Phase 1)

## What We Did and Why

Cyclopentane is widely used as a blowing agent in foam insulation, but its flammability and modest thermal performance create demand for alternatives. We built an end-to-end machine learning pipeline to screen novel fluorinated molecules as potential replacements.

The approach: train ML models to predict PC-SAFT equation-of-state parameters (m, σ, ε/k) from molecular structure alone, then rank candidate molecules by how close their predicted parameters are to cyclopentane's known values. Molecules with similar PC-SAFT parameters should exhibit similar thermodynamic behavior — similar vapor pressure and liquid density — making them candidate drop-in replacements.

Over nine development steps, we built and evaluated four prediction models, deployed a production-grade serving stack (REST API, Docker, Kubernetes, automated retraining pipeline), and validated the screening strategy thermodynamically.

---

## What We Found

### Model Accuracy Is Moderate

The best model — a Random Forest trained on 2,218 molecular features — explains 62% of variance in the chain-length parameter (m), but only 35% and 33% of variance in the size (σ) and energy (ε/k) parameters. The latter two are the ones that most directly drive vapor pressure. In practical terms, the average error in ε/k is approximately ±27 K on a scale that typically ranges from 150–600 K.

A pre-trained chemical language model (ChemBERTa) ranked second with R² of 0.53/0.25/0.27 on m/σ/ε/k, outperforming a custom neural network (0.47/0.11/0.14). All ML approaches substantially beat the traditional group-contribution method (GC-PC-SAFT), which actually performed worse than predicting the mean for σ and ε/k.

### Parameter-Space Ranking Is a Weak Proxy for Thermodynamic Similarity

This is the most important finding of Phase 1. We validated the core screening assumption by running EOS calculations (via teqp) on all 645 filtered candidates and comparing their computed vapor pressures and liquid densities to cyclopentane's.

The Spearman rank correlation between parameter-space distance and property-space distance was **ρ = 0.372** — statistically significant but moderate. In the top-20 candidates by parameter distance, only **2 of 20 survived** when re-ranked by thermodynamic property distance. The #1 parameter-space candidate (chlorocyclopropene) failed the equation-of-state calculation entirely. Candidates that look similar in (m, σ, ε/k) space can have vapor pressures 2–58× higher than cyclopentane.

### Two Candidates Survived All Filters

Combining parameter proximity, EOS convergence, thermodynamic property similarity, and synthetic accessibility, exactly two candidates passed every criterion:

- **(1R,3S)-1,3-difluorocyclobutane** — `F[C@H]1C[C@@H](F)C1`
- **(1R,3R)-1,3-difluorocyclobutane** — `F[C@H]1C[C@H](F)C1`

Both have predicted vapor pressures approximately 1.1× cyclopentane's, SA scores ≤ 3.5 (readily synthesizable), and converging EOS calculations. They rank 10th and 11th by parameter distance but 1st and 2nd by thermodynamic property distance — illustrating why EOS validation is essential.

**Important caveat**: These rankings are based on ML-predicted PC-SAFT parameters, not experimentally fitted ones. The model has ~33% explained variance on the most critical parameter. Experimental characterization is required before any synthesis commitment.

---

## What We Built

Beyond the ML models, we delivered a complete MLOps stack:

- **REST API** (FastAPI): batch prediction endpoint with uncertainty estimates, applicability domain flagging, and an associating-molecule warning
- **Containerized deployment** (Docker + Kubernetes): 2-replica deployment with liveness/readiness probes
- **Automated retraining pipeline** (Kubeflow): triggers when ≥10 new experimental data points are submitted; uses champion/challenger promotion
- **Researcher portal** (Streamlit): predict parameters for any SMILES, submit experimental data, view model dashboard

128 tests pass across all components.

---

## Key Limitations Before Acting on Results

1. **Accuracy on σ and ε/k is low** (R² ≈ 0.33–0.35). These parameters dominate vapor pressure; errors propagate nonlinearly through the EOS.
2. **Parameter-space ranking ≠ property-space ranking** (ρ = 0.37). Always run EOS validation on candidates before prioritizing.
3. **Training data is biased** toward common industrial chemicals. Novel fluorinated rings are underrepresented; ~8% of test molecules are flagged out-of-domain, with performance dropping sharply for those molecules.
4. **No experimental validation** of ML-predicted parameters vs. measured thermodynamic data.
5. **Association parameters not predicted**. Molecules with OH groups or N-H bonds require additional PC-SAFT parameters that are out of scope.
6. **Single temperature** (298 K). Blowing agent performance spans ~230–330 K; candidate ranking may change at operating conditions.

---

## Phase 2 Priorities

Phase 2 targets the most impactful gaps:

1. **Integrate ML-SAFT dataset** — roughly doubles training data; scaffolding is already in place
2. **Add honest metrics** — Mean Absolute Relative Error (MARE) and coverage-at-threshold alongside R²
3. **Multi-temperature candidate audit** — run EOS at 273/298/323 K for the difluorocyclobutane candidates
4. **Flexible portal comparisons** — allow researchers to compare against any molecule with known parameters, not just cyclopentane
5. **Improved applicability domain** — Tanimoto similarity AD and Williams plot for chemically interpretable OOD flagging
6. **Better models** — benchmark XGBoost and graph neural network (chemprop) against the RF baseline
