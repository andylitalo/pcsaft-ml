# Model Comparison: Full Results and Production Rationale

This document provides the current model comparison results across the evaluated datasets and the rationale used so far for selecting Random Forest (RF) for production deployment. Some sections below are explicitly provisional until the Step 47-50 validation work is complete.

---

## Why These Models?

The problem — predicting three PC-SAFT parameters from molecular structure — is a multi-output regression task on ~1,800 training molecules represented as ~2,200-dimensional fixed-length feature vectors (170 cleaned RDKit descriptors + 2,048-bit Morgan fingerprints). That problem structure, combined with the low-data regime, determines which model families are worth evaluating.

### Random Forest as baseline

RF has been the default strong baseline for QSAR (quantitative structure–activity/property relationship) modeling since Svetnik et al. (2003, *J. Chem. Inf. Comput. Sci.* **43**, 1947–1958) demonstrated it consistently outperformed other methods (SVMs, ridge regression, neural nets) across multiple pharmaceutical datasets. It became the standard baseline in MoleculeNet (Wu et al. 2018, *Chem. Sci.* **9**, 513–530) and remains competitive on small-to-medium molecular datasets.

More broadly, Grinsztajn et al. (2022, *NeurIPS* Datasets and Benchmarks) showed that tree-based models (RF, XGBoost, GBM) match or beat deep learning on medium-sized tabular datasets across 45 benchmarks — exactly the regime of this project. The advantages that make RF a natural first choice here:

- **High-dimensional sparse features**: RF handles 2,048 binary Morgan fingerprint bits alongside 170 continuous descriptors without requiring feature scaling or dimensionality reduction. Feature subsampling at each split (sqrt(n_features) ≈ 47 candidates per split) acts as built-in regularization in this high-dimensional space.
- **Variance reduction without bias increase**: Bagging (bootstrap aggregation) averages 100 independently trained trees, reducing prediction variance while preserving the low bias of deep individual trees. This is effective in the low-sample regime where individual models overfit.
- **No extrapolation**: RF predictions are bounded by training data range (leaf-node averages), which is conservative but prevents wild predictions on out-of-distribution molecules — appropriate for a screening application where false positives are costly.
- **Built-in uncertainty**: Per-prediction tree variance provides a natural uncertainty estimate without requiring separate calibration infrastructure (though it requires local calibration for deployment; see [rf_inductive_bias.md](rf_inductive_bias.md)).

### XGBoost as the natural alternative tree ensemble

XGBoost (Chen & Guestrin, 2016, *KDD*) represents the other dominant tree ensemble paradigm: sequential gradient boosting vs. RF's parallel bagging. The two algorithms make fundamentally different optimization choices:

- **RF (bagging)**: trains trees independently on bootstrap samples, then averages. Each tree sees the full target signal but a perturbed view of the data. Variance reduction is the primary mechanism.
- **XGBoost (boosting)**: trains trees sequentially, where each new tree fits the residual errors of the ensemble so far. This iterative error correction can capture higher-order feature interactions that bagging misses, and the additive structure with regularized objectives (L1/L2 on leaf weights) provides fine-grained control over model complexity.

On many tabular benchmarks (Kaggle competitions, OpenML suites), XGBoost edges out RF because boosting's sequential residual fitting extracts more signal from complex feature interactions. XGBoost is the standard "second tree ensemble to try" after RF.

On this dataset, however, the two tied: R^2 = 0.33 vs 0.33 on epsilon/k, 0.64 vs 0.62 on m, 0.36 vs 0.35 on sigma. **That tie is informative, but not definitive.** It is consistent with the hypothesis that the current fixed-feature representation is becoming the limiting factor, but it does not by itself prove a representation bottleneck. Other explanations remain possible, including limited sample size, split noise, or remaining hyperparameter headroom.

### SVR as algorithm-diversity probe (Step 47)

SVR with an RBF kernel represents a fundamentally different inductive bias from tree-based methods: it implicitly maps features to infinite-dimensional space via the kernel trick, rather than partitioning the feature space axis-aligned. Across 10 repeated stratified outer splits, SVR achieved R^2 = 0.325 ± 0.058 on epsilon/k — within 0.005 of both RF and XGBoost single-split estimates. This extends the algorithm-diversity evidence from two families (bagging, boosting) to three (bagging, boosting, kernel methods), **strengthening the case that the RDKit + Morgan representation, not the learning algorithm, is the primary bottleneck** for epsilon_k prediction.

### Ridge as linear baseline (Step 47)

Ridge regression establishes the performance floor of a regularized linear model on the same features. Its epsilon_k R^2 = -0.43 ± 1.55 is catastrophically unstable across splits, with some folds achieving R^2 ~ 0.45 and others as low as -4.34. The severe instability arises from collinearity in the 2,200-dimensional feature space. The **~0.76 R² gap between Ridge and RF on epsilon/k** quantifies the value of non-linear representations: tree ensembles and kernel methods capture structure–property interactions (ring formation, branching, halogenation) that linear models cannot.

XGBoost would become the preferred tree ensemble if richer features (e.g., 3D conformer descriptors) or substantially more training data introduced interaction effects that boosting could exploit. For this project, the RF = XGBoost tie was one of the signals that motivated testing learned representations (GNNs) and adding more disciplined follow-up checks, not a final causal conclusion on its own.

### GNNs as learned-representation challenger

GNNs bypass the fixed-feature representation entirely: they operate directly on the molecular graph (atoms as nodes, bonds as edges) and learn task-specific representations via message-passing. This is the natural "next rung" on the representation ladder for two reasons:

1. **Physics-matching inductive bias**: GNN message-passing mirrors PC-SAFT's own physical model. PC-SAFT treats molecules as chains of spherical segments whose dispersion energy (epsilon/k) depends on local chemical environment. After k GNN layers, each node encodes a k-hop neighborhood — exactly the local context that determines segment contributions. This makes GNNs a data-driven generalization of group-contribution methods, capable of learning non-additive interactions between functional groups ([architecture details](gnn_architectures.md)).
2. **Standard experimental design**: Comparing a feature-engineered baseline (RF on RDKit + Morgan) against an end-to-end learned representation (GNN on molecular graphs) is the standard experimental design in ML for molecular properties. It directly tests whether hand-crafted chemical knowledge or data-driven representation learning is more effective at the current data scale.

Two GNN architectures were tested: GINEConv (Xu et al. 2019; Hu et al. 2020) for atom-level message-passing, and chemprop D-MPNN (Yang et al. 2019) for bond-level directed message-passing. Both consistently outperformed RF on epsilon/k (+0.06–0.08 R^2 on the same Esper data), confirming that graph representations capture structure–property relationships that fingerprints miss — but neither survived external fluorinated validation.

### What was excluded and why

- **Plain MLP**: A PyTorch multi-layer perceptron was tested on the same RDKit + Morgan features. It achieved R^2 = 0.14 on epsilon/k — substantially worse than RF (0.33) — likely due to overfitting in the low-data regime without the variance-reduction benefits of bagging.
- **ChemBERTa** (SMILES transformer): Tested but underperformed feature-engineered models (R^2 = 0.27 on epsilon/k, 185x slower inference). With ~1,800 fine-tuning examples, the pretrained language model could not match hand-crafted chemical features ([details](smiles_chemberta_issues.md)).
- **ChemBERTa-2** (`DeepChem/ChemBERTa-77M-MTR`): Evaluated for inclusion in Step 48 and declined. It was listed as a candidate in the Step 04 guide and uses multi-task pre-training on molecular properties, which could yield marginal improvement over ChemBERTa-1. However, the fundamental bottleneck is unchanged: ~1,800 fine-tuning examples is insufficient for any SMILES transformer to surpass RDKit + Morgan fingerprints at this data scale. Adding it would require hours of fine-tuning compute for a model that is not a deployment candidate. The existing ChemBERTa-1 result already represents the SMILES-transformer angle in the comparison.
- **GNNePCSAFT** (PyPI: `gnnepcsaft`, [HuggingFace](https://huggingface.co/wildsonbbl/gnnepcsaft)): A published, open-source PNA-based GNN model built specifically to predict ePC-SAFT parameters, trained on the Esper dataset and integrated with the FeOs thermodynamic library. It was not used during the original model-building steps because the project goal was to build and evaluate models from scratch — calling a pre-trained third-party inference API demonstrates no ML engineering skills for a portfolio. However, it is a legitimate external benchmark. **Step 48 evaluated GNNePCSAFT on the fluorinated validation set**, finding it comparable to RF: epsilon/k MAE 16.6 K (vs RF 14.3 K), m MAE 0.838 (vs RF 0.852). On the Esper holdout, GNNePCSAFT is substantially better (R² = 0.84 vs 0.33), but this comparison is confounded by train-set overlap — GNNePCSAFT was trained on Esper. The fluorinated set, which is genuinely external for both models, is the fair comparison, and there they are comparable. For non-associating molecules the ePC-SAFT and PC-SAFT non-associating parameters (m, sigma, epsilon/k) are identical; only the association term differs.

---

## All Models Evaluated

### On Esper Test Set (n=361, 80/20 split of 1,801 molecules)


| Model                               | m R^2 | sigma R^2 | epsilon/k R^2 | m MAE | sigma MAE | epsilon/k MAE (K) |
| ----------------------------------- | ----- | --------- | ------------- | ----- | --------- | ----------------- |
| GC-PC-SAFT                          | 0.40  | -1.16     | -0.04         | 1.00  | 0.49      | 44.6              |
| RF                                  | 0.62  | 0.35      | 0.33          | 0.59  | 0.19      | 26.8              |
| NN (PyTorch)                        | 0.47  | 0.11      | 0.14          | 0.77  | 0.24      | 32.9              |
| ChemBERTa                           | 0.53  | 0.25      | 0.27          | 0.77  | 0.23      | 31.2              |
| XGBoost                             | 0.64  | 0.36      | 0.33          | 0.61  | 0.20      | 27.0              |
| Chemprop D-MPNN                     | 0.54  | 0.33      | 0.39          | 0.69  | 0.21      | 26.8              |
| GINEConv (Combined)                 | 0.69  | 0.34      | 0.41          | 0.76  | 0.23      | 28.2              |
| SVR (RBF kernel) ‡                 | 0.59  | 0.25      | 0.33          | 0.73  | 0.22      | 29.9              |
| Ridge Regression ‡                 | 0.52  | -0.00     | -0.43         | 0.73  | 0.24      | 32.6              |
| GNNePCSAFT (external, pre-trained) †| 0.94  | 0.78      | 0.84          | 0.14  | 0.06      | 6.0               |

† GNNePCSAFT **was** trained on the Esper corpus (it is one of its training sources). Its Esper holdout metrics are inflated by train-set overlap and should not be compared directly against the other models' holdout results. The fluorinated external validation (below) is the fair head-to-head comparison.

‡ SVR and Ridge metrics are mean values across 10 repeated stratified outer splits (seeds 42–51). All other models use a single 80/20 split. See `docs/reports/47_svm_benchmark.md` for interval estimates.


### External Fluorinated Validation (15 refrigerants with published PC-SAFT params)

Step 48 evaluated chemprop and GNN on the fluorinated validation set alongside RF. Step 51 resolved the XGBoost feature-dimension mismatch (2216 vs 2218 features due to different `clean_descriptors` outputs) and added SVR.

| Model                                  | epsilon/k MAE (K) | Boiling Point MAE (K) | epsilon/k R^2 |
| -------------------------------------- | ----------------- | --------------------- | ------------- |
| RF (Esper, 1,801 mol)                  | 14.3              | 8.2                   | -0.47         |
| XGBoost (Esper, 1,801 mol)             | 15.4              | 13.4                  | -0.44         |
| GNNePCSAFT (external, pre-trained) †  | 16.6              | ---                   | -0.82         |
| Chemprop D-MPNN                        | 19.6              | 19.3                  | -1.22         |
| GINEConv (Esper)                       | 17.4              | 23.9                  | -0.84         |
| SVR (Esper, 1,801 mol)                 | 26.9              | 54.5                  | -4.01         |
| GNN (unified, 13,764 mol)              | 77.6              | 133.7                 | -25.08        |

† GNNePCSAFT (v0.3.1, PNA checkpoint from HuggingFace `wildsonbbl/gnnepcsaft`) predicts ePC-SAFT parameters. For these non-associating fluorinated compounds the m/sigma/epsilon_k values are directly comparable to PC-SAFT. GNNePCSAFT was trained on the same Esper dataset, making this fluorinated set a genuine external test for both models. Full parameter MAEs: m 0.838, sigma 0.081, epsilon/k 16.6 K. BP MAE not computed (requires piping ePC-SAFT parameters through the EOS solver). Both RF and GNNePCSAFT share the same systematic failure mode: m over-predicted by 30-60% for small fluorinated molecules.

**Tier 1 verdict (Step 48, confirmed Step 51):** RF wins decisively with 8.2 K BP MAE. XGBoost is the runner-up at 13.4 K (gap: 5.2 K, exceeds 5 K threshold). SVR (54.5 K BP MAE) demonstrates that the Esper R²=0.33 tie among RF, XGBoost, and SVR does **not** extend to the fluorinated deployment domain — SVR's kernel-based generalization fails catastrophically on out-of-distribution fluorinated chemistry.


### On Unified Dataset (Esper + ML-SAFT + SPT-PCSAFT)


| Model                   | m R^2 | sigma R^2 | epsilon/k R^2 |
| ----------------------- | ----- | --------- | ------------- |
| GNN (unified)           | 0.79  | 0.77      | 0.73          |
| GNN (Esper subset only) | 0.59  | 0.03      | 0.36          |
| GNN (SPT subset only)   | 0.90  | 0.90      | 0.90          |


---

## Validation Hierarchy

Model selection uses a three-tier validation hierarchy, ordered by relevance to the deployment task (screening fluorinated HFO blowing agents):

1. **Tier 1 — Fluorinated external validation (15 refrigerants)**: The decisive metric. Tests the full pipeline (SMILES → PC-SAFT → boiling point via EOS). Boiling point MAE is primary because the screening filter window is ~35 K wide. A model with >50 K BP MAE is not useful for screening.
2. **Tier 2 — Esper holdout (361 molecules, 80/20 stratified split)**: General parameter prediction accuracy with bootstrap 95% CIs. Without CIs, the epsilon/k MAE values for RF (26.8 K), XGBoost (27.0 K), and chemprop (26.8 K) are too close to distinguish.
3. **Tier 3 — Cross-validated Q^2 (5-fold)**: Robustness check against split sensitivity. Confirms single-split results are not artifacts.

Tier 1 takes precedence. If Tier 1 is tied, Tier 2 paired uncertainty analysis breaks the tie. If still tied, practical considerations (inference speed, uncertainty calibration, deployment simplicity) decide.

> **Scope note:** The validation evidence summarized here is based on the project's current random split and shuffled CV protocols. That supports internal reproducibility and disciplined model selection, but it should not be presented as proof of scaffold-level chemistry generalization without an additional chemistry-aware split.

> **Note:** Step 48 applied the full three-tier framework to RF, chemprop, GNN, and GNNePCSAFT. Tier 1 was decisive: RF won by 11.2 K BP MAE over the runner-up. GNNePCSAFT was comparable to RF on fluorinated parameter accuracy (ε/k MAE 16.6 vs 14.3 K).

---

## Why RF Was Chosen for Production (Confirmed, Step 48)

Step 48 evaluated RF, chemprop, and GNN on the fluorinated validation set with paired bootstrap comparisons, confirming the provisional Step 38d decision.

1. **Best on the deployment domain (Tier 1 decisive)**: RF boiling point MAE = 8.2 K on 15 fluorinated refrigerants; chemprop was 2.4x worse at 19.3 K; GNN at 23.9 K. The paired bootstrap 95% CI for the RF-vs-chemprop gap excludes zero, and the 11.2 K gap exceeds the predeclared 5 K decisiveness threshold.
2. **Best fluorinated epsilon/k accuracy**: RF epsilon/k MAE = 14.3 K, vs chemprop 19.6 K and GNN 17.4 K.
3. **Robust on general data (Tier 3)**: 5-fold Q^2 on epsilon/k: RF 0.378 ± 0.064, XGBoost 0.337 ± 0.057. RF edges out XGBoost on cross-validated robustness.
4. **Data quality > quantity > architecture**: On the same ~1,900 Esper molecules, architectural differences between RF, XGBoost, chemprop, and GNN produced ±0.08 R^2 differences. With 7x more training data, GNN gained +0.40 R^2 on epsilon/k — but the larger dataset carried systematic bias.
5. **Built-in uncertainty**: 100-tree ensemble provides per-prediction tree-variance uncertainty (though it overestimates actual error by ~6x for well-represented chemistry and must be calibrated locally).
6. **Fast inference**: 182 mol/s (RF) vs 3,894 mol/s (GNN). RF is slower than GNN on raw throughput but faster than chemprop and ChemBERTa, and adequate for screening workloads.
7. **Interpretable features**: RDKit descriptors provide explainable feature importances.
8. **Convergence validated (Step 49)**: OOB R^2 plateaus by n_estimators=100 for all three targets, confirming the production ensemble size is sufficient.

---

## Where Other Models Might Be Valid

- **GNN (GINEConv, chemprop)**: Both consistently outperformed RF on epsilon/k (+0.06-0.08 R^2) on the same data. With a larger, clean experimental dataset (>5,000-10,000 molecules), GNNs should surpass RF. The crossover is data-limited, not architecture-limited.
- **XGBoost**: Matched RF exactly (0.33 vs 0.33 on epsilon/k) on the current holdout, which is consistent with but does not prove that boosting adds little over bagging on these features. Could be preferred if feature interaction effects become important with richer data.
- **ChemBERTa**: Lower accuracy but requires no feature engineering. With pre-training on larger chemical corpora and more fine-tuning data, SMILES-based models could become competitive.
- **SVR (RBF kernel)**: Matched RF on epsilon/k (R^2 = 0.33 across 10 splits), confirming the representation bottleneck. Could be preferred if feature engineering produces a lower-dimensional, denser feature set where kernel methods excel. Currently offers no advantage over RF on these features.
- **Ridge Regression**: Catastrophically unstable on epsilon/k (R^2 = -0.43 ± 1.55) due to collinearity in 2,200 features. Confirms that non-linear models provide substantial value. Not a deployment candidate.
- **GC-PC-SAFT**: Zero data needed. Useful as a baseline or for novel functional groups with no training data at all.
- **Log-target transform (Step 52)**: Training RF and XGBoost on log-transformed targets (making MSE approximate mean squared relative error) was tested as a cheap proxy for relative-error loss functions. Result: **rejected**. On the fluorinated validation set, RF log-corrected epsilon/k MAE = 13.39 K vs RF raw 14.24 K (bootstrap 95% CI [-2.78, +0.64] spans zero). XGBoost log showed a significant fluorinated improvement (15.90 → 13.98 K), but this does not change the RF production decision. Repeated CV (5×3) confirmed: m benefits slightly from log-transform (p=0.017 for RF R²), but epsilon/k and sigma show no significant improvement. The result narrows the hypothesis space: MSE-on-raw-targets is not a major contributor to the RF epsilon/k accuracy gap.

---

## Training Convergence and Hyperparameter Validation

Step 49 instruments all project-trained models with convergence diagnostics. The key insight is that "converged" means different things for different model families.

| Model | Diagnostic Type | Status | Key Finding |
|-------|----------------|--------|-------------|
| RF (production) | OOB R² vs n_estimators | Validated | OOB R² plateaus by n=100 for all targets (m: 0.671, sigma: 0.404, epsilon_k: 0.447) |
| NN (MLP) | Train/val loss curves | Overfitting | Val loss flatlines from epoch ~3; best epoch 14/34; large train-val gap |
| ChemBERTa | Train/eval loss curves | Mild overfitting | Best eval loss at epoch 8/13; slight increases thereafter |
| XGBoost | CV heatmap + boosting curves | Deferred | CV results and boosting history not saved; requires retraining with instrumented code |
| GNN (GINEConv) | Train/val loss curves | Deferred | History artifacts not saved; requires retraining with instrumented train_gnn.py |
| chemprop D-MPNN | Train/val loss curves | Deferred | Lightning logger was disabled; requires retraining with CSVLogger enabled |
| SVR | CV heatmap | Partial | cv_results.json saved but insufficient grid points for full heatmap visualization |

See `figures/49_convergence_diagnostics/` for diagnostic plots and `docs/reports/49_convergence_diagnostics.md` for the full assessment.

---

## Unified Model Comparison (Step 51)

This table consolidates all model results across Esper internal validation and fluorinated external validation into a single reference.

| Model | Esper ε/k R² | Esper ε/k MAE (K) | Fluor ε/k MAE (K) | Fluor ε/k R² | Fluor BP MAE (K) | EOS Conv. | Protocol |
| ----- | ------------ | ----------------- | ----------------- | ------------ | --------------- | --------- | -------- |
| GC-PC-SAFT | -0.04 | 44.6 | --- | --- | --- | --- | Deterministic |
| Ridge | -0.43 | 32.6 | --- | --- | --- | --- | 10 outer splits (mean) |
| NN (MLP) | 0.14 | 32.9 | --- | --- | --- | --- | Single split (seed=42) |
| ChemBERTa | 0.27 | 31.2 | --- | --- | --- | --- | Single split (seed=42) |
| SVR | 0.33 | 29.9 | 26.9 | -4.01 | 54.5 | 6/15 | 10 outer splits (mean) |
| RF | 0.33 | 26.8 | 14.3 | -0.47 | 8.2 | 6/15 | Single split (seed=42) |
| XGBoost | 0.33 | 27.0 | 15.4 | -0.44 | 13.4 | 6/15 | Single split (seed=42) |
| chemprop (D-MPNN) | 0.39 | 26.8 | 19.6 | -1.22 | 19.3 | 6/15 | Single split (seed=42) |
| GNN (GINEConv) | 0.41 | 28.2 | 17.4 | -0.84 | 23.9 | 4/15 | Single split (seed=42) |
| GNNePCSAFT † | 0.84 † | 6.0 † | 16.6 | -0.82 | --- | --- | External pre-trained |

**Notes:**
- SVR and Ridge Esper metrics are means across 10 repeated stratified outer splits; all others use a single 80/20 split (seed=42).
- GC-PC-SAFT, NN (MLP), ChemBERTa, and Ridge were not evaluated on the fluorinated set because they are not deployment candidates. GC-PC-SAFT is a deterministic group-contribution method; NN and ChemBERTa underperformed RF significantly; Ridge was catastrophically unstable.
- Negative R² on the fluorinated set is expected: 15 compounds with a narrow ε/k range makes the variance denominator small, so even modest MAE produces negative R².
- Fluorinated BP MAE is the decisive Tier 1 metric for model selection (see Step 48).
- † GNNePCSAFT Esper metrics are inflated by train-set overlap (GNNePCSAFT was trained on the Esper corpus). The fluorinated columns are the fair comparison: on that genuinely external test, GNNePCSAFT (ε/k MAE 16.6 K) is comparable to RF (14.3 K). BP MAE not computed for GNNePCSAFT.

See `figures/51_unified_model_comparison/` for consolidated comparison figures.
