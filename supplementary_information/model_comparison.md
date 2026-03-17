# Model Comparison: Full Results and Production Rationale

This document provides the complete model comparison results across all evaluation sets and explains the rationale for selecting Random Forest (RF) for production deployment.

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

On this dataset, however, the two tied: R^2 = 0.33 vs 0.33 on epsilon/k, 0.64 vs 0.62 on m, 0.36 vs 0.35 on sigma. **That tie is informative.** When both bagging and boosting converge to the same accuracy on identical features, the bottleneck is the feature representation, not the tree optimization strategy. Boosting's advantage — capturing residual patterns that bagging averages out — requires residual structure *in the features* to exploit. When the features are already saturated (the signal they encode has been fully extracted), switching from bagging to boosting adds no new information.

XGBoost would become the preferred tree ensemble if richer features (e.g., 3D conformer descriptors) or substantially more training data introduced interaction effects that boosting could exploit. For this project, the RF = XGBoost tie was one of the signals that motivated testing learned representations (GNNs) rather than further tuning tree hyperparameters.

### GNNs as learned-representation challenger

GNNs bypass the fixed-feature representation entirely: they operate directly on the molecular graph (atoms as nodes, bonds as edges) and learn task-specific representations via message-passing. This is the natural "next rung" on the representation ladder for two reasons:

1. **Physics-matching inductive bias**: GNN message-passing mirrors PC-SAFT's own physical model. PC-SAFT treats molecules as chains of spherical segments whose dispersion energy (epsilon/k) depends on local chemical environment. After k GNN layers, each node encodes a k-hop neighborhood — exactly the local context that determines segment contributions. This makes GNNs a data-driven generalization of group-contribution methods, capable of learning non-additive interactions between functional groups ([architecture details](gnn_architectures.md)).
2. **Standard experimental design**: Comparing a feature-engineered baseline (RF on RDKit + Morgan) against an end-to-end learned representation (GNN on molecular graphs) is the standard experimental design in ML for molecular properties. It directly tests whether hand-crafted chemical knowledge or data-driven representation learning is more effective at the current data scale.

Two GNN architectures were tested: GINEConv (Xu et al. 2019; Hu et al. 2020) for atom-level message-passing, and chemprop D-MPNN (Yang et al. 2019) for bond-level directed message-passing. Both consistently outperformed RF on epsilon/k (+0.06–0.08 R^2 on the same Esper data), confirming that graph representations capture structure–property relationships that fingerprints miss — but neither survived external fluorinated validation.

### What was excluded and why

- **Linear models** (ridge, LASSO): structure–property relationships for PC-SAFT parameters are non-linear (e.g., ring formation, branching, and halogenation have non-additive effects on dispersion energy). Linear models cannot capture these interactions without extensive manual feature crossing.
- **Plain MLP**: A PyTorch multi-layer perceptron was tested on the same RDKit + Morgan features. It achieved R^2 = 0.14 on epsilon/k — substantially worse than RF (0.33) — likely due to overfitting in the low-data regime without the variance-reduction benefits of bagging.
- **SVMs**: Not tested, but would operate on the same feature representation as RF/XGBoost and face the same representation bottleneck. SVMs are also less natural for multi-output regression and provide no built-in uncertainty estimate.
- **ChemBERTa** (SMILES transformer): Tested but underperformed feature-engineered models (R^2 = 0.27 on epsilon/k, 185x slower inference). With ~1,800 fine-tuning examples, the pretrained language model could not match hand-crafted chemical features ([details](smiles_chemberta_issues.md)).

---

## All Models Evaluated

### On Esper Test Set (n=361, 80/20 split of 1,801 molecules)


| Model                | m R^2 | sigma R^2 | epsilon/k R^2 | m MAE | sigma MAE | epsilon/k MAE (K) |
| -------------------- | ----- | --------- | ------------- | ----- | --------- | ----------------- |
| GC-PC-SAFT           | 0.40  | -1.16     | -0.04         | 1.00  | 0.49      | 44.6              |
| RF                   | 0.62  | 0.35      | 0.33          | 0.59  | 0.19      | 26.8              |
| NN (PyTorch)         | 0.47  | 0.11      | 0.14          | 0.77  | 0.24      | 32.9              |
| ChemBERTa            | 0.53  | 0.25      | 0.27          | 0.77  | 0.23      | 31.2              |
| XGBoost              | 0.64  | 0.36      | 0.33          | 0.61  | 0.20      | 27.0              |
| Chemprop D-MPNN      | 0.54  | 0.33      | 0.39          | 0.69  | 0.21      | 26.8              |
| GINEConv (Combined)  | 0.69  | 0.34      | 0.41          | 0.76  | 0.23      | 28.2              |


### External Fluorinated Validation (15 refrigerants with published PC-SAFT params)


| Model                     | epsilon/k MAE (K) | Boiling Point MAE (K) | epsilon/k R^2 |
| ------------------------- | ----------------- | --------------------- | ------------- |
| RF (Esper, 1,801 mol)     | 14.3              | 8.2                   | -0.47         |
| GINEConv (Esper)          | 72.4              | 142.3                 | -20.1         |
| GNN (unified, 13,764 mol) | 77.6              | 133.7                 | -25.08        |


### On Unified Dataset (Esper + ML-SAFT + SPT-PCSAFT)


| Model                   | m R^2 | sigma R^2 | epsilon/k R^2 |
| ----------------------- | ----- | --------- | ------------- |
| GNN (unified)           | 0.79  | 0.77      | 0.73          |
| GNN (Esper subset only) | 0.59  | 0.03      | 0.36          |
| GNN (SPT subset only)   | 0.90  | 0.90      | 0.90          |


---

## Why RF Was Chosen for Production

1. **Best on the deployment domain**: RF boiling point MAE = 8.2 K on 15 fluorinated refrigerants; GNN was 16.4x worse at 133.7 K. The entire screening filter window is only 35 K wide.
2. **Data quality > quantity > architecture**: On the same ~1,900 Esper molecules, architectural differences between RF, XGBoost, chemprop, and GNN produced ±0.08 R^2 differences. With 7x more training data, GNN gained +0.40 R^2 on epsilon/k -- but the larger dataset carried systematic bias.
3. **Built-in uncertainty**: 100-tree ensemble provides per-prediction tree-variance uncertainty (though it overestimates actual error by ~6x for well-represented chemistry and must be calibrated locally).
4. **Fast inference**: ~2 seconds for 361 molecules vs 185 seconds for ChemBERTa.
5. **Interpretable features**: RDKit descriptors provide explainable feature importances.

---

## Where Other Models Might Be Valid

- **GNN (GINEConv, chemprop)**: Both consistently outperformed RF on epsilon/k (+0.06-0.08 R^2) on the same data. With a larger, clean experimental dataset (>5,000-10,000 molecules), GNNs should surpass RF. The crossover is data-limited, not architecture-limited.
- **XGBoost**: Matched RF exactly (0.33 vs 0.33 on epsilon/k); gradient boosting added no signal over bagging on these features. Could be preferred if feature interaction effects become important with richer data.
- **ChemBERTa**: Lower accuracy but requires no feature engineering. With pre-training on larger chemical corpora and more fine-tuning data, SMILES-based models could become competitive.
- **GC-PC-SAFT**: Zero data needed. Useful as a baseline or for novel functional groups with no training data at all.

