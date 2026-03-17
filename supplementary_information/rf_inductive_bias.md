# Random Forest Inductive Bias for Low-Data Chemical Property Prediction

This document explains why Random Forest (RF) was chosen as the baseline model for PC-SAFT parameter prediction and what inductive biases make it effective for small datasets.

---

## Why Random Forest?

The project uses a 100-tree ensemble trained on a ~2,200-dimensional feature space: 170 cleaned RDKit descriptors plus 2,048-bit Morgan fingerprints. In the low-data regime (~1,800 molecules), hand-crafted features encoding chemical knowledge outperform learned representations.

RF provides a strong inductive bias: the model does not need to discover what "electronegativity" or "molecular weight" means from examples—it receives these as explicit features. The RDKit descriptor set encodes known chemical structure–property relationships (e.g., topological indices, atom counts, functional group indicators). Morgan fingerprints capture substructural patterns that correlate with thermodynamic behavior. With limited training data, models that must learn these concepts from scratch (e.g., graph neural networks) lack enough examples to generalize reliably. RF leverages domain knowledge already embedded in the feature design.

---

## Inductive Biases That Prevent Overfitting

Several built-in mechanisms make RF resistant to overfitting on small datasets:

1. **Bagging (Bootstrap Aggregation)**  
   Each tree is trained on a bootstrap sample of the data (~63% of unique samples per tree). Averaging across trees reduces variance without increasing bias. Trees that overfit to noise in their bootstrap sample are downweighted when combined with the ensemble.

2. **Feature Subsampling**  
   At each split, only sqrt(n_features) candidates are considered. This decorrelates trees and prevents any single feature from dominating all splits. It also acts as a form of regularization in high-dimensional spaces.

3. **Ensemble Averaging**  
   100 trees produce a smooth prediction surface. Individual tree noise cancels out, and the final prediction is robust to outliers in any single tree's training set.

4. **No Extrapolation**  
   RF predictions are bounded by the training data range. Predictions are averages of training labels in leaf nodes, so the model cannot extrapolate beyond observed values. This is conservative but prevents wild predictions on out-of-distribution molecules.

5. **Built-in Uncertainty**  
   Tree-level variance across the ensemble provides a natural uncertainty estimate. In practice, this tends to overestimate actual error by roughly 6x for well-represented chemistry and must be calibrated locally for deployment.

---

## RF vs Other Models on This Dataset

| Model | R² (ε/k) on Esper Test | Notes |
|-------|------------------------|-------|
| Random Forest | 0.33 | Baseline with RDKit + Morgan features |
| XGBoost | 0.33 | Gradient boosting adds no additional signal over RF on these features |
| GINEConv GNN | 0.41 | Better on ε/k; 142 K boiling point MAE on external fluorinated validation |

Key insight: RDKit descriptors provide strong inductive bias at low data volumes. On ~1,800 training molecules, RF and XGBoost perform equivalently, suggesting the feature representation dominates model choice. The GNN improves on ε/k but suffers on external fluorinated validation, indicating sensitivity to distribution shift when training data is limited.

---

## When RF Would Lose

RF's advantages diminish as data volume grows and as the problem shifts toward learning structure–property mappings that are not well captured by hand-crafted descriptors.

- **Scale crossover**: With >5,000–10,000 training molecules, GNN message-passing can learn features that outperform hand-crafted ones. On ~1,900 molecules, RF and GNN differ by about ±0.08 R²; with roughly 7x more data, the GNN gained ~+0.40 R² on ε/k.

- **Data quality matters**: The larger dataset that favored the GNN carried systematic bias, which made the model about 16x worse on the deployment domain. Model choice alone is insufficient; data curation and domain alignment are critical.

**Conclusion**: RF is well-suited for the current low-data regime and explicit chemical features. As more high-quality, domain-representative data becomes available, learned representations (e.g., GNNs) may overtake RF. For now, RF's inductive biases and conservative extrapolation behavior make it a reliable baseline.
