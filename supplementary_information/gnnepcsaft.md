# GNNePCSAFT External Benchmark

GNNePCSAFT ([PyPI](https://pypi.org/project/gnnepcsaft/), [HuggingFace](https://huggingface.co/wildsonbbl/gnnepcsaft), [GitHub](https://github.com/wildsonbbl/gnnepcsaft)) is a published, open-source GNN model built specifically to predict ePC-SAFT parameters from molecular structure. It was benchmarked against this project's models in Step 48 to answer: *"How does our approach compare to the published state-of-the-art model built for exactly this problem?"*

## Model Details

- **Architecture**: PNA (Principal Neighbourhood Aggregation) with 6 message-passing layers, hidden dim 256, sum pooling, and OGB atom/bond encoders
- **Training data**: Esper dataset (~1,801 molecules), confirmed from checkpoint hyperparameters (`dataset: esper`)
- **Training compute**: 1,400,000 optimization steps at batch size 512 (~125-400x more than this project's GNN)
- **Loss function**: Huber loss on relative percentage error (delta=0.01), not MSE on raw targets
- **Validation**: Physics-informed — checkpoint selected by density and vapor pressure MAPE through FeOs EOS, not parameter-level metrics
- **Output**: ePC-SAFT parameters (m, sigma, epsilon/k, plus association terms kappa_AB, epsilon_AB). For non-associating molecules, the m/sigma/epsilon_k values are identical to standard PC-SAFT
- **Checkpoint**: `model-hlrn7lqv` (msigmae), `assoc_model-j7isfrga` (association)
- **Version evaluated**: 0.3.1 (PyPI)

## Why GNNePCSAFT Outperforms Our GNN on Esper

Both models were trained on the same Esper dataset, but GNNePCSAFT achieves substantially better general accuracy (Esper holdout R² = 0.84 vs our GINEConv R² = 0.41). The key differences:

| | This project's GNN (GINEConv) | GNNePCSAFT (PNA) |
|---|---|---|
| Convolution | GINEConv (single sum aggregation) | PNA (mean + min + max + std, 3 scalers) |
| Depth | 3-4 layers | 6 layers |
| Graph encoding | Linear projection of 28 hand-crafted atom features | OGB learned categorical embeddings (9 atom properties) |
| Training steps | ~3,600-11,200 | 1,400,000 |
| Loss | MSE on z-score normalized targets | Huber on relative percentage error |
| weight_decay | 1e-5 to 1e-4 | 0.01 (100x stronger) |
| Validation signal | Parameter R²/MAE | Density MAPE + vapor pressure MAPE via EOS |
| Output bounds | None | Clipped to physical ranges |

The most impactful factors are likely the training duration (125-400x more steps), the relative loss function (scale-invariant across targets), and the more expressive PNA convolution.

## Benchmark Results

### Fluorinated External Validation (15 compounds)

This is the fair head-to-head comparison — both models are genuinely out-of-distribution on fluorinated refrigerants.

| Model | m MAE | sigma MAE | epsilon/k MAE (K) | epsilon/k R² |
|---|---|---|---|---|
| RF | 0.852 | 0.072 | **14.3** | -0.47 |
| GNNePCSAFT | **0.838** | 0.081 | 16.6 | -0.82 |

RF is slightly better on epsilon/k and sigma; GNNePCSAFT is slightly better on m. Both models share the same systematic failure: m is over-predicted by 30-60% for small fluorinated molecules, pointing to a training-data gap rather than an architecture limitation.

### Esper Holdout (388 molecules)

**Caveat**: GNNePCSAFT was trained on the Esper corpus, so these results are inflated by train-set overlap. They should not be compared directly against other models' holdout results.

| Parameter | RF R² | GNNePCSAFT R² | RF MAE | GNNePCSAFT MAE |
|---|---|---|---|---|
| m | 0.914 | 0.944 | 0.320 | 0.135 |
| sigma | 0.690 | 0.778 | 0.111 | 0.056 |
| epsilon/k | 0.695 | 0.839 | 15.3 K | 6.0 K |

### Candidate Molecule Predictions

GNNePCSAFT was also run on the chlorobutene near-hit candidates and their structural analogs to provide independent corroboration of the screening results. Full three-parameter predictions:

| Molecule | GNNePCSAFT m | GNNePCSAFT σ (Å) | GNNePCSAFT ε/k (K) | RF ε/k (K) | Esper lit ε/k (K) |
|---|---|---|---|---|---|
| 1-chlorobut-1-ene | 2.632 | 3.655 | 263.9 | 267.8 | — |
| (Z)-2-chloro-2-butene | 2.613 | 3.693 | 260.1 | 267.1 | — |
| (E)-2-chloro-2-butene | 2.575 | 3.737 | 262.6 | 267.1 | — |
| cyclopentane (reference) | 2.273 | 3.744 | 271.8 | 281.3 | 288.7 |
| 1-chlorobutane (analog) | 2.820 | 3.638 | 258.6 | 259.6 | 256.9 |
| 2-chlorobutane (analog) | 2.593 | 3.766 | 261.6 | — | — |
| 2-chloropropene (analog) | 1.925 | 3.830 | 275.4 | 266.1 | 273.2 |

GNNePCSAFT predicts the chlorobutenes 5-7 K lower than RF (260-264 K vs 267-268 K). Both models agree that the chlorobutenes are near cyclopentane in epsilon/k. GNNePCSAFT's predictions for the structural analogs with known literature values are accurate: 1-chlorobutane error +0.7%, 2-chloropropene error +0.8%. RF's analog predictions are also reasonable: 1-chlorobutane error +1.1%, 2-chloropropene error -2.6%, cyclopentane error -2.6%. (Note: all three analogs are in the RF training set, so these errors reflect in-distribution performance.)

GNNePCSAFT under-predicts cyclopentane by 17 K (271.8 vs 288.7 K); RF under-predicts by 7.5 K (281.3 vs 288.7 K). Despite this, the cross-interaction deficit between the chlorobutenes and cyclopentane under GNNePCSAFT's predictions is only ~2-4%, which still places them in the near-hit regime. The near-hit designation is robust across both models.

## Implications for This Project

**The project's contribution is not a better point-prediction model.** GNNePCSAFT is a purpose-built, carefully engineered model for this exact task, and it achieves comparable (fluorinated) to substantially better (general Esper) accuracy. The project's contribution is:

1. **Validation methodology**: The tiered validation framework (external fluorinated > Esper holdout > CV) revealed that in-distribution R² is misleading for deployment decisions. GNNePCSAFT's Esper R² = 0.84 does not translate to better fluorinated accuracy than RF's Esper R² = 0.33.
2. **End-to-end screening pipeline**: GNNePCSAFT gives parameters; this project builds the full workflow from SMILES through EOS to Henry's constant ranking, with locally calibrated uncertainty.
3. **Independent corroboration**: Two models built independently with different architectures converge to similar predictions on the candidate molecules, strengthening the near-hit designation.
4. **Honest confrontation with limitations**: Both models share the same systematic failure mode on small fluorinated molecules (m over-prediction), pointing to a data gap that no architecture can fix without more fluorinated training data.

## Dependency Note

GNNePCSAFT carries a heavy transitive dependency chain: PyTorch, PyTorch Geometric, OGB, Lightning, FeOs, xgboost, ray, absl-py, ml-collections. It was installed temporarily for benchmarking and should not be a permanent project dependency.
