# Results Report: PyTorch Multi-Task Neural Network (Esper Dataset, 1,801 molecules)

## Model Performance

A multi-task neural network (PCSAFTNet) was trained on the Esper PC-SAFT dataset using combined features (2,048-bit Morgan fingerprints + 170 cleaned RDKit 2D descriptors = 2,218 features). The architecture uses a shared trunk ([2218] -> 512 -> 256 -> 128, BatchNorm + ReLU + Dropout(0.2)) branching into three task-specific heads ([128] -> 64 -> ReLU -> 1) for m, sigma, and epsilon_k. Targets were normalized to zero mean / unit variance. Training used AdamW (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau (patience=10, factor=0.5), early stopping (patience=20), gradient clipping (max_norm=1.0), and batch size 64. The model converged at epoch 14 (early stopped at epoch 34).

### PCSAFTNet (Multi-Task NN)

| Parameter | MAE | RMSE | R2 (test) | Train samples | Test samples |
|-----------|-----|------|-----------|---------------|--------------|
| m (segments) | 0.816 | 1.438 | 0.42 | 1,440 | 361 |
| sigma (A) | 0.244 | 0.395 | 0.04 | 1,440 | 361 |
| epsilon_k (K) | 33.7 | 59.6 | 0.09 | 1,440 | 361 |

The NN underperforms the RF baseline across all three targets. This is a significant and informative result: with only 1,440 training samples and 2,218 features, the small-data regime favors the RF's implicit feature selection and ensemble averaging over the NN's gradient-based optimization. The NN struggles particularly with sigma (R2=0.04) and epsilon_k (R2=0.09), where signal is weak relative to the high-dimensional input.

## Comparison to Baseline

| Method | R2 (m) | R2 (sigma) | R2 (epsilon_k) | MAE (m) | MAE (sigma) | MAE (epsilon_k) |
|--------|--------|------------|-----------------|---------|-------------|-----------------|
| GC-PC-SAFT (domain baseline) | 0.40 | -1.16 | -0.04 | 1.001 | 0.494 | 44.6 |
| RF (RDKit only) | 0.61 | 0.34 | 0.31 | 0.634 | 0.192 | 26.1 |
| RF (Combined) | 0.62 | 0.35 | 0.33 | 0.585 | 0.189 | 26.8 |
| NN (PCSAFTNet) | 0.42 | 0.04 | 0.09 | 0.816 | 0.244 | 33.7 |

### Delta from RF (Combined) to NN

| Parameter | R2 (RF) | R2 (NN) | Delta R2 | MAE (RF) | MAE (NN) | Delta MAE |
|-----------|---------|---------|----------|----------|----------|-----------|
| m | 0.62 | 0.42 | -0.20 | 0.585 | 0.816 | +0.231 |
| sigma | 0.35 | 0.04 | -0.31 | 0.189 | 0.244 | +0.055 |
| epsilon_k | 0.33 | 0.09 | -0.24 | 26.8 | 33.7 | +6.9 |

## Key Findings

- **RF outperforms the NN on all three targets in this small-data regime.** This is not surprising: Random Forests naturally handle high-dimensional sparse features (like 2048-bit Morgan FPs) through split-based feature selection, while neural networks require more data to learn equivalent representations from scratch. With n=1,440 training samples and d=2,218 features, the NN is over-parameterized (~500K parameters vs 1.4K samples).

- **Early stopping triggered at epoch 34 (best=14), indicating overfitting.** The validation loss plateaued quickly while training loss continued to decrease, a classic sign of memorization. The gap between train and val loss suggests the model capacity is too large for the dataset size. Future work could explore stronger regularization (higher dropout, weight decay), dimensionality reduction (PCA on features), or data augmentation.

- **The NN still beats GC-PC-SAFT on m.** Despite its relatively poor performance, the NN predicts m (R2=0.42) better than the group-contribution baseline (R2=0.40), validating that it does learn some structure-property relationships. For sigma and epsilon_k, performance is poor but still positive R2 (unlike GC-PC-SAFT which has negative R2 for both).

- **MC Dropout uncertainty estimates are functional.** The predict_with_uncertainty method produces per-molecule standard deviations (mean std: m=0.298, sigma=0.062, epsilon_k=8.85 K). These are smaller than the RF tree-disagreement uncertainties (m=0.811, sigma=0.238, epsilon_k=29.6 K), which is expected given the NN's lower diversity compared to 100 independent trees.

- **Applicability Domain model trained successfully.** An Isolation Forest (contamination=0.05) was trained on the 1,440-sample training feature matrix and saved. This AD check is model-agnostic -- it operates on the shared feature space and flags out-of-domain molecules for both RF and NN predictions.

- **The architecture is well-positioned for Step 04 (ChemBERTa).** The PCSAFTNet's shared-trunk + task-head design will serve as a blueprint when fine-tuning a pre-trained language model. With ChemBERTa providing richer molecular embeddings (~768-dim contextual representations), the NN approach is expected to improve significantly.

## Figures

See `figures/02_pytorch_nn/` for:
- `nn_parity_plots.png` -- Parity plots for the NN (3 panels: m, sigma, epsilon_k)
- `rf_vs_nn_r2_comparison.png` -- Bar chart comparing R2 between RF and NN by parameter
- `nn_loss_curves.png` -- Training and validation loss curves with best-epoch marker

## Deviations

- Training converged much faster than expected (34 epochs vs max 200), with the best model found at epoch 14. The ReduceLROnPlateau scheduler reduced the learning rate from 1e-3 to 5e-4, and early stopping prevented further overfitting. This rapid convergence is consistent with the small dataset size.

## Readiness Check

1. [x] The NN trains to convergence with stable loss curves (no instability or NaN losses)
2. [x] Side-by-side RF vs NN metrics are available in the comparison table above
3. [x] The model saves and loads correctly: checkpoint contains model_state_dict, model_config, target_scalers, feature_scaler, and feature_config
4. [x] The performance gap is understood: RF's implicit feature selection and ensemble diversity give it an advantage in the small-data, high-dimensional regime; the NN's shared-trunk multi-task architecture needs more data or richer input representations to outperform
5. [x] Both RF and NN have predict_with_uncertainty() methods: RF uses tree disagreement, NN uses MC Dropout (n_forward=30)
6. [x] AD model (Isolation Forest) is trained and saved as ad_model.joblib
