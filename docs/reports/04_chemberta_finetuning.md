# Results Report: ChemBERTa Fine-Tuning (Esper Dataset, 1,801 molecules)

## Model Performance

ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`, ~85M parameters) was fine-tuned for PC-SAFT parameter regression on the Esper dataset. The model reads raw SMILES strings through a pretrained RoBERTa tokenizer, extracts a [CLS] representation from the pretrained encoder (768 hidden dim), and passes it through a regression head (768 -> 256 -> ReLU -> Dropout(0.1) -> 64 -> ReLU -> 3) predicting normalized m, sigma, and epsilon_k jointly.

Training used HuggingFace Trainer with AdamW (lr=2e-5, weight_decay=0.01, warmup_ratio=0.1), batch size 32, and early stopping (patience=5 epochs on eval loss). The model early-stopped at epoch 8 (best eval loss) out of 50 maximum epochs, with total CPU training time of ~2.5 minutes. Target normalization (zero mean, unit variance) was applied during training and reversed at inference.

| Parameter | MAE | RMSE | R2 (test) | Train samples | Test samples |
|-----------|-----|------|-----------|---------------|--------------|
| m (segments) | 0.766 | 1.283 | 0.53 | 1,440 | 361 |
| sigma (A) | 0.226 | 0.349 | 0.25 | 1,440 | 361 |
| epsilon_k (K) | 31.2 | 53.7 | 0.27 | 1,440 | 361 |

ChemBERTa is the clear second-best model behind RF on all three parameters. On m, ChemBERTa (R2=0.53) substantially outperforms the NN (R2=0.47) and approaches RF (R2=0.62). On sigma and epsilon_k, ChemBERTa (R2=0.25, 0.27) significantly outperforms the NN (R2=0.11, 0.14) and narrows the gap with RF (R2=0.35, 0.33). The improvement over the NN is notable given that ChemBERTa consumes raw SMILES without any feature engineering, while the NN relies on 2,218 hand-crafted Morgan + RDKit features.

## Comparison to Baseline (Four-Way)

### R2 Comparison

| Method | R2 (m) | R2 (sigma) | R2 (epsilon_k) |
|--------|--------|------------|-----------------|
| GC-PC-SAFT | 0.40 | -1.16 | -0.04 |
| RF (Combined features) | **0.62** | **0.35** | **0.33** |
| NN (PCSAFTNet) | 0.47 | 0.11 | 0.14 |
| ChemBERTa | 0.53 | 0.25 | 0.27 |

### MAE Comparison

| Method | MAE (m) | MAE (sigma) | MAE (epsilon_k) |
|--------|---------|-------------|-----------------|
| GC-PC-SAFT | 1.001 | 0.494 | 44.6 |
| RF (Combined features) | **0.585** | **0.189** | **26.8** |
| NN (PCSAFTNet) | 0.767 | 0.238 | 32.9 |
| ChemBERTa | 0.766 | 0.226 | 31.2 |

### Model Ranking

1. **RF** -- Best on all three targets. Benefits from implicit feature selection via random splits in the high-dimensional (2,218-feature) space.
2. **ChemBERTa** -- Strong second. Pretrained chemical language knowledge transfers well even with 1,440 fine-tuning samples.
3. **NN (PCSAFTNet)** -- Third. The multi-task architecture struggles to exploit the high-dimensional feature space with limited data.
4. **GC-PC-SAFT** -- Baseline. Fails on sigma (R2<0) and epsilon_k (R2~0) due to non-additive group interactions.

### Applicability Domain Analysis

ChemBERTa uses a tokenizer-based SMILES representation, so its "feature space" is fundamentally different from the Morgan FP space used for the Isolation Forest AD model. The AD labels from the fingerprint-based model were applied to ChemBERTa for comparison purposes, but their validity for this architecture is limited.

| Model | Parameter | R2 (in-domain) | R2 (OOD) | Fraction OOD |
|-------|-----------|----------------|----------|--------------|
| ChemBERTa | m | 0.534 | 0.351 | 8.0% |
| ChemBERTa | sigma | 0.253 | 0.110 | 8.0% |
| ChemBERTa | epsilon_k | 0.303 | 0.015 | 8.0% |

ChemBERTa shows the expected pattern: in-domain R2 is higher than OOD R2, suggesting that even a fingerprint-based AD check has some cross-architecture validity. However, a proper ChemBERTa AD model would operate in the CLS embedding space rather than Morgan FP space.

## Key Findings

- **Transfer learning pays off even with 1,440 samples.** ChemBERTa's pretrained representations from 77M SMILES provide a meaningful boost over the from-scratch NN. The model reads raw SMILES strings without any feature engineering and achieves R2=0.53 on m (vs 0.47 for the feature-engineered NN).
- **RF remains the champion in the small-data regime.** The RF's ensemble averaging and implicit feature selection via random subspace sampling give it a decisive edge with ~1,440 training samples and 2,218 features. This is expected: tree-based methods are hard to beat in low-data, high-dimensional settings.
- **ChemBERTa converges fast on CPU.** Training early-stopped at epoch 8 in ~2.5 minutes on an M3 MacBook CPU. The rapid convergence is partly due to the pretrained weights providing a strong initialization. The training loss decreased from ~1.0 to ~0.3, while validation loss reached its minimum at 0.71 (normalized MSE).
- **MC Dropout uncertainty is available but narrower than RF.** ChemBERTa's MC Dropout produces mean predicted std of m=0.27, sigma=0.05, epsilon_k=7.5 K. These are systematically lower than RF's tree-disagreement uncertainty (m=0.81, sigma=0.24, epsilon_k=29.6 K), consistent with the single-dropout-layer architecture providing limited epistemic uncertainty estimation.
- **ChemBERTa's AD limitations are real.** The Morgan FP-based AD model does not directly apply to ChemBERTa's tokenized SMILES representations. For production deployment, a ChemBERTa-specific AD check using the CLS embedding space would be needed.

## Expanded Screening Results (Section 4A)

The screening pipeline was expanded with additional scaffold families beyond HFOs:
- **HCFOs** (hydrochlorofluoroolefins): 10 Cl-substituted propene/butene scaffolds
- **HFEs** (hydrofluoroethers): 23 ether-linked fluorinated C1-C4 scaffolds
- **Unsaturated hydrocarbons**: 25 C2-C5 alkenes and cycloalkenes
- **Cyclic fluorinated**: 18 C3-C6 fluorinated ring scaffolds

With F+Cl halogen variant generation and stereoisomer enumeration, the broad scaffold set produced 805 unique candidates, of which 645 passed the SA Score filter (threshold 4.5).

### Screening Pipeline Summary

| Stage | Candidates In | Candidates Out |
|-------|---------------|----------------|
| 1. Candidate generation (broad) | -- | 805 |
| 2. SA Score <= 4.5 | 805 | 645 |
| 3. Patent check | skipped | -- |
| 4. PC-SAFT ranking (RF model) | 645 | 645 (ranked) |

### Top 10 Candidates (Broad Screening)

Cyclopentane reference: m=2.37, sigma=3.71 A, epsilon_k=288.8 K

| Rank | SMILES | SA Score | m | sigma (A) | epsilon_k (K) | Distance |
|------|--------|----------|---|-----------|---------------|----------|
| 1 | ClC1=CC1 | 2.88 | 2.33 | 3.55 | 299.6 | 0.079 |
| 2 | C1=CCCC1 | 2.73 | 2.31 | 3.66 | 276.0 | 0.082 |
| 3 | ClC1C=C1 | 3.33 | 2.37 | 3.43 | 300.8 | 0.105 |
| 4 | Cl[C@@H]1C=CC1 | 3.38 | 2.57 | 3.59 | 299.0 | 0.111 |
| 5 | Cl[C@H]1C=CC1 | 3.38 | 2.57 | 3.59 | 299.0 | 0.111 |
| 6 | FC1(Cl)CC1 | 3.31 | 2.62 | 3.53 | 290.3 | 0.117 |
| 7 | ClC1=CCC1 | 2.61 | 2.55 | 3.66 | 303.6 | 0.119 |
| 8 | C1=CCC1 | 2.06 | 2.20 | 3.59 | 273.1 | 0.121 |
| 9 | C1=CCCCC1 | 2.42 | 2.65 | 3.81 | 289.8 | 0.124 |
| 10 | F[C@H]1C[C@@H](F)C1 | 3.30 | 2.58 | 3.55 | 275.9 | 0.127 |

The expanded screening reveals that chlorinated cyclopropenes (ClC1=CC1, distance=0.079) and cyclopentene (C1=CCCC1, distance=0.082) are the closest to cyclopentane in PC-SAFT parameter space. This is a broader and more diverse candidate set than the original HFO-only screening.

## Figures

See `figures/04_chemberta/` for:

- `chemberta_parity_plots.png` -- Parity plots (predicted vs true) for ChemBERTa on m, sigma, epsilon_k
- `four_way_r2_comparison.png` -- Bar chart comparing R2 across GC-PC-SAFT, RF, NN, and ChemBERTa for all three targets
- `chemberta_loss_curves.png` -- Training and validation loss curves during ChemBERTa fine-tuning, showing early stopping
- `screening_results.png` -- Histogram of candidate distances to cyclopentane and scatter of predicted m vs epsilon_k

## Deviations

- **Training epochs**: The model early-stopped at epoch 8 out of 50 maximum epochs, which is faster than expected. This is typical for fine-tuning pretrained models on small datasets -- the pretrained weights provide such a strong initialization that the regression head converges quickly.
- **Scaffold count**: The broad scaffold set generated 805 candidates (645 after SA filtering), which meets the 500-2000 target. The F+Cl halogen variant generation was added to all scaffolds in broad mode (not just HCFOs) to boost diversity.

## Readiness Check

1. [x] Four-way comparison table (GC-PC-SAFT vs RF vs NN vs ChemBERTa): completed with MAE, RMSE, R2 for all three targets
2. [x] Can explain trade-offs: RF is fast and best in small-data regime; ChemBERTa leverages pretrained chemical knowledge for strong second place; NN needs richer representations; GC-PC-SAFT is the physics baseline
3. [x] Best model chosen for deployment: RF (R2: m=0.62, sigma=0.35, epsilon_k=0.33), with ChemBERTa as the recommended next-best for transfer-learning scenarios
4. [x] Evaluation harness includes all four models seamlessly via `python -m model.evaluate --models gc_pcsaft rf nn chemberta`
5. [x] Expanded screening results generated: 645 candidates ranked from broad scaffold set
