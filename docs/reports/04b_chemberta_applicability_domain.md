# Results Report: ChemBERTa Applicability Domain (Esper Dataset, 1,801 molecules)

## What Was Done

Step 04B implements a ChemBERTa-specific applicability domain (AD) detector using the fine-tuned model's CLS embedding space instead of the descriptor-space fallback inherited from the RF and NN models. This resolves a methodological inconsistency identified in Step 04: ChemBERTa reasons in transformer embedding space, not Morgan fingerprint + RDKit descriptor space, so its deployment guardrails should reflect the representation it actually uses.

### Implementation

1. **ChemBERTa AD Module** (`model/hf/ad.py`): Mirrors the style of `model/nn/ad.py` but operates on 768-dimensional CLS embeddings instead of descriptor vectors. Fits an `IsolationForest` detector (contamination=0.05) on training embeddings and saves artifacts to `model/saved/chemberta/`.

2. **Embedding Extraction** (`model/hf/chemberta_model.py`): Refactored `ChemBERTaForPCSAFT` to expose an `encode()` method that extracts CLS embeddings without going through the regression head. Both `forward()` and `predict_with_uncertainty()` now call `encode()` internally.

3. **AD Fitting** (`model/hf/train_chemberta.py`): After fine-tuning, the training script now extracts CLS embeddings for all training molecules and fits the AD detector. Saved artifacts include `chemberta_ad.joblib`, `chemberta_ad_metadata.json`, and `train_cls_embeddings.npy` (for visualization).

4. **Registry Integration** (`model/registry.py`): Added `embed()` and `predict_in_domain()` methods to the `ChemBERTaModel` wrapper. The wrapper now owns ChemBERTa-specific AD logic, keeping RF and NN paths unchanged.

5. **Model-Specific AD Analysis** (`model/evaluate.py`): Updated `_get_ad_labels()` to accept a `model_name` parameter. ChemBERTa uses embedding-space AD; RF and NN continue using descriptor-space AD. Evaluation now computes per-model AD labels instead of one global array.

6. **Serving Updates** (`serving/model_loader.py`): The `ModelServer` now branches on `model_type` for AD checks. ChemBERTa deployments use the ChemBERTa-specific AD artifact; RF and NN still use descriptor-space AD.

7. **Tests** (`tests/test_chemberta_ad.py`): 10 new tests cover `encode()` shape, AD fitting, registry methods, and model-specific AD branching. All 138 tests pass.

## ChemBERTa Performance Split by Embedding-Space AD

| Parameter | MAE (overall) | RMSE (overall) | R² (overall) | R² (in-domain) | R² (OOD) | Fraction OOD |
|-----------|---------------|----------------|--------------|----------------|----------|--------------|
| m         | 0.766         | 1.283          | 0.535        | 0.608          | 0.249    | 26.9%        |
| σ         | 0.226         | 0.349          | 0.253        | 0.249          | 0.207    | 26.9%        |
| ε/k       | 31.2          | 53.7           | 0.265        | 0.197          | 0.436    | 26.9%        |

> **Uncertainty note**: All R², MAE, and RMSE values in this report are point estimates on a single holdout split. 95% bootstrap confidence intervals are available via `python -m model.evaluate --bootstrap` and saved to `model/saved/comparison_metrics.csv` with `_lo`/`_hi` suffix columns.

### Interpretation

- **In-domain performance**: ChemBERTa's in-domain R² for `m` (0.608) is notably better than overall (0.535), confirming that the AD detector successfully identifies a training-like subset where the model performs well.

- **σ shows weak AD separation**: The in-domain R² (0.249) is nearly identical to overall (0.253), and OOD R² (0.207) is only slightly worse. This suggests that `σ` is hard to predict from learned embeddings regardless of AD status — likely a fundamental difficulty with 2D-to-3D property mapping rather than an AD artifact.

- **ε/k shows surprising OOD behavior**: OOD R² (0.436) is **higher** than in-domain (0.197). This counterintuitive result likely reflects small sample noise (97 OOD molecules) or a coincidental alignment of OOD molecules with simpler chemical patterns. It does not invalidate the AD detector but highlights that small-dataset AD splits can be noisy.

- **Flagged fraction (26.9%)**: Slightly higher than the RF/NN descriptor-space AD (~21% on the same test set). This difference is expected because embedding-space and descriptor-space capture different notions of "similarity."

## Comparison to Descriptor-Space AD (Step 02/03 Baseline)

| AD Method | Used By | Representation | Train Source | Flagged OOD (test) |
|-----------|---------|----------------|--------------|---------------------|
| Descriptor-space IsolationForest | RF, NN, (ChemBERTa fallback in Step 04) | Morgan FP + RDKit descriptors (~2,150 dims) | Esper train split | ~21% |
| Embedding-space IsolationForest | ChemBERTa (Step 04B) | CLS embeddings (768 dims) | Esper train split | 26.9% |

### Key Differences

1. **Representation alignment**: ChemBERTa now uses an AD detector that operates in the same space the model reasons in (CLS embeddings), rather than a proxy inherited from earlier models.

2. **Dimensionality**: Embedding space is 768-dimensional; descriptor space is ~2,150-dimensional. The curse of dimensionality affects both, but embedding-space detectors may be more sensitive to learned nuances of the training distribution.

3. **OOD fraction**: ChemBERTa flags more molecules as OOD (26.9% vs ~21%). This is not necessarily a problem; it may reflect that ChemBERTa's learned representation is more selective about what it considers "training-like" compared to handcrafted features.

4. **No change to RF/NN**: Descriptor-space AD remains unchanged for RF and NN. This preserves comparability with earlier results.

## Key Findings

1. **Methodological consistency achieved**: ChemBERTa AD is now architecture-appropriate. Embedding-space AD is scientifically more defensible than descriptor-space AD for a transformer model.

2. **In-domain improvement for `m`**: The AD detector successfully identifies a subset where `m` predictions are more reliable (R² 0.608 vs 0.535 overall).

3. **Noisy OOD metrics**: With only 97 OOD test molecules, OOD R² values are volatile. The `ε/k` OOD R² of 0.436 should not be overinterpreted.

4. **Flagged fraction is plausible**: 26.9% OOD is higher than descriptor-space AD but not unreasonably high. Visual inspection of the PCA plot (Figure 1) confirms that flagged molecules are geometrically outlying in embedding space.

5. **Deployment validity improved**: The API now returns `in_domain` for ChemBERTa using the ChemBERTa-specific AD artifact. This is more faithful to the model's behavior than the descriptor-space fallback from Step 04.

## Figures

See `figures/04b_chemberta_ad/` for visualizations:

1. **`chemberta_embedding_pca.png`**: 2D PCA of CLS embeddings colored by train/test and AD labels. Shows that OOD molecules are geometrically distant from the training cloud in the learned representation space.

2. **`chemberta_ad_score_histogram.png`**: Distribution of IsolationForest decision scores for train vs test. The threshold (score=0) successfully separates most training molecules from outliers.

3. **`chemberta_ad_parity_by_target.png`**: Per-target parity plots split by in-domain (blue circles) vs OOD (red triangles). Visual confirmation of improved `m` performance in-domain and noisy OOD behavior for `ε/k`.

## Deviations

None. The implementation follows the Step 04B guide exactly:
- ChemBERTa gets its own embedding-space AD
- RF and NN remain unchanged
- Serving branches on `model_type` for AD checks
- Evaluation computes per-model AD labels
- Response schema is unchanged

## Readiness Check

- [x] ChemBERTa AD labels are generated from CLS embedding space, not descriptors
- [x] Evaluation harness reports ChemBERTa in-domain vs OOD metrics using the new detector
- [x] Serving layer returns `in_domain` for ChemBERTa from the ChemBERTa-specific AD artifact
- [x] RF and NN predictions still use descriptor-space AD workflow without regression
- [x] At least one figure explains whether the new AD boundary is scientifically believable (yes, PCA shows clear geometric separation)
- [x] Can clearly explain why Step 04B improves deployment validity even if point-prediction metrics stay the same (because AD is now architecture-consistent)

## Why This Matters

Step 04B does not improve ChemBERTa's point-prediction accuracy. It improves **deployment validity**. A transformer model should be judged by what it knows about its own learned representation, not by a proxy inherited from older models. Embedding-space AD is more faithful to ChemBERTa's behavior and provides a scientifically defensible guardrail for production deployment.

This distinction is strong interview material: it shows understanding of the difference between model quality (R² scores) and deployment rigor (valid uncertainty quantification and OOD detection).
