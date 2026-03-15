# Step 4B: ChemBERTa Applicability Domain

## Purpose

Step 4 showed that ChemBERTa is a strong second-best predictor for PC-SAFT parameters, but it also exposed a mismatch in the current applicability-domain (AD) workflow. The project's existing AD model is an `IsolationForest` trained on Morgan fingerprints + RDKit descriptors, which is appropriate for the RF and NN because those models consume that feature space directly. ChemBERTa does not. It tokenizes raw SMILES and reasons in transformer embedding space.

This branch step closes that gap. Instead of asking, "Is this molecule close to the training set in fingerprint space?", Step 4B asks, "Is this molecule close to the training set in the representation ChemBERTa actually uses?" The goal is to build a ChemBERTa-specific AD check using the fine-tuned model's `CLS` embedding space, then wire that detector into evaluation and serving.

This is a useful later-phase extension because it sharpens the scientific story: the transformer should be judged, uncertainty-labeled, and deployment-gated in its own learned representation space rather than by a proxy inherited from older models.

## What It Adds Over Step 4

| After Step 4 | After This Step |
|--------------|----------------|
| ChemBERTa predictions are available, but AD labels still come from Morgan/RDKit feature space | ChemBERTa has its own embedding-space AD detector |
| ChemBERTa OOD metrics are approximate and only for rough comparison | ChemBERTa in-domain vs OOD metrics are architecture-consistent |
| Serving returns `in_domain` for ChemBERTa using descriptor-space fallback | Serving uses a ChemBERTa-specific AD artifact for ChemBERTa deployments |
| Step 4 report documents the limitation but does not resolve it | The limitation is implemented, measured, and reportable |

## Skills Demonstrated

- **Representation learning intuition**: understanding that OOD checks should live in the model's own feature space
- **Transformer internals**: extracting `CLS` embeddings from a fine-tuned encoder
- **OOD / anomaly detection**: fitting a lightweight detector such as `IsolationForest` or k-nearest-neighbor distance in embedding space
- **ML systems design**: keeping model-specific AD logic compatible with a shared registry, evaluation harness, and serving layer
- **Scientific rigor**: separating "comparison convenience" from "production-valid model behavior"

## Candidate AD Methods

All of the methods below are reasonable for ChemBERTa, but this step should start simple and explainable.

| Method | Idea | Pros | Cons |
|--------|------|------|------|
| `IsolationForest` on `CLS` embeddings | Fit an anomaly detector on the training embedding cloud | Reuses Step 02 intuition, easy to explain, no labels required | Decision boundary can be hard to interpret geometrically |
| kNN distance | Measure distance to nearest training embeddings | Intuitive and local | Requires storing training embeddings and choosing `k` |
| Centroid distance | Flag molecules far from the embedding centroid | Very cheap and easy to visualize | Too crude for multi-modal chemistry distributions |
| Mahalanobis distance | Distance with covariance scaling | More statistically grounded | Covariance can be unstable on small datasets or collinear embeddings |

**Recommended baseline**: start with `IsolationForest` on the fine-tuned `CLS` embeddings. It aligns with the current project language, stays CPU-friendly, and is straightforward to compare against the existing descriptor-space AD workflow.

**Recommended follow-up comparison**: if time permits later, compare `IsolationForest` against kNN distance or Mahalanobis distance to see whether local vs global density assumptions matter in this chemical space.

## Implementation Guide

### 1. Project structure

Add a small ChemBERTa AD helper module and keep all artifacts under the existing ChemBERTa save directory:

```text
model/
  hf/
    chemberta_model.py         # expose CLS embeddings
    train_chemberta.py         # fine-tune + fit embedding-space AD
    ad.py                      # ChemBERTa-specific AD helpers

model/saved/
  chemberta/
    chemberta_pcsaft.pt
    tokenizer/
    target_scalers.json
    chemberta_ad.joblib
    chemberta_ad_metadata.json
    train_cls_embeddings.npy   # optional; save if size is acceptable
```

Do not create a second heavy model. The AD detector should be a lightweight artifact derived from the already fine-tuned ChemBERTa encoder.

### 2. Expose ChemBERTa embeddings (`model/hf/chemberta_model.py`)

Right now the model extracts the `CLS` representation internally and immediately feeds it to the regression head:

```python
outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
cls_output = outputs.last_hidden_state[:, 0, :]
predictions = self.regression_head(cls_output)
```

Refactor this so embedding extraction is its own method. A future implementation should support something like:

```python
def encode(self, input_ids, attention_mask) -> torch.Tensor:
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state[:, 0, :]
```

Then `forward()` can call `encode(...)`, and the same method can be reused during AD fitting and serving. This avoids duplicating transformer logic and makes the representation explicit.

### 3. Train the AD detector after fine-tuning (`model/hf/train_chemberta.py`)

After fine-tuning completes and the best weights are loaded, compute `CLS` embeddings for the training split only. The AD detector must be fit on the same molecules used to train ChemBERTa, not on the test set.

Recommended workflow:

1. Re-tokenize the training SMILES with the saved tokenizer.
2. Run the fine-tuned encoder in evaluation mode.
3. Extract one `CLS` embedding per training molecule.
4. Fit a lightweight detector such as `IsolationForest(contamination=0.05)`.
5. Save the detector and any metadata needed for inference.

Suggested saved artifacts:

- `model/saved/chemberta/chemberta_ad.joblib`
- `model/saved/chemberta/chemberta_ad_metadata.json`
- optionally `model/saved/chemberta/train_cls_embeddings.npy`

The metadata file should record at least:

- detector type
- contamination or threshold hyperparameters
- embedding dimension
- number of training molecules
- train split source (`esper`, `combined`, etc.)

### 4. Create ChemBERTa AD helpers (`model/hf/ad.py`)

Keep the detector fitting and prediction logic separate from the training script so the code is reusable and testable. This helper module can mirror the style of `model/nn/ad.py`, but it should operate on embedding matrices rather than descriptor matrices.

Useful functions:

- `fit_chemberta_ad(embeddings: np.ndarray, contamination: float = 0.05)`
- `predict_chemberta_ad(ad_model, embeddings: np.ndarray) -> np.ndarray`
- `load_chemberta_ad_model()`

Returning boolean `True/False` or `1/-1` labels is fine as long as the convention matches the rest of the project.

### 5. Extend the registry wrapper (`model/registry.py`)

The `ChemBERTaModel` wrapper already knows how to load weights, tokenize SMILES, and return predictions. It should also become the home for ChemBERTa-specific AD behavior.

Future implementation options:

- expose a public embedding method such as `embed(smiles_list)`
- expose a model-specific AD method such as `predict_in_domain(smiles_list)`
- or do both, with the registry using embeddings internally and other modules calling a clean boolean AD API

At minimum, the wrapper should know how to:

1. load the ChemBERTa AD artifact if present,
2. embed incoming SMILES,
3. return in-domain / OOD flags in a consistent format.

This keeps the architecture clean: `rf` and `nn` remain descriptor-based, while `chemberta` owns its own embedding-space AD path.

### 6. Make AD analysis model-aware (`model/evaluate.py`)

The current evaluation harness computes a single shared set of AD labels from `model/saved/ad_model.joblib`, then applies those labels to every model. That was acceptable for Step 3 and only approximate for Step 4.

Step 4B should make this model-specific:

- `rf` and `nn` continue to use descriptor-space AD labels from Step 02
- `chemberta` uses embedding-space AD labels from the new ChemBERTa artifact
- `gc_pcsaft` can either reuse descriptor-space labels for comparison or be excluded from AD-specific reporting; document whichever choice you make

Practical refactor:

- replace a single `_get_ad_labels(smiles_list)` helper with something like `_get_ad_labels(smiles_list, model_name)`
- allow plots and metrics to consume per-model AD labels rather than one global array

This is the key scientific fix in the branch. Once this change lands, ChemBERTa in-domain vs OOD tables are actually about ChemBERTa's representation space.

### 7. Update serving to branch on model type (`serving/model_loader.py`)

Serving currently loads one generic AD model and applies descriptor-space feature computation unconditionally. That means `PCSAFT_MODEL_TYPE=chemberta` still reports `in_domain` from the wrong representation.

For a production-valid ChemBERTa deployment:

- if `model_type in {"rf", "nn"}`, keep the existing descriptor-space AD logic
- if `model_type == "chemberta"`, load the ChemBERTa AD artifact and compute `in_domain` from the embedding-space detector
- if the artifact is missing, fail gracefully and either mark all predictions as in-domain or return a clearly logged fallback, matching the current project's fault-tolerant style

The most important requirement is that the response schema does not change. `in_domain` stays a boolean; only the internal logic changes.

### 8. Preserve Step 4 comparability

Step 4B is a branch step, not a rewrite of Step 4. Preserve the original Step 4 report and results so you can still tell the story:

- Step 4: ChemBERTa improves prediction quality over the NN
- Step 4B: ChemBERTa gets its own architecture-appropriate deployment guardrail

This distinction is strong interview material because it shows you understand the difference between model quality and deployment validity.

## Science Improvements (Tier 1+2)

### 4B-A. Embedding Visualization

Create at least one figure that visualizes the ChemBERTa embedding space and AD boundary behavior. Options:

- 2D PCA of `CLS` embeddings colored by train/test and AD flag
- UMAP of embeddings with OOD molecules highlighted
- histogram of AD scores for train vs test molecules

This figure is not just for presentation polish; it helps catch failure modes like the detector flagging an implausibly large fraction of chemically ordinary molecules.

### 4B-B. Detector Comparison (Optional)

If the baseline `IsolationForest` works but gives ambiguous results, compare one alternative detector:

- kNN distance with a chosen `k`
- Mahalanobis distance on the embedding cloud

Only do this if the baseline is already implemented. The main deliverable of Step 4B is a working ChemBERTa-specific AD path, not an exhaustive OOD-method benchmark.

## Evaluation & Success Criteria

### Metrics to compare

Report ChemBERTa metrics both overall and split by the new AD labels:

| Model | Target | R2 (overall) | R2 (in-domain) | R2 (OOD) | Fraction OOD |
|-------|--------|--------------|----------------|----------|--------------|
| ChemBERTa | m | — | — | — | — |
| ChemBERTa | sigma | — | — | — | — |
| ChemBERTa | epsilon_k | — | — | — | — |

If you compare detectors, add a second table showing how the OOD fraction and in-domain / OOD gap changes by detector type.

### What success looks like

- `ChemBERTaModel` can produce `CLS` embeddings without going through the regression head
- a ChemBERTa-specific AD artifact exists under `model/saved/chemberta/`
- `python -m model.evaluate --models chemberta` reports embedding-space AD metrics for ChemBERTa
- serving returns `in_domain` for ChemBERTa using the ChemBERTa-specific detector rather than descriptor-space fallback
- the RF and NN still use the existing Step 02 AD path unchanged
- at least one figure in `figures/04b_chemberta_ad/` communicates the detector behavior clearly

### Realistic expectations

Do not expect a dramatic jump in ChemBERTa predictive accuracy from this step. AD is a reliability layer, not a performance booster. The win is that out-of-domain warnings become more faithful to the model's own learned representation.

Also do not assume the ChemBERTa AD split will be cleaner than the descriptor-space split on every metric. With only ~1,800 molecules, OOD subsets may be small and noisy. What matters is methodological consistency and an honest discussion of whether the new detector produces a more believable boundary.

### What to watch for

- **Pooling choice**: Step 4 uses the first-token `CLS` representation. Keep that consistent unless you deliberately test mean pooling as a follow-up.
- **Contamination sensitivity**: the fraction flagged OOD may swing noticeably with the `IsolationForest` contamination hyperparameter.
- **Embedding collapse**: if many molecules map to a narrow embedding region, the detector may provide little separation.
- **Split leakage**: never fit the detector on test embeddings.
- **Serving latency**: ChemBERTa AD requires an encoder pass at inference, so avoid duplicate forward passes where possible.
- **Fallback behavior**: missing AD artifacts should not crash the API silently or produce mislabeled outputs.

### Deliverables

- `model/hf/ad.py`
- updated `model/hf/chemberta_model.py`
- updated `model/hf/train_chemberta.py`
- updated `model/registry.py`
- updated `model/evaluate.py`
- updated `serving/model_loader.py`
- `model/saved/chemberta/chemberta_ad.joblib`
- `figures/04b_chemberta_ad/`
- `docs/reports/04b_chemberta_applicability_domain.md`

### Out of scope

To keep this branch step manageable, explicitly defer:

- conformal prediction or calibrated abstention
- retraining ChemBERTa to optimize OOD separation directly
- a large benchmark of many OOD detectors
- changes to RF or NN AD beyond preserving their current behavior
- replacing ChemBERTa with a larger pretrained model solely for AD research

### When to move on

You're ready to close Step 4B when:

1. ChemBERTa's AD labels are generated from `CLS` embedding space, not Morgan/RDKit descriptors
2. The evaluation harness reports ChemBERTa in-domain vs OOD metrics using the new detector
3. The serving layer returns `in_domain` for ChemBERTa from the ChemBERTa-specific AD artifact
4. RF and NN predictions still use the existing descriptor-space AD workflow without regression
5. You have at least one figure and one report section that explain whether the new AD boundary is scientifically believable
6. You can clearly explain why Step 4B improves deployment validity even if point-prediction metrics stay the same
