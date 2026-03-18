# Step 49: Training Convergence Diagnostics

> **Presentation priority: HIGH.** "How do you know your models converged?" is a first-round question from any MLE reviewer. Right now, only 2 of 7 models have saved convergence evidence — and those two (NN and ChemBERTa) are the weakest performers. The models that matter for the narrative (RF, XGBoost, GNN) have no saved diagnostic artifacts. This creates the impression that convergence was assumed rather than verified.

## Objective

Instrument all training scripts to persist **training and hyperparameter-selection diagnostics** (loss curves, early-stopping metadata, OOB scores, boosting-round traces, CV detail), regenerate the missing artifacts, and produce figures and a report that answer two separate questions:

1. **Did iterative training actually converge without using the test set for model selection?**
2. **For non-iterative models, do we have enough search diagnostics to justify the chosen hyperparameters?**

## Motivation

The project trains 7 models but has limited saved diagnostic evidence. An ML-for-chemistry audience will ask how we know the models converged or whether the selected hyperparameters were reasonable, and right now the honest answer is: for most models, we do not have the artifacts to show it cleanly.

| Model | Current convergence evidence | Gap |
|-------|------------------------------|-----|
| RF | None | `n_estimators=100` used without OOB diagnostic |
| XGBoost | Best params in `step15_summary.json` | Full `GridSearchCV.cv_results_` not saved; no boosting-round trace for the selected config |
| SVM (SVR) | Not yet trained (Step 47) | Will use `GridSearchCV`; needs CV detail saving from the start |
| NN (MLP) | `model/saved/nn_history.json` + figure | Only model with full evidence (and it shows overfitting) |
| GINEConv GNN | Report claims "epoch 47" | No training history saved; `train_gnn.py` doesn't persist loss curves |
| chemprop D-MPNN | Early stopping configured | Lightning logger disabled (`logger=False`); no history saved |
| ChemBERTa | `model/saved/chemberta/training_history.json` (13 epochs) | Artifact exists; needs verification and re-plotting with convergence annotations |
| GNNePCSAFT | External pre-trained benchmark (added in Step 48) | Not applicable — model was trained by third-party authors on external databases; no project training history to instrument or save. Inference only. |

Note the optics: the only two models with saved history (NN, ChemBERTa) are the lowest-performing models in the comparison. The three models an MLE cares about — RF (production), XGBoost (tied for best on Esper), GNN (best R² on Esper) — have no diagnostic artifacts. This is the gap that this step closes.

## What It Adds

| Current | After This Step |
|---------|----------------|
| 2 of 7 project-trained models have saved training history (NN + ChemBERTa) | All 7 project-trained models have defensible diagnostic artifacts |
| 1 learning curve figure (for the weakest model) | 9 convergence figures covering all project-trained models |
| No RF ensemble-size validation | OOB R² vs n_estimators shows whether 100 trees has plateaued |
| No XGBoost or SVM CV detail | Full grid-search results saved and visualized for both |

GNNePCSAFT (Step 48 external benchmark) is explicitly excluded from this step: it is a pre-trained third-party model with no project-side training history. Its "convergence" is the responsibility of its original authors (see the GNNePCSAFT GitHub repository for training details). This step covers only models trained within this project.

## Dependencies

- Requires: all model training infrastructure from Steps 01, 02, 02b, 04, 15, 47
- No new packages needed (SVM uses sklearn which is already installed)

**Generalization caveat**: The diagnostics in this step validate optimization behavior, early stopping, and hyperparameter search quality under the project's current random-split training regime. They strengthen confidence that the models were trained correctly, but they do **not** by themselves establish scaffold-level chemistry generalization.

## Implementation Guide

### 49.1 — Instrument GNN training to save history

Modify `model/gnn/train_gnn.py` to accumulate per-epoch losses and save them after training.

Critical requirement: **do not use the held-out test split for `val_loss`, early stopping, or convergence plots**. The current implementation evaluates on `test_loader` inside the training loop, which is acceptable for exploratory debugging but not for scientific evidence. Before adding history saving, restructure the data split so the model trains on `train_df`, early-stops on a validation subset carved from `train_df`, and evaluates once on the untouched test set after model selection.

Recommended split:

1. Use the existing `split_data(df)` call to define the final train/test partition.
2. Split `train_df` again into `train_inner_df` and `val_df` using a fixed seed.
3. Build `train_loader` from `train_inner_df`, `val_loader` from `val_df`, and keep `test_loader` only for the final post-selection evaluation.

The training loop (lines 149-199) already computes `train_loss` and a validation-like loss per epoch but discards them. After introducing a true validation loader, add two accumulator lists before the loop and save them at the end, matching the format in `model/nn/trainer.py` (lines 259-268).

**Before the training loop** (after line 147), add:

```python
train_losses: list[float] = []
val_losses: list[float] = []
```

**Inside the loop**, after `val_loss /= max(n_val, 1)`, add:

```python
train_losses.append(train_loss)
val_losses.append(val_loss)
```

**After saving the model checkpoint** (after line 258), add:

```python
best_epoch = epoch - no_improve  # reconstruct from early stopping state
history = {
    "epochs": list(range(1, len(train_losses) + 1)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "best_epoch": best_epoch,
    "final_lr": scheduler.get_last_lr()[0],
    "source": source,
}
history_path = SAVED_DIR / f"gnn_history_{source}.json"
history_path.write_text(json.dumps(history, indent=2) + "\n")
print(f"Training history saved to {history_path}")
```

Note: save with source suffix (`gnn_history_combined.json`, `gnn_history_all.json`) because the GNN is trained on two different datasets.

Also persist split provenance in the history JSON:

```python
"split_role": {
    "train": len(train_inner_dataset),
    "val": len(val_dataset),
    "test": len(test_dataset),
    "early_stopping_monitor": "validation_loss",
}
```

Without this change, any resulting learning curve would be a test-set-tuned artifact and should not be cited as convergence evidence.

### 49.2 — Instrument chemprop to save history

Modify `model/chemprop_model/chemprop_wrapper.py` `fit()` method.

The Lightning Trainer is currently configured with `logger=False` (line 168). Replace with a CSVLogger so that per-epoch metrics are persisted:

```python
import lightning as L

csv_logger = L.pytorch.loggers.CSVLogger(
    save_dir=str(SAVED_DIR),
    name="chemprop_training",
)

trainer = L.Trainer(
    max_epochs=max_epochs,
    accelerator="cpu",
    enable_progress_bar=True,
    logger=csv_logger,
    enable_checkpointing=False,
    callbacks=callbacks,
)
```

After `trainer.fit(...)` completes, extract the logged metrics and save as JSON for consistency with other models:

```python
import json

metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
if metrics_path.exists():
    import csv as csv_mod
    with open(metrics_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)
    history = {
        "epochs": [int(float(r.get("epoch", i))) for i, r in enumerate(rows)],
        "train_loss": [float(r["train_loss"]) for r in rows if "train_loss" in r and r["train_loss"]],
        "val_loss": [float(r["val_loss"]) for r in rows if "val_loss" in r and r["val_loss"]],
    }
    history_path = SAVED_DIR / "chemprop_training_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n")
```

If extracting from the CSVLogger proves unreliable (Lightning sometimes writes sparse CSV rows), an alternative is to use a custom callback that records `trainer.callback_metrics["train_loss"]` and `trainer.callback_metrics["val_loss"]` at each epoch end.

### 49.3 — Verify ChemBERTa history artifact

The file `model/saved/chemberta/training_history.json` already exists (5.7 KB, 13 epochs of train/eval loss). The `_save_training_history()` function in `model/hf/train_chemberta.py` (lines 230-254) produced it during a prior training run.

Action: verify the existing artifact is complete and usable for plotting. The file should contain `train_losses` (list of `{epoch, loss}` objects), `eval_losses` (list of `{epoch, eval_loss}` objects), and `epochs`. No retraining is needed unless the file is malformed or incomplete.

Add a guard to `_save_training_history()` so future retraining does not fail silently if `log_history` is empty:

```python
log_history = trainer.state.log_history
if log_history:
    _save_training_history(log_history)
else:
    logger.warning("No training history available from Trainer state")
```

The existing artifact shows eval loss reaching a minimum at epoch 8 (0.689) with slight increases by epoch 13 — evidence of mild overfitting. This should be noted in the convergence assessment (49.9).

### 49.4 — Add RF OOB convergence diagnostic

Create a function in `scripts/step49_convergence_diagnostics.py` (the step's driver script) that evaluates whether 100 trees is sufficient:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import json
import numpy as np

def rf_oob_convergence(X_train, y_train, target_names):
    """Train RF with warm_start, recording OOB R² at each n_estimators checkpoint."""
    n_estimators_schedule = [10, 25, 50, 75, 100, 150, 200, 300]
    results = {name: [] for name in target_names}
    results["n_estimators"] = n_estimators_schedule

    for target_name, y_col in zip(target_names, y_train.T):
        rf = RandomForestRegressor(
            n_estimators=10,
            random_state=42,
            n_jobs=-1,
            warm_start=True,
            oob_score=True,
        )
        for n_est in n_estimators_schedule:
            rf.set_params(n_estimators=n_est)
            rf.fit(X_train, y_col)
            results[target_name].append(rf.oob_score_)

    return results
```

`warm_start=True` allows incrementally adding trees without retraining from scratch. `oob_score=True` computes the out-of-bag R² as a free byproduct of bagging (each tree's OOB predictions on the ~37% of samples not in its bootstrap). If OOB R² plateaus before `n_estimators=100`, the choice is validated. If it's still improving at 100, more trees may be warranted.

Save as `model/saved/rf_oob_convergence.json`.

### 49.5 — Save XGBoost CV detail

Modify `model/xgb/xgb_model.py` `fit()` method. After `cv.fit(X_train, y_train)` (line 84), save the full CV results:

```python
import json

cv_results = {
    k: v.tolist() if hasattr(v, "tolist") else v
    for k, v in cv.cv_results_.items()
    if not k.startswith("param_")  # skip non-serializable param objects
}
cv_results["best_params"] = cv.best_params_
cv_results["best_score"] = float(cv.best_score_)

SAVED_DIR.mkdir(parents=True, exist_ok=True)
cv_results_path = SAVED_DIR / "cv_results.json"
cv_results_path.write_text(json.dumps(cv_results, indent=2) + "\n")
logger.info("XGBoost CV results saved to %s", cv_results_path)
```

Also save the `params` key separately since `GridSearchCV.cv_results_["params"]` contains dict objects:

```python
cv_results["params"] = [
    {k: str(v) for k, v in p.items()} for p in cv.cv_results_["params"]
]
```

### 49.5a — Save XGBoost boosting-round learning curve

XGBoost is a boosting model whose convergence story is about **boosting rounds** (number of sequential trees), not just CV hyperparameter selection. The CV heatmap from 49.5 shows which hyperparameters are best, but not whether the best configuration used enough boosting rounds.

After the `GridSearchCV` completes (49.5), retrain the best configuration with a **validation subset carved from the training partition** to produce a per-round learning curve:

```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Use best params from GridSearchCV
best_params = cv.best_params_
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42,
)

xgb = XGBRegressor(
    **{k.replace("estimator__", ""): v for k, v in best_params.items()},
    random_state=42, verbosity=0,
    early_stopping_rounds=20,
)
xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=False,
)

# Extract per-round metrics
evals_result = xgb.evals_result()
xgb_history = {
    "train_rmse": evals_result["validation_0"]["rmse"],
    "val_rmse": evals_result["validation_1"]["rmse"],
    "n_rounds": list(range(1, len(evals_result["validation_0"]["rmse"]) + 1)),
    "best_iteration": xgb.best_iteration,
    "best_params": best_params,
}
xgb_history_path = SAVED_DIR / "xgb_boosting_history.json"
xgb_history_path.write_text(json.dumps(xgb_history, indent=2) + "\n")
```

This produces a train/val loss vs boosting round curve analogous to the neural model learning curves. If val loss is still decreasing at the configured `n_estimators`, the model may be undertrained. If it plateaus early, the configuration is validated.

Note: this must be done per target since XGBoost uses `MultiOutputRegressor` (one XGBRegressor per target). Run for `epsilon_k` at minimum since it is the deployment-critical target; run for all three targets if time permits. Treat this as **boosting convergence evidence for the chosen configuration**, distinct from the CV heatmap, which is a hyperparameter-search diagnostic.

### 49.5b — Save SVM CV detail

Step 47 creates `model/svm/svm_model.py` with a `GridSearchCV` over 18 SVR hyperparameter combinations (C, gamma, epsilon). Apply the same CV-results-saving pattern as XGBoost (49.5).

**Note on SVM "convergence"**: SVMs solve a convex QP to optimality — there is no iterative training trajectory like neural networks or boosting rounds like XGBoost. For SVMs, "convergence" means: (a) the QP solver converged (always true for RBF kernel with properly scaled features, which the SVM pipeline's `StandardScaler` ensures), and (b) the CV surface is smooth (performance is not sensitive to hyperparameter choice). The CV heatmap serves a different diagnostic purpose than neural learning curves: it validates hyperparameter selection, not training adequacy. State this distinction explicitly in the report (49.9).

In `model/svm/svm_model.py`, after the `GridSearchCV.fit()` call, save the full CV results:

```python
import json

cv_results = {
    k: v.tolist() if hasattr(v, "tolist") else v
    for k, v in cv.cv_results_.items()
    if not k.startswith("param_")
}
cv_results["params"] = [
    {k: str(v) for k, v in p.items()} for p in cv.cv_results_["params"]
]
cv_results["best_params"] = {k: str(v) for k, v in cv.best_params_.items()}
cv_results["best_score"] = float(cv.best_score_)

svm_saved_dir = Path(__file__).resolve().parent.parent / "saved" / "svm"
svm_saved_dir.mkdir(parents=True, exist_ok=True)
cv_results_path = svm_saved_dir / "cv_results.json"
cv_results_path.write_text(json.dumps(cv_results, indent=2) + "\n")
logger.info("SVM CV results saved to %s", cv_results_path)
```

If Step 47 has not yet been executed when this step runs, instrument the SVM model class at creation time so that convergence diagnostics are baked in from the start. This avoids the pattern that caused the gap in the first place (training first, adding diagnostics later).

### 49.6 — Retrain all models with instrumentation

After applying the code changes from 49.1-49.5b, retrain each model to produce diagnostic artifacts.

**RF OOB convergence** (new diagnostic, does not replace the existing trained RF):
```bash
python scripts/step49_convergence_diagnostics.py --section oob
```

**XGBoost** (retrain with CV detail saving):
```bash
python -m model.train --model xgboost
```
Or invoke the XGBoost fit path through whatever training entrypoint the project uses. The key is that the modified `fit()` now saves `cv_results.json`.

**GNN on combined data**:
```bash
python -m model.gnn.train_gnn --source combined --epochs 100 --patience 15
```

**GNN on unified data**:
```bash
python -m model.gnn.train_gnn --source all --epochs 100 --patience 15
```

**ChemBERTa**: No retraining needed only if `model/saved/chemberta/training_history.json` can be tied to the current training code, dataset, and split convention. Verify the artifact is loadable and that its provenance is documented before using it in the report. If provenance is unclear, retrain and regenerate the artifact rather than mixing incomparable runs.

**chemprop**:
The chemprop model is trained as part of the Step 15 pipeline. Retrain with the modified wrapper that now logs history.

**SVM**:
If Step 47 has already been executed, retrain the SVM with the CV-detail-saving code from 49.5b. If Step 47 has not been executed yet, skip this — the instrumentation will be in place when the SVM is first trained.
```bash
python scripts/train_step47.py
```

**NN (MLP)**: Same rule as ChemBERTa. Reuse `model/saved/nn_history.json` only if its provenance matches the current pipeline; otherwise retrain or clearly label it as historical evidence rather than directly comparable evidence.

After each run, verify the expected artifact exists:

| Model | Expected artifact path |
|-------|----------------------|
| RF (OOB) | `model/saved/rf_oob_convergence.json` |
| XGBoost | `model/saved/xgb/cv_results.json` |
| SVM | `model/saved/svm/cv_results.json` |
| NN (MLP) | `model/saved/nn_history.json` (already exists) |
| GNN (combined) | `model/saved/gnn_history_combined.json` |
| GNN (unified) | `model/saved/gnn_history_all.json` |
| chemprop | `model/saved/chemprop/chemprop_training_history.json` |
| ChemBERTa | `model/saved/chemberta/training_history.json` |

### 49.7 — Generate diagnostic figures

Create a plotting section in `scripts/step49_convergence_diagnostics.py` that reads each model's saved diagnostics and produces publication-quality figures. Save all figures to `figures/49_convergence_diagnostics/`.

**Required figures:**

1. **`rf_oob_convergence.png`** — Line plot: OOB R² (y-axis) vs n_estimators (x-axis), one line per target (m, sigma, epsilon_k). Vertical dashed line at n=100 (the production value). If the curves plateau before 100, the choice is validated. If they are still rising, flag it.

2. **`nn_learning_curves.png`** — Train loss and val loss vs epoch from `nn_history.json`. Mark best epoch (14) with a vertical line. Annotate the gap between train and val loss as evidence of overfitting. This is a re-plot of `figures/03_evaluation_harness/nn_learning_curves.png` with convergence-focused annotations.

3. **`gnn_learning_curves_combined.png`** — Train/val loss vs epoch for GNN on combined data from `gnn_history_combined.json`. Mark best epoch. Assess whether val loss is still decreasing or has plateaued.

4. **`gnn_learning_curves_unified.png`** — Same as above but for unified data from `gnn_history_all.json`.

5. **`chemprop_learning_curves.png`** — Train/val loss vs epoch from chemprop history. Mark where early stopping triggered.

6. **`chemberta_learning_curves.png`** — Train/eval loss vs epoch from ChemBERTa history.

7. **`xgb_cv_heatmap.png`** — Heatmap of mean CV R² across the XGBoost hyperparameter grid from `xgb/cv_results.json`. Rows: `n_estimators`, columns: `max_depth`. Best configuration marked with a star or bold border. This is a **selection diagnostic**, not a learning curve.

8. **`xgb_boosting_curves.png`** — Train/val RMSE vs boosting round for the best XGBoost configuration on epsilon_k (and optionally m, sigma). Mark the `best_iteration` from early stopping. This is the XGBoost analog of neural learning curves — it answers "did boosting converge?" rather than "which hyperparameters are best?"

9. **`svm_cv_heatmap.png`** — Heatmap of mean CV R² across the SVM hyperparameter grid from `svm/cv_results.json`. Axes: `C` vs `gamma`. Best configuration marked. If Step 47 has not been executed, skip this figure and note its absence in the report. This is a **selection diagnostic**, not an optimization trajectory.

**Plotting conventions:**
- Use `matplotlib` with `plt.style.use("seaborn-v0_8-whitegrid")` or equivalent
- Title each plot with the model name and dataset
- Include axis labels with units
- For learning curves: train loss in blue, val loss in orange, best-epoch marker in red
- Save at 150 DPI, tight layout

### 49.8 — Create the driver script

Create `scripts/step49_convergence_diagnostics.py` with the following structure:

```python
"""Step 49: Training Convergence Diagnostics.

Generates convergence artifacts and figures for all trained models.

Usage:
    python scripts/step49_convergence_diagnostics.py [--section SECTION]

Sections:
    oob         RF OOB convergence analysis
    figures     Generate all learning curve figures
    report      Write the report
    all         Run everything (default)
"""
```

Sections:
- `section_49_4_oob()` — RF OOB convergence
- `section_49_7_figures()` — all figures
- `section_49_8_report()` — write report markdown

### 49.9 — Write report

Write `docs/reports/49_convergence_diagnostics.md` following the standard report template. Required sections:

1. **Title**: `# Results Report: Training Convergence Diagnostics`
2. **Summary**: One paragraph stating which models have clean convergence evidence and which don't.
3. **Diagnostic framework**: Define the model-type-specific criteria used to classify each artifact, and state clearly that the same word "converged" does not mean the same thing for every model family.

   | Model type | Evidence | What can be claimed |
   |-----------|----------|---------------------|
   | RF (bagging) | OOB R² vs `n_estimators` | Whether ensemble size has approximately plateaued |
   | Neural (GNN, chemprop, ChemBERTa, NN) | Train/validation loss vs epoch on a true validation split | Whether optimization stabilized, whether early stopping triggered, and whether overfitting is visible |
   | XGBoost | CV surface + boosting-round validation trace | Whether the searched configuration is reasonable and whether boosting rounds were sufficient |
   | SVM | CV surface and solver completion | Whether the selected hyperparameters lie in a stable region; not an optimization trajectory claim |

   Avoid hard universal thresholds such as "gap > 2x final val loss = overfitting" unless they are empirically justified for this dataset. Prefer reporting the observed quantities and a cautious verdict.

4. **Per-model diagnostic assessment**: For each of the 7 models, a subsection with:
   - Training configuration (epochs, patience, LR schedule)
   - Diagnostic verdict (`clean convergence`, `overfitting visible`, `selection diagnostic only`, or `insufficient evidence`)
   - Key metric from the convergence artifact (e.g., "OOB R² plateaus at n=75" or "val loss still decreasing at early stop")
   - Reference to the figure
5. **Summary table**:

| Model | Diagnostic Status | Best Epoch / n_est | Train-Val Gap | Notes |
|-------|-------------------|-------------------|---------------|-------|
| RF | ... | n=100 (OOB plateau at ?) | N/A (bagging) | ... |
| XGBoost | ... | Best config + best boosting round | ... | Distinguish CV search from boosting trace |
| SVM (SVR) | ... | Best CV config | N/A | Hyperparameter diagnostic, not learning curve |
| NN (MLP) | Overfitting | 14 / 34 | Large | Val loss flat from epoch 3 |
| GNN (combined) | ... | ... | ... | ... |
| GNN (unified) | ... | ... | ... | ... |
| chemprop | ... | ... | ... | ... |
| ChemBERTa | ... | ... | ... | ... |

6. **Key findings**: Bullet points.
7. **Figures**: Reference all 9 figures with paths.
8. **Deviations**: If any model couldn't be retrained or instrumented, explain why.
9. **Readiness check**: Checklist.

### 49.10 — Update supplementary materials

After the report is written:

1. Add a new section to `supplementary_information/model_comparison.md` titled **"Training Convergence and Hyperparameter Validation"** that summarizes the per-model convergence status and links to the figures in `figures/49_convergence_diagnostics/`.

2. In `presentation/outline.md`, in the "Metrics of Success" section (around line 104), add a parenthetical link: `([convergence diagnostics](supplementary_information/model_comparison.md#training-convergence-and-hyperparameter-validation))`.

## Key Files

| File | Purpose |
|------|---------|
| `model/gnn/train_gnn.py` | Modify: add history saving (49.1) |
| `model/chemprop_model/chemprop_wrapper.py` | Modify: enable Lightning logger (49.2) |
| `model/hf/train_chemberta.py` | Verify: history saving works (49.3) |
| `model/xgb/xgb_model.py` | Modify: save CV results (49.5) |
| `model/svm/svm_model.py` | Modify: save CV results (49.5b) — created in Step 47 |
| `model/nn/trainer.py` | Reference only: already saves history |
| `model/saved/nn_history.json` | Existing artifact: NN training history |
| `scripts/step49_convergence_diagnostics.py` | Create: driver script (49.8) |

## Artifacts

| File | Description |
|------|-------------|
| `scripts/step49_convergence_diagnostics.py` | Driver script |
| `docs/reports/49_convergence_diagnostics.md` | Results report |
| `figures/49_convergence_diagnostics/` | All 9 convergence figures |
| `model/saved/rf_oob_convergence.json` | RF OOB R² vs n_estimators |
| `model/saved/gnn_history_combined.json` | GNN training history (combined data) |
| `model/saved/gnn_history_all.json` | GNN training history (unified data) |
| `model/saved/chemprop/chemprop_training_history.json` | chemprop training history |
| `model/saved/chemberta/training_history.json` | ChemBERTa training history |
| `model/saved/xgb/cv_results.json` | XGBoost grid-search CV results |
| `model/saved/xgb/xgb_boosting_history.json` | XGBoost per-round train/val loss for best config |
| `model/saved/svm/cv_results.json` | SVM grid-search CV results |

## Success Criteria

- [ ] `model/gnn/train_gnn.py` modified to save `gnn_history_{source}.json`
- [ ] `model/chemprop_model/chemprop_wrapper.py` modified to log and save training history
- [ ] ChemBERTa `training_history.json` verified as complete and provenance-documented (or regenerated)
- [ ] RF OOB convergence analysis saved to `rf_oob_convergence.json`
- [ ] XGBoost `cv_results.json` saved after grid search
- [ ] XGBoost boosting-round learning curve saved (`xgb_boosting_history.json`) for best config on epsilon_k
- [ ] SVM `cv_results.json` saved after grid search (or deferred if Step 47 not yet run)
- [ ] All 9 figures generated in `figures/49_convergence_diagnostics/` (SVM heatmap conditional on Step 47)
- [ ] Report written at `docs/reports/49_convergence_diagnostics.md`
- [ ] `supplementary_information/model_comparison.md` updated with convergence section
- [ ] `presentation/outline.md` updated with convergence link
- [ ] Any new neural-model learning curve uses a true validation split distinct from the held-out test set
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- Every model has a saved diagnostic artifact and a corresponding figure
- The report honestly distinguishes optimization convergence evidence from hyperparameter-selection evidence
- No learning curve or early-stopping claim is based on the held-out test set
- The supplementary materials link to the convergence evidence so it is available for presentation Q&A
- If any model's convergence status is unsatisfactory (e.g., val loss still improving at early stop), the report flags it with a recommendation (e.g., "increase patience" or "add data")

## Budget

90 minutes. The bulk of time is retraining (49.6) — the GNN and chemprop runs each take several minutes on CPU; SVM GridSearchCV adds ~5-10 minutes. Code instrumentation (49.1-49.5b) is fast. Plotting and report writing (49.7-49.10) should take ~30 minutes total. If ChemBERTa or NN provenance is unclear and they must be rerun for consistency, budget additional time.
