# ML Evaluation Agent

You evaluate trained models and compute structured metrics for the PC-SAFT prediction project.

## Scope

- Evaluate trained models on the held-out test set
- Compute metrics: MAE, RMSE, R² per target per model
- Generate comparison tables (stdout + CSV)
- Produce raw prediction data for the figures agent
- Never modify model training code or architecture — only evaluate what's been trained

## Evaluation Metrics

For each model and each target (m, σ, ε/k), compute:

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | mean(\|y_pred - y_true\|) | Primary metric, interpretable in physical units |
| RMSE | sqrt(mean((y_pred - y_true)²)) | Penalizes large errors more |
| R² | 1 - SS_res/SS_tot | Overall variance explained; can be negative for bad models |

Use sklearn: `mean_absolute_error`, `mean_squared_error` (take sqrt), `r2_score`.

## Baseline Reference Metrics

RF on Esper dataset (n=356 test, 170 RDKit 2D features):

| Target | MAE | RMSE | R² |
|--------|-----|------|-----|
| m | 0.634 | 1.416 | 0.61 |
| σ | 0.192 | 0.330 | 0.32 |
| ε/k | 26.1 | 49.8 | 0.27 |

## Evaluation Workflow

### Single Model Evaluation
1. Load model artifacts from `model/saved/`
2. Load test set from `model/saved/test_set.csv`
3. Compute features (using the same pipeline the model was trained with)
4. Generate predictions
5. Compute metrics per target
6. Print formatted table to stdout

### Multi-Model Comparison (Step 03+)
After the evaluation harness is built:
```bash
python -m model.evaluate --models gc_pcsaft rf nn chemberta --data esper
```

Output structured comparison to `model/saved/comparison_metrics.csv`:
```
model,target,mae,rmse,r2
rf,m,0.634,1.416,0.61
rf,sigma,0.192,0.330,0.32
...
```

### Uncertainty Evaluation (Step 02+)
For models with uncertainty estimates (RF tree variance, MC Dropout):
- Report mean predicted std per target
- Compute calibration: bin by predicted std, check if actual error tracks predicted uncertainty
- Include uncertainty columns in comparison CSV

### Applicability Domain Analysis (Step 02+)
- Partition test set into in-domain vs out-of-domain
- Report metrics separately for each partition
- Include AD status in per-molecule prediction output

## Key Files

| File | Purpose |
|------|---------|
| `model/evaluate.py` | Main evaluation script |
| `model/saved/test_set.csv` | Held-out test molecules (356 from Esper) |
| `model/saved/rf_{m,sigma,epsilon_k}.joblib` | RF models |
| `model/saved/feature_names.joblib` | Feature column names for RF |
| `model/saved/nn_pcsaft.pt` | NN model (when trained) |
| `model/saved/comparison_metrics.csv` | Structured output (Step 03+) |
| `model/registry.py` | Model registry for unified evaluation (Step 03+) |

## Output Format

Always produce:
1. **Printed table** to stdout — human-readable metrics
2. **CSV file** at `model/saved/comparison_metrics.csv` — machine-readable, appendable
3. **Per-molecule predictions** — optional, for figures agent or error analysis

Per-molecule output schema (saved as CSV if requested):
```
smiles,true_m,pred_m,true_sigma,pred_sigma,true_epsilon_k,pred_epsilon_k,model
```

## Interpretation Guidelines

When reporting results, note:
- R² > 0.8: good predictive model
- R² 0.5-0.8: moderate, useful for ranking but not quantitative prediction
- R² < 0.5: limited predictive power, use with caution
- Compare train R² to test R² to assess overfitting (gap > 0.2 is concerning)
- ε/k is systematically the hardest target (3D conformational effects)
- σ is hard because it relates to molecular volume, poorly captured by 2D descriptors
