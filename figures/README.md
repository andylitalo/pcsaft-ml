# Figures

Parity plots for Random Forest PC-SAFT parameter predictions, organized by training dataset.

## Directory Structure

| Directory | Dataset | Train / Test | Description |
|-----------|---------|-------------|-------------|
| `fallback/` | 34 hand-curated molecules | 27 / 7 | Initial validation run |
| `esper/` | 1,801 Esper et al. molecules | 1,410 / 356 | Full-scale training run |

## Files (per directory)

| File | Description |
|------|-------------|
| `parity_all_parameters.png` | Combined 3-panel parity plot for m, sigma, and epsilon/k |
| `parity_m.png` | Segment number (m) -- predicted vs. true |
| `parity_sigma.png` | Segment diameter (sigma) -- predicted vs. true |
| `parity_epsilon_k.png` | Dispersion energy (epsilon/k) -- predicted vs. true |

## How to Read a Parity Plot

Each plot shows true (literature) values on the x-axis and model predictions on the y-axis. The red dashed line is y = x -- a perfect model would place all points on this line. The inset box reports R², MAE, and RMSE for that parameter.

## Regenerating

```bash
# For the Esper dataset (recommended)
python -m model.train --data esper --tune
python -m model.evaluate

# For the fallback dataset
python -m model.train --data fallback
python -m model.evaluate
```

The `evaluate` command saves a combined parity plot to `model/saved/parity_plots.png`. The individual figures in the subdirectories here were generated separately for presentation use with enlarged fonts.
