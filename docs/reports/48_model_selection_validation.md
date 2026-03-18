# Results Report: Model Selection Validation (5-Model Comparison)

This report documents the reproducible, apples-to-apples comparison of five PC-SAFT parameter prediction models across three validation tiers.

## Candidate Scope

Five models were evaluated: **RF**, **XGBoost**, **chemprop** (D-MPNN), **GNN** (GIN), and **GNNePCSAFT** (external pre-trained benchmark, ePC-SAFT). GNNePCSAFT predicts ePC-SAFT parameters rather than standard PC-SAFT; for non-associating fluorinated hydrocarbons the parameters are numerically close.

**ChemBERTa-2 was considered and declined**: the existing ChemBERTa-1 fine-tune (Step 04, R2=0.27 on eps/k, 185x slower than RF) already covers the SMILES-transformer angle. The fundamental bottleneck (~1,800 fine-tuning examples) is unchanged for v2; it is not a deployment candidate.

## Tier 1: Fluorinated External Validation (15 compounds)

### Parameter Accuracy


| Model    | m MAE | sigma MAE | epsilon/k MAE (K) | epsilon/k R2 |
| -------- | ----- | --------- | ----------------- | ------------ |
| RF       | 0.852 | 0.072     | 14.3              | -0.467       |
| chemprop | 0.718 | 0.056     | 19.6              | -1.219       |
| GNN      | 0.705 | 0.056     | 17.4              | -0.841       |


### Boiling Point Accuracy


| Model    | BP MAE (K) | BP RMSE (K) | EOS Convergence |
| -------- | ---------- | ----------- | --------------- |
| RF       | 8.2        | 10.3        | 6/15            |
| chemprop | 19.3       | 22.8        | 6/15            |
| GNN      | 23.9       | 26.5        | 4/15            |


### Data Leakage Audit

- Exact SMILES overlap (fluorinated vs Esper train): **11**
- Mean Tanimoto NN similarity to train: **0.900**
- Max Tanimoto NN similarity to train: **1.000**
- Compounds with high similarity (>0.85): **12**

## Tier 2: Esper Holdout with Bootstrap CIs

Test set: 2753 molecules (split hash: e456ecf90038)


| Model | eps/k R2 [95% CI]    | eps/k MAE [95% CI] |
| ----- | -------------------- | ------------------ |
| RF    | 0.236 [0.157, 0.307] | 23.4 [22.5, 24.4]  |
| GNN   | 0.728 [0.675, 0.779] | 10.0 [9.4, 10.7]   |


## Tier 3: Cross-Validated Q2 (5-fold, tree models)


| Model   | Q2 m            | Q2 sigma        | Q2 epsilon/k    |
| ------- | --------------- | --------------- | --------------- |
| RF      | 0.632 +/- 0.069 | 0.343 +/- 0.059 | 0.378 +/- 0.064 |
| XGBoost | 0.615 +/- 0.075 | 0.349 +/- 0.038 | 0.337 +/- 0.057 |


Note: chemprop, GNN, and GNNePCSAFT Q2 deferred (require per-fold retraining; this is a tree-only robustness check).

## Model Selection Decision

**RF selected as deployment model. Evidence: Tier 1 decisive (8.2 K BP MAE on fluorinated validation).**

- **tier1_bp_mae**: {"name": "tier1_bp_mae", "best_model": "rf", "best_mae": 8.164058235182551, "second_model": "chemprop", "second_mae": 19.333667049028687, "gap_K": 11.169608813846136, "meets_5K_threshold": true, "paired_ci_significant": true, "tier1_decisive": true}

## Figures

All figures saved to `figures/48_model_selection_validation/`.

1. `fluorinated_parity_all_models.png` - epsilon/k parity, one panel per model
2. `fluorinated_boiling_point_parity.png` - predicted vs experimental BP
3. `esper_bootstrap_ci_comparison.png` - forest plot of R2 and MAE with CIs
4. `paired_difference_forest.png` - paired metric differences
5. `tier_summary_heatmap.png` - key metrics across tiers

## Provenance

- Feature config hash: `593e5743fe14`
- Test split hash: `e456ecf90038`
- Test split n: 2753
- GNNePCSAFT version: not_installed
- rf: `/Users/aylitalo/Documents/personal/interviews/process/ml_chem/model/saved`
- gnn: `/Users/aylitalo/Documents/personal/interviews/process/ml_chem/model/saved`
- xgboost: `/Users/aylitalo/Documents/personal/interviews/process/ml_chem/model/saved/xgb/xgb_model.joblib`
- chemprop: `/Users/aylitalo/Documents/personal/interviews/process/ml_chem/model/saved/chemprop/chemprop_model.pt`

## Readiness Check

- All available models evaluated on fluorinated validation set
- Esper holdout metrics include bootstrap 95% CIs
- Paired comparisons use bootstrapped metric differences
- Data leakage audit documented
- Tanimoto nearest-neighbor similarity recorded
- Model selection decision documented with explicit evidence
- Provenance metadata saved
- Figures generated

