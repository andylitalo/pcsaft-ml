# Results Report: RF vs GNN Formal Comparison (Step 38d, 15-Compound Fluorinated Validation)

## Experiment Design

Steps 38-38c established that both the unified GNN (trained on ~12,800 molecules from Esper+ML-SAFT+SPT-PCSAFT) and the Esper-retrained GNN (trained on ~1,801 experimentally fitted molecules) catastrophically fail on fluorinated compounds. This step performs the definitive three-way comparison (RF vs GNN-Esper vs GNN-Unified) on the 15-compound fluorinated validation set and the 10,700-candidate HFO screening pipeline, then documents the model selection decision.

The three models compared:
- **RF**: Random Forest on RDKit 2D descriptors, trained on Esper dataset (1,801 molecules, 170 features)
- **GNN-Esper**: GINEConv (3-layer, 128-dim), retrained on Esper-only data (1,801 molecules, 195K params)
- **GNN-Unified**: GINEConv (4-layer, 256-dim), trained on unified corpus (~12,800 molecules)

## Fluorinated Validation Results

### Boiling Point MAE (Primary Metric)

| Model | BP MAE (K) | BP RMSE (K) | EOS Converged | RF Advantage |
|-------|-----------|-------------|---------------|--------------|
| **RF** | **8.17** | **10.32** | 6/15 | -- |
| GNN-Esper | 142.30 | 145.59 | 7/15 | 17.4x |
| GNN-Unified | 133.72 | 139.91 | 6/15 | 16.4x |

The RF achieves 8.2 K boiling point MAE -- a 16-17x advantage over both GNN variants. The GNN-Esper is actually 8.6 K *worse* than the GNN-Unified, confirming that the failure is architectural, not caused by SPT data contamination.

### Parameter MAE (vs Literature, n=15)

| Parameter | RF MAE | GNN-Esper MAE | GNN-Unified MAE | RF Advantage |
|-----------|--------|---------------|-----------------|--------------|
| m (segments) | 0.852 | 1.696 | 1.575 | 1.8-2.0x |
| sigma (A) | 0.072 | 0.409 | 0.397 | 5.5-5.7x |
| epsilon_k (K) | 14.30 | 72.38 | 77.60 | 5.1-5.4x |

The RF's advantage is most dramatic for sigma (5.7x) and epsilon_k (5.4x), the parameters that dominate boiling point through the PC-SAFT equation of state. The m parameter advantage is smaller (1.8x) but still decisive.

Both GNN variants show systematic over-prediction of all three parameters for fluorinated compounds. The GNN learns a "generic organic molecule average" rather than correctly capturing the distinct parameter signatures of small, heavily fluorinated refrigerants (C1-C4, sigma ~2.8-3.5 A, m ~1.3-2.8).

## Screening Rank Agreement

### Filter Funnel (Three-Way)

| Stage | RF | GNN-Esper | GNN-Unified |
|-------|-----|-----------|-------------|
| Initial candidates | 10,700 | 10,700 | 10,700 |
| Boiling point [288-323K] | 1,190 | 168 | 729 |
| Has C=C bond | 10,700 | 10,700 | 10,700 |
| No chlorine | 2,558 | 2,558 | 2,558 |
| F count >= 2 | 10,132 | 10,132 | 10,132 |
| SA score <= 4.5 | 4,663 | 4,663 | 4,663 |
| Fluorine mass fraction >= 65% | 330 | 330 | 330 |
| No reactive fluorination sites | 3,511 | 3,511 | 3,511 |
| **Final (all filters)** | **50** | **23** | **65** |

The boiling point filter is the key divergence point. RF places 1,190 candidates in the [288, 323] K window; GNN-Esper places only 168 (7x fewer). This is because the GNN systematically over-predicts boiling points, pushing most candidates above the 323 K upper bound. The GNN-Unified places an intermediate 729, but with 133 K MAE these predictions are unreliable.

Structural filters (C=C bond, no chlorine, F count, SA score, F mass fraction, reactive sites) are identical across models because they depend only on SMILES structure.

### Candidate Overlap

| Comparison | Filter-Passing Overlap | Top-10 | Top-20 |
|------------|----------------------|--------|--------|
| RF vs GNN-Esper | 16 | 3 | 11 |
| RF vs GNN-Unified | 39 | 1 | 7 |
| All three models | 10 | -- | -- |

### Spearman Rank Correlations (Shared Candidates)

| Pair | Spearman rho | p-value | n |
|------|-------------|---------|---|
| RF vs GNN-Esper | 0.247 | 0.356 | 16 |
| RF vs GNN-Unified | 0.289 | 0.075 | 39 |
| GNN-Esper vs GNN-Unified | 0.565 | 0.035 | 16 |

RF and GNN rankings are essentially uncorrelated (rho < 0.3, not significant at p=0.05). The two GNN variants agree more with each other (rho=0.57, p=0.035) than either does with RF, consistent with shared architectural bias in the GNN predictions.

The mean absolute rank shift between RF and GNN-Esper (on shared candidates) is 5.0 positions, reflecting substantial disagreement in ordering.

## Model Disagreement Analysis

### Tanimoto Similarity to Training Set

The 15 validation compounds have a mean Tanimoto similarity of 0.906 to the Esper training set. This indicates they are structurally well-represented in the training data -- the failure is not a simple domain distance problem.

Correlation between Tanimoto similarity and prediction error:
- RF epsilon_k error vs similarity: rho = +0.41 (more similar = larger RF error, counterintuitive)
- GNN epsilon_k error vs similarity: rho = -0.37 (more similar = smaller GNN error, expected)
- Model disagreement vs similarity: rho = -0.41 (more similar = GNN does relatively better)

The positive correlation for RF may reflect that the most "similar" fluorinated compounds in the training set are non-fluorinated analogs, and RF's fingerprint-based features handle fluorine substitution effects reasonably well even at lower similarity. The GNN's message-passing architecture does benefit from structural similarity but still produces catastrophic errors even for high-similarity molecules.

## Uncertainty Comparison

### RF Tree-Level Uncertainty (15 validation compounds)

| Parameter | Mean Std | Mean Error | Std/Error Ratio | 1-sigma Coverage | 2-sigma Coverage |
|-----------|----------|------------|----------------|------------------|------------------|
| m | 0.303 | 0.852 | 0.36 | 0% | 7% |
| sigma | 0.136 | 0.072 | 1.90 | 87% | 100% |
| epsilon_k | 18.20 | 14.30 | 1.27 | 80% | 93% |

### GNN MC Dropout Uncertainty (15 validation compounds)

| Parameter | Mean Std | Mean Error | Std/Error Ratio | 1-sigma Coverage | 2-sigma Coverage |
|-----------|----------|------------|----------------|------------------|------------------|
| m | 0.387 | 1.696 | 0.23 | 0% | 7% |
| sigma | 0.064 | 0.409 | 0.16 | 7% | 13% |
| epsilon_k | 9.55 | 72.38 | 0.13 | 7% | 13% |

The RF uncertainty is reasonably calibrated for sigma (87% 1-sigma coverage vs 68% expected) and epsilon_k (80% vs 68% expected). The m parameter is underestimated by both models. The GNN MC Dropout uncertainty is catastrophically miscalibrated: for epsilon_k, the mean uncertainty is 9.5 K but the actual error is 72.4 K (7.6x underestimation). The GNN "confidently" makes wrong predictions.

**Key finding**: The RF's epsilon_k std/error ratio of 1.27 means its uncertainty approximately tracks actual errors. The GNN's ratio of 0.13 means its uncertainty is 7.6x too small. For uncertainty-aware screening (Steps 39-42), only RF uncertainty is usable.

## Model Selection Decision

**Selected model: RF (Random Forest)**

The decision is based on three criteria:

1. **Accuracy**: RF achieves 8.2 K boiling point MAE, 16.4x better than the best GNN variant (133.7 K). This is not a marginal difference -- it is the difference between screening-useful predictions and catastrophic failure.

2. **Parameter fidelity**: RF achieves 5.1x better epsilon_k MAE (14.3 K vs 72.4 K). Since epsilon_k dominates the HFO distance ranking (5:2:1 weighting), this directly translates to more reliable candidate rankings.

3. **Uncertainty calibration**: RF epsilon_k uncertainty has 80% 1-sigma coverage (vs 68% expected, slightly conservative). GNN has 7% (catastrophically miscalibrated). Only RF uncertainty supports the uncertainty-aware shortlisting needed for Steps 39-42.

The GNN's failure is architectural, not due to data contamination:
- GNN-Esper (clean data only) is WORSE than GNN-Unified (142.3 vs 133.7 K)
- 1,801 molecules is insufficient for a 195K-parameter GNN to learn effective fluorinated-compound representations
- RF succeeds because RDKit descriptors provide strong inductive bias at low data volumes

## Lessons Learned

1. **Inductive bias matters at small data scale**: With ~1,800 training molecules, hand-crafted molecular descriptors (RDKit features) decisively outperform learned representations (GNN message passing). The RF's 170 features encode decades of chemical knowledge that the GNN must learn from scratch.

2. **GNN failure mode is systematic, not random**: Both GNN variants show consistent over-prediction of all three PC-SAFT parameters. This is a representation failure, not noise.

3. **Uncertainty calibration is a critical differentiator**: Even if two models had similar accuracy, the one with calibrated uncertainty is far more useful for screening. The GNN's 7.6x underestimation of epsilon_k uncertainty would lead to false confidence in incorrect predictions.

4. **Data contamination was a red herring**: Step 38c showed that removing SPT data made the GNN *worse*, not better. The root cause is architectural limitation, not training data quality.

5. **Simpler models can win on specific domains**: The RF is a simpler model with fewer parameters, but its domain-appropriate feature engineering makes it the right tool for this specific application.

## Impact on Steps 39-42

| Step | Impact |
|------|--------|
| **39: Uncertainty-Aware Shortlist** | Use RF tree-level std; propagate through boiling point calculation |
| **40: Property-Space Re-Ranking** | Use RF-predicted PC-SAFT parameters for vapor pressure and density ratios |
| **41: Safety/Environmental Filtering** | Use RF boiling points for flash point estimation; RF predictions are screening-reliable |
| **42: Final Report** | Document RF as the selected model; archive GNN results as negative evidence |

The 50-candidate RF ranked list (`screening/results/hfo_rf_ranked.csv`) is the starting point for Steps 39-42. The GNN ranked lists are archived for comparison but not used for decision-making.

## Readiness Check

- [x] Three-way validation comparison computed (RF vs GNN-Esper vs GNN-Unified)
- [x] Boiling point MAE: RF=8.2 K, GNN-Esper=142.3 K, GNN-Unified=133.7 K
- [x] Screening rank agreement analyzed (Spearman rho, top-N overlap, rank shifts)
- [x] Filter funnel three-way comparison documented
- [x] Model disagreement correlated with Tanimoto domain distance
- [x] Uncertainty comparison: RF calibrated (80% at 1sigma), GNN miscalibrated (7%)
- [x] Model selection decision documented with evidence
- [x] 6 figures generated in `figures/38d_rf_vs_gnn_comparison/`
- [x] Metrics saved to `model/saved/step38d_rf_vs_gnn_comparison.json`
- [x] All existing tests pass
- [x] `ruff check .` passes

## Figures

See `figures/38d_rf_vs_gnn_comparison/` for:
1. `validation_boiling_point_comparison.png` -- Three-panel boiling point parity (RF, GNN-Esper, GNN-Unified)
2. `parameter_parity_comparison.png` -- 3x3 grid of parameter parity plots (3 params x 3 models)
3. `screening_rank_agreement.png` -- Rank scatter: RF vs GNN-Esper and RF vs GNN-Unified
4. `filter_funnel_comparison.png` -- Three-way filter funnel bar chart
5. `uncertainty_vs_error.png` -- Uncertainty vs actual error for RF and GNN-Esper
6. `hfo_distance_rf_vs_gnn.png` -- HFO distance agreement colored by Tanimoto similarity
