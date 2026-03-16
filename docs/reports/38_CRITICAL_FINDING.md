# CRITICAL FINDING: GNN Fluorinated Screening Invalid

**Date**: 2026-03-16
**Step**: 38 (GNN Fluorinated Validation)
**Status**: ❌ **FAILED** — Step 37 GNN screening results are invalid for fluorinated compounds

## Executive Summary

The Step 38 validation of the GNN model on 15 known fluorinated refrigerants reveals **catastrophic failure** that invalidates the Step 37 GNN screening results:

- **GNN boiling point MAE**: 133.72 K (16× worse than RF's 8.17 K)
- **GNN parameter errors**: Systematic over-prediction of all parameters by large margins
- **Filter reliability**: Compromised (133 K error vs. 35 K filter window)
- **Uncertainty calibration**: Severely poor (0-6.7% 1σ coverage vs. expected 68%)

**Recommendation**: **Do NOT use Step 37 GNN screening outputs**. Re-run screening with RF model instead.

## The Numbers

| Metric | GNN | RF | RF Advantage |
|--------|-----|-----|-------------|
| **Boiling Point MAE** | **133.72 K** | **8.17 K** | **16.4×** |
| m parameter MAE | 1.57 | 0.85 | 1.8× |
| σ parameter MAE | 0.40 Å | 0.07 Å | 5.7× |
| ε/k parameter MAE | 77.60 K | 14.30 K | 5.4× |
| EOS Convergence Rate | 40% | 40% | Same |

## Why This Matters

1. **Step 37 ranked list is unreliable**: The GNN predicts HFO-1234yf (a commercial refrigerant) at 446.6 K when its actual boiling point is 243.7 K (+203 K error).

2. **Filter window is compromised**: The 288-323 K filter (35 K wide) cannot function with 133 K MAE.

3. **Screening sanity checks fail**: R-23 (T_b=191 K, far below target) ranks #1 instead of HFO candidates.

4. **Steps 39-42 cannot proceed**: Uncertainty-aware shortlist work requires calibrated uncertainty, which the GNN does not provide.

## Root Cause: Source Mismatch

The GNN was trained on SPT-PC-SAFT-derived parameters, which systematically differ from literature-optimized PC-SAFT parameters. The GNN learned to reproduce SPT's over-estimation pattern, making it unsuitable for real-world fluorinated screening.

**Evidence**:
- Step 31 reported R²(ε/k)=0.91 on fluorinated test set → likely SPT-derived
- Step 38 shows R²(ε/k)=-25.08 on literature-optimized parameters → catastrophic failure

## Immediate Action Required

### Option A: Re-run Step 37 with RF Model (Recommended)

- RF achieves 8.2 K boiling point MAE on fluorinated validation set
- RF parameters are well-calibrated to literature values
- RF uncertainty (tree ensemble variance) supports Steps 39-42

### Option B: Retrain GNN on Literature Data Only

- Exclude all SPT-PC-SAFT training data
- Use only experimentally validated PC-SAFT parameters
- Re-validate on Step 38 set before any screening use

### Option C: Abandon Fluorinated Screening Temporarily

- Focus on non-fluorinated chemistry where model performance is better
- Revisit fluorinated screening after model improvements

## Impact on Project Timeline

- **Steps 39-42**: Blocked pending model decision
- **Step 37 rerun**: Required if proceeding with Option A (~1 hour)
- **GNN retrain**: Required if proceeding with Option B (~6-8 hours)

## Files to Review

- **Full report**: `docs/reports/38_gnn_fluorinated_validation.md`
- **Validation script**: `scripts/step38_gnn_fluorinated_validation.py`
- **Figures**: `figures/38_gnn_fluorinated_validation/`
- **Validation set**: `model/saved/gnn_fluorinated_validation_set.csv`
- **Results**: `model/saved/gnn_fluorinated_validation_results.csv`

## Lessons Learned

1. **Benchmark contamination risk**: Step 31's fluorinated test set may have included SPT-derived parameters, inflating GNN metrics.

2. **Validation on real targets is essential**: Model performance on held-out training distribution != model performance on deployment targets.

3. **Small molecule refrigerants are different**: The C1-C4 highly fluorinated regime may require specialized treatment.

4. **Parameter source matters**: SPT-PC-SAFT vs. literature-optimized PC-SAFT are different parameterization philosophies.

---

**Next conversation**: Decide on Option A, B, or C before proceeding.
