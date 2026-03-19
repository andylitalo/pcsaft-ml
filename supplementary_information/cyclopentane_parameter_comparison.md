# Cyclopentane PC-SAFT Parameters: Cross-Source Comparison

Six independent sources provide PC-SAFT parameters for cyclopentane — the reference blowing agent throughout this project. This document compares them to assess inter-source consistency and contextualize how well each method captures the experimentally fitted values.

---

## Parameter Comparison


| Source               | m      | σ (Å)  | ε/k (K) | Provenance                                          |
| -------------------- | ------ | ------ | ------- | --------------------------------------------------- |
| Esper (experimental) | 2.3655 | 3.7114 | 288.84  | Gross & Sadowski 2001; `model/data/pcsaft_data.csv` |
| RF (this project)    | 2.467  | 3.743  | 281.3   | `python -m model.predict "C1CCCC1"`                 |
| GNNePCSAFT           | 2.273  | 3.744  | 271.8   | wildsonbbl/gnnepcsaft v0.3.1 (PNA, HuggingFace)     |
| GC-PC-SAFT           | 2.434  | 3.238  | 275.3   | `model/gc_pcsaft.py` (5×CH2 + ring correction)      |
| ML-SAFT              | 2.396  | 3.692  | 263.8   | Felton et al. 2024; `model/data/mlsaft_pcsaft.csv`  |
| SPT-PCSAFT           | 2.674  | 3.594  | 246.4   | Winter et al. 2025; `model/data/spt_pcsaft.csv`     |


## Error vs Esper Reference


| Source     | Δm              | Δσ (Å)          | Δε/k (K)       |
| ---------- | --------------- | --------------- | -------------- |
| RF         | +0.101 (+4.3%)  | +0.032 (+0.9%)  | −7.5 (−2.6%)   |
| GNNePCSAFT | −0.093 (−3.9%)  | +0.033 (+0.9%)  | −17.0 (−5.9%)  |
| GC-PC-SAFT | +0.069 (+2.9%)  | −0.474 (−12.8%) | −13.5 (−4.7%)  |
| ML-SAFT    | +0.031 (+1.3%)  | −0.019 (−0.5%)  | −25.1 (−8.7%)  |
| SPT-PCSAFT | +0.309 (+13.1%) | −0.117 (−3.2%)  | −42.4 (−14.7%) |


---

## Discussion

### σ (segment diameter)

RF and GNNePCSAFT are nearly identical on σ (3.743 vs 3.744 Å), both within 1% of the Esper value. ML-SAFT is also close (−0.5%). GC-PC-SAFT badly misses σ (−12.8%), consistent with its known failure on this parameter (R² = −1.16 on the Esper test set). The group-additivity assumption cannot capture the compact ring geometry of cyclopentane: the ring correction (−3.0 on m·σ³) is insufficient to offset the accumulated CH2 contributions.

### ε/k (dispersion energy)

RF is closest (−2.6%), followed by GC-PC-SAFT (−4.7%) and GNNePCSAFT (−5.9%). ML-SAFT and SPT-PCSAFT underpredict ε/k substantially (−8.7% and −14.7%, respectively). The SPT-PCSAFT bias is consistent with the systematic ε/k underprediction documented in `supplementary_information/spt_pcsaft_issues.md`.

For the screening application, the ε/k error matters most because it propagates exponentially through the Boltzmann factor into Henry's constant. RF's −7.5 K error on cyclopentane is within the ±9 K locally calibrated 95% CI established in Step 43.

### m (segment number)

All sources are within ~4% of Esper except SPT-PCSAFT (+13.1%). GNNePCSAFT slightly underpredicts m while RF slightly overpredicts. The m parameter has the least impact on the screening application because Henry's constant is dominated by the cross-interaction energy (ε_ij), not the chain length.

---

## Caveats

- **Both RF and GNNePCSAFT were trained on the Esper dataset**, which contains cyclopentane. These are in-distribution predictions for both models — they do not represent out-of-distribution generalization accuracy.
- **GC-PC-SAFT is deterministic** (no training data needed) but its group-additivity assumption limits accuracy on ring compounds.
- **ML-SAFT parameters** (Felton et al. 2024) are programmatically regressed from simulation data, not experimentally fitted. Inter-dataset variability between Esper and ML-SAFT is documented in `docs/reports/10_mlsaft_integration.md`.
- **SPT-PCSAFT parameters** (Winter et al. 2025) are themselves ML-predicted (SMILES transformer), not experimentally fitted. The ε/k values carry a known systematic low bias for specific chemical classes.

---

## Data Sources

- **Esper**: Gross, J.; Sadowski, G. *Ind. Eng. Chem. Res.* **2001**, *40*, 1244–1260. Parameters from Esper et al. 2023 Figshare dataset.
- **RF**: This project's Random Forest (100 trees, RDKit + Morgan features, trained on 1,801 Esper molecules).
- **GNNePCSAFT**: Wildson et al., PNA architecture, trained on Esper. PyPI: `gnnepcsaft` v0.3.1. HuggingFace: `wildsonbbl/gnnepcsaft`.
- **GC-PC-SAFT**: Simplified group-contribution method based on Sauer et al. 2014 and Gross & Sadowski 2001/2002.
- **ML-SAFT**: Felton, K.; Heid, E. et al. 2024. Programmatically regressed PC-SAFT parameters.
- **SPT-PCSAFT**: Winter, B. et al. *Digital Discovery* **2025**, *4*, 1142–1157.

