# Supplementary Information: SPT-PCSAFT Dataset Issues for PC-SAFT Parameter Training

## 1. What is SPT-PCSAFT?

SPT-PCSAFT is a published dataset of 13,646 ML-predicted PC-SAFT parameters generated using a SMILES transformer model. It was introduced by Winter et al. in *Digital Discovery* 2025, **4**, 1142–1157 (DOI: 10.1039/D4DD00077C). The dataset is released under CC-BY-NC-SA 4.0 license.

PC-SAFT (Perturbed-Chain Statistical Associating Fluid Theory) is an equation-of-state model that characterizes molecules by three segment parameters: segment number *m*, segment diameter σ (Å), and dispersion energy ε/*k* (K). These parameters are typically fitted to experimental thermodynamic data (e.g., vapor pressure, density). SPT-PCSAFT instead predicts them directly from molecular structure via a machine-learning model trained on existing fitted parameters.

## 2. The Promise

The SPT-PCSAFT dataset offered a substantial increase in training data for downstream ML models. The baseline Esper dataset contains 1,801 molecules with experimentally fitted PC-SAFT parameters. SPT-PCSAFT provides approximately **7× more** molecules (13,646 vs 1,801).

Training a graph neural network (GNN) on a unified corpus (Esper + ML-SAFT + SPT-PCSAFT; 13,764 molecules after deduplication) yielded dramatic apparent gains:

- **Overall**: R²(ε/*k*) = 0.73 on the pooled test set, up from R² = 0.33 for a Random Forest trained on Esper alone
- **Fluorinated subset**: R²(ε/*k*) = 0.91 on the fluorinated test slice (297 molecules, ~90% SPT-PCSAFT by source)

These metrics suggested that the GNN, trained on the expanded corpus, could reliably predict PC-SAFT parameters for fluorinated refrigerants—the primary product target for blowing agent screening.

## 3. The Systematic Bias Discovered

A source-stratified evaluation exposed a critical flaw. The unified test set is dominated by SPT-PCSAFT-derived data (87.5% of test molecules). Performance differed sharply by data source:

| Test subset | Source | GNN R² (ε/*k*) | RF R² (ε/*k*) |
|-------------|--------|----------------|---------------|
| SPT-PCSAFT (n=2,409) | Model-predicted | **0.90** | 0.01 |
| Esper (n=317) | Experimentally fitted | 0.36 | **0.78** |

The GNN achieved R² = 0.90 on SPT-PCSAFT test data but only R² = 0.36 on Esper test data. Conversely, the RF trained solely on Esper performed excellently on Esper (R² = 0.78) and failed on SPT-PCSAFT (R² = 0.01).

**Parameter distribution analysis** revealed the root cause. Esper fluorinated molecules have mean ε/*k* = 208.7 K; SPT-PCSAFT (all) has mean ε/*k* = 284.4 K. **SPT predicts ε/*k* approximately 75 K higher** than Esper for fluorinated compounds. The GNN, trained on 87% SPT-PCSAFT data, learned to reproduce SPT’s smooth but systematically biased distribution rather than the experimentally anchored Esper distribution.

## 4. External Validation Collapse

The decisive test was external validation on 15 fluorinated refrigerants with independently published, literature-optimized PC-SAFT parameters (Liang 2014, Raabe 2013, Konnova 2014, and others). These reference parameters were fitted to experimental vapor pressure and density data—the intended deployment target for predictions.

| Model | Training data | ε/*k* MAE (K) | Boiling point MAE (K) | R²(ε/*k*) |
|-------|---------------|---------------|------------------------|-----------|
| GNN (unified) | 13,764 molecules (87% SPT) | 77.6 | **133.7** | **-25.08** |
| RF (Esper-only) | 1,801 molecules | **14.3** | **8.2** | -0.47 |

The GNN showed catastrophic failure: boiling point MAE of 133.7 K vs 8.2 K for the RF. R²(ε/*k*) = -25.08 indicates that GNN predictions are far worse than predicting the mean. The RF advantage on boiling point is **16.4×**, despite being trained on 7× fewer molecules.

Example: HFO-1234yf (experimental T*_b* = 243.7 K) was predicted by the GNN at 446.6 K—an error of +203 K. The GNN learned SPT-PCSAFT’s systematic over-estimation; when evaluated against literature-optimized parameters, that learned bias became a massive failure.

## 5. Inter-Dataset Variability Analysis

A separate analysis quantified the inherent variability in PC-SAFT parameter fitting. The Esper and ML-SAFT datasets share **745 molecules** with independently fitted parameters (matched by InChI). Both datasets use experimental thermodynamic data, but different fitting protocols and data sources.

For the 567 non-associating overlap molecules:

| Parameter | Median disagreement | P95 disagreement |
|-----------|---------------------|------------------|
| *m* | 3.2% | **47.6%** |
| σ | 1.2% | 22.4% |
| ε/*k* | 1.9% | **31.5%** |

Median parameter disagreement is 1–3%, but heavy tails push P95 to **31% for ε/*k*** and **48% for *m***. This is the first published quantification of inter-method variability in PC-SAFT parameter fitting for the same molecules. It establishes an irreducible noise floor: different valid fitting procedures can yield substantially different parameters for identical structures.

## 6. Key Lesson

**Data quality > data quantity > model architecture.**

- Expanding training data 7× with SPT-PCSAFT degraded real-world performance because the added data carried systematic bias relative to the deployment target (literature-optimized parameters).
- A simpler model (Random Forest) trained on 1,801 high-quality experimentally fitted parameters outperformed a GNN trained on 13,764 molecules by 16.4× on boiling point prediction for fluorinated refrigerants.
- Model architecture improvements cannot compensate for distribution shift between training labels (SPT-predicted) and deployment targets (literature-fitted). Source-stratified evaluation and external validation on deployment-relevant molecules are essential before promoting any model.

---

## References

- Winter, B. et al., *Digit. Discov.* 2025, **4**, 1142–1157. DOI: 10.1039/D4DD00077C (SPT-PCSAFT dataset).
- Internal reports: `docs/reports/31_unified_gnn_retrain.md`, `docs/reports/38_CRITICAL_FINDING.md`, `docs/reports/38_gnn_fluorinated_validation.md`, `docs/reports/10_mlsaft_integration.md`.
