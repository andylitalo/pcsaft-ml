# The Bitter Lesson in ML for Chemistry

Supplementary note on the tension between learned representations and hand-crafted features in molecular property prediction, with evidence from this project.

---

## The Bitter Lesson in Brief

In a widely cited 2019 essay, Rich Sutton argued that the biggest lesson from 70 years of AI research is that **general methods that leverage computation** (search and learning) ultimately win over methods that leverage human knowledge. Hand-crafted features, expert rules, and domain-specific heuristics may dominate early, but as compute and data scale, methods that learn from data surpass them.

In machine learning terms: models that learn representations directly from data tend to outperform models built on hand-crafted features, given sufficient data and compute. The bitter part is that human expertise, however valuable for framing the problem, is often superseded by brute-force learning when the latter has enough resources.

---

## Evidence From This Project

### Where the Bitter Lesson Holds

**Graph representations outperform fingerprints on ε/k.** When trained on the same ~1,900 Esper molecules, GNN message-passing (chemprop D-MPNN, GINEConv) consistently outperformed fingerprint-based Random Forest on ε/k: chemprop R²(ε/k)=0.39 vs RF 0.33 (+0.06), GINEConv 0.41 vs 0.33 (+0.08). The message-passing architecture captures segment interactions—the physical basis of dispersion energy—better than hashed Morgan fingerprints.

**Data scale drives large gains.** With roughly 7× more training data (unified dataset: Esper 1,801 + ML-SAFT 870 + SPT-PCSAFT 13,646 ≈ 13,764 unique molecules after deduplication), the GNN's R²(ε/k) leapt from 0.41 (combined Esper+ML-SAFT) to 0.73 (all data)—+0.40 R² relative to the Esper-only RF baseline (0.33). On the same limited data, architectural differences between RF (0.33), XGBoost (0.33), chemprop (0.39), and GNN (0.41) produced only ±0.08 R².

**Data–architecture interaction.** Below ~5,000 molecules, architectural choice yields modest improvements. Above that scale, learned representations dominate: the GNN on the expanded dataset outperformed every Esper-only model by a wide margin on all three PC-SAFT parameters.

**Hand-crafted group-contribution fails.** GC-PC-SAFT (fully hand-crafted group additivity) achieved R²(ε/k) = −0.04 and MAE(ε/k) = 44.6 K on the Esper test set (n=361)—worse than predicting the mean, and 67% higher ε/k error than RF (26.8 K). For PC-SAFT parameters, domain knowledge alone, without learning from data, is inadequate.

### Where the Bitter Lesson Does Not (Yet) Hold

**Data quality outweighs quantity and architecture.** RF trained on 1,801 clean experimental molecules achieved boiling point MAE of 8.2 K on 15 fluorinated refrigerants. The GNN trained on 13,764 molecules achieved 133.7 K MAE on the same set—about 16× worse. On HFO-1234yf (actual boiling point 243.7 K), the GNN predicted 446.6 K—a +203 K error. The screening filter window (288–323 K target, 35 K wide) cannot function with 133 K MAE. The GNN has ~195K parameters; Esper contains only ~210 fluorinated molecules—insufficient signal for the model to learn effective representations. The decisive factor was the quality and representativeness of the training data, not model capacity or dataset size. The GNN's extra data came largely from SPT-PCSAFT, whose parameters differ systematically from literature-optimized fits, and the model failed to generalize to real fluorinated compounds.

**Low-data regime favors hand-crafted features.** RF uses ~2,200 dimensions: Morgan fingerprints (2048-bit) plus RDKit 2D descriptors encoding electronegativity (e.g., EState indices), topology (ring counts, path lengths), and polarity (TPSA, Labute ASA). The GNN (195K parameters, GINEConv) must rediscover these patterns from the graph. With 1,801 molecules and only ~210 fluorinated examples, the GNN lacked sufficient signal for C1–C4 fluorinated refrigerants (out-of-distribution relative to Esper's broader coverage). RF ε/k MAE was 14.3 K vs. GNN 77.6 K on the fluorinated validation set; RF and GNN candidate rankings were uncorrelated (Spearman ρ < 0.3).

**Approximate crossover point.** On combined Esper+ML-SAFT (~2,600 trainable molecules), GNN R²(ε/k)=0.41—only +0.08 over RF. On the full unified set (~11K train), GNN reached 0.73. The crossover is estimated at 5,000–10,000 molecules of clean, representative experimental data; below that, hand-crafted features are more efficient; above it, learned representations dominate.

---

## Where the Bitter Lesson Has Succeeded in Chemistry

Several molecular property prediction tasks now have datasets large enough for end-to-end learning to compete with or surpass physics-based and feature-engineered methods.

1. **Aqueous solubility (logS).** AqSolDB (~10,000 molecules), ESOL, and similar datasets support SMILES-based models that match or exceed traditional physics-based predictions. Multiple Kaggle competitions and benchmarks reflect this.

2. **Lipophilicity (logP).** ChEMBL and PubChem contain millions of measurements. Deep learning models predict logP from SMILES with MAE < 0.5 log units in standard benchmarks.

3. **Drug-like properties (ADMET).** Absorption, distribution, metabolism, excretion, and toxicity datasets of 100K+ molecules from pharmaceutical screening exist. MoleculeNet benchmarks show GNNs competitive with or better than fingerprint-based methods.

4. **Protein–ligand binding affinity.** PDBbind (~20,000 complexes) supports 3D GNNs and transformers approaching practically useful accuracy for drug discovery.

5. **Molecular energy and forces.** ANI-1x (~5M conformations), QM9 (~130K molecules). SchNet, DimeNet, and PaiNN achieve near-chemical accuracy from learned representations without explicit quantum chemistry in the forward pass.

6. **Reaction yield prediction.** USPTO dataset (~1M reactions). Transformers such as Molecular Transformer predict reaction outcomes from SMILES at useful accuracy.

These are well-known benchmarks in the ML-for-chemistry community and are not exhaustive.

---

## Implication for This Project

PC-SAFT parameter prediction sits in a middle ground: the available clean experimental data (~1,800 molecules) is too small for the Bitter Lesson to apply. The main constraint is not algorithm choice but data.

**Implications:**

- Curating larger, clean experimental PC-SAFT datasets would unlock the full potential of GNN-style models. With 5,000–10,000+ high-quality fits, learned representations would likely surpass RF.
- Until then, feature engineering and domain expertise remain the most efficient use of the available data.
- The pipeline design is forward-compatible: the model registry (`model.registry.get_model("gnn")` vs `"rf"`) allows swapping RF for GNN when data catches up, without redesigning serving, screening, or the portal. Production screening currently uses RF (50 candidates pass all filters; ranked list at `screening/results/hfo_rf_ranked.csv`).

The Bitter Lesson applies when data and compute are sufficient; for PC-SAFT parameters today, the bottleneck is data.
