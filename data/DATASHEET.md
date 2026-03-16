# Datasheet: PC-SAFT Novel Predictions Library v1.0

This datasheet follows the framework proposed by Gebru et al. (2021) "Datasheets for Datasets" to document the PC-SAFT Novel Predictions Library, a **model-generated predictive library** of PC-SAFT equation-of-state parameters for screening candidate refrigerant molecules.

**CRITICAL NOTICE**: This is NOT an experimental dataset. All PC-SAFT parameters (m, σ, ε/k) are **model predictions**, not laboratory measurements. Use only for computational screening and prioritization, not direct engineering applications.

---

## Motivation

**For what purpose was the dataset created?**

This dataset was created to enable rapid computational screening of novel refrigerant candidates by providing PC-SAFT equation-of-state parameters for molecules not present in existing experimental databases. The primary use case is identifying promising blowing agent alternatives to cyclopentane and HFO-1234yf that balance environmental performance, thermodynamic properties, and synthetic accessibility.

**Who created the dataset and on whose behalf?**

Created as part of an ML-driven refrigerant discovery pipeline. The dataset is derived from ensemble machine learning models (Random Forest + Graph Neural Network) trained on experimental PC-SAFT parameters from literature sources.

**Who funded the creation of the dataset?**

Research project funded as a technical demonstration of ML-accelerated molecular property prediction for industrial chemistry applications.

---

## Composition

**What do the instances represent?**

Each instance represents a single organic molecule (defined by SMILES string) with:
1. Molecular identity (SMILES, InChI, InChIKey, IUPAC name where available)
2. Predicted PC-SAFT parameters: m (segment number), σ (segment diameter, Å), ε/k (dispersion energy, K)
3. Uncertainty estimates: standard deviations for each parameter from ensemble predictions
4. Applicability domain flags: Tanimoto similarity to nearest training neighbor, domain flag
5. Molecular descriptors: MW, elemental composition, synthetic accessibility score
6. Derived thermodynamic properties: boiling point, vapor pressure ratio, Henry's constant ratio (where computed)
7. Screening metrics: distance to reference molecules (cyclopentane, HFO-1234yf)

**How many instances are there?**

**4,612+ instances** (after deduplication by InChIKey).

**Does the dataset contain all possible instances or is it a sample?**

This is a **targeted sample** from a larger chemical space. The dataset includes:
- Molecules enumerated from systematic combinatorial generation (halogenated C3-C6 hydrocarbons)
- Molecules selected via similarity search to cyclopentane and HFO-1234yf
- Molecules filtered for synthetic accessibility (SA score < 6) and property constraints

The dataset does NOT represent:
- Exhaustive coverage of all refrigerant chemical space
- Random sampling from any parent distribution
- Complete coverage of any molecular class

**What data does each instance consist of?**

26 fields per molecule:
- **Identity**: `smiles`, `inchi`, `inchikey`, `name` (IUPAC, often empty)
- **Classification**: `mol_class` (e.g., HFO, HCFO, Cl-olefin, cyclopentane-like), `backbone` (e.g., acyclic_C5, cyclopropane_C3)
- **Descriptors**: `mw`, `n_fluorine`, `n_chlorine`, `n_carbon`, `has_double_bond`, `sa_score`
- **Predicted PC-SAFT**: `m`, `sigma`, `epsilon_k` (ALL MODEL-GENERATED)
- **Uncertainty**: `m_std`, `sigma_std`, `epsilon_k_std` (ensemble standard deviations)
- **Provenance**: `parameter_source` (all = "model_predicted")
- **Applicability Domain**: `tanimoto_nn` (nearest neighbor similarity), `ad_in_domain` (boolean flag)
- **Thermodynamics**: `boiling_point_K`, `vp_298K_Pa`, `density_298K_mol_m3`, `henrys_ratio` (EOS-derived, many missing)
- **Screening**: `hfo_distance`, `cyc_distance` (feature-space distances to reference molecules)

**Is there a label or target?**

No. This is a **predictive library**, not a supervised learning dataset. The PC-SAFT parameters are the predicted outputs, not ground-truth labels.

**Is any information missing from individual instances?**

Yes, common missing fields:
- `name`: IUPAC names often unavailable (RDKit does not generate systematic names)
- `backbone`: Only populated for molecules in HFO-centric screening subset (~500 molecules)
- `boiling_point_K`, `vp_298K_Pa`, `density_298K_mol_m3`: Only computed for HFO-centric subset
- `henrys_ratio`: Sparse coverage, depends on EOS calculations
- `hfo_distance`: Only populated for HFO-centric subset

Missing values are represented as `NaN` or empty strings.

**Are relationships between instances made explicit?**

Implicit relationships via:
- `mol_class`: Groups molecules by halogenation pattern
- `backbone`: Groups molecules by carbon skeleton
- `cyc_distance` / `hfo_distance`: Quantifies similarity to reference molecules

No explicit parent-child or reaction network relationships are encoded.

**Are there recommended data splits?**

Not applicable. This is a screening library, not a training dataset.

**Are there any errors, sources of noise, or redundancies?**

**Errors and noise**:
- **Model prediction errors**: All PC-SAFT parameters are predicted, not measured. Typical test-set R² ranges: m (0.62), σ (0.35), ε/k (0.33) for RF baseline; m (0.76), σ (0.77), ε/k (0.73) for GNN.
- **Extrapolation risk**: Many molecules fall outside the training set's applicability domain (see `ad_in_domain` flag).
- **Uncertainty underestimation**: Ensemble uncertainties (`*_std`) may underestimate true error, especially for ε/k.
- **Stereoisomer duplication**: Some molecules appear multiple times as different stereoisomers with identical predicted parameters.

**Redundancies**:
- Stereoisomers with identical SMILES hashes may have separate InChIKeys but identical predictions.
- After InChIKey deduplication, ~50 duplicates removed from original 4,663 rows.

**Is the dataset self-contained?**

Yes. All molecular identifiers and predictions are included in the CSV file.

**Does the dataset contain data that might be considered confidential?**

No. All molecules are hypothetical structures generated computationally. No proprietary formulations or trade secrets are included.

---

## Collection Process

**How was the data collected?**

This is a **model-generated dataset**, not a collection of empirical measurements.

**Process**:
1. **Training data**: Experimental PC-SAFT parameters collected from literature (Esper et al., Thol et al., Dortmund Data Bank) — 13,764 molecules.
2. **Model training**: Random Forest (100 trees) and Graph Neural Network (4-layer GIN, 964K params) trained on experimental data.
3. **Candidate generation**:
   - Systematic enumeration: Halogenated C3-C6 hydrocarbons with up to 2 double bonds
   - Similarity search: MACCS fingerprint neighbors of cyclopentane and HFO-1234yf
   - Filtering: SA score < 6, molecular weight < 200 Da
4. **Prediction**: Ensemble (inverse-variance weighted RF+GNN) generates PC-SAFT parameters for 4,663 novel molecules.
5. **Thermodynamic enrichment**: Subset (HFO-centric) passed through teqp EOS calculations for derived properties.

**Who was involved in the data collection process?**

Automated computational pipeline. No human labeling or curation beyond setting enumeration rules.

**Over what timeframe was the data collected?**

Model training and prediction: January 2025. Training data spans literature from 1990-2024.

**Were any ethical review processes conducted?**

Not applicable — no human subjects or sensitive data involved.

---

## Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling done?**

Yes:
1. **SMILES canonicalization**: All molecules canonicalized via RDKit to ensure consistent representation.
2. **InChI/InChIKey generation**: Computed from canonical SMILES for unique identification.
3. **Deduplication**: Removed duplicate InChIKeys (50 molecules).
4. **Descriptor calculation**: RDKit 2D descriptors, SA score, elemental counts computed.
5. **Applicability domain tagging**: Tanimoto similarity to nearest training-set neighbor computed; `ad_in_domain` flag set for similarity >= 0.4.

**Was the "raw" data saved?**

Yes. Intermediate files retained:
- `model/saved/novel_pcsaft_predictions.csv` (raw model output, 4,663 rows)
- `screening/results/hfo_centric_ranked.csv` (HFO-centric subset with thermodynamics, 500 rows)
- `screening/results/ranked_candidates_systematic.csv` (cyclopentane-like subset, 1,000 rows)

**Is the software used to preprocess the data available?**

Yes. Open-source pipeline in this repository:
- `scripts/step28_prepare_dataset.py` (dataset assembly)
- `model/gnn/predict.py` (GNN predictions)
- `model/baseline.py` (RF predictions)
- RDKit 2023.09.1 (molecular descriptors)

---

## Uses

**Has the dataset been used for any tasks already?**

Yes:
1. **Refrigerant candidate screening**: Identify molecules with boiling point 280-320 K, low GWP (fluorine < 3), high synthetic accessibility.
2. **Thermodynamic validation**: Compare EOS-derived properties to experimental refrigerant data for known molecules.
3. **Model evaluation**: Assess extrapolation behavior of PC-SAFT prediction models.

**What (other) tasks could the dataset be used for?**

- **Inverse design**: Train generative models to propose molecules with target PC-SAFT parameters.
- **Active learning**: Prioritize molecules for experimental synthesis based on uncertainty estimates.
- **Transfer learning**: Fine-tune models for related property prediction tasks (e.g., critical properties, surface tension).
- **Chemical space visualization**: Map structure-property relationships in halogenated hydrocarbon space.

**Is there anything about the composition that might impact future uses?**

**Limitations**:
- **Model bias**: Training set dominated by simple hydrocarbons; predictions for exotic functional groups (e.g., siloxanes, heavy halogens) unreliable.
- **No associating compounds**: All predictions assume non-associating PC-SAFT; alcohols, amines, acids not suitable.
- **Stereoisomer redundancy**: Multiple entries for stereoisomers may inflate apparent diversity.
- **Sparse thermodynamic data**: Only ~500 molecules have boiling point / vapor pressure data.
- **No experimental validation**: Zero molecules in this dataset have been synthesized or measured.

**Are there tasks for which the dataset should not be used?**

**DO NOT USE FOR**:
- Direct engineering design (e.g., selecting refrigerant charge quantities, safety calculations)
- Regulatory compliance or environmental risk assessment
- Physical property databases (present as predictions, not experimental data)
- Training new PC-SAFT prediction models (this is model-generated data — would create circular dependencies)

---

## Distribution

**Will the dataset be distributed to third parties?**

Yes. Dataset released under CC-BY-NC-SA 4.0 as part of public GitHub repository.

**How will the dataset be distributed?**

- **Primary**: CSV file in GitHub repository (`data/pcsaft_novel_predictions_v1.csv`)
- **Metadata**: This datasheet (`data/DATASHEET.md`)
- **Documentation**: Model cards in `docs/model_cards/`

**When will the dataset be distributed?**

Released alongside project completion (March 2026).

**Will the dataset be distributed under a copyright or IP license?**

**CC-BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike). The dataset includes predictions from models trained on SPT-PCSAFT data (CC-BY-NC-SA 4.0), which propagates non-commercial and share-alike requirements. See `LICENSE-DATA` for full terms.

**Do any export controls or regulatory restrictions apply?**

No. Public chemical data, no dual-use concerns, no ITAR/EAR restrictions.

---

## Maintenance

**Who will support/host/maintain the dataset?**

Project maintainer. GitHub repository serves as primary hosting platform.

**How can the owner/curator be contacted?**

Via GitHub Issues in repository.

**Is there an erratum?**

Not yet. Known issues tracked in `docs/decisions.md` and GitHub Issues.

**Will the dataset be updated?**

Versioned updates planned:
- **v1.1**: Add thermodynamic properties (BP, VP, density) for all molecules via batch teqp calculations
- **v2.0**: Re-predict with improved models (e.g., fine-tuned ChemBERTa, larger training set)
- **v3.0**: Expand chemical space (C7-C8 hydrocarbons, sulfur-containing compounds)

Version history will be maintained in `data/CHANGELOG.md`.

**If the dataset relates to people, are there limits on retention?**

Not applicable — dataset contains only hypothetical molecular structures.

**Will older versions be supported?**

Yes. All versions tagged in Git and archived via Zenodo DOI.

**If others want to extend/augment/build on the dataset, is there a mechanism?**

Yes. Pull requests accepted via GitHub. Contributions should:
1. Add new molecules as separate CSV files (not modify v1.0)
2. Document data provenance in commit message
3. Update DATASHEET.md with new version section

---

## References

- Gebru, T., et al. (2021). "Datasheets for Datasets." *Communications of the ACM*, 64(12), 86-92.
- Esper, G., et al. (2017). "PC-SAFT Parameters from Literature." Figshare Collection 6821654.
- This project repository: `docs/model_cards/`, `docs/steps/`, `docs/decisions.md`

---

**Last Updated**: 2026-03-15
**Dataset Version**: v1.0
**Datasheet Version**: 1.0
