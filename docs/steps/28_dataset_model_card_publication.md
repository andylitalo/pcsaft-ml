# Step 28: Predictive Library and Model Card Publication

## Objective

Package the 4,612+ novel PC-SAFT parameter predictions and all trained models with
standardized metadata so they can be cited, reproduced, and reused by the
thermodynamic modeling community.

This step must clearly distinguish between:
- measured or literature-fitted data
- ML-predicted PC-SAFT parameters
- EOS-derived properties computed from predicted parameters
- heuristic screening metrics

Do **not** present the release artifact as a reference-quality experimental
dataset. It is a **model-generated predictive library** intended for screening,
benchmarking, and follow-up prioritization.

## Motivation

The project produces two distinct publishable assets:

1. **A novel predictions library**: 4,612 fluorinated/chlorinated cycloalkene PC-SAFT
   parameter predictions, none of which exist in published datasets. Plus the HFO-centric
   screening results from Step 25 with boiling points and Henry's constants.

2. **Trained ML models**: GNN (R² > 0.70 on all 3 params), RF, ChemBERTa, ensemble —
   each with quantified uncertainty and applicability domain checking.

Neither asset is useful without proper documentation of provenance, limitations,
intended use, and citation information. The ML community has converged on two
standard formats for this: **Datasheets for Datasets** (Gebru et al. 2021) and
**Model Cards** (Mitchell et al. 2019, adopted by Hugging Face).

For this project, the datasheet must explicitly acknowledge that the main tabular
artifact is a **predicted library**, not an observed dataset in the usual
scientific sense.

## Dependencies

- Steps 20-23 (novel predictions CSV): `model/saved/novel_pcsaft_predictions.csv`
- Steps 24-25 (HFO screening results): `screening/results/hfo_centric_ranked.csv`
- Step 02b (GNN model): `model/saved/gnn_pcsaft.pt`
- Step 01 (RF models): `model/saved/rf_*.joblib`
- Step 04 (ChemBERTa): `model/saved/chemberta/`
- Step 26 (`pcsaft-predict` package): for reporting package-level capabilities

## Implementation Guide

### 28.1 Standardize the novel predictions CSV

Create `scripts/step28_prepare_dataset.py` that reads the existing CSVs and
produces a publication-ready predictive library at
`data/pcsaft_novel_predictions_v1.csv`.

In all public-facing text, describe this file as a **model-generated predictive
library** or **predicted screening library**, not as an experimental reference
dataset.

Required columns (in this order):

| Column | Type | Description |
|--------|------|-------------|
| smiles | str | Canonical SMILES (RDKit) |
| inchi | str | InChI string |
| inchikey | str | InChIKey (for deduplication) |
| name | str | IUPAC name (from RDKit, or empty) |
| mol_class | str | hfo / hcfo / cl_olefin |
| backbone | str | Parent alkene backbone SMILES |
| mw | float | Molecular weight (g/mol) |
| n_fluorine | int | Number of F atoms |
| n_chlorine | int | Number of Cl atoms |
| n_carbon | int | Number of C atoms |
| has_cc_double_bond | bool | Contains C=C bond |
| parameter_source | str | Source of PC-SAFT parameters, e.g. `rf_prediction` |
| m | float | PC-SAFT segment number (RF prediction) |
| sigma | float | PC-SAFT segment diameter in Angstrom (RF prediction) |
| epsilon_k | float | PC-SAFT dispersion energy in Kelvin (RF prediction) |
| m_std | float | RF tree std for m |
| sigma_std | float | RF tree std for σ |
| epsilon_k_std | float | RF tree std for ε/k |
| tanimoto_nn | float | Max Tanimoto similarity to training set |
| ad_in_domain | bool | Tanimoto ≥ 0.4 |
| sa_score | float | Synthetic accessibility score |
| boiling_point_K | float | Normal boiling point computed from predicted parameters (NaN if EOS fails) |
| vp_298K_Pa | float | Vapor pressure at 298.15 K computed from predicted parameters |
| density_298K_mol_m3 | float | Liquid molar density at 298.15 K computed from predicted parameters |
| henrys_ratio | float | H/H(cyclopentane) in hexane at 298 K, computed from predicted parameters |
| hfo_distance | float | Weighted Euclidean distance to HFO-1336mzz(Z) |
| cyc_distance | float | Weighted Euclidean distance to cyclopentane |

Add a header comment row (line starting with `#`) documenting: prediction date,
model version, training dataset, and a DOI/URL placeholder.

The header comment must also state, in plain language:
- this file is a model-generated predictive library
- PC-SAFT parameters are predicted, not experimentally measured
- thermodynamic properties are EOS-derived from predicted parameters
- the file is intended for screening and prioritization, not direct engineering use

### 28.2 Write the Datasheet (`data/DATASHEET.md`)

Follow the Datasheets for Datasets template (Gebru et al. 2021). Sections:

**Motivation**
- Purpose: provide model-generated PC-SAFT parameter hypotheses for fluorinated/chlorinated cycloalkenes
  where no experimental or literature-fitted parameters exist
- Creators: this project
- Funding: none (personal project)

**Composition**
- 4,612+ instances (molecules)
- Each instance: molecular identity + predicted PC-SAFT params + uncertainty + AD + derived properties
- No personally identifiable information
- Molecules generated by systematic combinatorial enumeration (Step 20)

**Collection Process**
- Molecular structures: algorithmic enumeration from 16 alkene backbones
- PC-SAFT parameters: predicted by RF model trained on Esper dataset (1,801 molecules)
- Derived properties: computed via teqp equation of state from predicted params
- No human annotation

**Preprocessing**
- RDKit sanitization and canonical SMILES deduplication
- SA score filtering (≤ 4.5)
- MW filtering (< 200 Da)

**Uses**
- Intended: screening and ranking fluorinated blowing agent candidates; identifying
  structural motifs that preserve dispersion energy; active learning prioritization
  for experimental measurement
- Not intended for: final material selection without experimental validation;
  predicting properties of associating compounds; quantitative Henry's constant
  estimation (k_ij = 0 assumption)
- Not intended for: use as a reference-quality ground-truth dataset

**Distribution**
- License: CC-BY-4.0 (data) / MIT (code)
- Format: CSV with header comment

**Maintenance**
- Updated when model is retrained or new experimental data is available
- Contact: GitHub issues

### 28.3 Write Model Cards

Create one model card per published model in `docs/model_cards/`:

- `docs/model_cards/gnn.md`
- `docs/model_cards/rf.md`
- `docs/model_cards/ensemble.md`

Follow the Hugging Face Model Card template. Each card contains:

**Model Details**
- Name, version, type (GNN / RF / ensemble)
- Architecture summary (e.g., "4-layer GIN with GINEConv, 256 hidden dim,
  residual connections, BatchNorm, Dropout(0.1), 964K parameters")
- Input: SMILES string
- Output: m, σ (Å), ε/k (K), plus per-parameter uncertainty
- Framework: PyTorch + PyTorch Geometric (GNN), scikit-learn (RF)

**Intended Use**
- Primary: predict non-associating PC-SAFT parameters for organic molecules
- Secondary: screen candidate molecules by thermodynamic similarity to a reference
- Out of scope: associating compounds (OH, NH, COOH), ionic liquids, polymers,
  inorganic compounds

**Training Data**
- GNN "all": 13,764 molecules (Esper + ML-SAFT + SPT-PCSAFT)
- GNN "combined": 1,926 molecules (Esper + ML-SAFT experimental only)
- RF: 1,801 molecules (Esper only)
- Cite sources: Esper et al. 2023, Felton et al. 2024, Winter et al. 2025
- State explicitly which labels are experimentally fitted, literature-regressed,
  or model-generated, and do not blur those categories

**Evaluation Results**

Table of R², MAE, RMSE for each target parameter on the test set. Include both
"combined" and "all" data results for GNN.

**Limitations and Biases**
- Training data dominated by common industrial chemicals; fluorinated ring systems
  underrepresented (~10.7% of Esper dataset)
- ε/k predictions are screening-quality (R² = 0.33 RF, 0.73 GNN-all), not
  reference-quality
- 0% of the 4,663 HFO/HCFO candidates fall in-domain (Tanimoto ≥ 0.4)
- GNN "all" dataset includes SPT-PCSAFT model-predicted parameters (not experimental)
- 3-parameter PC-SAFT only; no association parameters predicted

**Ethical Considerations**
- Predictions should not be used for safety-critical decisions without experimental
  validation (toxicity, flammability, environmental fate are not predicted)

### 28.4 Add BibTeX citation

Create `CITATION.cff` (Citation File Format, supported by GitHub) at repo root:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "pcsaft-predict: ML-driven PC-SAFT parameter prediction"
version: 1.0.0
date-released: 2026-XX-XX
url: "https://github.com/<org>/pcsaft-predict"
type: software
authors:
  - family-names: "<last>"
    given-names: "<first>"
keywords:
  - PC-SAFT
  - equation of state
  - molecular property prediction
  - graph neural network
  - thermodynamics
license: MIT
```

Also add a BibTeX block to the README:

```bibtex
@software{pcsaft_predict_2026,
  title={pcsaft-predict: ML-driven PC-SAFT parameter prediction},
  author={<author>},
  year={2026},
  url={https://github.com/<org>/pcsaft-predict},
  version={1.0.0}
}
```

### 28.5 Tests

1. `test_dataset_csv_columns`: load `data/pcsaft_novel_predictions_v1.csv`, verify
   all required columns present and no duplicate InChIKeys
2. `test_dataset_no_null_smiles`: no null SMILES in the predictive library
3. `test_dataset_parameter_ranges`: m > 0, σ > 0, ε/k > 0 for all non-NaN rows
4. `test_model_card_files_exist`: all 3 model cards exist and have required sections

## Key Outputs

| Artifact | Path |
|----------|------|
| Publication-ready predictive library | `data/pcsaft_novel_predictions_v1.csv` |
| Datasheet | `data/DATASHEET.md` |
| GNN model card | `docs/model_cards/gnn.md` |
| RF model card | `docs/model_cards/rf.md` |
| Ensemble model card | `docs/model_cards/ensemble.md` |
| Citation file | `CITATION.cff` |
| Dataset preparation script | `scripts/step28_prepare_dataset.py` |
| Tests | `tests/test_dataset_publication.py` |
| Report | `docs/reports/28_dataset_model_card_publication.md` |

## Acceptance Criteria (When to Move On)

- [ ] `data/pcsaft_novel_predictions_v1.csv` exists with all 26 required columns
- [ ] No duplicate InChIKeys in the predictive library
- [ ] `data/DATASHEET.md` follows Datasheets for Datasets format with all sections
- [ ] Model cards exist for GNN, RF, and ensemble in `docs/model_cards/`
- [ ] Each model card has: Model Details, Intended Use, Training Data, Evaluation Results, Limitations
- [ ] Public-facing text describes the CSV as a model-generated predictive library, not a ground-truth reference dataset
- [ ] Header comment explicitly separates predicted parameters from EOS-derived properties
- [ ] Model cards explicitly separate experimental/literature/model-generated training-label provenance
- [ ] `CITATION.cff` exists at repo root
- [ ] BibTeX block in README
- [ ] Dataset preparation script is reproducible (`python scripts/step28_prepare_dataset.py`)
- [ ] Tests pass (≥ 4 tests)
- [ ] Report at `docs/reports/28_dataset_model_card_publication.md`
