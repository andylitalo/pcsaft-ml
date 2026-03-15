# Step 16: Fluorinated Compound Data Expansion

## Purpose

The training data (Esper + ML-SAFT) is dominated by common industrial chemicals — alkanes, alcohols, aromatics, ethers. Novel blowing agent candidates (fluorinated cyclopropanes, cyclobutanes, HFOs) are underrepresented. This step curates additional PC-SAFT parameters for fluorinated compounds from published literature to reduce the OOD rate for screening candidates and improve model accuracy in the chemical space that matters most.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| ~2,500 combined molecules, few fluorinated rings | Additional 50–200 fluorinated compounds with fitted PC-SAFT parameters |
| 8% OOD on test set; likely >30% on HFO candidates | Reduced OOD rate for fluorinated candidates |
| No refrigerant-specific data | HFC/HFO parameters from refrigerant literature |

## Skills Demonstrated

- **Literature data curation**: Systematic extraction of data from published papers
- **Data integration**: Merging heterogeneous sources with consistent schema
- **Domain knowledge**: Understanding which chemical classes are relevant for blowing agent screening

## Dependencies

- Requires: Steps 01, 10 (data loading infrastructure)
- Does not require: any new packages

## Implementation Guide

### 16.1 Curate Refrigerant PC-SAFT Parameters from Literature

Published PC-SAFT parameter sets for HFCs/HFOs/refrigerants are scattered across individual papers. Key sources:

**High priority (parameters published with SMILES or CAS):**
- Gross & Sadowski (2001) *Ind. Eng. Chem. Res.* 40:1244 — original PC-SAFT paper, ~100 compounds including some halocarbons
- Economou et al. (2007) *Fluid Phase Equilib.* 261:343 — PC-SAFT parameters for refrigerants
- Liang et al. (2014) — PC-SAFT for HFC/HFO refrigerants (R-1234yf, R-1234ze, R-32, R-134a, etc.)
- Raabe (2013) *J. Chem. Eng. Data* 58:3212 — PC-SAFT parameters for HFOs
- Papers from the Sadowski group (TU Dortmund) on fluorinated compound thermodynamics

**Medium priority (require SMILES lookup):**
- DIPPR 801 database entries for fluorinated compounds (if academic access available)
- FeOs parameter tables (Rehner & Gross, GitHub: `feos-org/feos`)

**Lower priority (require refitting):**
- NIST TDE experimental VP/density data for fluorinated compounds → refit PC-SAFT parameters using `teqp` or `FeOs`
- DDB (Dortmund Data Bank) entries for fluorinated compounds

### 16.2 Create Supplementary CSV

Create `model/data/fluorinated_pcsaft.csv` with columns: smiles, m, sigma, epsilon_k, name, source (literature reference).

Quality requirements:
- All SMILES must be canonical RDKit SMILES
- Source column must include paper DOI or database identifier
- Parameters must be for the same PC-SAFT variant used in training (non-associating, 3 parameters)
- For associating fluorinated compounds, include only if the non-associating 3-parameter fit is published

### 16.3 Integrate into Data Pipeline

Update `model/data/load.py` to support a `fluorinated` source or include fluorinated CSV in the `combined` loader:

```python
FLUORINATED_CSV = DATA_DIR / "fluorinated_pcsaft.csv"
```

### 16.4 Characterize Chemical Coverage

Report:
- Number of fluorinated rings (cyclopropane, cyclobutane, cyclopentane derivatives)
- Number of HFOs, HFCs, HCFOs
- Tanimoto similarity distribution of new compounds to existing training set
- Expected reduction in OOD rate for screening candidates

## Acceptance Criteria

- [ ] `model/data/fluorinated_pcsaft.csv` exists with ≥ 30 fluorinated compounds
- [ ] Each entry has a literature source reference
- [ ] Combined loader integrates fluorinated data
- [ ] Tanimoto analysis shows improved coverage of screening candidate chemical space
- [ ] Report at `docs/reports/16_fluorinated_data_expansion.md`
