# PC-SAFT Background

## What is PC-SAFT?

Perturbed-Chain Statistical Associating Fluid Theory (PC-SAFT) is an equation of state that models the thermodynamic properties of fluids. It represents molecules as chains of spherical segments interacting through dispersion forces.

## Key Parameters

### Segment Number (m)

- Number of spherical segments per molecule
- Higher m → longer/larger molecules
- Typical range: 1.0 (methane) to 5+ (long-chain alkanes)
- Governs molecular size contribution to phase behavior

### Segment Diameter (σ, Angstroms)

- Diameter of each spherical segment
- Typical range: 2.5–4.5 Å
- Controls the molecular volume
- Generally larger for heavier/bulkier molecules

### Dispersion Energy (ε/k, Kelvin)

- Depth of the pair potential well, divided by Boltzmann constant
- Typical range: 100–400 K
- Higher ε/k → stronger intermolecular attractions
- Critical for predicting boiling points and vapor pressures

## Why These Parameters Matter for Foaming

In polyurethane foam production, the blowing agent must:

1. **Vaporize at the right temperature** — governed by vapor pressure (linked to ε/k)
2. **Have appropriate liquid density** — linked to m and σ
3. **Remain in gas phase within cells** — phase behavior from all three parameters

Matching cyclopentane's PC-SAFT parameters means matching its thermodynamic behavior during foam rise and curing.

## Cyclopentane Reference Values


| Parameter | Value  | Unit     |
| --------- | ------ | -------- |
| m         | 2.3655 | segments |
| σ         | 3.7114 | Å        |
| ε/k       | 288.84 | K        |


Source: Gross & Sadowski, Ind. Eng. Chem. Res., 2001.

### Closeness Criteria (rules of thumb, subject to change)

When comparing a candidate's PC-SAFT parameters to cyclopentane:

- **σ**: < 2% relative difference is considered "close"
- **ε/k**: < 5% relative difference is considered "close"

These thresholds are rough empirical estimates. A candidate meeting both criteria would have similar liquid density (via σ) and similar vapor pressure / boiling behavior (via ε/k) to cyclopentane, making it a plausible drop-in replacement for foam blowing.

## Esper Dataset

Esper et al. (2023) regressed PC-SAFT parameters for 1,842 pure substances using 551,172 experimental data points (vapor pressure and liquid density). This is the largest systematically-fitted PC-SAFT parameter set available, with good coverage of:

- Nonpolar molecules (alkanes, cycloalkanes)
- Dipolar molecules (ethers, ketones)
- Associating molecules (alcohols, acids)
- Fluorinated compounds (HFCs, HFOs)

## Literature References

### Core PC-SAFT and Data

- **Gross & Sadowski (2001)**: "Perturbed-Chain SAFT: An Equation of State Based on a
  Perturbation Theory for Chain Molecules." *Ind. Eng. Chem. Res.*, 40(4), 1244-1260.
  Original PC-SAFT EOS and cyclopentane parameter fit used throughout this project.
- **Esper et al. (2023)**: Systematic PC-SAFT parameter regression for 1,842 substances.
  Primary training dataset.
- **Felton et al. (2024)**: "ML-SAFT: A Machine Learning Framework for PCP-SAFT Parameter
  Prediction." *Chem. Eng. J.* 988-molecule curated dataset for ML training of PC-SAFT.
- **Sauer et al. (2014)** / **Gross & Sadowski (2002)**: Group-contribution PC-SAFT methods.
  Basis for the GC-PC-SAFT baseline implemented in Step 01.

### ML for SAFT Parameter Prediction (SOTA context)

- **Habicht et al. (2023)**: "Machine Learning Estimation of PC-SAFT Pure Component
  Parameters from Molecular Structure." Demonstrates GNNs (graph neural networks) on the
  Esper dataset, achieving R-squared ~0.97 (m), ~0.93 (sigma), ~0.85 (epsilon/k). Sets the
  current state-of-the-art benchmark for this exact task. Our project uses fingerprint+RF and
  NN architectures (simpler to implement, better suited to interview scope) but should cite
  these results as the SOTA ceiling.
- **Winter et al. (2022)**: "SPT-NRTL: A physics-guided machine learning model." Demonstrates
  that multi-task learning across thermodynamic parameters with shared molecular representations
  outperforms independent regressors. Justifies our multi-task NN architecture (Step 02).
- **Jirasek et al. (2023)**: "Machine Learning of SAFT-VR Mie Force Field Parameters."
  Shows that transfer learning from large molecular databases improves SAFT parameter
  prediction, directly supporting the ChemBERTa fine-tuning approach (Step 04).

### Blowing Agent Context

- **Calm (2008)**: "The next generation of refrigerants -- Historical review, considerations,
  and outlook." *Int. J. Refrigeration.* Context for HFO selection as low-GWP alternatives.
- **Mota-Babiloni et al. (2017)**: Review of HFO refrigerants and blowing agents as
  next-generation replacements for high-GWP compounds. Supports the HFO screening rationale.
- **HFO-1233zd / HFO-1336mzz**: Honeywell Solstice and Chemours Opteon product families.
  Commercial precedent for HFOs as polyurethane foam blowing agents.

### QSAR Methodology

- **Tropsha (2010)**: Best practices for QSAR model development and validation. Referenced
  for applicability domain methodology.
- **Sahigara et al. (2012)**: Conformal prediction for QSAR. Relevant to our uncertainty
  quantification approach (MC Dropout, RF tree variance).
- **Ertl & Schuffenhauer (2009)**: Synthetic Accessibility Score. Used in screening Stage 2.
  Note: SA Score is known to be less reliable for fluorinated compounds; SCScore (Coley et al.
  2018) is a more modern alternative but requires a trained model.

### Why Not GNNs?

This project uses fingerprint-based and transformer-based approaches rather than GNNs despite
GNNs being SOTA for this task. Rationale:
1. **Scope**: The project demonstrates a progressive ML pipeline (RF -> NN -> Transformer)
   rather than optimizing a single architecture.
2. **Data size**: With ~2,800 molecules (Esper + ML-SAFT), GNNs may not strongly outperform
   simpler architectures. Habicht et al. used data augmentation strategies not replicated here.
3. **Implementation complexity**: GNNs require PyTorch Geometric or DGL, adding significant
   dependency and implementation overhead vs. the interview timeline.
4. **Future work**: A GNN comparison (SchNet, DimeNet, or MPNN) remains a natural extension.

