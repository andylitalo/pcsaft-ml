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

