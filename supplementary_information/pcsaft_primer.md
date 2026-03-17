# PC-SAFT Primer: Supplementary Information

This document explains the Perturbed-Chain Statistical Associating Fluid Theory (PC-SAFT) equation of state and its parameters for a technical audience working on AI-for-chemistry applications.

---

## 1. What is PC-SAFT?

PC-SAFT is an equation of state that models thermodynamic properties of fluids by representing molecules as **chains of spherical segments** interacting through dispersion forces. Unlike cubic equations of state (e.g., Peng-Robinson), PC-SAFT uses a more physically grounded representation: each molecule is a chain of tangent hard spheres with attraction modeled via a perturbative treatment of the Lennard-Jones potential. This representation captures molecular size, shape (chain length), and intermolecular attraction in a way that generalizes well to mixtures and diverse chemical structures.

The theory was developed by Gross and Sadowski (2001) and has since become widely used in chemical engineering for phase equilibrium, density, vapor pressure, and solubility predictions across hydrocarbons, polar molecules, and fluorinated compounds.

---

## 2. The Three Parameters We Predict

The non-associating PC-SAFT model requires three pure-component parameters per molecule. These are the regression targets for our ML models.

### m (Segment Number)

- **Definition**: Number of spherical segments per molecule
- **Physical meaning**: Governs molecular size and chain length. Higher m corresponds to longer or bulkier molecules.
- **Range**: 1.0 for methane to 5+ for long-chain alkanes
- **Role**: Determines the chain-contribution to free energy and phase behavior; larger m increases the repulsive (hard-sphere) contribution and affects liquid-phase packing

### σ (Segment Diameter, Ångströms)

- **Definition**: Diameter of each spherical segment
- **Physical meaning**: Controls molecular volume and the effective size of the molecule.
- **Range**: Typically 2.5–4.5 Å; generally larger for heavier or bulkier molecules
- **Role**: Sets the hard-sphere diameter used in the reference and perturbation terms; directly influences liquid density and packing efficiency

### ε/k (Dispersion Energy, Kelvin)

- **Definition**: Depth of the pair potential well divided by the Boltzmann constant
- **Physical meaning**: Quantifies intermolecular attraction strength. Higher ε/k means stronger dispersion (van der Waals) interactions.
- **Range**: Typically 100–400 K
- **Role**: Critical for vapor pressure, boiling point, and phase coexistence; dominates mixture thermodynamics through its appearance in cross-interaction energies and the chemical potential

---

## 3. Why Predict Parameters Instead of Properties Directly

Predicting PC-SAFT parameters rather than thermodynamic properties (e.g., vapor pressure, density) offers several advantages.

### Physical Consistency

Parameters propagate through the equation of state in a thermodynamically consistent way. All properties (vapor pressure, liquid density, enthalpy, fugacity, Henry's constant) are derived from the same underlying model, so Maxwell relations and thermodynamic identities are satisfied by construction.

### Mixture Behavior via Combining Rules

Pure-component parameters extend to mixtures through the **Lorentz-Berthelot combining rules**:
- σ_ij = (σ_i + σ_j) / 2
- ε_ij/k = √(ε_i/k · ε_j/k)

Cross-interaction parameters ε_ij and σ_ij determine mixture phase behavior, solubility, and activity coefficients. Two molecules with similar (m, σ, ε/k) will have similar cross-interaction parameters with any solvent—enabling mixture-level screening without requiring experimental binary data for each candidate.

### Multi-Level Validation

The pipeline supports a tiered validation strategy:
1. **Parameter-level**: ML predictions vs. reference fits (e.g., Esper dataset)
2. **Property-level**: EOS-derived vapor pressure and density vs. experiment
3. **Screening-level**: Parameter-space distance as a coarse filter
4. **Mixture-level**: Henry's constant in a model solvent as the final thermodynamic validation

### Uncertainty Propagation

Parameter uncertainties (e.g., from RF tree disagreement or MC dropout) propagate through known EOS equations. Per-tree propagation yields confidence intervals on derived quantities (vapor pressure ratio, Henry's constant ratio) without black-box uncertainty estimation.

### Closure of the Scientific Question

Henry's constant analysis—computable only because we predict parameters—quantitatively closed the project's central screened question: within the enumerated halogenated-olefin space, no pure HFO serves as a drop-in thermodynamic replacement for cyclopentane in rigid polyurethane foam. The Boltzmann-exponential sensitivity of Henry's constant to ε_ij showed that a 22% ε/k deficit translates to roughly 10⁷-fold higher Henry's constant, explaining why commercial HFOs are far from cyclopentane-like in this mixture metric despite matching vapor pressure in some regimes.

---

## 4. Why Only Three Non-Associating Parameters?

We omit association parameters (κ^AB, ε^AB/k) and binary interaction parameters (k_ij) for the following reasons.

### Limited Data for Association Parameters

Only approximately 200–500 compounds in the open literature have fitted association parameters. Gold-standard databases (DDB, DIPPR) that contain such fits are commercially licensed. The Esper dataset does not include association parameters, so our primary training source does not support these regression targets.

### Acceptable Scope

The reference compound (cyclopentane) and most blowing agent candidates (HFOs, HCFOs) are **non-associating**. Cyclopentane has no hydrogen-bond donors or acceptors; fluorinated molecules with only C, F, and H do not form strong H-bonds. For non-polar and moderately polar molecules, the 3-parameter model is adequate.

### Limitations Introduced

- **Strongly polar or H-bonding compounds**: Alcohols, acids, amines, and strongly dipolar molecules require association terms. Their ε/k values in 3-parameter fits conflate dispersion with association energy, making such molecules incomparable to non-associating compounds.
- **Fluorinated molecules**: Strong C–F dipoles are partially conflated with dispersion energy in the 3-parameter model. The fit compensates for missing polarity by inflating ε/k. This contributes to prediction error and parameter-space bias for fluorinated compounds.
- **Accuracy ceiling**: For molecules where polarity or H-bonding dominates, the 3-parameter model has a fundamental accuracy limit.

### Pragmatic Choice

Extending to association parameters would require (a) a different EOS backend (teqp does not support association; `pcsaft` or a custom implementation would be needed), (b) a training set with sufficient labeled association parameters—which does not exist in open data—and (c) a stricter screening criterion (five parameters instead of three), reducing the chance of finding viable candidates. Given that the screening already fails to find a close 3-parameter match to cyclopentane among HFOs, adding association parameters would not improve outcomes for the current use case.

---

## References

- Gross, J. & Sadowski, G. (2001). Perturbed-Chain SAFT: An Equation of State Based on a Perturbation Theory for Chain Molecules. *Ind. Eng. Chem. Res.*, 40(4), 1244–1260.
- Esper, T. et al. (2023). Systematic PC-SAFT parameter regression for 1,842 substances. Primary training dataset.
