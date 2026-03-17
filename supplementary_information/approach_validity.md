# Why Predict Parameters, Not Properties Directly?

## 1. Physical Consistency

PC-SAFT predictions satisfy thermodynamic identities: Maxwell relations, the Gibbs-Duhem equation, and other constraints that derive from fundamental thermodynamics. When parameters are predicted and fed into the equation of state (EOS), all derived properties are internally consistent by construction.

A black-box property predictor has no such structure. A separate model for boiling point may disagree with one for vapor pressure (e.g., boiling point and vapor pressure that don't satisfy Clausius-Clapeyron). Predictions for density and phase equilibria can violate thermodynamic stability. Such inconsistencies violate thermodynamics and undermine trust in downstream screening.

## 2. Mixture Behavior

PC-SAFT parameters connect to mixture thermodynamics through Lorentz-Berthelot combining rules: \(\epsilon_{ij}/k = \sqrt{(\epsilon_i/k)(\epsilon_j/k)}\). Henry's constant depends on the residual chemical potential through a Boltzmann factor: \(H \propto \exp(-\epsilon_{ij}/kT)\). This link enabled the Henry's constant analysis that quantitatively closed the project's screened question: within the enumerated halogenated-olefin space, no pure HFO behaves as a thermodynamic drop-in for cyclopentane.

A direct boiling-point predictor would miss this exponential sensitivity entirely. It yields a number but no insight into dissolution in polyol or foam behavior.

## 3. Multi-Level Validation

Parameters can be validated at multiple independent levels:

- **Parameter accuracy**: R² and MAE against experimental fits
- **Property accuracy**: boiling point MAE against known values (8.2 K for RF)
- **Screening utility**: do candidates pass physically reasonable filters?
- **Mixture behavior**: are Henry's constant ratios consistent with parameter proximity?

Each level provides independent evidence of model quality.

## 4. Uncertainty Propagation

Errors in predicted parameters can be propagated through the EOS to derived quantities (vapor pressure, density, Henry's constant). Tree-level variance gives per-prediction uncertainty that can be propagated. Local calibration—comparing predictions to structurally similar known molecules—yields the most reliable intervals. A direct property predictor typically outputs only a point estimate.

## 5. The Boltzmann Sensitivity Insight

The core physics insight of this project: Henry's constant depends exponentially on cross-interaction energy:

- A **4%** \(\epsilon_{ij}\) deficit (chlorobutenes) gives \(H/H_{\text{cyclopentane}} \approx 1.05\) (near-identical dissolution)
- A **16%** deficit (HCFO-1233zd) gives \(H/H_{\text{cyclopentane}} \approx 61\) (two orders of magnitude)
- A **22%** deficit (HFO-1234ze) gives \(H/H_{\text{cyclopentane}} \approx 29{,}000{,}000\) (seven orders of magnitude)

This cliff—not a slope—in dissolution behavior is only visible through the parameter → EOS → mixture property pipeline. A direct H predictor would give a number but no understanding of why the number is what it is.

---

## Evidence of Approach Validity in This Project

1. RF boiling point MAE = 8.2 K on 15 fluorinated refrigerants (property-level validation)
2. Chlorobutene predictions form a locally benchmarked thermodynamic near-hit across independent quantities (VP ratio, Henry's constant, temperature stability), pending direct experimental confirmation
3. 16× more accurate than SPT-PCSAFT GNN on deployment domain
4. Henry's constant analysis provided quantitative closure within the screened candidate space
5. Cross-class analysis (Step 46) showed supporting evidence for the fluorination–ε/k pattern in Esper experimental data across multiple functional group classes
