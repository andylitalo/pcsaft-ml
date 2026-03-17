# Why n-Hexane as the Model Solvent for Henry's Constant?

## The Question

The Henry's constant analysis uses n-hexane at 298 K as the solvent, not a polyol or an alcohol. Real polyurethane foam uses polyol blends (polypropylene glycol / polyethylene glycol, MW 200–6,000) as the matrix. Why not use hexanol, butanol, or another alcohol that is structurally closer to a polyol?

---

## 1. The 3-Parameter PC-SAFT Model Cannot Represent Alcohols

The project uses the non-associating 3-parameter PC-SAFT model (m, σ, ε/k). This captures chain length, segment size, and dispersion (van der Waals) attraction — but not hydrogen bonding.

Alcohols, polyols, and any molecule with -OH groups are **strongly associating**: they form hydrogen bonds that dominate their liquid-phase thermodynamics. Representing these interactions requires two additional association parameters (κ^AB, ε^AB/k) that our model does not include and our EOS backend (teqp) does not support.

If hexanol were used as the solvent in the 3-parameter model, its fitted ε/k would **conflate dispersion energy with association energy**. The resulting Henry's constants would be physically meaningless — the model would be trying to represent hydrogen bonding through a parameter that is only supposed to capture van der Waals interactions. From the PC-SAFT primer:

> Alcohols, acids, amines, and strongly dipolar molecules require association terms. Their ε/k values in 3-parameter fits conflate dispersion with association energy, making such molecules incomparable to non-associating compounds.

This is not a choice between a good proxy and a better proxy. It is a choice between a physically grounded calculation (hexane) and a physically broken one (hexanol). See [why association parameters are out of scope](association_parameters_out_of_scope.md) for the full justification.

---

## 2. Why Hexane Specifically?

n-Hexane was chosen because it satisfies all of the following:

1. **Non-associating**: well-described by 3-parameter PC-SAFT with experimentally fitted parameters from the literature (m = 3.058, σ = 3.798 Å, ε/k = 236.8 K)
2. **Liquid at 298 K**: VLE converges reliably at the computation temperature
3. **Well-studied**: one of the most thoroughly characterized molecules in PC-SAFT; parameter uncertainty is negligible
4. **Hydrocarbon**: dispersion-dominated interactions, so the combining rules (Lorentz-Berthelot) are on their firmest theoretical ground

---

## 3. Why the Specific Solvent Matters Less Than It Appears

The screening uses the **ratio** H_candidate / H_cyclopentane, not the absolute Henry's constant. In this ratio, solvent-dependent contributions partially cancel.

The Henry's constant for solute *i* in solvent *s* is:

H_i = φ_i^∞ · P_sat,s

where φ_i^∞ is the fugacity coefficient at infinite dilution and P_sat,s is the solvent's saturation pressure. The fugacity coefficient depends on the cross-interaction energy ε_is through a Boltzmann-like factor. When taking the ratio:

H_candidate / H_cyclopentane = φ_candidate^∞ / φ_cyclopentane^∞

The solvent saturation pressure cancels exactly. What remains depends on how each solute's parameters differ in their interaction with the solvent — dominated by the solute's ε/k through the combining rule ε_is = √(ε_i · ε_s). Since ε_s appears in both numerator and denominator, the ratio is driven primarily by the difference in solute dispersion energies, not by the absolute solvent properties.

This is why a 22% ε/k deficit produces ~10^7-fold higher Henry's constant regardless of the solvent: the exponential Boltzmann sensitivity to ε_ij is a property of the solute, amplified by the thermodynamic framework, not an artifact of the solvent choice.

---

## 4. Sensitivity Analysis: Hexane vs. Decane

The polyol sensitivity analysis (`scripts/polyol_sensitivity.py`) was designed to test this cancellation directly by recomputing Henry's constants with n-decane (m = 4.66, σ = 3.84 Å, ε/k = 243.9 K) — a higher-molecular-weight, higher-ε/k hydrocarbon that is structurally further from hexane while remaining non-associating.

In practice, the decane VLE computation failed at the high chain-length parameters needed for some candidates (teqp non-convergence). This is documented in the orchestrator log: "Polyol sensitivity failed (VLE non-convergent at high m)."

However, the theoretical argument for rank preservation is strong: because the H ratio depends primarily on the solute's ε/k through the combining rule, any non-associating solvent that does not introduce specific chemical interactions (hydrogen bonding, charge transfer) should produce the same relative ranking. The 7-orders-of-magnitude gap between chlorobutenes and HFOs is far too large to be reversed by a solvent change.

---

## 5. What Would Change With a Real Polyol?

A real polyol (PPG, MW 2,000) would introduce:

1. **Association interactions**: hydrogen bonding between polyol -OH groups and the solute. For non-associating solutes like HFOs and chlorobutenes, this contribution is small (no H-bond donors on the solute), so the polyol's self-association affects the absolute H values but not the relative ranking of non-associating solutes.

2. **Combinatorial entropy effects**: the large MW difference between solute (~100 Da) and polyol (~2,000 Da) introduces a Flory-Huggins-like combinatorial contribution to the activity coefficient. This shifts all Henry's constants in the same direction and largely cancels in the ratio.

3. **Non-zero k_ij**: binary interaction parameters for fluorocarbon–polyether pairs can range from −0.05 to +0.10. The k_ij sensitivity analysis (documented in `model/saved/uncertainty_propagated.csv`) sweeps k_ij over this range and shows it shifts absolute H values but does not reverse the order-of-magnitude ranking differences.

The honest limitation: we cannot **prove** that the hexane ranking is identical to the polyol ranking without either (a) extending to 5-parameter PC-SAFT with association ([details](association_parameters_out_of_scope.md)), or (b) comparing against experimental solubility data in actual polyol blends. But the physics of why the ranking should be preserved — solvent cancellation in the ratio, Boltzmann sensitivity dominated by solute ε/k — is well-grounded, and the magnitude of the gaps between candidate classes (1.05× vs. 10^7×) is far too large for any physically reasonable solvent effect to overturn.

---

## 6. Why Not Use COSMO-RS or Another Solvation Model?

COSMO-RS (conductor-like screening model for real solvents) can handle associating solvents directly from quantum-chemical calculations, without fitted association parameters. It would be the natural tool for computing activity coefficients in polyol blends. However:

- COSMO-RS requires DFT-optimized geometries and σ-profiles for each molecule (~hours per compound), defeating the throughput advantage of the ML → PC-SAFT pipeline (~seconds per compound)
- The project's value proposition is demonstrating that ML-predicted PC-SAFT parameters are sufficient for screening — introducing a second thermodynamic framework would complicate the narrative without strengthening the core claim
- COSMO-RS would be appropriate for Tier 4 validation of the final 3–5 shortlisted candidates, not for batch screening of 4,663 molecules

---

## Summary

| Solvent option | Feasible? | Problem |
|---|---|---|
| **n-Hexane** | **Yes** | Not a polyol, but ratio-based ranking is robust |
| Hexanol / butanol | No | Association parameters required; 3-parameter model gives unphysical ε/k |
| PPG polyol (MW 2,000) | No | Association parameters required; teqp does not support association |
| n-Decane | Partially | VLE non-convergence for some candidates; attempted but failed |
| COSMO-RS in polyol | In principle | ~hours/molecule; defeats the ML throughput advantage |

n-Hexane is not the ideal solvent — it is the best solvent the model can handle. The ratio-based comparison mitigates the solvent mismatch by cancelling solvent-specific terms to first order, and the 7-orders-of-magnitude gap between candidate classes provides a large margin against any residual solvent effect.
