# Step 21: Henry's Constant Mixture Analysis

## Objective

Validate that PC-SAFT parameter-space proximity implies similar **mixture behavior**
(not just similar pure-component properties) by computing Henry's constants at
infinite dilution using teqp binary PC-SAFT models.

## Motivation

The Spearman ρ = 0.245 between parameter distance and VP distance (Step 20) raised
the question of whether parameter proximity is the right screening metric. The
answer depends on what property we care about:

- **Pure-component VP**: can be matched by compensating parameter combinations
  (different m/σ/ε/k giving similar VP). Parameter proximity is a moderate proxy.
- **Mixture behavior (Henry's constant, solubility)**: governed by Lorentz-Berthelot
  cross-parameters ε_ij = √(ε_i × ε_j) and σ_ij = (σ_i + σ_j)/2. These depend
  **directly** on the pure-component parameters — similar parameters guarantee
  similar cross-interactions with any solvent.

For blowing agents, mixture behavior (dissolution in polyol, phase separation during
foaming, cell nucleation) is the decision-relevant property, not pure-component VP.
Henry's constant at infinite dilution is the simplest quantitative probe of this.

## Implementation

### Henry's constant computation

Using teqp's binary PCSAFTEOS:

1. Create pure solvent model, compute liquid density ρ_L and P_sat via `pure_VLE_T`
2. Create binary model [solute, solvent], set ρ_vec = [ε × ρ_L, (1−ε) × ρ_L]
3. Compute fugacity coefficient: φ_i^∞ = exp(`get_fugacity_coefficients(T, ρ_vec)`[0])
4. H_i = φ_i^∞ × P_sat

### Model solvent

n-Hexane (m = 3.0576, σ = 3.7983 Å, ε/k = 236.77 K) at 298.15 K. A simplification
of real polyol, but the relative rankings (H_candidate / H_cyclopentane) are
insensitive to solvent choice since cross-interaction differences dominate.

### Validation

- VLE self-consistency verified: P_L = P_V at coexistence
- Fugacity balance verified: f_L = f_V for pure solvent
- Mixture pressure at infinite dilution equals P_sat (to numerical precision)

## Molecules Evaluated

1. Cyclopentane (reference)
2. Verified candidates from Step 20 (1-chlorobut-1-ene, 2-chloro-2-butene)
3. Top HFOs by parameter distance (fluorocyclobutene, fluoromethylenecyclopropane, etc.)
4. Top HCFOs by parameter distance
5. HFOs with good VP proximity (relaxed screening)
6. Known commercial agents (HFO-1234ze, HFO-1234yf, HCFO-1233zd, HFO-1336mzz)

## Key Metrics

| Group | H/H(cyclopentane) range | Interpretation |
|-------|------------------------|----------------|
| Verified Cl-olefins | 1.02–1.05 | Near-identical mixture behavior |
| Top HFOs (param dist) | 0.56–1.27 | Within 2× |
| Top HCFOs (param dist) | 0.55–0.65 | ~35-45% less soluble |
| HFOs with VP proximity | 0.90–0.96 | Excellent match despite different ε/k |
| Commercial HFOs | 35–2.9 × 10⁷ | Categorically different |

## When to Move On

- [x] Henry's constant computed for representative molecules across all candidate classes
- [x] Fugacity coefficient convention verified (VLE self-consistency)
- [x] Physical explanation (Boltzmann exponential sensitivity to ε_ij) documented
- [x] Implications for screening strategy documented (4-tier recommendation)
- [x] Limitations acknowledged (model solvent, predicted parameters, no k_ij, single T)
- [x] Report written at `docs/reports/21_henrys_constant_analysis.md`
- [x] `docs/limitations.md` Section 7 updated with actual Henry's constant results
