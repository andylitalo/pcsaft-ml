# Results Report: EOS Robustness & Property-Space Validation (Step 40, 50 RF Candidates)

## Summary

Step 40 validates the 50 RF-ranked HFO candidates (from Step 39's credibility table) using thermodynamic property calculations from PC-SAFT EOS via teqp. Multi-temperature EOS convergence, vapor pressure and density ratios relative to cyclopentane, and Henry's constants in hexane are computed. Four thermodynamic screening gates filter candidates based on physical plausibility and proximity to cyclopentane's property profile.

## EOS Convergence Statistics

| Convergence Count | Candidates | Fraction |
|-------------------|-----------|----------|
| 0 of 3 temperatures | 0 | 0% |
| 1 of 3 temperatures | 0 | 0% |
| 2 of 3 temperatures | 1 | 2% |
| 3 of 3 temperatures | 49 | 98% |

### Convergence by Credibility Tier

| Credibility | Total | Converge at all 3T | Fraction |
|-------------|-------|--------------------|----------|
| screening_ready | 21 | 21 | 100% |
| screening_with_warning | 28 | 27 | 96% |
| high_extrapolation_risk | 1 | 1 | 100% |

### Cyclopentane Reference Properties

| Temperature (K) | VP (Pa) | Liquid Density (mol/m3) |
|-----------------|---------|------------------------|
| 273.15 | 4786.6 | 11180.1 |
| 298.15 | 16248.4 | 10865.5 |
| 323.15 | 44324.4 | 10548.0 |

## Property-Space Results (Top 20 by Property Distance at 298 K)

| SMILES | HFO Dist | Credibility | VP Ratio 298 | Rho Ratio 298 | Prop Dist 298 | H Ratio | All Gates |
|--------|----------|-------------|-------------|--------------|--------------|---------|-----------|
| F/C=C/C(F)(F)[C@H](F)C(F)(F)F | 0.566 | screening_ready | 2.412 | 0.622 | 1.462 | 2.123 | FAIL |
| F/C=C/C(F)(F)[C@@H](F)C(F)(F)F | 0.566 | screening_ready | 2.412 | 0.622 | 1.462 | 2.123 | FAIL |
| F/C=C\C(F)(F)[C@H](F)C(F)(F)F | 0.566 | screening_ready | 2.412 | 0.622 | 1.462 | 2.123 | FAIL |
| F/C=C\C(F)(F)[C@@H](F)C(F)(F)F | 0.566 | screening_ready | 2.412 | 0.622 | 1.462 | 2.123 | FAIL |
| FC(F)=CCC(F)(F)C(F)(F)F | 0.574 | screening_with_warning | 2.447 | 0.626 | 1.494 | 2.134 | FAIL |
| CC(F)(F)C(=C(F)F)C(F)(F)F | 0.528 | screening_with_warning | 2.551 | 0.573 | 1.609 | 2.916 | FAIL |
| C=C(F)C(F)(F)[C@@H](F)C(F)(F)F | 0.557 | screening_with_warning | 2.711 | 0.632 | 1.750 | 2.561 | FAIL |
| C=C(F)C(F)(F)[C@H](F)C(F)(F)F | 0.557 | screening_with_warning | 2.711 | 0.632 | 1.750 | 2.561 | FAIL |
| C=C(C(F)(F)F)C(F)(F)C(F)F | 0.540 | screening_with_warning | 3.003 | 0.675 | 2.030 | 2.812 | FAIL |
| C/C(F)=C(\F)C(F)(F)C(F)(F)F | 0.545 | screening_with_warning | 3.063 | 0.609 | 2.100 | 3.637 | FAIL |
| C/C(F)=C(/F)C(F)(F)C(F)(F)F | 0.545 | screening_with_warning | 3.063 | 0.609 | 2.100 | 3.637 | FAIL |
| F/C(=C/CC(F)(F)F)C(F)(F)F | 0.530 | screening_ready | 3.148 | 0.644 | 2.178 | 3.452 | FAIL |
| F/C(=C\CC(F)(F)F)C(F)(F)F | 0.530 | screening_ready | 3.148 | 0.644 | 2.178 | 3.452 | FAIL |
| CC(=C(F)F)C(F)(F)C(F)(F)F | 0.495 | screening_with_warning | 3.163 | 0.598 | 2.200 | 4.458 | FAIL |
| F/C(=C/C(F)(F)F)CC(F)(F)F | 0.527 | screening_ready | 3.172 | 0.642 | 2.201 | 3.553 | FAIL |
| F/C(=C\C(F)(F)F)CC(F)(F)F | 0.527 | screening_ready | 3.172 | 0.642 | 2.201 | 3.553 | FAIL |
| C=C(F)C(C(F)(F)F)C(F)(F)F | 0.521 | screening_with_warning | 3.183 | 0.624 | 2.215 | 3.873 | FAIL |
| F/C=C(/F)C(F)(F)CF | 0.439 | screening_with_warning | 3.329 | 0.867 | 2.333 | 3.123 | FAIL |
| F/C=C(\F)C(F)(F)CF | 0.439 | screening_with_warning | 3.329 | 0.867 | 2.333 | 3.123 | FAIL |
| C/C(=C(\F)C(F)(F)F)C(F)(F)F | 0.484 | screening_with_warning | 3.333 | 0.606 | 2.366 | 4.981 | FAIL |

## Henry's Constant Results

- Computed for 50/50 candidates (those converging EOS at 298 K)
- H ratio median: 5.019
- H ratio mean: 20.636
- H ratio range: [2.123, 140.482]
- 33/50 candidates have H ratio in (0.1, 10.0) (within order of magnitude of cyclopentane)

## Thermodynamic Screening Gates

| Gate | Pass | Fail | Pass Rate |
|------|------|------|-----------|
| EOS convergence (all 3T) | 49 | 1 | 98% |
| VP ratio in (0.3, 3.0) at 298 K | 8 | 42 | 16% |
| Property distance < 1.0 at 298 K | 0 | 50 | 0% |
| Henry's ratio in (0.1, 10.0) | 33 | 17 | 66% |
| All gates | 0 | 50 | 0% |

## Key Findings

1. **EOS convergence is high**: 49/50 candidates (98%) converge at all 3 temperatures, indicating the RF-predicted PC-SAFT parameters are physically plausible for most candidates.

2. **Thermodynamic gates**: 0/50 candidates (0%) pass all four thermodynamic screening gates. The most restrictive gate filters candidates whose vapor pressure or Henry's constant deviates too far from cyclopentane.

3. **Parameter-property correlation**: Pearson r = -0.709 between parameter-space distance (HFO distance) and property-space distance at 298 K. 
   This strong negative correlation is counter-intuitive: candidates closer in parameter space actually have *larger* property-space distances. This arises because the HFO distance metric weights epsilon_k heavily, while VP depends nonlinearly on all three PC-SAFT parameters. Thermodynamic validation cannot be replaced by parameter distance alone.

4. **Discordant VP/Henry's candidates**: 0 candidates show discordant behavior (high VP ratio but low Henry's ratio or vice versa). 
   The absence of discordant candidates suggests VP and solubility behavior are directionally consistent across this candidate set.

## Figures

See `figures/40_eos_property_validation/` for:
- `eos_convergence_by_credibility.png` -- Stacked bar: convergence counts by credibility tier
- `vp_ratio_temperature_sweep.png` -- VP ratio vs temperature for top 20 candidates
- `property_distance_vs_hfo_distance.png` -- Parameter distance vs property distance scatter
- `henrys_vs_vp_ratio.png` -- Henry's ratio vs VP ratio, with discordant candidates highlighted

## Readiness Check

- [x] Multi-temperature EOS (273, 298, 323 K) run for all 50 candidates
- [x] VP ratio, density ratio, and property-space distance computed
- [x] Henry's constant in hexane at 298 K computed
- [x] Four thermodynamic screening gates applied
- [x] Results merged with Step 39 credibility labels
- [x] Enriched CSV saved to screening/results/hfo_rf_thermo_validated.csv
- [x] All 4 figures generated
