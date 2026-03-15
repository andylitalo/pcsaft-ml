# Step 19: Temperature Sweep Validation

## Purpose

All current thermodynamic validation is at 298.15 K. Blowing agent performance depends on behavior across the polyurethane foam operating range (230–330 K). This step extends the Tier 2 validation to a temperature sweep, checking whether the relative ranking of candidates is stable across the operating range.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| Single-temperature validation (298.15 K) | VP and density computed at 230, 260, 290, 320, 330 K |
| Unknown if rankings shift with temperature | Rank stability analysis across T range |

## Implementation Guide

### 19.1 Temperature Sweep

For the top-50 candidates from Step 09:
1. Compute VP and density at T = [230, 250, 270, 290, 310, 330] K
2. Compute property-space distance to cyclopentane at each T
3. Re-rank by property distance at each T

### 19.2 Rank Stability Analysis

- Compute Spearman ρ between rankings at T=298 K and each other temperature
- Identify candidates whose rank changes by more than 10 positions
- If rankings are stable (ρ > 0.8), single-T validation is sufficient for screening

### 19.3 Phase Diagram Check

For the top-5 candidates, plot VP vs T (Clausius-Clapeyron) and compare slope to cyclopentane. Blowing agents need similar VP-temperature sensitivity, not just similar VP at one point.

## Acceptance Criteria

- [ ] VP computed at 6 temperatures for top-50 candidates
- [ ] Rank stability Spearman ρ reported
- [ ] VP vs T plot for top-5 candidates + cyclopentane reference
- [ ] Report at `docs/reports/19_temperature_sweep_validation.md`
