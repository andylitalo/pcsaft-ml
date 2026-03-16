# Results Report: Fluorinated Safety & Environmental Gate (Step 41, 50 RF Candidates)

## Safety Assessment Methodology

This step applies structural heuristics to screen the 50 RF-ranked HFO/HCFO blowing agent candidates for safety, environmental impact, and toxicity concerns. **All classifications are proxy-based heuristics derived from molecular structure, NOT regulatory determinations.** Real safety assessment requires:

- Flash point and lower flammability limit (LFL) testing
- Limiting oxygen concentration (LOC) measurement
- Cardiac sensitization studies (NOAEL determination)
- Atmospheric chemistry modelling (OH radical rate constants)
- Full ASHRAE 34 classification protocol

### Heuristic Methods

1. **Fluorine mass fraction**: Computed from molecular formula. Higher F content correlates with lower flammability.
2. **GWP proxy**: Based on presence of C=C bonds (enabling OH radical attack) and C-H bonds. HFOs with C=C bonds have atmospheric lifetimes of days (GWP < 10), analogous to HFO-1234yf.
3. **Flammability proxy**: ASHRAE 34 heuristic based on F mass fraction thresholds (>= 65% -> A1, >= 50% -> A2L, >= 30% -> A2, < 30% -> A3).
4. **Toxicity red flags**: Substructure search for epoxides, peroxides, isocyanates, and acid fluorides.
5. **Composite safety score** (0-5, higher = safer): +1 for each of F >= 50%, C=C present, no reactive F, not associating, no tox flags; -1 per reactive pattern found.

### Gate Criteria

- **PASS**: safety_score >= 3 AND gwp_class in [ultra_low, low] AND flammability_class in [A1, A2L]
- **FLAG**: safety_score >= 2 AND gwp_class in [ultra_low, low, medium]
- **FAIL**: everything else

## Results: Safety Profile (Top 20)

| Rank | SMILES | F frac | C=C | React F | Flamm | GWP | Score | Gate | Credibility |
|------|--------|--------|-----|---------|-------|-----|-------|------|-------------|
| 1 | CC(F)(F)C(F)=C(F)F | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | screening_ready |
| 2 | FC(F)=CCC(F)(F)F | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 3 | F/C=C(\F)CC(F)(F)F | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 4 | F/C=C(/F)CC(F)(F)F | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 5 | F/C=C\C(F)(F)C(F)F | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 6 | F/C=C/C(F)(F)C(F)F | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 7 | FCC(F)(F)C=C(F)F | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 8 | FC(F)=C(F)CC(F)F | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | high_extrapolation_risk |
| 9 | FC(F)=CC(F)(F)C(F)(F)F | 0.73 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 10 | FC(F)=C(F)CC(F)(F)F | 0.70 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 11 | F/C(=C/C(F)(F)F)C(F)(F)F | 0.73 | 1 | N | A1 | ultra_low | 5 | PASS | screening_ready |
| 12 | F/C(=C\C(F)(F)F)C(F)(F)F | 0.73 | 1 | N | A1 | ultra_low | 5 | PASS | screening_ready |
| 13 | F/C=C(/F)C(F)(F)CF | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 14 | F/C=C(\F)C(F)(F)CF | 0.65 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 15 | FCC(F)(F)C(F)=C(F)F | 0.70 | 1 | N | A1 | ultra_low | 5 | PASS | screening_ready |
| 16 | CC(F)=C(C(F)(F)F)C(F)(F)F | 0.68 | 1 | N | A1 | ultra_low | 5 | PASS | screening_with_warning |
| 17 | F/C=C(/F)C(F)(F)C(F)F | 0.70 | 1 | N | A1 | ultra_low | 5 | PASS | screening_ready |
| 18 | F/C=C(\F)C(F)(F)C(F)F | 0.70 | 1 | N | A1 | ultra_low | 5 | PASS | screening_ready |
| 19 | FC(F)=C(C(F)(F)F)C(F)(F)F | 0.76 | 1 | N | A1 | low | 5 | PASS | screening_ready |
| 20 | F/C(=C(\F)C(F)(F)F)C(F)(F)F | 0.76 | 1 | N | A1 | low | 5 | PASS | screening_ready |

## Summary Statistics

### Safety Gate Distribution

| Gate | Count | Fraction |
|------|-------|----------|
| PASS | 50 | 100% |
| FLAG | 0 | 0% |
| FAIL | 0 | 0% |

- Mean safety score: 5.00
- Median safety score: 5.0

### GWP Class Distribution

| GWP Class | Count |
|-----------|-------|
| ultra_low | 46 |
| low | 4 |
| medium | 0 |
| high | 0 |

### Atmospheric Lifetime Classification

| Lifetime Class | Count |
|----------------|-------|
| very_short | 46 |
| short | 4 |
| moderate | 0 |
| long | 0 |

### Flammability Class Distribution

| Flammability | Count |
|--------------|-------|
| A1 | 50 |
| A2L | 0 |
| A2 | 0 |
| A3 | 0 |

### Reactive Fluorination Sites

- Candidates with reactive F patterns: 0/50 (0%)

### Toxicity Red Flags

- Candidates with toxicity red flags: 0/50
- No candidates triggered any toxicity red flag patterns (epoxide, peroxide, isocyanate, acid fluoride).

**Note**: Absence of these substructural red flags does NOT confirm safety. Most fluorinated olefins have low acute toxicity (cf. HFO-1234yf, NOAEL > 50,000 ppm), but cardiac sensitization is a known concern class for all halogenated hydrocarbons and should be evaluated experimentally for any candidate.

## Key Findings

1. **50 of 50 candidates pass the safety gate** (100%), with 0 flagged for review and 0 failing.

2. **All candidates have C=C bonds**: True. This is expected since they passed the HFO filter in Step 38b, confirming ultra-low GWP potential (atmospheric lifetime of days).

3. **Flammability**: 50 candidates classified as A1 (non-flammable heuristic) and 0 as A2L (mildly flammable). The fluorine mass fraction range determines this distribution.

4. **Reactive fluorination**: 0 candidates have reactive fluorination patterns (allylic CHF or CH2F adjacent to C=C). These could indicate thermal instability but are common in commercial HFOs (e.g., HFO-1234yf has allylic CHF).

5. **No toxicity red flags**: None of the 50 candidates contain epoxide, peroxide, isocyanate, or acid fluoride substructures. This is expected for simple fluorinated olefins.

## Figures

See `figures/41_safety_environmental_gate/` for:
- `safety_score_distribution.png` -- Histogram of safety scores colored by pass/flag/fail gate
- `flammability_vs_gwp.png` -- 2D scatter of flammability class vs GWP class, with point size proportional to candidate count and color representing mean safety score
- `safety_profile_top20.png` -- Horizontal stacked bar chart showing which safety criteria each top-20 candidate passes/fails

## Caveats

**These are structural heuristics, NOT regulatory classifications.** The following limitations apply:

1. **Flammability**: The F mass fraction threshold is a rough correlation. Actual ASHRAE 34 classification requires standardized flash point testing, burning velocity measurement, and heat of combustion data.
2. **GWP**: The atmospheric lifetime proxy assumes OH radical reactivity based on C=C bond presence. True GWP requires atmospheric chemistry modelling with measured OH rate constants, IR absorption cross-sections, and radiative efficiency calculations.
3. **Toxicity**: Substructure screening catches only known-bad patterns. Cardiac sensitization (the primary toxicity concern for halogenated hydrocarbons) cannot be predicted from structure alone and requires in-vivo testing.
4. **Thermal stability**: The reactive fluorination patterns are heuristic. Some commercial HFOs (e.g., HFO-1234yf) contain these patterns and are thermally stable under normal conditions.
5. **Environmental persistence**: Trifluoroacetic acid (TFA) formation from HFO degradation is a growing environmental concern not captured by GWP alone.

## Readiness Check

- [x] Structural safety assessment for all 50 candidates
- [x] GWP and atmospheric lifetime proxy classification
- [x] Flammability classification proxy
- [x] Toxicity red flag screening
- [x] Composite safety score computed
- [x] Safety gate applied: 50 PASS, 0 FLAG, 0 FAIL
- [x] Enriched CSV saved with all safety columns
- [x] 3 figures generated
- [x] Heuristic nature of all classifications documented
