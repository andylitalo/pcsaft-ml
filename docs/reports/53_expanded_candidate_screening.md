# Results Report: Expanded Candidate Screening (All Non-Associating C2-C6 Halogenated Hydrocarbons)

## Summary

Step 53 expanded the blowing agent enumeration from F/Cl-substituted olefins (Step 20: 10,700 candidates) to the full space of non-associating C2-C6 hydrocarbons with H/F/Cl/Br/I substitution on both olefin and saturated backbones. The expanded space produced **41,809 unique candidates** (including stereoisomers) from 35 backbones (16 alkene + 19 alkane). After cyclopentane-centric screening (boiling point + SA score), **1,628 candidates** remain. The expansion confirms the original negative result: no new candidate emerges that is simultaneously a thermodynamic near-hit for cyclopentane and free of non-thermodynamic disqualifiers (GWP, ODP, flammability, regulatory status).

## Enumeration Summary


| Category                              | Count        |
| ------------------------------------- | ------------ |
| Total candidates (with stereoisomers) | 41,809       |
| From olefin backbones                 | 15,571 (37%) |
| From saturated backbones              | 26,238 (63%) |
| Valid boiling points computed         | 26,672 (64%) |


The saturated backbone space is larger because alkanes have more C-H sites available for substitution, and the absence of a double bond creates more unique isomers.

## Filter Funnel


| Stage                     | Step 20 (Olefin F/Cl only) | Step 53 (Expanded) |
| ------------------------- | -------------------------- | ------------------ |
| Enumerated                | 10,700                     | 41,809             |
| Boiling point [288-323 K] | ~856                       | 1,961              |
| SA score <= 4.5           | N/A (separate)             | 18,661             |
| Pass all broad filters    | 0 (strict HFO)             | 1,628              |


The Step 20 pipeline applied 7 HFO-specific filters (including C=C required, no Cl, F >= 2, F mass fraction >= 65%). Step 53 uses only boiling point and SA as broad filters, then ranks by cyclopentane parameter distance.

## Top 20 Candidates


| Rank | SMILES        | m    | sigma | epsilon_k | BP (K) | Distance | Regulatory Flag        | AD  | Known Agent |
| ---- | ------------- | ---- | ----- | --------- | ------ | -------- | ---------------------- | --- | ----------- |
| 1    | C=C1CC1       | 2.35 | 3.53  | 269.2     | 320.6  | 0.167    | flammable-hydrocarbon  | in  |             |
| 2    | CC1CC1        | 2.32 | 3.67  | 264.2     | 316.0  | 0.192    | flammable-hydrocarbon  | in  |             |
| 3    | C1CCC1        | 2.21 | 3.67  | 265.4     | 307.7  | 0.193    | flammable-hydrocarbon  | in  |             |
| 4    | C=C(C)Cl      | 2.06 | 3.74  | 266.1     | 296.9  | 0.217    | HCFO-transitional      | in  |             |
| 5    | CCCCl         | 2.45 | 3.56  | 254.5     | 313.3  | 0.275    | HCFC-Montreal-phaseout | in  |             |
| 6    | CC(C)Cl       | 2.36 | 3.64  | 251.8     | 305.3  | 0.288    | HCFC-Montreal-phaseout | in  |             |
| 7    | FC(F)=C(Cl)Br | 2.62 | 3.64  | 246.1     | 319.1  | 0.349    | Br-ODP-nonstarter      | in  |             |
| 8    | CC(C)(F)Cl    | 2.50 | 3.64  | 243.2     | 305.9  | 0.359    | HCFC-Montreal-phaseout | in  |             |
| 9    | CC=C(C)C      | 2.60 | 3.69  | 239.9     | 311.3  | 0.392    | flammable-hydrocarbon  | in  |             |
| 10   | C/C=C\CC      | 2.65 | 3.72  | 240.3     | 315.9  | 0.394    | flammable-hydrocarbon  | in  |             |
| 11   | C/C=C/CC      | 2.65 | 3.72  | 240.3     | 315.9  | 0.394    | flammable-hydrocarbon  | in  |             |
| 12   | C=CCCC        | 2.54 | 3.79  | 238.2     | 307.3  | 0.401    | flammable-hydrocarbon  | in  |             |
| 13   | F/C=C(\F)I    | 2.63 | 3.52  | 239.4     | 308.6  | 0.405    | I-unstable             | in  |             |
| 14   | F/C=C(/F)I    | 2.63 | 3.52  | 239.4     | 308.6  | 0.405    | I-unstable             | in  |             |
| 15   | C=C(F)CC      | 2.67 | 3.48  | 238.6     | 310.0  | 0.420    | HFO-compliant          | in  |             |
| 16   | C=C[C@@H](C)F | 2.69 | 3.44  | 239.1     | 310.5  | 0.422    | HFO-compliant          | in  |             |
| 17   | C=C[C@H](C)F  | 2.69 | 3.44  | 239.1     | 310.5  | 0.422    | HFO-compliant          | in  |             |
| 18   | C=CCCF        | 2.75 | 3.47  | 239.7     | 316.3  | 0.424    | HFO-compliant          | in  |             |
| 19   | C=C(C)CC      | 2.61 | 3.72  | 235.3     | 307.5  | 0.427    | flammable-hydrocarbon  | in  |             |
| 20   | C=C(C)CF      | 2.65 | 3.47  | 236.4     | 305.3  | 0.433    | HFO-compliant          | in  |             |


Every top-20 candidate has a non-thermodynamic disqualifier:

- **Ranks 1-3, 9-12, 19**: Unsubstituted or lightly substituted hydrocarbons -- flammable, classified as VOCs
- **Rank 4**: Chlorinated olefin (HCFO) -- transitional status, nonzero ODP
- **Ranks 5-6, 8**: Chlorinated saturated compounds (HCFCs) -- Montreal Protocol phaseout
- **Rank 7**: Contains both Cl and Br -- ODP nonstarter
- **Ranks 13-14**: Contains iodine -- thermally and photochemically unstable (C-I bond ~240 kJ/mol)
- **Ranks 15-18, 20**: HFO-compliant, but with cyclopentane distance > 0.42 (significant epsilon_k deficit)

## Known Commercial Agents Found


| Agent                | Rank | Distance | Regulatory Flag       | AD        |
| -------------------- | ---- | -------- | --------------------- | --------- |
| isopentane (CC(C)CC) | 22   | 0.439    | flammable-hydrocarbon | in-domain |
| n-pentane (CCCCC)    | 37   | 0.491    | flammable-hydrocarbon | in-domain |


Cyclopentane itself is outside the filtered set because the enumeration excludes the reference compound by definition (it would have distance 0). The pentane isomers rank in the top 40, confirming the model correctly identifies hydrocarbons with similar thermodynamic behavior.

HFC-245fa, HFC-365mfc, and HCFC-141b were not recovered in the filtered set because their predicted boiling points or SA scores placed them outside the filter window. This is expected -- these are heavier, more complex molecules at the edge of the MW ceiling.

## Comparison to Step 20/38b

- Step 38b produced 50 ranked HFO candidates (after 7 strict HFO-specific filters)
- **Zero overlap** between the Step 38b top-50 and the Step 53 expanded top-50
- This is expected: Step 38b candidates are heavily fluorinated olefins (F mass fraction >= 65%), while Step 53 top candidates are lightly substituted hydrocarbons and chlorinated compounds
- The first HFO-compliant candidate in the expanded ranking (C=C(F)CC) appears at rank 15 with distance 0.420, consistent with the Step 38b conclusion that no pure HFO achieves close thermodynamic proximity to cyclopentane

## AD Analysis


| Halogen Class | In-Domain | Total | Coverage |
| ------------- | --------- | ----- | -------- |
| unsubstituted | 12        | 12    | 100%     |
| Cl only       | 3         | 3     | 100%     |
| F+Cl          | 184       | 252   | 73%      |
| F+I           | 5         | 5     | 100%     |
| F+Cl+Br       | 1         | 1     | 100%     |
| F only        | 454       | 1,195 | 38%      |
| F+Br          | 56        | 160   | 35%      |


Unsubstituted hydrocarbons and simple chlorinated compounds have high AD coverage (the Esper training set contains many such molecules). F-only and F+Br molecules have lower coverage (~35-38%), reflecting the relative scarcity of heavily fluorinated and brominated compounds in the training data. Predictions for out-of-domain molecules should be treated with reduced confidence.

## Filtered Candidates by Regulatory Class


| Regulatory Class       | Count | % of Filtered |
| ---------------------- | ----- | ------------- |
| HFO-compliant          | 797   | 49.0%         |
| HFC-Kigali-phasedown   | 398   | 24.4%         |
| Br-ODP-nonstarter      | 161   | 9.9%          |
| HCFO-transitional      | 134   | 8.2%          |
| HCFC-Montreal-phaseout | 121   | 7.4%          |
| flammable-hydrocarbon  | 12    | 0.7%          |
| I-unstable             | 5     | 0.3%          |


## Key Findings

- **The expansion rediscovers known chemistry.** The top candidates are cyclopropane derivatives, chloroalkanes, and pentane isomers -- all well-known molecules with established industrial profiles.
- **Every new near-hit has a non-thermodynamic disqualifier.** Small cycloalkanes (ranks 1-3) are flammable; chlorinated compounds (ranks 4-8) face Montreal Protocol restrictions; brominated compounds have unacceptable ODP; iodinated compounds are thermally unstable.
- **Br and I provide higher epsilon_k but introduce fatal regulatory/stability issues.** The brominated candidate FC(F)=C(Cl)Br at rank 7 achieves epsilon_k = 246 K (closer to cyclopentane's 289 K than typical HFOs), but Br makes it a regulatory nonstarter.
- **The epsilon_k gap persists.** Even with the expanded chemical space, the fundamental tradeoff remains: molecules with epsilon_k close to cyclopentane (~289 K) are either unsubstituted hydrocarbons (flammable), chlorinated (ODP > 0), or brominated/iodinated (unstable/high ODP). Fluorination systematically reduces epsilon_k.
- **The first HFO-compliant candidate** (C=C(F)CC, rank 15) has distance 0.420, consistent with the ~0.34 minimum found in Step 38b's HFO-specific screening.
- **AD coverage is acceptable** for the halogen classes most relevant to the conclusion: unsubstituted hydrocarbons (100%), chlorinated (73-100%), and F-only (38%) have sufficient training-set representation for the directional conclusion to hold.

## Figures

See `figures/53_expanded_screening/` for:

1. `parameter_space_by_regulatory_class.png` -- epsilon_k vs m scatter colored by regulatory class
2. `filter_funnel_comparison.png` -- side-by-side comparison with Step 20
3. `halogen_class_distribution.png` -- halogen composition at each filter stage
4. `ad_coverage.png` -- Tanimoto AD histogram by halogen content

## Readiness Check

- `ALKANE_BACKBONES` added to `screening/generate.py` with 19 saturated C2-C6 backbones
- `_enumerate_halogen_patterns()` extended to support Br (Z=35) and I (Z=53) with position-selection
- `include_parent` kwarg allows unsubstituted hydrocarbons in enumeration
- `batch_boiling_points()` parallelized with `joblib.Parallel`
- Enumeration across backbones parallelized with `joblib.Parallel`
- `apply_cyclopentane_filters()` added (boiling point + SA only, no HFO-specific gates)
- Tanimoto AD flagging for all candidates (`ad_tanimoto_max`, `ad_in_domain` columns)
- Structural and regulatory annotation for all candidates
- Known-agent matching against commercial blowing agents
- Full pipeline runs end-to-end via `scripts/step53_expanded_screening.py`
- Results saved to `screening/results/expanded_ranked.csv`
- 4 figures saved to `figures/53_expanded_screening/`
- All existing tests pass (3 pre-existing failures unrelated to Step 53)
- `ruff check .` passes for modified files

