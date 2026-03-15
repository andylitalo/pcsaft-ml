# Step 20: Systematic HFO/HCFO Enumeration

## Objective

Replace the hand-curated scaffold + single-substitution approach with exhaustive
combinatorial enumeration of all possible F/Cl substitution patterns on bare
alkene backbones. This generates the complete HFO (hydrofluoroolefin) and HCFO
(hydrochlorofluoroolefin) chemical space for C2–C6 carbon skeletons.

## Motivation

Step 12 (candidate verification) found only two candidates passing all five
criteria, both with severe practical limitations. The root cause is that the
original generation approach (`_generate_halogen_variants`) only substitutes
**one H at a time**, so it cannot reach multi-substituted molecules like
HFO-1234ze (4 fluorines) or HCFO-1233zd (3 fluorines + 1 chlorine) from a bare
propene backbone. The industry-standard blowing agents are precisely these
multi-substituted olefins.

## Implementation

### New backbone set (`ALKENE_BACKBONES`)

16 systematic alkene backbones covering:
- Acyclic C2–C5 (ethene, propene, butene isomers, pentene isomers)
- Endocyclic C3–C6 (cyclopropene through cyclohexene)
- Exocyclic C4–C5 (methylenecyclopropane, methylenecyclobutane)

### New enumeration function (`_enumerate_halogen_patterns`)

For each backbone:
1. Parse SMILES, add explicit H atoms
2. Identify all H atoms bonded to carbon
3. Enumerate all combinations of {H, F} for HFOs and {H, F, Cl} for HCFOs
4. Constraint: at most 1 Cl (HCFOs); at least 1 F or Cl (no parent hydrocarbons)
5. MW filter (< 200 Da) to exclude non-volatile heavy molecules
6. Canonicalize via RDKit, deduplicate

### Integration

- New `"systematic"` scaffold set in `generate_hfo_candidates()`
- CLI flags: `--scaffold-set systematic --max-cl 1 --max-mw 200`
- Backward compatible: `"hfo"`, `"broad"`, `"all"` still work unchanged

## Key Metrics

- **10,700** unique candidates generated (16× the previous 645 from "broad")
- **4,663** pass SA ≤ 4.5 filter
- **All 9** known commercial blowing agents found in the enumerated space
- **3** candidates pass all 5 verification criteria (up from 2)
- **131** HFOs and **605** HCFOs pass VP proximity when parameter proximity relaxed

## When to Move On

- [x] `generate_systematic_candidates()` produces >5,000 unique candidates
- [x] All known commercial HFOs/HCFOs appear in the enumerated space
- [x] Full screening pipeline runs end-to-end with `--scaffold-set systematic`
- [x] 15 new tests pass
- [x] Existing 121 core tests still pass
- [x] Report written with comparison to Step 12 baseline
