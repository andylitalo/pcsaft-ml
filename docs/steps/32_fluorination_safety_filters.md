# Step 32: Fluorination Safety Filters

## Objective

Add three chemically motivated fluorination heuristics to the HFO screening pipeline that go beyond the current heuristic of "F count >= 2". These rules are intended to support exploratory ranking and exclusion decisions for fluorinated hydrocarbon candidates; they are **not** non-flammability guarantees, stability certifications, or substitutes for property testing.

## Motivation

The current HFO screening in `screening/hfo_screening.py` applies five filters, but only one relates to fluorination: `n_fluorine >= 2`. This is insufficient for safety-critical refrigerant screening:

1. **Fluorination content heuristic**: The ASHRAE 34 non-flammability classification for fluorinated compounds often correlates with high fluorine mass fraction. Simply counting fluorine atoms does not account for molecular weight.
2. **Thermal stability preference**: Terminal CF3 groups (-CF3) are often more stable fluorinated motifs than isolated C-F substitution patterns.
3. **Reactive sites**: Certain fluorination patterns are thermally or chemically unstable:
   - Isolated tertiary fluorines on sp3 carbons (no neighboring F to stabilize)
   - Allylic -CHF- or -CH2F adjacent to C=C bonds (risk of HF acid elimination)

These filters are orthogonal to the ML prediction quality and can be applied regardless of which model generates the PC-SAFT parameters. They should remain clearly secondary to the core model-validation and deployment path in Steps 31, 33, and 34.

## Dependencies

- `screening/hfo_screening.py` exists with `apply_hfo_filters()`
- `screening/filters.py` exists with existing filter infrastructure
- RDKit available for SMARTS matching and molecular weight calculation
- No dependency on Step 31 (can run in parallel)

## Implementation Guide

### 32.1 Fluorine mass fraction filter

Add to `screening/filters.py` or `screening/hfo_screening.py`:

```python
from rdkit import Chem
from rdkit.Chem import Descriptors

def fluorine_mass_fraction(smiles: str) -> float:
    """Compute mass fraction of fluorine in a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    mw = Descriptors.ExactMolWt(mol)
    n_fluorine = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    return (n_fluorine * 18.998) / mw if mw > 0 else 0.0

def passes_fluorine_mass_fraction(smiles: str, threshold: float = 0.65) -> bool:
    """Check if molecule has sufficient fluorine mass fraction for non-flammability."""
    return fluorine_mass_fraction(smiles) >= threshold
```

The threshold of 0.65 (65 wt% F) comes from the correlation between fluorine content and ASHRAE 34 A1 classification in commercial refrigerants. This is a screening heuristic, not a rigorous flammability test, and must not be described elsewhere in the roadmap as a "guarantee."

### 32.2 CF3 terminal group detection

```python
CF3_SMARTS = Chem.MolFromSmarts("[CX4](F)(F)F")

def count_cf3_groups(smiles: str) -> int:
    """Count terminal CF3 groups in a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    matches = mol.GetSubstructMatches(CF3_SMARTS)
    return len(matches)

def has_cf3_group(smiles: str) -> bool:
    """Check if molecule has at least one CF3 terminal group."""
    return count_cf3_groups(smiles) > 0
```

CF3 groups should be used as a **ranking bonus** rather than a hard filter. Add a `cf3_count` column to screening results and use it as a tiebreaker when ranking candidates.

Before finalizing the SMARTS, verify that the pattern actually captures a **terminal** CF3 motif rather than any tetrahedral carbon with three fluorines. If necessary, refine the SMARTS or rename the feature to match what is truly detected.

### 32.3 Reactive site ban

Define SMARTS patterns for unstable fluorination motifs, but treat the first implementation as provisional until checked against curated examples or literature-backed cases:

```python
# Isolated tertiary fluorine: sp3 carbon with exactly one F and no other F neighbors
ISOLATED_TERT_F = Chem.MolFromSmarts("[CH0;X4;!$([CH0](F)(F))](F)")

# Allylic -CHF- adjacent to C=C
ALLYLIC_CHF = Chem.MolFromSmarts("[C]=[C][CH1]F")

# Allylic -CH2F adjacent to C=C
ALLYLIC_CH2F = Chem.MolFromSmarts("[C]=[C][CH2]F")

REACTIVE_PATTERNS = [
    (ISOLATED_TERT_F, "isolated_tertiary_fluorine"),
    (ALLYLIC_CHF, "allylic_CHF"),
    (ALLYLIC_CH2F, "allylic_CH2F"),
]

def has_reactive_fluorine(smiles: str) -> tuple[bool, list[str]]:
    """Check if molecule has reactive fluorination sites.

    Returns (has_reactive, list_of_pattern_names).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, []
    found = []
    for pattern, name in REACTIVE_PATTERNS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            found.append(name)
    return len(found) > 0, found
```

This is a **hard filter** only after the SMARTS definitions are manually validated on representative molecules. Log which pattern triggered the exclusion for the report. If the chemistry review shows that a SMARTS pattern is too broad or ambiguous, downgrade it to a warning-only annotation instead of silently excluding molecules.

### 32.4 Integrate into `apply_hfo_filters()`

Add the new filters to the existing cascade in `screening/hfo_screening.py::apply_hfo_filters()`. The order should be:

1. Boiling point range [288, 323] K (existing)
2. Has C=C bond (existing)
3. No chlorine (existing)
4. F count >= 2 (existing)
5. SA score <= 4.5 (existing)
6. **NEW**: Fluorine mass fraction >= 65%
7. **NEW**: No reactive fluorination sites
8. Add `cf3_count` column for ranking (not a hard filter)

Return a stats dict that includes how many candidates were removed by each new filter.

### 32.5 Tests and chemistry sanity checks

Create `tests/test_fluorination_filters.py` with tests for each filter:

**Known-good SMILES** (should pass all filters):
- `FC(F)=C(F)C(F)(F)F` (perfluoro-2-butene: high F fraction, has CF3, no reactive sites)
- `C(=C(F)F)C(F)(F)F` (1,1,3,3,3-pentafluoropropene: CF3 group, high F fraction)

**Known-bad SMILES** (should fail specific filters):
- `C=CC(F)F` (low F mass fraction, allylic -CH2F-like)
- `CC(F)=CC` (isolated tertiary fluorine, low F fraction)
- `C=CCF` (allylic -CH2F)

Test each filter independently and verify the cascade behavior.

In addition to unit tests, include a short chemistry sanity table in the report:

- representative molecules expected to match each SMARTS
- representative near-misses expected **not** to match
- one sentence explaining why each example does or does not match

This guards against a common failure mode where the code is syntactically correct but the SMARTS do not encode the intended chemistry.

### 32.6 Re-run HFO screening

Execute the full screening pipeline with the new filters enabled:

```bash
uv run python scripts/step25_hfo_centric_screening.py
```

Record:
- How many candidates pass the original 5 filters
- How many are removed by the fluorine mass fraction filter
- How many are removed by the reactive site ban
- How many have CF3 groups (ranking bonus)
- Final ranked list size

### 32.7 Report

Write `docs/reports/32_fluorination_safety_filters.md` with:
- Description of each filter and its chemical rationale
- Explicit disclaimer that the filters are heuristic screening rules, not formal flammability or stability certification
- Impact table showing candidate counts before/after each filter
- Example molecules removed by each filter (with SMILES and reason)
- Example molecules that gain CF3 ranking bonus
- Comparison to previous screening results (Step 25)

## Artifacts

| File | Description |
|------|-------------|
| `screening/filters.py` or `screening/hfo_screening.py` | New filter functions |
| `tests/test_fluorination_filters.py` | Filter unit tests |
| `docs/reports/32_fluorination_safety_filters.md` | Step report |

## Success Criteria

- [ ] Fluorine mass fraction filter implemented and tested
- [ ] CF3 detection implemented and tested
- [ ] Reactive site ban implemented with SMARTS patterns and tested
- [ ] SMARTS patterns manually checked against curated positive and negative examples
- [ ] Filters integrated into `apply_hfo_filters()` cascade
- [ ] Screening re-run with impact statistics documented
- [ ] All existing tests pass
- [ ] `ruff check .` passes
- [ ] Report documents each filter's impact on candidate pool and clearly labels the filters as heuristics

## When to Move On

- All three filters pass unit tests with known-good and known-bad SMILES
- SMARTS definitions are chemically defensible enough to avoid obvious false positives/false negatives
- Screening report shows per-filter impact statistics
- No existing tests broken

## Budget

15 minutes. This is primarily SMARTS pattern definition and testing; no model training required.
