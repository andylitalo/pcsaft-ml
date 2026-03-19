# Step 53: Expanded Candidate Screening — All Non-Associating C2–C6 Halogenated Hydrocarbons

## Objective

Expand the blowing agent enumeration from F/Cl-substituted olefins to the full space of non-associating C2–C6 hydrocarbons with H/F/Cl/Br/I substitution. Run a cyclopentane-centric screening on the expanded space, annotate candidates with applicability domain and regulatory status, and determine whether the expansion changes the screening conclusion.

## Motivation

Step 20 enumerated 10,700 F/Cl-substituted olefins and concluded that no pure HFO is a thermodynamic drop-in for cyclopentane. Steps 38b–42 confirmed this through the full screening cascade. But two scope restrictions were imposed a priori:

1. **Only olefin backbones** (C=C required for low GWP). This excluded saturated alkanes and their halogenated derivatives — including known commercial blowing agents like HFC-245fa, HFC-365mfc, and HCFC-141b, as well as the reference class itself (n-pentane, isopentane, cyclopentane).

2. **Only F and Cl substituents**. This excluded Br (which is more polarizable than Cl, potentially yielding higher ε/k) and I (which is highly polarizable but has weak C–I bonds).

Both restrictions were scientifically motivated (atmospheric chemistry, stability, modeling scope). But a completeness argument demands that we check whether relaxing them produces new thermodynamic near-hits. The expected outcome is that every new near-hit has a non-thermodynamic disqualifier (GWP, ODP, flammability, instability, or regulatory status), strengthening the negative result. If any genuinely novel candidate appears, that is also valuable.

### Model applicability for Br/I

The Esper training set contains 67 Br-containing molecules (~3.7%) and 16 I-containing molecules (~0.9%). RDKit 2D descriptors and Morgan fingerprints handle all halogens natively. Predictions for molecules with multiple Br or I atoms will have sparse training-set support and should be flagged via the Tanimoto AD check (threshold 0.4).

## Dependencies

- Requires: Step 20 (`screening/generate.py` with `_enumerate_halogen_patterns()` and `generate_systematic_candidates()`)
- Requires: Step 38b (RF screening pipeline, `screening/hfo_screening.py`)
- Requires: `model/ad_tanimoto.py` (Tanimoto AD, from Step 14)
- No new packages needed (`joblib` is already available via scikit-learn)

## Implementation Guide

### 53.1 — Add saturated backbones to enumeration

In `screening/generate.py`, add a new `ALKANE_BACKBONES` list alongside the existing `ALKENE_BACKBONES`:

```python
ALKANE_BACKBONES: list[str] = [
    # Acyclic — C2
    "CC",                # ethane
    # Acyclic — C3
    "CCC",              # propane
    # Cyclic — C3
    "C1CC1",            # cyclopropane
    # Acyclic — C4
    "CCCC",             # butane
    "CC(C)C",           # isobutane
    # Cyclic — C4
    "C1CCC1",           # cyclobutane
    "CC1CC1",           # methylcyclopropane
    # Acyclic — C5
    "CCCCC",            # pentane
    "CC(C)CC",          # isopentane
    "CC(C)(C)C",        # neopentane
    # Cyclic — C5
    "C1CCCC1",          # cyclopentane
    "CC1CCC1",          # methylcyclobutane
    # Acyclic — C6
    "CCCCCC",           # hexane
    "CC(C)CCC",         # 2-methylpentane
    "CCC(C)CC",         # 3-methylpentane
    "CC(C)(C)CC",       # 2,2-dimethylbutane
    "CC(C)C(C)C",       # 2,3-dimethylbutane
    # Cyclic — C6
    "C1CCCCC1",         # cyclohexane
    "CC1CCCC1",         # methylcyclopentane
]

ALL_BACKBONES: list[str] = ALKENE_BACKBONES + ALKANE_BACKBONES
```

Verify all SMILES parse and canonicalize correctly with RDKit. Ensure no duplicates between `ALKENE_BACKBONES` and `ALKANE_BACKBONES` (there should be none — alkenes all have C=C, alkanes do not).

### 53.2 — Extend halogen enumeration to Br and I

The current `_enumerate_halogen_patterns()` handles Cl atoms via position-selection (`combinations`), then fills remaining sites with H or F via binary `product`. Extend this pattern to Br (Z=35) and I (Z=53) with additional position-selection loops to avoid a 5^n combinatorial blowup:

```python
def _enumerate_halogen_patterns(
    backbone_smi: str,
    *,
    max_cl: int = 1,
    max_br: int = 2,
    max_i: int = 1,
    max_mw: float = 200.0,
    include_parent: bool = False,
) -> set[str]:
    """Exhaustively enumerate halogen substitution patterns on a backbone.

    For each H bonded to a carbon atom, assign H, F, Cl, Br, or I.
    Heavy halogens (Cl, Br, I) are enumerated by position-selection
    (combinations) to avoid 5^n blowup. Remaining sites get H or F
    (binary product, 2^k).
    """
    mol = Chem.MolFromSmiles(backbone_smi)
    if mol is None:
        return set()

    mol = Chem.AddHs(mol)

    h_on_c_indices: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 6:
                h_on_c_indices.append(atom.GetIdx())
                break

    n = len(h_on_c_indices)
    if n == 0:
        return set()

    # Precompute backbone MW for early pruning
    backbone_mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(backbone_smi))
    CL_MW, BR_MW, I_MW = 34.45, 78.9, 125.9  # added MW per substitution (halogen - H)

    results: set[str] = set()

    for n_cl in range(min(max_cl, n) + 1):
        for n_br in range(min(max_br, n - n_cl) + 1):
            for n_i in range(min(max_i, n - n_cl - n_br) + 1):
                # Early MW pruning: even if all remaining are F (lightest halogen)
                min_added = n_cl * CL_MW + n_br * BR_MW + n_i * I_MW
                if backbone_mw + min_added > max_mw:
                    continue

                for cl_sites in combinations(range(n), n_cl):
                    cl_set = set(cl_sites)
                    remaining_after_cl = [i for i in range(n) if i not in cl_set]

                    for br_sites in combinations(remaining_after_cl, n_br):
                        br_set = set(br_sites)
                        remaining_after_br = [i for i in remaining_after_cl if i not in br_set]

                        for i_sites in combinations(remaining_after_br, n_i):
                            i_set = set(i_sites)
                            remaining = [i for i in remaining_after_br if i not in i_set]

                            for f_pattern in product((1, 9), repeat=len(remaining)):
                                n_heavy = n_cl + n_br + n_i
                                n_f = sum(1 for z in f_pattern if z == 9)
                                if not include_parent and n_heavy == 0 and n_f == 0:
                                    continue

                                ed = Chem.RWMol(Chem.Mol(mol))

                                for si in cl_sites:
                                    ed.GetAtomWithIdx(h_on_c_indices[si]).SetAtomicNum(17)
                                for si in br_sites:
                                    ed.GetAtomWithIdx(h_on_c_indices[si]).SetAtomicNum(35)
                                for si in i_sites:
                                    ed.GetAtomWithIdx(h_on_c_indices[si]).SetAtomicNum(53)
                                for si, z in zip(remaining, f_pattern):
                                    if z == 9:
                                        ed.GetAtomWithIdx(h_on_c_indices[si]).SetAtomicNum(9)

                                try:
                                    Chem.SanitizeMol(ed)
                                    if Descriptors.ExactMolWt(ed) > max_mw:
                                        continue
                                    ed_clean = Chem.RemoveHs(ed)
                                    results.add(Chem.MolToSmiles(ed_clean))
                                except Exception:
                                    pass

    return results
```

**Key design decisions**:
- Heavy halogens (Cl, Br, I) are enumerated by position-selection with `combinations()`. With max_cl=1, max_br=2, max_i=1, the outer loops are tiny (0-1 or 0-2 iterations each).
- The inner loop remains binary (H vs F) at 2^k — same as the current implementation.
- MW ≤ 200 Da constraint naturally limits the number of heavy halogens. Br is 80 Da and I is 127 Da, so most backbones can accept at most 1-2 heavy halogens.
- Early MW pruning before entering the `combinations()` loops avoids wasted work.
- `include_parent=True` allows unsubstituted hydrocarbons (e.g., pentane, cyclopentane) into the candidate pool.

Update `generate_systematic_candidates()` to accept the new parameters and pass them through:

```python
def generate_systematic_candidates(
    backbones: list[str] | None = None,
    *,
    max_cl: int = 1,
    max_br: int = 2,
    max_i: int = 1,
    max_mw: float = 200.0,
    include_parent: bool = False,
) -> list[tuple[str, Chem.Mol]]:
    if backbones is None:
        backbones = ALKENE_BACKBONES  # preserve default for backward compat

    # ... rest of function passes new kwargs to _enumerate_halogen_patterns ...
```

### 53.3 — Parallelize the pipeline for speed

The expanded candidate space (estimated 30,000–80,000 molecules) makes the sequential bottlenecks in the current pipeline impractical. Use `joblib.Parallel` (already available via scikit-learn) to parallelize within the script:

**Target 1 — Boiling point computation (primary bottleneck)**: `batch_boiling_points()` in `screening/hfo_screening.py` currently uses a sequential `iterrows()` loop. Each `compute_boiling_point()` call is independent (scipy `brentq` root-finding on teqp VLE). Replace with:

```python
from joblib import Parallel, delayed

def batch_boiling_points(candidates_df, n_jobs=-1):
    rows = [row for _, row in candidates_df.iterrows()]
    bps = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_boiling_point)(row["m"], row["sigma"], row["epsilon_k"])
        for row in rows
    )
    df = candidates_df.copy()
    df["boiling_point_K"] = bps
    return df
```

On a machine with N cores, this achieves ~Nx speedup on the dominant cost. The `verbose=10` flag provides progress output.

**Target 2 — Enumeration across backbones**: `generate_systematic_candidates()` loops over ~36 backbones sequentially. Each backbone's enumeration is independent. Parallelize:

```python
from joblib import Parallel, delayed

all_pattern_sets = Parallel(n_jobs=-1)(
    delayed(_enumerate_halogen_patterns)(
        bb, max_cl=max_cl, max_br=max_br, max_i=max_i,
        max_mw=max_mw, include_parent=include_parent,
    )
    for bb in backbones
)
all_smiles: set[str] = set()
for patterns in all_pattern_sets:
    all_smiles.update(patterns)
```

**Target 3 — SA score computation**: If SA scoring on 50k+ molecules is slow, replace the `.apply()` with a `Parallel` call. Otherwise leave as-is (SA scoring is generally fast).

RF prediction (`predict_pcsaft`) already uses `n_jobs=-1` internally — no change needed.

**Why hardware parallelization, not multi-agent coordination**: PLAN.md file ownership rules assign all `screening/*` files to a single agent. `joblib.Parallel` achieves the same throughput as splitting across agents with 3 lines of code and no branch coordination complexity.

### 53.4 — Broader screening function

Add `apply_cyclopentane_filters()` to `screening/hfo_screening.py`. This applies only the universally applicable filters, with no HFO-specific structural gates:

```python
def apply_cyclopentane_filters(df):
    """Apply broad screening filters for cyclopentane-centric ranking.

    Unlike apply_hfo_filters(), this does NOT require:
    - C=C double bond
    - No chlorine
    - Minimum fluorine count
    - Fluorine mass fraction threshold
    - Reactive fluorination site check

    Only applies:
    1. Boiling point in range [288, 323] K
    2. SA score <= 4.5
    """
    initial_count = len(df)
    stats = {}

    mask_bp = (df["boiling_point_K"] >= 288) & (df["boiling_point_K"] <= 323)
    stats["boiling_point"] = mask_bp.sum()

    if "sa_score" in df.columns:
        mask_sa = df["sa_score"] <= 4.5
    else:
        from rdkit.Contrib.SA_Score import sascorer
        def compute_sa(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10.0
            try:
                return sascorer.calculateScore(mol)
            except Exception:
                return 10.0
        sa_scores = df["smiles"].apply(compute_sa)
        mask_sa = sa_scores <= 4.5
    stats["sa_score"] = mask_sa.sum()

    final_mask = mask_bp & mask_sa
    stats["total"] = final_mask.sum()

    filtered = df[final_mask].copy()

    print(f"\nCyclopentane-centric Filter Results:")
    print(f"  Initial candidates: {initial_count}")
    print(f"  Boiling point [288-323K]: {stats['boiling_point']}")
    print(f"  SA score <= 4.5: {stats['sa_score']}")
    print(f"  Final passing all filters: {stats['total']}")

    return filtered, stats
```

Candidates passing these filters are ranked by `filter_pcsaft_similarity()` (already in `screening/filters.py`) using cyclopentane as the reference with weights `{m: 1, sigma: 2, epsilon_k: 5}`.

### 53.5 — Applicability domain flagging

For each candidate, compute the maximum Tanimoto similarity to the nearest Esper training molecule using the existing `model/ad_tanimoto.py`. Flag molecules with max Tanimoto < 0.4 as "outside applicability domain."

Add two columns to the output DataFrame:
- `ad_tanimoto_max`: max Tanimoto similarity to any training molecule
- `ad_in_domain`: boolean (`ad_tanimoto_max >= 0.4`)

Load the Tanimoto AD model:

```python
import joblib
tanimoto_ad = joblib.load("model/saved/tanimoto_ad.joblib")
```

Or compute directly using the training set fingerprints if the saved model is unavailable. The key function is in `model/ad_tanimoto.py` or `pcsaft_predict/_ad.py`.

### 53.6 — Annotation function

Add a function that annotates each candidate with structural and regulatory flags:

```python
def annotate_candidates(df):
    """Add structural, regulatory, and known-agent annotations."""
    from rdkit import Chem

    pattern_cc = Chem.MolFromSmarts("C=C")
    pattern_cl = Chem.MolFromSmarts("[Cl]")
    pattern_br = Chem.MolFromSmarts("[Br]")
    pattern_i = Chem.MolFromSmarts("[#53]")

    def _annotate_row(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        has_cc = mol.HasSubstructMatch(pattern_cc)
        has_cl = mol.HasSubstructMatch(pattern_cl)
        has_br = mol.HasSubstructMatch(pattern_br)
        has_i = mol.HasSubstructMatch(pattern_i)
        n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
        n_cl = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17)
        n_br = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 35)
        n_i = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 53)
        n_halogens = n_f + n_cl + n_br + n_i

        # Regulatory classification
        if has_i:
            reg = "I-unstable"
        elif has_br:
            reg = "Br-ODP-nonstarter"
        elif has_cl and not has_cc:
            reg = "HCFC-Montreal-phaseout"
        elif has_cl and has_cc:
            reg = "HCFO-transitional"
        elif n_halogens == 0:
            reg = "flammable-hydrocarbon"
        elif not has_cc and n_f > 0:
            reg = "HFC-Kigali-phasedown"
        elif has_cc and n_f > 0:
            reg = "HFO-compliant"
        else:
            reg = "other"

        return {
            "has_double_bond": has_cc,
            "contains_cl": has_cl,
            "contains_br": has_br,
            "contains_i": has_i,
            "n_halogens": n_halogens,
            "regulatory_flag": reg,
        }

    annotations = df["smiles"].apply(_annotate_row).apply(pd.Series)
    return pd.concat([df, annotations], axis=1)
```

Also add known-agent matching against a lookup table of ~15 commercial blowing agents:

```python
KNOWN_AGENTS = {
    "C1CCCC1": "cyclopentane",
    "CCCCC": "n-pentane",
    "CC(C)CC": "isopentane",
    "FC(F)CC(F)(F)F": "HFC-245fa",
    "CC(F)C(F)C(F)(F)F": "HFC-365mfc",
    "CC(F)(F)Cl": "HCFC-141b",
    "F/C=C/C(F)(F)F": "HFO-1234ze(E)",
    "C=C(F)C(F)(F)F": "HFO-1234yf",
    "F/C(=C\\C(F)(F)F)C(F)(F)F": "HFO-1336mzz(Z)",
    "ClC=CC(F)(F)F": "HCFO-1233zd",
    # Add more as appropriate
}
```

Add a `known_agent` column: the commercial name if matched, else empty string.

### 53.7 — Driver script

Create `scripts/step53_expanded_screening.py` that runs the full pipeline:

1. **Enumerate**: Call `generate_systematic_candidates(backbones=ALL_BACKBONES, max_cl=1, max_br=2, max_i=1, max_mw=200, include_parent=True)`. Log candidate count breakdown by backbone type (olefin vs saturated) and halogen composition (F-only, F+Cl, F+Br, F+I, Cl-only, Br-only, etc.).

2. **Predict**: Run RF `predict_pcsaft()` on all candidates. This is already batch-optimized with `n_jobs=-1`.

3. **Boiling points**: Call parallelized `batch_boiling_points()` with `n_jobs=-1`.

4. **SA scores**: Compute SA scores for all candidates.

5. **Broad filter**: Apply `apply_cyclopentane_filters()` (boiling point + SA only).

6. **Rank**: Compute cyclopentane parameter distance with weights `{m: 1, sigma: 2, epsilon_k: 5}`.

7. **AD check**: Run Tanimoto AD, add `ad_tanimoto_max` and `ad_in_domain` columns.

8. **Annotate**: Add structural flags, regulatory flags, and known-agent matching.

9. **Save**: Write full results to `screening/results/expanded_ranked.csv`.

10. **Summary**: Print summary table: how many candidates pass each filter stage, broken down by halogen class and backbone type. Print top 20 candidates with annotations.

11. **Comparison**: Load existing `screening/results/hfo_rf_ranked.csv` (Step 38b) and compare: how many new candidates appear in the top 50? Are any genuinely novel (not known agents, not regulatory dead-ends)?

### 53.8 — Generate figures

Save all figures to `figures/53_expanded_screening/`.

**Required:**

1. **`parameter_space_by_regulatory_class.png`** — Scatter plot of ε/k vs m for all candidates passing boiling point filter, colored by `regulatory_flag`. Cyclopentane reference point marked. Shows where each regulatory class sits in parameter space.

2. **`filter_funnel_comparison.png`** — Side-by-side filter funnel: Step 20 olefin-only (10,700 → ... → final) vs Step 53 expanded (N → ... → final). Bar chart showing candidate counts at each filter stage.

3. **`halogen_class_distribution.png`** — Stacked bar chart showing the halogen composition of candidates at each filter stage (F-only, F+Cl, F+Br, F+I, Cl-only, Br-only, unsubstituted, etc.).

4. **`ad_coverage.png`** — Histogram of `ad_tanimoto_max` values for all candidates, with the 0.4 threshold marked. Separate histograms or color-coding for molecules containing Br, I, or neither.

### 53.9 — Write report

Write `docs/reports/53_expanded_candidate_screening.md` following the standard template:

1. **Title**: `# Results Report: Expanded Candidate Screening (All Non-Associating C2–C6 Halogenated Hydrocarbons)`
2. **Enumeration summary**: Total candidates generated, breakdown by backbone type and halogen class.
3. **Filter funnel**: Table showing candidate counts at each filter stage, compared to Step 20.
4. **Top candidates**: Table of top 20 by cyclopentane parameter distance, with annotations (regulatory flag, AD status, known-agent match).
5. **Comparison to Step 20**: Which top candidates are new vs already found? Are any genuinely novel?
6. **Known agents found**: List all commercial blowing agents recovered in the expanded space, with their rankings and regulatory status.
7. **AD analysis**: What fraction of candidates are in-domain vs out-of-domain? How does this break down by halogen class?
8. **Key findings**: Bullet points. Expected: the expansion rediscovers known chemistry; every new near-hit has a non-thermodynamic disqualifier.
9. **Figures**: Reference all figure paths.
10. **Readiness check**: Checklist.

### 53.10 — Update supplementary materials

In `supplementary_information/screening_methodology.md`:

1. **Revise the "Why not bromine or iodine?" subsection**: Keep the regulatory and stability concerns but note that the expanded screening (Step 53) now includes Br and I for completeness, with AD flagging for prediction confidence.

2. **Add a new section "Expanded Screening: All Non-Associating C2–C6 Halogenated Hydrocarbons"** after the existing "Results Summary" section:
   - Enumeration scope and candidate counts
   - Key results by halogen class
   - Comparison to olefin-only screening
   - Which top-ranked candidates are already known commercial agents
   - Conclusion: whether the expansion changes the screening outcome

## Key Files

| File | Purpose |
|------|---------|
| `screening/generate.py` | Enumeration: add `ALKANE_BACKBONES`, extend `_enumerate_halogen_patterns()` |
| `screening/hfo_screening.py` | Add `apply_cyclopentane_filters()`, parallelize `batch_boiling_points()` |
| `screening/filters.py` | `filter_pcsaft_similarity()` for cyclopentane-centric ranking (no changes needed) |
| `model/predict.py` | RF prediction (no changes needed, already `n_jobs=-1`) |
| `model/ad_tanimoto.py` | Tanimoto AD check (no changes needed) |
| `screening/results/hfo_rf_ranked.csv` | Step 38b results for comparison |
| `supplementary_information/screening_methodology.md` | Documentation update |

## Artifacts

| File | Description |
|------|-------------|
| `scripts/step53_expanded_screening.py` | Driver script |
| `screening/results/expanded_ranked.csv` | Full annotated results |
| `docs/reports/53_expanded_candidate_screening.md` | Results report |
| `figures/53_expanded_screening/` | 4 figures |
| `supplementary_information/screening_methodology.md` | Updated documentation |

## Success Criteria

- [ ] `ALKANE_BACKBONES` added to `screening/generate.py` with ~20 saturated C2–C6 backbones
- [ ] `_enumerate_halogen_patterns()` extended to support Br (Z=35) and I (Z=53) with position-selection
- [ ] `include_parent` kwarg allows unsubstituted hydrocarbons in enumeration
- [ ] `batch_boiling_points()` parallelized with `joblib.Parallel`
- [ ] Enumeration across backbones parallelized with `joblib.Parallel`
- [ ] `apply_cyclopentane_filters()` added (boiling point + SA only, no HFO-specific gates)
- [ ] Tanimoto AD flagging for all candidates (`ad_tanimoto_max`, `ad_in_domain` columns)
- [ ] Structural and regulatory annotation for all candidates
- [ ] Known-agent matching against ~15 commercial blowing agents
- [ ] Full pipeline runs end-to-end via `scripts/step53_expanded_screening.py`
- [ ] Results saved to `screening/results/expanded_ranked.csv`
- [ ] At least 4 figures saved to `figures/53_expanded_screening/`
- [ ] Report written at `docs/reports/53_expanded_candidate_screening.md`
- [ ] `supplementary_information/screening_methodology.md` updated with expanded screening section
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- The expanded enumeration has been run with all four halogens on both olefin and saturated backbones
- A clear comparison to Step 20's olefin-only screening is documented
- Known commercial agents are identified in the expanded space with their regulatory status
- The screening conclusion is re-evaluated: does the expansion change the negative result?
- Supplementary documentation is updated for presentation readiness

## Budget

60–90 minutes. The enumeration and parallelized boiling point computation are the dominant costs. RF prediction is fast (batch, `n_jobs=-1`). On an 8-core machine, the parallelized boiling point loop should process 50,000 candidates in ~15–30 minutes (vs ~2–4 hours sequential). The annotation, AD check, figure generation, and report writing are fast follow-ups.
