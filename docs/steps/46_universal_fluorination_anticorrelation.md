# Step 46: Cross-Class Fluorination Context in the Esper Dataset

## Objective

Test whether the fluorination-lowers-`epsilon_k` pattern observed in the halogenated
olefin screen also appears across a broader set of **carbon-containing Esper molecules**
that are at least loosely relevant to the project chemistry. This step is **supporting
context**, not a proof of a universal physical law across all molecule classes.

The goal is to strengthen the project's negative result in a disciplined way:

- **Allowed conclusion**: the Esper corpus is consistent with the project narrative that
  heavily fluorinated, candidate-like organics tend to sit below cyclopentane's
  `epsilon_k` regime.
- **Not allowed**: claiming that "no molecule of any class can work" from a descriptive
  sweep of the entire Esper dataset.

## Motivation

Steps 20–23 established the key anti-correlation inside the enumerated halogenated-olefin
search space. A fair follow-up question is:

**"Is this just a peculiarity of halogenated olefins, or does the same trend appear more
broadly in experimental PC-SAFT data?"**

The Esper dataset is useful for that purpose because it contains 1,801 experimental fits
covering diverse carbon-containing chemistries. But it also contains molecules that are not
meaningful analogs for the project's blowing-agent question. This step therefore uses Esper
as a **context dataset**, with careful filtering, rather than treating every entry as an
equally relevant candidate.

## Scope Rules

1. **Keep the analysis on carbon-containing molecular species.**
   Exclude entries with no carbon and clearly non-comparable species before drawing any
   conclusion about "organic chemistry."
2. **Treat class labels as descriptive bins, not authoritative ontology.**
   If classes are inferred from SMARTS or atom presence, state that clearly and do not
   present them as native Esper annotations.
3. **Use the project's existing flammability proxy.**
   The primary fluorination axis for this step is **fluorine mass fraction**, aligned with
   Step 41:
   - `>= 0.65` -> A1-like heuristic
   - `0.50–0.65` -> A2L-like heuristic
   - `< 0.50` -> more flammable side of the project's screen
4. **Do not use `F / (F + H)` as the sole "non-flammability" threshold.**
   It may be reported as a secondary descriptive variable if useful, but it cannot be the
   main project-facing safety boundary.
5. **Inspect apparent counterexamples manually.**
   Any molecule in the high-`epsilon_k`, high-fluorination region must be listed explicitly
   and checked for relevance before being used in the narrative.

## Method

1. **Filter the Esper corpus to a project-relevant comparison set.**
   At minimum require:
   - carbon-containing molecules
   - discrete molecular species rather than elemental or obviously inorganic entries
   - no silent inclusion of entries outside the intended "organic context" scope
2. **Annotate each retained molecule with a coarse functional class.**
   These labels are for visualization and subgroup summaries only.
3. **Compute fluorine mass fraction** for every retained molecule.
   Optionally also compute `F / (F + H)` as a secondary descriptor, but keep fluorine mass
   fraction as the project-facing axis.
4. **Plot `epsilon_k` vs fluorine mass fraction**, colored by coarse functional class.
   Mark:
   - cyclopentane reference (`epsilon_k = 288.8 K`)
   - A2L-like boundary (`F mass fraction = 0.50`)
   - A1-like boundary (`F mass fraction = 0.65`)
5. **Run within-class or paired-family comparisons** only where sample size is adequate.
   Require a minimum sample size per subgroup and report the counts directly.
6. **Define a candidate-like subset** for the strongest claim.
   For example: carbon-containing organics in a blowing-agent-like molecular-weight window
   and without obvious off-scope chemistries. Quantify how many such molecules combine high
   `epsilon_k` with A2L-like or A1-like fluorination.
7. **List and classify all apparent hits manually.**
   Separate:
   - relevant candidate-like organics
   - borderline cases
   - obvious off-scope artifacts or chemically irrelevant entries

The intended output is a **careful supporting argument**, not a slogan about a universal
forbidden quadrant.

## Key Deliverables

1. `scripts/step46_fluorination_anticorrelation.py` — analysis + figure generation
2. `figures/46_universal_anticorrelation/epsk_vs_fluorination.png` — scatter plot using the
   filtered, carbon-containing comparison set
3. `figures/46_universal_anticorrelation/class_comparison_bars.png` — class-level or
   family-level comparison chart with subgroup counts shown
4. `docs/reports/46_universal_fluorination_anticorrelation.md` — results report

The report must explicitly document:

- filtering rules and exclusions
- fluorination metric definitions
- sample counts for every subgroup
- every apparent high-`epsilon_k`, high-fluorination hit and whether it is actually relevant
- a final conclusion framed as **supporting context for the project narrative**

## When to Move On

- [ ] The filtered analysis set is documented and excludes non-carbon or otherwise off-scope
      species
- [ ] The main fluorination axis is fluorine mass fraction, consistent with Step 41
- [ ] Every apparent high-`epsilon_k`, high-fluorination molecule is listed and manually
      triaged
- [ ] Within-class or family-level comparisons report subgroup counts and avoid overclaiming
- [ ] The report states a restrained conclusion about **broader Esper context**, not a
      universal law
- [ ] The presentation framing is updated only with wording that the filtered analysis
      actually supports
