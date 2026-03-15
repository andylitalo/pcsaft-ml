# Step 29: Documentation and Examples

## Objective

Write external-facing documentation so that both target audiences — thermodynamic
modelers/engineers (Audience A) and ML-for-chemistry researchers (Audience B) — can
self-serve without reading the internal project docs, step guides, or orchestrator
logs.

## Motivation

The project currently has extensive internal documentation (23+ step reports,
limitations doc, executive summary), but all of it is written for the project lead
reviewing agent work. An external user who discovers the repo needs:

1. A README that answers "what does this do and how do I use it" in 60 seconds
2. A quickstart notebook they can run immediately
3. A screening workflow notebook that demonstrates the full value proposition
4. A contribution guide if they want to add models or data
5. API reference generated from docstrings

The external docs should be clear and useful without overstating scientific
validation. Examples may be polished, but they must remain conservative about
OOD predictions, domain-specific heuristics, and the difference between model-
generated outputs and measured data.

The current README is project-internal and references the 8-step MLOps roadmap,
which is irrelevant to a user who wants to predict PC-SAFT parameters.

## Dependencies

- Step 26 (`pcsaft-predict` package): the public API that documentation references
- Step 27 (generalized portal): the Streamlit UI that screenshots reference
- Step 28 (dataset + model cards): citation info and dataset documentation

## Implementation Guide

### 29.1 Rewrite `README.md`

Replace the current project-internal README with an external-facing one. Structure:

**Title + Badge row**
```markdown
# pcsaft-predict

Predict PC-SAFT equation-of-state parameters from molecular SMILES using graph
neural networks.

[![PyPI](https://img.shields.io/pypi/v/pcsaft-predict)](https://pypi.org/project/pcsaft-predict/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/<org>/pcsaft-predict/actions/workflows/ci.yml/badge.svg)](...)
```

**One-paragraph pitch**: "pcsaft-predict is an open-source Python library for
predicting PC-SAFT equation-of-state parameters (m, σ, ε/k) from molecular SMILES
strings. It provides uncertainty quantification via MC Dropout, applicability domain
checking via Tanimoto similarity, and optional thermodynamic property computation
(boiling point, Henry's constant). The default GNN model achieves R² > 0.70 on all
three parameters, trained on 13,764 molecules."

Immediately after the pitch, add a short caution box that explains:
- predictions are intended for screening and prioritization, not direct engineering decisions
- some release models may rely on mixed-provenance labels
- thermodynamic properties are derived from predicted parameters unless otherwise stated

**Quick install**
```bash
pip install pcsaft-predict
```

**Quick usage (3 examples)**

Example 1 — Python API:
```python
from pcsaft_predict import predict

df = predict(["CCO", "c1ccccc1", "C1CCCC1"])
print(df[["smiles", "m", "sigma", "epsilon_k"]])
```

Example 2 — With uncertainty:
```python
from pcsaft_predict import predict_with_uncertainty

df = predict_with_uncertainty(["FC(F)=CF"])
print(df[["smiles", "epsilon_k", "epsilon_k_std", "tanimoto_nn"]])
```

Example 3 — Streamlit portal:
```bash
pip install pcsaft-predict[all] streamlit
streamlit run portal/app.py
```

**Performance table**

| Model | R²(m) | R²(σ) | R²(ε/k) | Training data |
|-------|-------|-------|---------|---------------|
| GNN (all data) | 0.76 | 0.77 | 0.73 | 13,764 molecules |
| RF | 0.62 | 0.35 | 0.33 | 1,801 molecules |
| Ensemble (RF+GNN) | — | — | — | Combined |

**Available models**: table with model name, description, when to use it.

**Novel predictions dataset**: brief description and link to
`data/pcsaft_novel_predictions_v1.csv` and `data/DATASHEET.md`.

**Project background**: 2-3 sentences. Link to the narrative report for the full
story. Frame the negative result honestly: "The project originated as an ML-driven
search for cyclopentane-like blowing agents among fluorinated olefins. Systematic
screening of 4,663 candidates proved that no fluorinated olefin can match
cyclopentane's dispersion energy — a fundamental physics constraint. The ML
pipeline and trained models have standalone value for the broader thermodynamic
modeling community."

**Citation**: BibTeX block from Step 28.

**License**: MIT for code, CC-BY-4.0 for data.

### 29.2 Create `examples/quickstart.ipynb`

Jupyter notebook that a user can run immediately after `pip install pcsaft-predict`.
Target runtime: < 2 minutes including model download.

Cells:

1. **Install** (markdown): `pip install pcsaft-predict`
2. **Import and predict**: predict params for 5 diverse molecules (ethanol, benzene,
   cyclopentane, R-134a, a novel HFO)
3. **Inspect results**: display DataFrame, explain each column
4. **Visualize uncertainty**: bar chart of ε/k ± ε/k_std for the 5 molecules
5. **Check applicability domain**: explain tanimoto_nn column, show which molecules
   are in-domain vs out-of-domain
6. **Compare models**: predict with RF and GNN, show side-by-side table
7. **Use a different model**: `predict(smiles, model="rf")` vs `model="gnn"`
8. **Batch prediction**: predict for 50 molecules from a CSV, time it

### 29.3 Create `examples/screening_workflow.ipynb`

Jupyter notebook demonstrating the full screening value proposition. This reproduces
the refrigerant screening use case (not blowing agents — to show generalizability),
but it must be labeled throughout as an **illustrative workflow**, not a validated
domain protocol.

Cells:

1. **Define the problem**: "We want to find novel fluorinated compounds with
   similar PC-SAFT parameters to R-134a (a common refrigerant being phased out)"
2. **Generate candidates**: enumerate fluorinated ethane/propane derivatives using
   RDKit (show the enumeration logic, ~200 candidates)
3. **Predict PC-SAFT parameters**: batch predict for all candidates
4. **Compute parameter distance**: weighted Euclidean to R-134a reference
5. **Filter by AD**: remove candidates with Tanimoto < 0.3
6. **Filter by uncertainty**: remove candidates where ε/k_std > 30 K
7. **Rank and visualize**: parity-style scatter of predicted ε/k vs R-134a ε/k,
   colored by Tanimoto similarity
8. **Top candidates table**: show top 10 with all properties
9. **Compute boiling points** (optional, requires teqp): add boiling point column
10. **Export**: save to CSV for further analysis
11. **Interpret**: "This workflow took 3 minutes and produced screening-quality
    predictions for 200 novel candidates. The ranked molecules are hypotheses for
    follow-up, not validated recommendations. The top candidates should be
    validated experimentally before further development."

### 29.4 Create `CONTRIBUTING.md`

Guide for external contributors:

**How to add a new model**:
1. Create a model wrapper class with `load()`, `predict()`, `predict_with_uncertainty()`
2. Register with `@register_model("name")` in `model/registry.py`
3. Add weights to `model/saved/`
4. Add entry to `pcsaft_predict/_registry.py`
5. Run `pytest tests/` to verify

**How to add training data**:
1. Prepare CSV with columns: smiles, m, sigma, epsilon_k
2. Add to `model/data/` with a download script
3. Update `model/data/load.py` to include the new source
4. Document provenance and license in the load function docstring

**How to run tests**:
```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
```

**Code style**: ruff with line-length 99, Python 3.10+. No type: ignore without
explanation. Logging, not print().

**Pull request process**: describe expected PR format, testing requirements.

### 29.5 Create API reference

Use pdoc (or mkdocs with mkdocstrings) to auto-generate API docs from the
`pcsaft_predict` package docstrings. Generate to `docs/api/`.

At minimum, document:
- `pcsaft_predict.predict()`
- `pcsaft_predict.predict_with_uncertainty()`
- `pcsaft_predict.list_models()`
- `pcsaft_predict.load_model()`

If using mkdocs, create `mkdocs.yml` at repo root. If using pdoc, add a Makefile
target: `make docs`.

### 29.6 Preserve internal documentation

The existing internal docs (step guides, reports, orchestrator log) are valuable
for the project narrative and should be preserved. Move them to a clearly labeled
subdirectory:

- Keep `docs/reports/` as-is (the reports tell the scientific story)
- Keep `docs/steps/` as-is (useful for understanding the development process)
- Add a note in the top-level README: "For the full project development narrative
  and step-by-step reports, see `docs/reports/`."

Do NOT delete any existing documentation.

## Key Outputs

| Artifact | Path |
|----------|------|
| README (rewritten) | `README.md` |
| Quickstart notebook | `examples/quickstart.ipynb` |
| Screening workflow notebook | `examples/screening_workflow.ipynb` |
| Contributing guide | `CONTRIBUTING.md` |
| API reference | `docs/api/` (auto-generated) |
| Report | `docs/reports/29_documentation_examples.md` |

## Acceptance Criteria (When to Move On)

- [ ] README answers "what, why, how" in < 60 seconds of reading
- [ ] README has: install, 3 usage examples, performance table, citation, license
- [ ] `examples/quickstart.ipynb` runs end-to-end with `pcsaft_predict` installed
- [ ] `examples/screening_workflow.ipynb` demonstrates the full predict→filter→rank pipeline
- [ ] Screening notebook uses a **non-blowing-agent** reference (e.g., R-134a) to demonstrate generality
- [ ] README includes a concise caution that outputs are model-generated and intended for screening, not direct engineering use
- [ ] Screening notebook is explicitly labeled as an illustrative workflow rather than a validated domain standard
- [ ] `CONTRIBUTING.md` explains how to add models and data
- [ ] API reference generated for `pcsaft_predict` public functions
- [ ] Existing internal docs preserved (not deleted or moved)
- [ ] Report at `docs/reports/29_documentation_examples.md`
