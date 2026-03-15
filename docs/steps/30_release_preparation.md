# Step 30: Release Preparation

## Objective

Prepare the repository for public release on GitHub and PyPI. Add a license, CI
pipeline, changelog, issue templates, and perform a pre-release audit.

## Motivation

Steps 26-29 created the package, generalized the portal, published datasets and
model cards, and wrote documentation. This final step handles the operational
requirements for a public open-source release: licensing, automated testing,
versioning, and distribution.

## Dependencies

- Step 26 (installable package): `pcsaft_predict/` with `pyproject.toml`
- Step 28 (model cards + citation): `CITATION.cff`, `data/DATASHEET.md`
- Step 29 (documentation): `README.md`, `CONTRIBUTING.md`, examples

## Implementation Guide

### 30.1 License selection

Add `LICENSE` file at repo root.

**Recommended: MIT License.**

Rationale:
- Maximally permissive — no barrier to adoption by industry or academia
- Compatible with all dependencies: RDKit (BSD-3), PyTorch (BSD-3),
  PyTorch Geometric (MIT), scikit-learn (BSD-3), teqp (MIT)

Important caveat: dependency compatibility alone is **not** sufficient to justify
public redistribution of model weights or derived data products. Before release,
perform a source-by-source provenance and redistribution audit covering:
- training data licenses and redistribution terms
- whether upstream model-generated labels can be used in redistributed downstream weights
- whether processed datasets and release artifacts can be sublicensed as proposed
- citation and attribution obligations for each upstream source

For the dataset (`data/pcsaft_novel_predictions_v1.csv`), use **CC-BY-4.0**. Add
a `LICENSE-DATA` file or a license field in the CSV header comment and in
`data/DATASHEET.md`.

Do not publish the predictive library or model weights until that provenance audit
confirms redistribution is allowed.

### 30.2 GitHub Actions CI

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check .

  test-core:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v -m "not slow" --tb=short

  test-package:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e "./pcsaft_predict[all]"
      - run: pip install pytest
      - run: pytest tests/test_pcsaft_predict.py -v --tb=short

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build
      - run: python -m build pcsaft_predict/
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: pcsaft_predict/dist/
```

Notes:
- `test-core` runs the full project test suite (excluding slow tests that need
  model weights)
- `test-package` tests the standalone package API
- `build` verifies the package builds correctly (sdist + wheel)
- Model weights are NOT downloaded in CI (tests marked `@pytest.mark.slow` are skipped)
- The `lint` job catches ruff violations before merge

### 30.3 Issue templates

Create `.github/ISSUE_TEMPLATE/`:

**`bug_report.md`**:
```markdown
---
name: Bug Report
about: Report a bug in pcsaft-predict
labels: bug
---

**Describe the bug**
A clear description of what happened.

**To reproduce**
```python
from pcsaft_predict import predict
# Paste minimal code that reproduces the issue
```

**Expected behavior**
What you expected to happen.

**Environment**
- OS:
- Python version:
- pcsaft-predict version:
- torch version:
- rdkit version:

**SMILES (if applicable)**
The molecule(s) that triggered the issue.
```

**`feature_request.md`**:
```markdown
---
name: Feature Request
about: Suggest an improvement
labels: enhancement
---

**Is your feature request related to a problem?**
Describe the use case.

**Proposed solution**
How you'd like this to work.

**Alternatives considered**
Other approaches you've thought about.
```

**`model_contribution.md`**:
```markdown
---
name: Model Contribution
about: Propose a new model or training data
labels: model
---

**Model type**
(e.g., new architecture, new training data, improved hyperparameters)

**Performance on test set**
| Parameter | R² | MAE | RMSE |
|-----------|----|-----|------|
| m         |    |     |      |
| σ         |    |     |      |
| ε/k       |    |     |      |

**Training data**
Description and license of training data used.

**Weights**
How to access the trained weights (link, or will upload).
```

### 30.4 Create `CHANGELOG.md`

```markdown
# Changelog

All notable changes to pcsaft-predict will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-XX-XX

### Added
- Initial public release
- GNN model (R² > 0.70 on all PC-SAFT parameters, trained on 13,764 molecules)
- RF model (R² = 0.62/0.35/0.33 on m/σ/ε/k, trained on 1,801 molecules)
- Ensemble model (inverse-variance-weighted RF + GNN)
- MC Dropout uncertainty quantification for GNN
- RF tree variance uncertainty quantification
- Tanimoto nearest-neighbor applicability domain checking
- Boiling point computation from PC-SAFT parameters via teqp
- Henry's constant estimation at infinite dilution
- Streamlit portal with application presets (blowing agents, refrigerants, solvents)
- Novel predictions dataset: 4,612 fluorinated/chlorinated cycloalkenes
- HFO-centric screening results with boiling points and Henry's constants

### Models
- `gnn`: 4-layer GIN (PCSAFTGraphNet), 964K params, trained on Esper + ML-SAFT + SPT-PCSAFT
- `rf`: Random Forest (100 trees per target), trained on Esper
- `ensemble`: Inverse-variance-weighted combination of RF + GNN
- `nn`: PyTorch MLP (PCSAFTNet), trained on Esper
- `chemberta`: Fine-tuned ChemBERTa-zinc-base-v1, trained on Esper
```

### 30.5 Pre-release audit

Run the following checks before tagging v1.0.0:

**Provenance / redistribution audit (blocking)**: create a release checklist that
lists every upstream source used in:
- model training
- curated validation tables
- packaged datasets
- bundled examples or screenshots

For each source, record:
- what was used
- whether it is raw data, fitted parameters, or model-generated labels
- license / terms of use
- whether redistribution of weights or derived tables is permitted
- required citation / attribution language

If any source has unclear redistribution status, do **not** attach the affected
weights or derived CSVs to the public release until it is resolved.

**Secrets audit**: Verify no API keys, tokens, or credentials in git history:
```bash
git log --all --diff-filter=A --name-only --pretty=format: | sort -u | grep -iE '\.env|credential|secret|token|key'
```
Inspect any matches. None expected in this project.

**Large file audit**: Check for accidentally committed large files:
```bash
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '/^blob/ {print $3, $4}' | sort -rn | head -20
```
Model weights should be in `.gitignore` and distributed via GitHub Releases, not
committed to the repo.

**`.gitignore` review**: Ensure model weights, data caches, and build artifacts
are ignored:
```
model/saved/*.pt
model/saved/*.joblib
model/saved/chemberta/
pcsaft_predict/data/*.pt
pcsaft_predict/data/*.joblib
__pycache__/
*.egg-info/
dist/
build/
.env
```

**Dependency pinning**: Verify `pyproject.toml` has version bounds (not unbounded
`>=` without `<`). For the installable package, use compatible release specifiers
where appropriate (`torch>=2.0,<3.0`).

### 30.6 Tag and release

```bash
git tag -a v1.0.0 -m "Initial public release of pcsaft-predict"
git push origin v1.0.0
```

Create GitHub Release:
- Title: "pcsaft-predict v1.0.0"
- Body: copy relevant section from CHANGELOG.md
- Attach release assets:
  - `gnn_pcsaft.pt` (GNN weights)
  - `rf_m.joblib`, `rf_sigma.joblib`, `rf_epsilon_k.joblib` (RF weights)
  - `feature_config.json`, `feature_names.joblib` (feature pipeline config)
  - `pcsaft_novel_predictions_v1.csv` (model-generated predictive library)

Only attach assets whose provenance / redistribution audit passed. If weight
redistribution is not clearly permitted, release the code and documentation first
and defer weight hosting.

### 30.7 PyPI publication

Option A — Manual (for first release):
```bash
cd pcsaft_predict
python -m build
twine upload dist/*
```

Option B — Trusted publisher (recommended for ongoing releases):
Configure GitHub Actions to publish to PyPI on tag push using the
`pypa/gh-action-pypi-publish` action with OIDC trusted publishing.

Add to `.github/workflows/ci.yml`:
```yaml
  publish:
    needs: [lint, test-core, test-package, build]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

## Key Outputs

| Artifact | Path |
|----------|------|
| License | `LICENSE` |
| Data license | `LICENSE-DATA` (or in DATASHEET.md) |
| CI workflow | `.github/workflows/ci.yml` |
| Bug report template | `.github/ISSUE_TEMPLATE/bug_report.md` |
| Feature request template | `.github/ISSUE_TEMPLATE/feature_request.md` |
| Model contribution template | `.github/ISSUE_TEMPLATE/model_contribution.md` |
| Changelog | `CHANGELOG.md` |
| Git tag | `v1.0.0` |
| GitHub Release | with model weights as assets |
| PyPI package | `pcsaft-predict` on pypi.org |
| Report | `docs/reports/30_release_preparation.md` |

## Acceptance Criteria (When to Move On)

- [ ] `LICENSE` file exists (MIT)
- [ ] `LICENSE-DATA` or data license documented (CC-BY-4.0)
- [ ] `.github/workflows/ci.yml` runs lint, test-core, test-package, build
- [ ] CI passes on main branch (or would pass — verify locally with `act` or manual run)
- [ ] Issue templates exist for bug reports, feature requests, and model contributions
- [ ] `CHANGELOG.md` documents v1.0.0 features
- [ ] Provenance / redistribution audit completed for training data, model weights, and packaged predictive-library artifacts
- [ ] No secrets or credentials in git history (audit passed)
- [ ] No large binary files committed to git (model weights in .gitignore)
- [ ] `.gitignore` covers all generated artifacts
- [ ] `v1.0.0` tag created
- [ ] GitHub Release created with only those model weights / assets whose redistribution status is confirmed
- [ ] `pip install pcsaft-predict` works from PyPI (or TestPyPI for dry run)
- [ ] Report at `docs/reports/30_release_preparation.md`
