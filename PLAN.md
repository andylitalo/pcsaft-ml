# PLAN.md — Orchestrator Playbook

This file is the single source of truth for step execution, agent coordination, and project state.
It is read by the orchestrator (Claude Code) at the start of every session. Background context
about the project (PC-SAFT, directory conventions, commit format, report template) lives in
`CLAUDE.md`, which is auto-loaded into every Cursor conversation.

---

## 1. Decisions Log

Scientific clarification answers from the project lead. The orchestrator records answers here
so future instances have context. **Do not proceed past Phase 0 until all questions are answered.**

| # | Question | Answer |
|---|----------|--------|
| Q1 | **Target property**: Closest PC-SAFT params to cyclopentane, or multi-criteria (vapor pressure, solubility, flammability, GWP)? | **Answered**: Parameter proximity to cyclopentane as primary objective. Step 09 validates whether this correlates with vapor pressure proximity. Multi-criteria screening deferred as future work. |
| Q2 | **GC-PC-SAFT baseline**: Implement group-contribution baseline (Sauer 2014 / Gross-Sadowski 2002) for comparison, or cite literature values only? | **Answered**: Implement a simplified GC calculation (~100 lines) in Step 01 as a running comparison baseline. More compelling than literature citation. |
| Q3 | **Distance metric weighting**: Weight epsilon/k more heavily? Proposed 3:1:1 (epsilon_k : sigma : m). | **Answered**: Use 3:1:1 (ε/k : σ : m) as configurable default. Step 09 thermodynamic validation will retroactively validate whether these weights are physically justified. |
| Q4 | **Applicability domain**: Use leverage/distance in Morgan FP space + flag out-of-distribution molecules? | **Answered**: Yes. Isolation Forest in Morgan FP space, flag OOD molecules. Implemented in Step 02 (2B), reported in Step 03 (3B). |
| Q5 | **Uncertainty quantification**: RF tree variance + MC Dropout for NN. Propagate into screening distance as confidence intervals? | **Answered**: Yes for UQ (RF variance + MC Dropout). Report as error bars and confidence intervals, but do not propagate into the distance metric itself — adds complexity for marginal benefit during prototyping. |
| Q6 | **Association parameters**: Acknowledge 3-param limitation (non-associating only). Flag OH/NH/COOH candidates? | **Answered**: Flag only. Acknowledge 3-param scope limitation honestly. `is_associating()` utility flags candidates with association sites. No attempt at 5-param prediction. |
| Q7 | **ML-SAFT dataset (Felton 2024)**: Integrate 988-molecule dataset alongside Esper (~1,842)? IP concerns? | **Answered**: Yes, integrate. Published academic dataset, no IP concerns for this project. Increases training set by ~50%. |
| Q8 | **Expanded candidate space**: Broaden beyond 63 HFOs to HCFOs, HFEs, unsaturated hydrocarbons, C3-C6 cyclic fluorinated? Target 500-2000? | **Answered**: Yes, broaden to 500-2000 candidates. Include HCFOs, HFEs, unsaturated hydrocarbons, C3-C6 cyclic fluorinated. Implemented in Step 04 (4A). |
| Q9 | **Thermodynamic validation**: Defer EOS closure to later pass, or include post-screening in Step 04? | **Answered**: Add as Step 09, a lightweight post-hoc validation using teqp (CPU-only, seconds per molecule). Does not block Steps 05-08. |

---

## 2. Repo Setup Checklist

Phase 1 and Phase 2 items. The orchestrator marks these off as they are completed.

### Phase 1: Repository Organization

- [ ] Pre-scaffold empty packages: `model/nn/`, `model/hf/`, `serving/`, `pipeline/`, `portal/`
- [ ] Create `scripts/check_gate.py`
- [ ] Create `Makefile` with per-step targets
- [ ] Create `state.yaml` for step tracking
- [ ] Create root `README.md`
- [ ] Fix `docs/steps/step_dependencies.md` (remove duplicates, remove contradictions)
- [ ] Decouple `screening/filters.py` from `model.predict` import
- [ ] Patch step guides 01-05 with science improvements (sections 1A-1D, 2A-2B, 3A-3B, 4A, 5-schema)
- [ ] Fill all optional dep groups in `pyproject.toml` with pinned versions
- [ ] Add `[tool.ruff]` config to `pyproject.toml`, run `ruff check --fix .`
- [ ] Create `tests/conftest.py` with mini-dataset fixture
- [ ] Create `tests/fixtures/mini_pcsaft.csv`
- [ ] Slim down `CLAUDE.md` (remove workflow sections, add pointer to PLAN.md)

### Phase 2: Version Control

- [ ] Commit Phase 1 changes to `main`
- [ ] Generate dependency lockfile via `uv lock`
- [ ] Add SHA-256 checksum verification to `model/data/download_esper.py`
- [ ] Add `model/saved/MANIFEST.json` pattern to `scripts/check_gate.py`
- [ ] Verify: `pytest tests/ -v` passes, `ruff check .` passes
- [ ] Commit Phase 2 changes

---

## 3. Orchestrator Operating Model

```
Orchestrator (Claude Code)
├── Read PLAN.md and state.yaml on startup
├── Determine next actionable step(s) from state.yaml
├── For each step:
│   ├── Create branch step-NN-<name>
│   ├── Read docs/steps/NN_<name>.md (includes science improvement sections)
│   ├── Run: python scripts/check_gate.py NN
│   ├── Spawn sub-agent to implement the step
│   ├── Sub-agent: code, test, train, evaluate, generate figures, write report
│   ├── Verify: tests pass, artifacts exist, metrics in expected range
│   ├── Present report to human for approval
│   ├── On approval: merge branch to main, update state.yaml
│   └── On rejection: sub-agent revises, re-submit
├── Proceed to next step(s) whose dependencies are satisfied
└── Repeat until all steps complete
```

### Branching Strategy

- `main` — approved, working code only
- `step-NN-<name>` — per-step work branches (e.g., `step-01-morgan-fps`)
- Merge to `main` only after human approval of the step report

### Approval Model

Per-step serial approval. The orchestrator stops after each step report and waits for the
human lead to approve before merging and proceeding. This is slower but gives tight control
during prototyping.

---

## 4. Context Management & Handoff

The orchestrator will eventually fill its context window. All state must live in files so a
fresh orchestrator instance can resume with zero prior context.

### Periodic State Persistence (after every step completion)

1. Update `state.yaml` with step status, artifact paths, key metrics
2. Append to `docs/orchestrator_log.md`: 5-10 line summary of what was completed, decisions
   made, issues encountered, and what the next action should be

### When Context Is Running Low

1. Write a final entry to `docs/orchestrator_log.md` with:
   - Current step in progress and its sub-step
   - Files currently being modified
   - Any uncommitted work on the current branch
   - The exact next action to take
2. Commit any WIP to the current step branch: `git commit -m "WIP: <description>"`
3. Update `state.yaml` with `status: in_progress` for the current step

### Resume Protocol (for a new orchestrator instance)

1. Read `PLAN.md` — understand the overall plan and decisions log
2. Read `state.yaml` — find all step statuses
3. Read `docs/orchestrator_log.md` — last 2-3 entries for recent context
4. Run `git branch` and `git status` — find any in-progress branches
5. Run `python scripts/check_gate.py` for the next incomplete step
6. Resume from where the previous orchestrator left off

---

## 5. Parallelization Policy & File Ownership

### What Can Run in Parallel

Given per-step serial approval, true parallelism is limited but possible:

- **Model stream** (01 → 02 → 03 → 04): strictly sequential, each needs approval
- **Infra scaffolding**: After Step 02 is approved, a sub-agent can start Step 05 with a
  stub model loader (mock predictions). Does not need Step 04 completion.
- **Rule**: Never have two agents modifying the same file simultaneously.

### File Ownership

| Files | Owner |
|-------|-------|
| `model/data/descriptors.py`, `model/train.py` | Model stream agent |
| `model/nn/*`, `model/hf/*` | Model stream agent |
| `model/evaluate.py`, `model/registry.py` | Model stream agent (Step 03+) |
| `screening/*` | Model stream agent |
| `serving/*`, `k8s/*` | Infra agent |
| `pipeline/*`, `portal/*` | Infra agent |
| `pyproject.toml` | Orchestrator only |
| `state.yaml`, `docs/orchestrator_log.md` | Orchestrator only |
| `PLAN.md`, `CLAUDE.md` | Orchestrator only |
| `tests/*` | Whoever owns the module under test |
| `docs/reports/*` | The agent that completed the step |

---

## 6. Per-Step Agent Template

When spawning a sub-agent, the orchestrator sends this prompt (fill in NN and name):

```
You are implementing Step NN: <name>.

1. Read docs/steps/NN_<name>.md in full. It includes both the original implementation
   guide AND science improvement sections (labeled 1A, 1B, etc.).
2. Read all files you will modify before editing.
3. Implement the features described, including the science improvement sections.
4. Run: pytest tests/ -v (all must pass).
5. Train and evaluate as specified in the guide.
6. Save figures to figures/<step-name>/.
7. Write report to docs/reports/NN_<name>.md following the report template in CLAUDE.md.
8. Return: list of files changed, metrics achieved, any blockers.

Constraints:
- Only modify files in your ownership set (see File Ownership in PLAN.md).
- Do not modify pyproject.toml; request dependency additions in your report.
- Keep tests minimal: one smoke test per new module to verify it doesn't crash.
- Use Python logging, not print().
```

---

## 7. Step Reference

Condensed from `docs/steps/step_dependencies.md`. The step guides themselves are the
authoritative source; this table is for quick orchestrator lookup.

| Step | Depends On | Key Artifacts Produced | Gates | Install |
|------|-----------|----------------------|-------|---------|
| 01 | — | `build_features()`, GC-PC-SAFT baseline, weighted distance, ML-SAFT data | 02 | `uv sync --extra dev` |
| 02 | 01 | `model/nn/`, `nn_pcsaft.pt`, UQ (MC Dropout), AD check | 03, 05 | `uv sync --extra dev --extra nn` |
| 03 | 01, 02 | `model/registry.py`, unified `evaluate.py`, comparison figures, UQ reporting, AD analysis | 04 | `uv sync --extra dev --extra nn` |
| 04 | 01, 02, 03 | `model/hf/`, `chemberta/`, 3-way comparison, expanded screening | 05 | `uv sync --extra dev --extra nn --extra hf` |
| 05 | 01, 02 (minimal) | `serving/` package, `/predict`, `/submit-data`, `/health` | 06, 07, 08 | `uv sync --extra dev --extra nn --extra serve` |
| 06 | 05 | Dockerfile, `k8s/` manifests, kind deployment | 07 | `uv sync --extra dev --extra nn --extra serve` |
| 07 | 05, 06 | `pipeline/` package, `pipeline.yaml`, champion/challenger | 08 | `uv sync --extra dev --extra nn --extra serve --extra pipeline` |
| 08 | 05 (minimal) | `portal/` package, full end-to-end demo | — | `uv sync --extra dev --extra portal` |
| 09 | 01 or 04 (screening results) | `model/thermodynamic.py`, property-space validation, rank correlation | — | `uv sync --extra dev --extra thermo` |

### Result Gates (what scripts/check_gate.py verifies)

| Step | Cannot Start Until |
|------|---------------------|
| 02 | `build_features()` exists in `model/data/descriptors.py` |
| 03 | `model/saved/nn_pcsaft.pt` exists |
| 04 | `model/registry.py` exists; `evaluate --models rf nn` works |
| 05 | At least one trained model artifact exists |
| 06 | `serving/app.py` exists and API starts |
| 07 | `k8s/` manifests exist |
| 08 | API is reachable; pipeline compiles |
| 09 | Screening results CSV exists |

---

## 8. Appendix: Report Template

Reports must follow this format (from CLAUDE.md). Sub-agents reference this when writing
`docs/reports/NN_<name>.md`.

### Required Sections

1. **Title**: `# Results Report: <Step Name> (<key context>)`
2. **Model Performance**: Opening paragraph + metrics table:

| Parameter | MAE | RMSE | R² (test) | Train samples | Test samples |
|-----------|-----|------|-----------|---------------|--------------|
| m (segments) | ... | ... | ... | ... | ... |
| σ (Å) | ... | ... | ... | ... | ... |
| ε/k (K) | ... | ... | ... | ... | ... |

Follow with interpretation paragraph.

3. **Comparison to Baseline/Previous Step**: Side-by-side delta table.
4. **Key Findings**: Bullet points on surprises, what worked, what didn't.
5. **Figures**: Reference paths (e.g., `See figures/01_morgan_fingerprints/`).
6. **Deviations**: Anything that diverged from the step guide. Omit if none.
7. **Readiness Check**: Confirm "when to move on" criteria as a checklist.

### Failure Handling

If a step produces worse results than expected:
- Document it honestly (valuable for interview discussion)
- Note hypotheses for why and what to try differently
- Wait for approval before committing
