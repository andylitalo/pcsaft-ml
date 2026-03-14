# ML-Driven Blowing Agent Screening via PC-SAFT Parameter Prediction

Screen novel hydrofluoroolefin (HFO) candidates as drop-in replacements for cyclopentane
in polyurethane foam blowing, using machine learning to predict thermodynamic (PC-SAFT)
parameters from molecular structure.

See [docs/project_overview.md](docs/project_overview.md) for full project description.
See [PLAN.md](PLAN.md) for orchestrator execution guide.

## Quick Start

```bash
uv sync --extra dev                    # install dependencies
uv run python -m model.data.download_esper   # fetch Esper dataset (~1,842 molecules)
uv run python -m model.train           # train RF baseline
uv run python -m model.evaluate        # evaluate and generate parity plots
uv run pytest tests/ -v                # run tests
```

## Project Structure

```
model/          ML models (RF baseline, NN, ChemBERTa)
screening/      Candidate generation and filtering pipeline
serving/        FastAPI prediction service (Step 05)
pipeline/       Kubeflow retrain pipeline (Step 07)
portal/         Streamlit researcher portal (Step 08)
docs/           Guides, reports, and background
figures/        Generated plots and visualizations
scripts/        Utility scripts (gate checking, etc.)
tests/          Test suite
```
