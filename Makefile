.PHONY: setup download-data lint test train-rf evaluate screen check-gate \
       step-01 step-02 step-03 step-04 step-05 step-06 step-07 step-08 step-09

# ── Setup ──────────────────────────────────────────────────────────────────────

setup:
	uv sync --extra dev

setup-nn:
	uv sync --extra dev --extra nn

setup-hf:
	uv sync --extra dev --extra nn --extra hf

setup-serve:
	uv sync --extra dev --extra nn --extra serve

setup-pipeline:
	uv sync --extra dev --extra nn --extra serve --extra pipeline

setup-portal:
	uv sync --extra dev --extra portal

setup-thermo:
	uv sync --extra dev --extra thermo

setup-all:
	uv sync --all-extras

download-data:
	uv run python -m model.data.download_esper

# ── Quality ────────────────────────────────────────────────────────────────────

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

test:
	uv run pytest tests/ -v

# ── Model ──────────────────────────────────────────────────────────────────────

train-rf:
	uv run python -m model.train --data esper

evaluate:
	uv run python -m model.evaluate

screen:
	uv run python -m screening.run_screening --skip-patents

# ── Gate Checks ────────────────────────────────────────────────────────────────

check-gate:
	uv run python scripts/check_gate.py all

check-gate-%:
	uv run python scripts/check_gate.py $*

# ── Per-Step Targets ───────────────────────────────────────────────────────────
# These are placeholders. Each step guide defines the actual commands.
# The orchestrator uses these as entry points.

step-01: setup
	@echo "Step 01: Morgan Fingerprints — see docs/steps/01_morgan_fingerprints.md"
	uv run python scripts/check_gate.py 01

step-02: setup-nn
	@echo "Step 02: PyTorch NN — see docs/steps/02_pytorch_multitask_nn.md"
	uv run python scripts/check_gate.py 02

step-03: setup-nn
	@echo "Step 03: Evaluation Harness — see docs/steps/03_evaluation_harness.md"
	uv run python scripts/check_gate.py 03

step-04: setup-hf
	@echo "Step 04: ChemBERTa — see docs/steps/04_chemberta_finetuning.md"
	uv run python scripts/check_gate.py 04

step-05: setup-serve
	@echo "Step 05: FastAPI Serving — see docs/steps/05_fastapi_serving.md"
	uv run python scripts/check_gate.py 05

step-06: setup-serve
	@echo "Step 06: Docker & K8s — see docs/steps/06_docker_kubernetes.md"
	uv run python scripts/check_gate.py 06

step-07: setup-pipeline
	@echo "Step 07: Kubeflow Pipeline — see docs/steps/07_kubeflow_retrain_pipeline.md"
	uv run python scripts/check_gate.py 07

step-08: setup-portal
	@echo "Step 08: Streamlit Portal — see docs/steps/08_streamlit_portal.md"
	uv run python scripts/check_gate.py 08

step-09: setup-thermo
	@echo "Step 09: Thermodynamic Validation — see docs/steps/09_thermodynamic_validation.md"
	uv run python scripts/check_gate.py 09
