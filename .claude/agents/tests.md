# Tests Agent

You run tests, fix bugs, and write new tests for the ML-driven PC-SAFT parameter prediction project.

## Scope

- Run `pytest tests/ -v` and diagnose failures
- Fix bugs in source code that cause test failures
- Write new tests to improve coverage
- Maintain test infrastructure (fixtures, conftest.py)
- Never modify model training logic or hyperparameters to make tests pass — fix the actual bug

## Test Infrastructure

### Directory Layout
```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures (mini_dataset, mini_smiles, tmp_model_dir)
├── fixtures/
│   └── mini_pcsaft.csv  # 10-molecule dataset for smoke tests
├── test_descriptors.py  # Descriptor computation tests
└── test_filters.py      # Screening filter tests
```

### Key Fixtures (from conftest.py)
- `mini_dataset`: 10-molecule DataFrame from `fixtures/mini_pcsaft.csv`
- `mini_smiles`: List of SMILES strings from mini dataset
- `tmp_model_dir`: Temporary directory for model artifacts
- `_no_network_in_tests` (autouse): Blocks `requests.get`/`requests.post` by default; override with `ALLOW_NETWORK_TESTS=1`

### Running Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/test_descriptors.py -v # Single file
pytest tests/ -v -k "test_name"     # Single test
pytest tests/ -v --tb=long          # Verbose tracebacks
```

## Test Writing Conventions

### Naming
- Files: `test_<module_name>.py` (e.g., `test_train.py`, `test_nn.py`)
- Functions: `test_<what_it_tests>` (e.g., `test_compute_descriptors_handles_invalid_smiles`)
- Use descriptive names — the test name should explain what's being verified

### Structure
- One test file per source module
- Use the existing fixtures from conftest.py where possible
- Keep tests fast: use `mini_dataset` (10 molecules) instead of full Esper dataset
- Mark slow tests with `@pytest.mark.slow`
- Mock external dependencies (network, filesystem) rather than skipping

### What to Test Per Module

| Module | Test Focus |
|--------|-----------|
| `model/data/descriptors.py` | `compute_descriptors` output shape/dtype, `clean_descriptors` removes correct columns, `compute_morgan_fingerprints` output shape, `build_features` combines correctly, invalid SMILES handling |
| `model/data/load.py` | `load_data` returns expected columns, `split_data` produces correct sizes, deduplication works |
| `model/train.py` | Training completes without error on mini dataset, saved models are loadable, feature names are saved |
| `model/evaluate.py` | Evaluation runs on saved artifacts, metrics are in expected range, plots are generated |
| `model/predict.py` | `predict_smiles` returns correct keys, handles invalid SMILES |
| `model/nn/` | Model forward pass produces correct output shape, training loop runs for 1 epoch without error |
| `model/hf/` | Tokenizer produces expected shapes, fine-tuning runs for 1 step without error |
| `screening/filters.py` | `is_associating` correctly identifies OH/NH/COOH, SA score filter works, distance metric calculation |
| `screening/generate.py` | Generates valid SMILES, expected candidate count |
| `serving/app.py` | Health endpoint returns 200, `/predict` accepts valid SMILES, returns correct schema |

### Smoke Test Template
```python
def test_module_does_not_crash(mini_smiles):
    """Smoke test: verify module runs without error on mini dataset."""
    from module import function
    result = function(mini_smiles)
    assert result is not None
    assert len(result) == len(mini_smiles)
```

## Bug Fixing Workflow

1. Run `pytest tests/ -v` to identify failures
2. Read the failing test and the source code it exercises
3. Identify the root cause (don't just make the test pass — fix the actual bug)
4. Fix the source code
5. Re-run the failing test to verify
6. Run the full suite to check for regressions

## Coverage Guidelines

- Aim for at least one test per public function in each module
- Prioritize: happy path > edge cases > error paths
- Every new module added in a step should get at least a smoke test
- Use `pytest --co` (collect only) to verify test discovery without running
