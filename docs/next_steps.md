# Next Steps

This roadmap builds progressively from the current Random Forest baseline toward a production-grade, continuously-improving prediction system. Each step adds a concrete capability and demonstrates a specific engineering skill set. Detailed implementation guides live in [`docs/steps/`](steps/).

## Phase A: PyTorch Foundation (Steps 1–3)

### Step 1. Morgan Fingerprint Feature Module

Add Morgan fingerprints (2048-bit, radius 2) as a molecular representation and combine them with selected RDKit 2D descriptors into a unified feature vector (~2,150 dimensions). This creates the input pipeline that all subsequent neural network steps depend on.

**Skills**: RDKit cheminformatics, feature engineering, NumPy/Pandas.
**Guide**: [01_morgan_fingerprints.md](steps/01_morgan_fingerprints.md)

### Step 2. PyTorch Multi-Task Neural Network

Design and train a feedforward neural network with a shared trunk and three task-specific heads (m, σ, ε/k). Multi-task learning exploits the physical correlation between PC-SAFT parameters and acts as an implicit regularizer on a small dataset.

**Skills**: PyTorch `nn.Module`, `DataLoader`, training loops, LR scheduling, early stopping, multi-task loss weighting.
**Guide**: [02_pytorch_multitask_nn.md](steps/02_pytorch_multitask_nn.md)

### Step 3. Evaluation Harness

Build a unified evaluation framework that compares Random Forest vs. PyTorch NN (and later, ChemBERTa) on identical test splits. Produce parity plots, learning curves, and a metrics summary table.

**Skills**: scikit-learn metrics, matplotlib, experiment tracking, reproducibility.
**Guide**: [03_evaluation_harness.md](steps/03_evaluation_harness.md)

## Phase B: HuggingFace Transfer Learning (Step 4)

### Step 4. ChemBERTa Fine-Tuning

Fine-tune a pretrained molecular language model (ChemBERTa or MoLFormer) from HuggingFace Hub for PC-SAFT regression. The model reads raw SMILES strings—no descriptor engineering needed—and leverages representations learned from ~77M molecules.

**Skills**: HuggingFace `Trainer`, `AutoModel`, `AutoTokenizer`, transfer learning, model hub publishing.
**Guide**: [04_chemberta_finetuning.md](steps/04_chemberta_finetuning.md)

## Phase C: Production Deployment (Steps 5–8)

### Step 5. FastAPI Model Serving

Wrap the best model in a FastAPI endpoint with Pydantic request/response schemas, health checks, and a `/submit-data` route for researchers to contribute experimental PC-SAFT measurements.

**Skills**: FastAPI, Pydantic, async Python, REST API design.
**Guide**: [05_fastapi_serving.md](steps/05_fastapi_serving.md)

### Step 6. Docker & Kubernetes Deployment

Containerize the serving application with a multi-stage Dockerfile, write Kubernetes manifests (Deployment, Service, ConfigMap), and deploy to a local kind cluster or cloud.

**Skills**: Docker, Kubernetes manifests, container orchestration, resource management.
**Guide**: [06_docker_kubernetes.md](steps/06_docker_kubernetes.md)

### Step 7. Kubeflow Retrain Pipeline

Build a Kubeflow Pipelines DAG that validates newly submitted data, merges it with the training set, retrains the model, evaluates against a champion, and promotes the challenger if it wins.

**Skills**: Kubeflow Pipelines SDK, pipeline components, conditional logic, model registry, MLOps.
**Guide**: [07_kubeflow_retrain_pipeline.md](steps/07_kubeflow_retrain_pipeline.md)

### Step 8. Streamlit Academic Portal

Create a researcher-facing web portal where chemists input SMILES (or draw molecules), receive predicted PC-SAFT parameters with uncertainty, and optionally submit experimental ground truth that feeds the online learning pipeline.

**Skills**: Streamlit, UX for scientific tools, end-to-end system integration.
**Guide**: [08_streamlit_portal.md](steps/08_streamlit_portal.md)

## Phase D: Open-Source Release (Steps 26–30)

The ML pipeline, trained models, and novel predictions have standalone value for the thermodynamic modeling community. No usable, open-source PC-SAFT parameter prediction tool currently exists. Phase D packages the project as a public tool.

### Step 26. Extract Installable Package (`pcsaft-predict`)

Extract the prediction core into a standalone pip-installable Python package. A user can `pip install pcsaft-predict` and call `predict(["CCO"])` without cloning the repo. Exposes GNN, RF, and ensemble models through a 3-function API with uncertainty quantification and applicability domain checking.

**Skills**: Python packaging, API design, weight management, dependency minimization.
**Guide**: [26_extract_installable_package.md](steps/26_extract_installable_package.md)

### Step 27. Generalize Portal for Multi-Domain Use

Extend the Streamlit portal with application presets (blowing agents, refrigerants, solvents, general), a batch screening tab, and a direct-to-library fallback that eliminates the FastAPI dependency for casual users.

**Skills**: Streamlit, UX generalization, multi-audience design.
**Guide**: [27_generalize_portal.md](steps/27_generalize_portal.md)

### Step 28. Dataset and Model Card Publication

Package the 4,612+ novel predictions and trained models with standardized metadata (Datasheets for Datasets, Hugging Face Model Cards) for citation and reuse. Add BibTeX and CITATION.cff.

**Skills**: Data curation, ML documentation standards, scientific metadata.
**Guide**: [28_dataset_model_card_publication.md](steps/28_dataset_model_card_publication.md)

### Step 29. Documentation and Examples

Rewrite README for external users, create quickstart and screening-workflow Jupyter notebooks, write CONTRIBUTING.md, generate API reference.

**Skills**: Technical writing, Jupyter notebooks, documentation generation.
**Guide**: [29_documentation_examples.md](steps/29_documentation_examples.md)

### Step 30. Release Preparation

Add MIT license, GitHub Actions CI, issue templates, CHANGELOG, and perform pre-release audit. Tag v1.0.0, create GitHub Release with model weights, publish to PyPI.

**Skills**: CI/CD, open-source release management, PyPI publishing.
**Guide**: [30_release_preparation.md](steps/30_release_preparation.md)

---

## Earlier Ideas (retained for reference)

The items below were explored before the current roadmap was defined. Several are subsumed by the steps above; others remain independent future directions.

### ML-SAFT Dataset Integration

The ML-SAFT dataset (Felton et al., Chem. Eng. J., 2024) contains 988 molecules curated specifically for ML training of PC-SAFT parameter prediction. Could be combined with the Esper dataset for a larger, more balanced training set. Some underlying data may be under commercial license—verify before use.

### Graph Neural Networks (GNN)

Use molecular graphs directly as input (atoms as nodes, bonds as edges) with message-passing architectures (SchNet, DimeNet, MPNN). Libraries: PyTorch Geometric, DGL. Potentially better generalization to novel chemistries but more complex to implement.

### Multi-Objective Optimization

Extend screening to optimize PC-SAFT similarity, GWP, toxicity, flammability, and synthesis cost simultaneously via Pareto optimization.

### Expanded Chemical Space

Include HCFOs, fluorinated ethers, cyclic fluorinated compounds, and combinatorial enumeration of substitution patterns on C3–C5 scaffolds.

### Association Parameter Prediction (out of scope)

Extending the model to predict PC-SAFT association parameters (κ^AB, ε^AB/k) for hydrogen-bonding compounds. Investigated and ruled out because: (1) only ~200–500 compounds have experimentally fitted association parameters in the open literature, far too few for our ~2,150-dim feature space; (2) the screening already cannot find a close 3-parameter match to cyclopentane, so requiring 5-parameter agreement would further reduce the chance of finding viable candidates; (3) teqp does not support association, requiring a backend switch; (4) the SOTA approach (Winter et al. 2025) relies on commercially licensed training data. See `docs/reports/17_nist_experimental_validation.md` for the full analysis.
