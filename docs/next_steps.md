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
