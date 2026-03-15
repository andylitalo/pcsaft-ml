# Step 2B: Graph Neural Network (GIN) for PC-SAFT Prediction

## Purpose

Step 02 showed that a feedforward NN operating on Morgan fingerprints + RDKit descriptors underperforms the RF baseline in the small-data regime (R² = 0.42/0.04/0.09 vs 0.62/0.35/0.33 on m/σ/ε/k). The bottleneck is the representation, not the learning algorithm: fingerprints flatten molecular topology into fixed-length bit vectors, losing connectivity information that is critical for physical properties like segment diameter (σ) and dispersion energy (ε/k).

This branch replaces the flat feature vector with a **molecular graph representation** processed by a **Graph Isomorphism Network (GIN)**, the architecture that underlies modern molecular property prediction (Xu et al. 2019, Hu et al. 2020). Instead of asking "which substructures are present?" (fingerprints), the GNN asks "how are atoms connected, and what local environments do they form?" — a strictly more informative question for PC-SAFT parameters.

Simultaneously, this step integrates two additional data sources (ML-SAFT: 870 mol, SPT-PCSAFT: 13,646 mol) to address the data scarcity that limited all Phase 1 models.

## What It Adds Over Step 2

| After Step 2 | After This Step |
|--------------|----------------|
| MLP on fingerprints (2,218-dim flat vector) | GIN on molecular graphs (atom/bond features, connectivity) |
| Input representation loses connectivity | Full molecular topology preserved |
| 1,801 training molecules (Esper only) | Up to 13,764 molecules (Esper + ML-SAFT + SPT-PCSAFT) |
| Best NN: R²=0.42/0.04/0.09 on m/σ/ε/k | GNN: R²=0.69/0.34/0.41 (combined) or 0.76/0.77/0.73 (all) |
| RF is clearly the best model | GNN overtakes RF on all three targets |

## Skills Demonstrated

- **Graph neural networks**: GINEConv message passing, edge-feature-aware convolutions, residual connections
- **Molecular featurization**: SMILES → PyG `Data` objects (atom features, bond features, edge indices)
- **PyTorch Geometric**: `InMemoryDataset`, `DataLoader`, batched graph operations
- **Data integration**: Downloading and merging heterogeneous PC-SAFT datasets with InChI deduplication
- **Multi-task graph regression**: Global pooling → shared MLP head for m, σ, ε/k simultaneously

## Dependencies

- Requires: Esper dataset (from Step 00), optionally ML-SAFT and SPT-PCSAFT
- New packages: `torch-geometric>=2.4`

```toml
[project.optional-dependencies]
gnn = ["torch>=2.0,<3.0", "torch-geometric>=2.4"]
```

Install: `uv sync --extra dev --extra gnn`

## Architecture

### Graph Featurizer (`model/gnn/graph_featurizer.py`)

Each molecule is converted to a PyG `Data` object:

**Atom features** (28-dim): atomic number (one-hot over top 10 elements + other), degree (one-hot 0-5+), formal charge (clipped), hybridization (one-hot sp/sp2/sp3/sp3d/sp3d2), aromaticity, number of Hs, in-ring flag.

**Bond features** (4-dim): bond type one-hot (single/double/triple/aromatic).

**Edge index**: undirected — each bond produces two directed edges (i→j, j→i).

### GIN Architecture (`model/gnn/architecture.py`)

```
SMILES → RDKit mol → atom/bond features + edge index
       ↓
  node_encoder: Linear(28 → 256)
  edge_encoder: Linear(4 → 256)
       ↓
  4× GINEConv layer (with residual connections)
     [MLP: 256→256→ReLU→256] + BatchNorm + ReLU + Dropout(0.1)
       ↓
  Global readout: mean_pool ⊕ max_pool → 512-dim
       ↓
  MLP head: 512 → 256 → ReLU → 128 → ReLU → 3
       ↓
  (m, σ, ε/k) predictions
```

~965K parameters. Trains in ~30s on CPU (combined dataset) or ~2.5 min (all dataset).

### Why GIN over other GNN architectures

- **GIN is maximally expressive** among message-passing NNs for distinguishing non-isomorphic graphs (Xu et al. 2019). This matters for PC-SAFT because structurally similar molecules (e.g., positional isomers) can have meaningfully different dispersion energies.
- **GINEConv** extends GIN with edge features, capturing bond-type information (single vs. double vs. aromatic) that directly influences σ and ε/k.
- **Simplicity**: GIN has fewer hyperparameters than attention-based alternatives (GAT, Transformer) and trains stably on small datasets without extensive tuning.

## Data Sources

| Source | Molecules | Type | Citation |
|--------|-----------|------|----------|
| Esper | 1,801 | Experimentally fitted | Esper et al. 2023, Ind. Eng. Chem. Res. |
| ML-SAFT | 870 | Programmatically regressed | Felton et al. 2024, Chem. Eng. J. |
| SPT-PCSAFT | 13,646 | Transformer-predicted | Winter et al. 2025, Digital Discovery |

`load_data("combined")` → Esper + ML-SAFT (1,926 mol after InChI dedup)
`load_data("all")` → Esper + ML-SAFT + SPT-PCSAFT (13,764 mol after InChI dedup)

Priority for duplicates: Esper > ML-SAFT > SPT-PCSAFT (experimentally fitted preferred).

**Caveat**: SPT-PCSAFT parameters are model predictions, not experimental fits. They add coverage and reduce variance but may introduce systematic bias from the source model.

## Training

```bash
# On experimentally-derived data only (fair comparison to RF)
python -m model.gnn.train_gnn --source combined --epochs 100 --patience 15

# On all available data (maximum performance)
python -m model.gnn.train_gnn --source all --epochs 100 --patience 15
```

Training details: AdamW (lr=1e-3, weight_decay=1e-5), cosine annealing LR, gradient clipping (max_norm=5.0), early stopping on validation MSE.

## Acceptance Criteria

- [x] `model/gnn/` package with `graph_featurizer.py`, `architecture.py`, `train_gnn.py`
- [x] GNN registered as `"gnn"` in model registry
- [x] `predict()` and `predict_with_uncertainty()` (MC Dropout) available via registry
- [x] GNN trained on combined data matches or beats RF on ≥2 of 3 targets
- [x] GNN trained on all data achieves R² > 0.70 on all three targets
- [x] Model checkpoint saves/loads correctly via `model/saved/gnn_pcsaft.pt`
- [x] Data download scripts work: `python -m model.data.download_mlsaft`, `python -m model.data.download_spt_pcsaft`
- [x] `pyproject.toml` updated with `gnn` optional dependency group
