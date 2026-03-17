# GNN Architectures for PC-SAFT Parameter Prediction: Supplementary Information

This document explains the two graph neural network architectures tested for predicting PC-SAFT parameters (m, σ, ε/k) from molecular structure. Both operate directly on molecular graphs, bypassing hand-crafted feature engineering.

---

## GINEConv (Graph Isomorphism Network with Edge features)

### References

- Xu et al. (2019) "How Powerful are Graph Neural Networks?" — original GIN architecture
- Hu et al. (2020) "Strategies for Pre-training Graph Neural Networks" — GIN with edge features (GINEConv)

### Architecture

GINEConv performs **atom-level message passing**. Each atom's representation is updated based on its bonded neighbors' features. The update incorporates edge (bond) features explicitly, allowing the model to distinguish single, double, triple, and aromatic bonds. This design gives GIN strong expressiveness for graph isomorphism testing and has been shown to be as powerful as the Weisfeiler-Lehman test in distinguishing graph structures.

### Configuration in This Project

The primary configuration uses 4 GINEConv layers with hidden_dim=256, totaling 964,867 parameters. Each layer is followed by BatchNorm, ReLU, and dropout (0.1). Residual connections connect layer outputs. A global readout concatenates mean pooling and max pooling over node representations, producing a 512-dim graph embedding fed to a multi-task MLP head that predicts m, σ, and ε/k jointly.

A smaller Esper-only configuration was also tested: 3 layers, hidden_dim=128, 195K parameters. This reduced model was designed for the 1,801-molecule Esper corpus to mitigate overfitting.

**Implementation**: `model/gnn/architecture.py` — `PCSAFTGraphNet` class.

---

## Chemprop D-MPNN (Directed Message Passing Neural Network)

### Reference

- Yang et al., *Journal of Chemical Information and Modeling* **59** (2019) — original D-MPNN

### Architecture

The D-MPNN performs **bond-level (directed edge) message passing**. Messages flow along directed bonds, with each directed edge maintaining its own message vector. This design avoids "information collision": in atom-level message passing, messages from multiple neighbors can interfere when aggregated at a node; the D-MPNN routes messages through bonds in a way that preserves more distinct information. Messages are updated as they traverse the molecular graph, and a final aggregation produces molecular representations.

### Configuration in This Project

Default configuration: depth=3, hidden_dim=300, 318K parameters. Training uses batch_size=64, early stopping with patience=5, and a 10% validation split. The model uses chemprop v2 with `BondMessagePassing` for the message-passing module, `MeanAggregation` for graph-level readout, and `RegressionFFN` for the multi-task prediction head.

**Implementation**: `model/chemprop_model/chemprop_wrapper.py` — `ChempropPCSAFT` class.

---

## The Physics Connection

GNN message-passing mirrors PC-SAFT's physical picture. PC-SAFT models molecules as chains of spherical segments; dispersion energy (ε/k) depends on how segments interact with their local chemical environment. The GNN's iterative aggregation — where each node (atom) encodes information from its k-hop neighborhood after k layers — captures exactly this local context. After 4 GINEConv layers, each node representation encodes a 4-hop neighborhood: the connectivity and bond types that determine how a segment contributes to m, σ, and ε/k.

This is conceptually a **data-driven generalization of the group-contribution method**. Traditional group-contribution schemes assign fixed increments per functional group (CH3, CH2, F, etc.) and sum them. GNNs learn analogous contributions from data but can capture non-additive interactions between groups — for example, how fluorine substitution on a ring differs from substitution on a chain. The learned representations are not interpretable in the same way as group contributions, but they encode richer structure–property relationships.

Both GNN variants consistently outperformed fingerprint-based models (RF, Morgan + RDKit descriptors) on ε/k, with R² gains of +0.06 to +0.08 on the same Esper data.

---

## Performance Comparison

| Model | Dataset | R²(ε/k) |
|-------|---------|---------|
| RF (Morgan + RDKit) | Esper (~1,800 mol) | 0.33 |
| Chemprop D-MPNN | Esper | 0.39 |
| GINEConv (4-layer, 256-dim) | Esper | 0.41 |
| GINEConv (4-layer, 256-dim) | Unified (~13,764 mol) | 0.73 |

On the Esper test set (n≈361), GINEConv achieves the best ε/k R² (0.41), followed by Chemprop (0.39), with RF at 0.33. The GNN advantage is real but modest at this data scale. Both architectures require more data to realize their full potential: with ~13,764 training molecules, the GINEConv model reaches R²(ε/k)=0.73 — a substantial jump driven by the improved data-to-parameter ratio and the graph representation's ability to exploit topology that fingerprints cannot encode.
