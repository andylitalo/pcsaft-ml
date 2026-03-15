"""Convert SMILES strings to PyTorch Geometric molecular graph objects.

Atom features (9-dim):
    - Atomic number (one-hot, top 10 elements + other)
    - Degree (one-hot, 0-5+)
    - Formal charge (clipped)
    - Hybridization (one-hot, sp/sp2/sp3/sp3d/sp3d2)
    - Aromaticity (binary)
    - Number of Hs (0-4+)
    - In ring (binary)

Bond features (4-dim):
    - Bond type (one-hot: single/double/triple/aromatic)
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

# Top 10 most common elements in organic molecules
ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53, 14]  # C N O F P S Cl Br I Si
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _one_hot(value, choices: list) -> list[float]:
    encoding = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1.0
    except ValueError:
        encoding[-1] = 1.0  # "other" bucket
    return encoding


def atom_features(atom: Chem.Atom) -> list[float]:
    """Compute feature vector for a single atom."""
    features = []
    features.extend(_one_hot(atom.GetAtomicNum(), ATOM_LIST))
    features.extend(_one_hot(min(atom.GetDegree(), 5), list(range(6))))
    features.append(float(np.clip(atom.GetFormalCharge(), -2, 2)))
    features.extend(_one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST))
    features.append(float(atom.GetIsAromatic()))
    features.append(float(min(atom.GetTotalNumHs(), 4)))
    features.append(float(atom.IsInRing()))
    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    """Compute feature vector for a single bond."""
    bt = bond.GetBondType()
    features = [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
    ]
    return features


ATOM_FEATURE_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_FEATURE_DIM = 4


def smiles_to_graph(
    smiles: str,
    y: np.ndarray | None = None,
) -> Data | None:
    """Convert a SMILES string to a PyG Data object.

    Parameters
    ----------
    smiles : str
        SMILES string.
    y : np.ndarray, optional
        Target values (m, sigma, epsilon_k) as shape (3,).

    Returns
    -------
    Data or None
        PyG Data object, or None if SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edge index and features (undirected: both directions)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Single-atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles

    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float).unsqueeze(0)

    return data


class MoleculeGraphDataset(InMemoryDataset):
    """In-memory dataset of molecular graphs for PC-SAFT prediction.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings.
    targets : np.ndarray, optional
        Target array of shape (n, 3) for [m, sigma, epsilon_k].
    """

    def __init__(
        self,
        smiles_list: list[str],
        targets: np.ndarray | None = None,
    ):
        super().__init__()
        data_list = []
        skipped = 0
        for i, smi in enumerate(smiles_list):
            y = targets[i] if targets is not None else None
            graph = smiles_to_graph(smi, y=y)
            if graph is not None:
                data_list.append(graph)
            else:
                skipped += 1
        if skipped > 0:
            import logging
            logging.getLogger(__name__).warning(
                "Skipped %d/%d unparseable SMILES", skipped, len(smiles_list)
            )
        self.data, self.slices = self.collate(data_list)
