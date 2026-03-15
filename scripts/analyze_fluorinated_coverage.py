"""Analyze the chemical coverage of fluorinated data expansion.

This script compares the fluorinated dataset to the existing training data
to quantify the improvement in chemical diversity, particularly for
screening candidate chemical space (fluorinated cycloalkanes, HFOs).
"""

from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import BulkTanimotoSimilarity

from model.data.load import load_data


def count_fluorine_atoms(smiles: str) -> int:
    """Count fluorine atoms in a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F")


def has_ring(smiles: str) -> bool:
    """Check if molecule has any ring."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.GetRingInfo().NumRings() > 0


def get_ring_sizes(smiles: str) -> list[int]:
    """Get sizes of all rings in a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [len(ring) for ring in mol.GetRingInfo().AtomRings()]


def is_fluorinated(smiles: str) -> bool:
    """Check if molecule contains fluorine."""
    return count_fluorine_atoms(smiles) > 0


def classify_compound(smiles: str, name: str = "") -> str:
    """Classify fluorinated compound by type."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"

    n_f = count_fluorine_atoms(smiles)
    if n_f == 0:
        return "non-fluorinated"

    # Check for specific classes
    if "HFO" in name.upper():
        return "HFO"
    if "R-" in name:
        return "HFC/HCFC"
    if "perfluoro" in name.lower():
        return "perfluorocarbon"

    # Ring classification
    ring_sizes = get_ring_sizes(smiles)
    if ring_sizes:
        if any(s <= 5 for s in ring_sizes):
            return "fluorinated_ring"
        if 6 in ring_sizes:
            # Check for aromatic
            if Descriptors.NumAromaticRings(mol) > 0:
                return "fluorinated_aromatic"

    # Other classifications
    n_atoms = mol.GetNumHeavyAtoms()
    if n_f / n_atoms > 0.7:
        return "highly_fluorinated"

    return "fluorinated_alkane/alkene"


def compute_similarity_to_training(
    fluorinated_smiles: list[str], training_smiles: list[str], radius: int = 2, n_bits: int = 2048
) -> dict[str, float]:
    """Compute Tanimoto similarity distribution of fluorinated compounds to training set.

    For each fluorinated compound, compute max Tanimoto similarity to any training compound.

    Parameters
    ----------
    fluorinated_smiles : list[str]
        SMILES of fluorinated compounds.
    training_smiles : list[str]
        SMILES of training compounds.
    radius : int
        Morgan fingerprint radius.
    n_bits : int
        Fingerprint size.

    Returns
    -------
    dict[str, float]
        Summary statistics: mean, median, min, max, pct_similar (>0.5).
    """
    # Generate fingerprints for training set
    training_fps = []
    for smi in training_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            training_fps.append(fp)

    if not training_fps:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "pct_similar": 0.0}

    # For each fluorinated compound, find max similarity to training
    max_similarities = []
    for smi in fluorinated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            sims = BulkTanimotoSimilarity(fp, training_fps)
            max_similarities.append(max(sims))

    if not max_similarities:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "pct_similar": 0.0}

    import numpy as np

    return {
        "mean": float(np.mean(max_similarities)),
        "median": float(np.median(max_similarities)),
        "min": float(np.min(max_similarities)),
        "max": float(np.max(max_similarities)),
        "pct_similar": float(sum(s > 0.5 for s in max_similarities) / len(max_similarities) * 100),
    }


def main():
    """Analyze fluorinated data coverage."""
    print("=" * 70)
    print("Fluorinated Compound Data Expansion: Coverage Analysis")
    print("=" * 70)

    # Load datasets
    df_fluorinated = load_data(source="fluorinated")
    try:
        df_esper = load_data(source="esper")
        df_mlsaft = load_data(source="mlsaft")
        df_combined_orig = pd.concat([df_esper, df_mlsaft], ignore_index=True)
        df_combined_orig = df_combined_orig.drop_duplicates(subset=["smiles"])
    except FileNotFoundError:
        print("Warning: Could not load all datasets, using fluorinated only\n")
        df_combined_orig = pd.DataFrame()

    print("\n1. Dataset Sizes")
    print(f"   Fluorinated compounds: {len(df_fluorinated)}")
    if not df_combined_orig.empty:
        print(f"   Original training data (Esper + ML-SAFT): {len(df_combined_orig)}")
        combined_total = len(df_combined_orig) + len(df_fluorinated)
        print(f"   Combined (with fluorinated): {combined_total} (before deduplication)")

    # Chemical class distribution in fluorinated dataset
    print("\n2. Fluorinated Dataset Chemical Classes")
    classes = [
        classify_compound(row["smiles"], row.get("name", ""))
        for _, row in df_fluorinated.iterrows()
    ]
    class_counts = Counter(classes)
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"   {cls}: {count}")

    # Fluorine count distribution
    print("\n3. Fluorine Atom Distribution in Fluorinated Dataset")
    f_counts = [count_fluorine_atoms(smi) for smi in df_fluorinated["smiles"]]
    f_count_dist = Counter(f_counts)
    for n_f, count in sorted(f_count_dist.items()):
        print(f"   {n_f} fluorine atoms: {count} compounds")

    # Ring analysis
    print("\n4. Ring Analysis in Fluorinated Dataset")
    rings_in_fluor = [smi for smi in df_fluorinated["smiles"] if has_ring(smi)]
    print(f"   Fluorinated rings: {len(rings_in_fluor)} / {len(df_fluorinated)}")

    ring_size_counts = Counter()
    for smi in rings_in_fluor:
        for size in get_ring_sizes(smi):
            ring_size_counts[size] += 1
    for size, count in sorted(ring_size_counts.items()):
        print(f"   {size}-membered rings: {count}")

    # Comparison to original training data
    if not df_combined_orig.empty:
        print("\n5. Fluorinated Coverage in Original Training Data")
        orig_fluorinated = [smi for smi in df_combined_orig["smiles"] if is_fluorinated(smi)]
        pct = len(orig_fluorinated) / len(df_combined_orig) * 100
        print(
            f"   Fluorinated compounds in original data: {len(orig_fluorinated)} / "
            f"{len(df_combined_orig)} ({pct:.1f}%)"
        )

        orig_fluor_rings = [smi for smi in orig_fluorinated if has_ring(smi)]
        print(f"   Fluorinated rings in original data: {len(orig_fluor_rings)}")

        print("\n6. Tanimoto Similarity to Original Training Data")
        sim_stats = compute_similarity_to_training(
            df_fluorinated["smiles"].tolist(), df_combined_orig["smiles"].tolist()
        )
        print(f"   Mean max similarity: {sim_stats['mean']:.3f}")
        print(f"   Median max similarity: {sim_stats['median']:.3f}")
        print(f"   Min/Max: {sim_stats['min']:.3f} / {sim_stats['max']:.3f}")
        print(f"   Compounds with >0.5 similarity to training: {sim_stats['pct_similar']:.1f}%")
        print(f"   Novel chemical space (<0.5 similarity): {100 - sim_stats['pct_similar']:.1f}%")

    # Summary for blowing agent screening
    print("\n7. Coverage for Blowing Agent Screening Candidates")
    fluor_cyclo_3_to_5 = []
    for _, row in df_fluorinated.iterrows():
        ring_sizes = get_ring_sizes(row["smiles"])
        if any(3 <= s <= 5 for s in ring_sizes) and is_fluorinated(row["smiles"]):
            fluor_cyclo_3_to_5.append(row)

    print(f"   Fluorinated 3-5 membered rings: {len(fluor_cyclo_3_to_5)}")
    if fluor_cyclo_3_to_5:
        print("   Examples:")
        for entry in fluor_cyclo_3_to_5[:5]:
            print(f"     - {entry.get('name', 'unknown')}: {entry['smiles']}")

    print("\n" + "=" * 70)
    print("Analysis complete. See docs/reports/16_fluorinated_data_expansion.md for full report.")
    print("=" * 70)


if __name__ == "__main__":
    main()
