"""Merge validated submissions with existing training data."""

from kfp import dsl


def merge_datasets_logic(
    valid_csv: str,
    existing_csv: str,
    merged_csv: str,
    run_id: str,
    version_csv: str,
) -> int:
    """Core logic for merging validated data with existing training data.

    Parameters
    ----------
    valid_csv : str
        Path to validated submissions CSV.
    existing_csv : str
        Path to existing training data CSV.
    merged_csv : str
        Path to save merged dataset.
    run_id : str
        Pipeline run ID for versioning.
    version_csv : str
        Path to save versioned snapshot.

    Returns
    -------
    int
        Number of rows in merged dataset.
    """
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi

    # Load datasets
    valid_df = pd.read_csv(valid_csv)
    existing_df = pd.read_csv(existing_csv)

    # Combine
    combined = pd.concat([existing_df, valid_df], ignore_index=True)

    # Deduplicate by InChI (canonical molecular identifier)
    inchis = []
    for smi in combined["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        inchis.append(MolToInchi(mol) if mol else None)

    combined["_inchi"] = inchis
    combined = combined.dropna(subset=["_inchi"]).drop_duplicates(subset=["_inchi"])
    combined = combined.drop(columns=["_inchi"])

    # Save merged dataset
    combined.to_csv(merged_csv, index=False)

    # Save versioned snapshot
    combined.to_csv(version_csv, index=False)

    return len(combined)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.1.0", "rdkit==2023.9.1"],
)
def merge_datasets(
    valid_csv: str,
    existing_csv: str,
    run_id: str,
    merged_csv: dsl.OutputPath(str),
    version_csv: dsl.OutputPath(str),
) -> int:
    """Merge validated submissions with existing training data.

    Deduplicates on canonical InChI to handle identical molecules
    with different SMILES representations.

    Parameters
    ----------
    valid_csv : str
        Path to validated submissions CSV.
    existing_csv : str
        Path to existing training data CSV.
    run_id : str
        Pipeline run ID for versioned snapshot.
    merged_csv : OutputPath(str)
        Path to save merged and deduplicated dataset.
    version_csv : OutputPath(str)
        Path to save versioned snapshot (data/versions/train_v{run_id}.csv).

    Returns
    -------
    int
        Number of rows in merged dataset.
    """
    return merge_datasets_logic(
        valid_csv,
        existing_csv,
        merged_csv,
        run_id,
        version_csv,
    )
