"""Tests for fluorinated compound data integration."""

import pandas as pd
import pytest
from rdkit import Chem

from model.data.load import FLUORINATED_CSV, load_data


def test_fluorinated_csv_exists():
    """Fluorinated CSV file exists."""
    assert FLUORINATED_CSV.exists(), f"{FLUORINATED_CSV} not found"


def test_fluorinated_csv_has_required_columns():
    """Fluorinated CSV has all required columns."""
    df = pd.read_csv(FLUORINATED_CSV)
    required_cols = {"smiles", "m", "sigma", "epsilon_k", "name", "source"}
    missing = required_cols - set(df.columns)
    assert required_cols.issubset(df.columns), f"Missing columns: {missing}"


def test_fluorinated_csv_minimum_size():
    """Fluorinated CSV has at least 30 compounds (acceptance criterion)."""
    df = pd.read_csv(FLUORINATED_CSV)
    assert len(df) >= 30, f"Expected >=30 compounds, got {len(df)}"


def test_fluorinated_smiles_are_valid():
    """All SMILES in fluorinated CSV are valid RDKit molecules."""
    df = pd.read_csv(FLUORINATED_CSV)
    invalid = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            invalid.append((idx, row["smiles"], row.get("name", "unknown")))

    assert not invalid, f"Invalid SMILES found: {invalid}"


def test_fluorinated_smiles_are_canonical():
    """All SMILES in fluorinated CSV are canonical RDKit SMILES."""
    df = pd.read_csv(FLUORINATED_CSV)
    non_canonical = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            canonical = Chem.MolToSmiles(mol)
            if canonical != row["smiles"]:
                non_canonical.append((idx, row["smiles"], canonical, row.get("name", "unknown")))

    assert not non_canonical, f"Non-canonical SMILES found: {non_canonical}"


def test_fluorinated_parameters_are_positive():
    """All PC-SAFT parameters are positive."""
    df = pd.read_csv(FLUORINATED_CSV)
    for param in ["m", "sigma", "epsilon_k"]:
        assert (df[param] > 0).all(), f"{param} contains non-positive values"


def test_fluorinated_parameters_in_reasonable_range():
    """PC-SAFT parameters are in physically reasonable ranges."""
    df = pd.read_csv(FLUORINATED_CSV)

    # m typically 1-10 for small molecules
    assert (df["m"] >= 0.5).all() and (df["m"] <= 15).all(), "m out of range"

    # sigma typically 2-5 Å
    assert (df["sigma"] >= 2.0).all() and (df["sigma"] <= 6.0).all(), "sigma out of range"

    # epsilon_k typically 100-500 K for small molecules
    assert (df["epsilon_k"] >= 50).all() and (df["epsilon_k"] <= 600).all(), (
        "epsilon_k out of range"
    )


def test_fluorinated_all_have_sources():
    """All entries have literature source references."""
    df = pd.read_csv(FLUORINATED_CSV)
    missing_source = df[df["source"].isna() | (df["source"] == "")]
    assert len(missing_source) == 0, f"Entries without sources: {missing_source['name'].tolist()}"


def test_fluorinated_sources_have_doi_or_reference():
    """All sources contain DOI or clear reference."""
    df = pd.read_csv(FLUORINATED_CSV)
    invalid_sources = []
    for idx, row in df.iterrows():
        source = row["source"]
        # Check for DOI pattern or year in parentheses (citation format)
        has_doi = "doi:" in source.lower() or "10." in source
        has_citation = "(" in source and ")" in source
        if not (has_doi or has_citation):
            invalid_sources.append((idx, source, row.get("name", "unknown")))

    assert not invalid_sources, f"Invalid source formats: {invalid_sources}"


def test_load_fluorinated_only():
    """load_data(source='fluorinated') returns fluorinated dataset."""
    df = load_data(source="fluorinated")
    assert len(df) >= 30, f"Expected >=30 compounds, got {len(df)}"
    assert "smiles" in df.columns
    assert "m" in df.columns
    assert "sigma" in df.columns
    assert "epsilon_k" in df.columns


def test_load_combined_includes_fluorinated(tmp_path):
    """Combined loader includes fluorinated data when available."""
    # This test assumes all datasets exist (esper, mlsaft, fluorinated)
    # If they don't all exist, it will test with whatever is available
    try:
        df_combined = load_data(source="combined")
        df_fluorinated = load_data(source="fluorinated")

        # Combined should have at least as many rows as fluorinated
        # (could have more if esper/mlsaft don't overlap)
        assert len(df_combined) >= len(df_fluorinated) * 0.5, \
            "Combined dataset doesn't seem to include fluorinated data"

    except FileNotFoundError:
        pytest.skip("Not all datasets available for combined test")


def test_fluorinated_deduplication_with_combined():
    """Fluorinated data is properly deduplicated when loaded with combined."""
    try:
        df_combined = load_data(source="combined")
        df_fluorinated = load_data(source="fluorinated")

        # Check if any fluorinated SMILES made it into combined
        fluorinated_smiles = set(df_fluorinated["smiles"])
        combined_smiles = set(df_combined["smiles"])

        # At least some fluorinated compounds should be in combined
        overlap = fluorinated_smiles & combined_smiles
        assert len(overlap) > 0, "No fluorinated compounds found in combined dataset"

    except FileNotFoundError:
        pytest.skip("Not all datasets available for deduplication test")


def test_fluorinated_chemical_diversity():
    """Fluorinated dataset contains diverse chemical classes."""
    df = pd.read_csv(FLUORINATED_CSV)

    # Check for key chemical classes by name patterns
    hfcs = df[df["name"].str.contains("R-", case=False, na=False)]
    hfos = df[df["name"].str.contains("HFO", case=False, na=False)]
    perfluoro = df[df["name"].str.contains("perfluoro", case=False, na=False)]
    cyclo = df[df["name"].str.contains("cyclo", case=False, na=False)]

    assert len(hfcs) > 0, "No HFC refrigerants found"
    assert len(hfos) > 0, "No HFO compounds found"
    assert len(perfluoro) > 0, "No perfluorocarbons found"
    assert len(cyclo) > 0, "No fluorinated cycloalkanes found"


def test_fluorinated_contains_fluorine():
    """All fluorinated compounds actually contain fluorine atoms."""
    df = pd.read_csv(FLUORINATED_CSV)
    missing_fluorine = []

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            has_fluorine = any(atom.GetSymbol() == "F" for atom in mol.GetAtoms())
            if not has_fluorine:
                missing_fluorine.append((idx, row["smiles"], row.get("name", "unknown")))

    assert not missing_fluorine, f"Compounds without fluorine: {missing_fluorine}"


def test_fluorinated_no_duplicate_smiles():
    """No duplicate SMILES in fluorinated dataset."""
    df = pd.read_csv(FLUORINATED_CSV)
    duplicates = df[df["smiles"].duplicated()]
    dup_records = duplicates[["smiles", "name"]].to_dict("records")
    assert len(duplicates) == 0, f"Duplicate SMILES found: {dup_records}"
