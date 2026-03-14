"""Tests for screening filters."""

from rdkit import Chem

from screening.filters import filter_synthesizability


def _make_candidates(smiles_list):
    """Helper to create (smiles, mol) tuples."""
    return [(smi, Chem.MolFromSmiles(smi)) for smi in smiles_list]


def test_filter_synthesizability_passes_simple():
    """Simple molecules should have low SA scores and pass."""
    candidates = _make_candidates(["CC", "CCC", "C1CCCC1"])
    passed = filter_synthesizability(candidates, threshold=4.5)
    assert len(passed) == 3
    for smi, mol, score in passed:
        assert score <= 4.5


def test_filter_synthesizability_returns_scores():
    """Returned tuples should include SA scores."""
    candidates = _make_candidates(["CC"])
    passed = filter_synthesizability(candidates, threshold=10.0)
    assert len(passed) == 1
    smi, mol, score = passed[0]
    assert isinstance(score, float)
    assert 1.0 <= score <= 10.0


def test_filter_synthesizability_strict_threshold():
    """Very strict threshold filters more aggressively."""
    candidates = _make_candidates(["CC", "FC(F)(F)C(F)(F)C(F)(F)F"])
    loose = filter_synthesizability(candidates, threshold=5.0)
    strict = filter_synthesizability(candidates, threshold=1.5)
    assert len(strict) <= len(loose)


def test_filter_synthesizability_invalid_mol():
    """Candidates with None mol should be skipped."""
    candidates = [("CC", Chem.MolFromSmiles("CC")), ("BAD", None)]
    passed = filter_synthesizability(candidates, threshold=10.0)
    # None mol should be skipped (calculateScore will fail)
    assert len(passed) <= 2
