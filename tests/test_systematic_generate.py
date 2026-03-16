"""Tests for systematic HFO/HCFO candidate generation."""

from rdkit import Chem

from screening.generate import (
    _enumerate_halogen_patterns,
    generate_hfo_candidates,
    generate_systematic_candidates,
)


def test_enumerate_halogen_patterns_ethene():
    """Ethene (C=C, 4 H on C) should produce known fluoroethylenes."""
    patterns = _enumerate_halogen_patterns("C=C", max_cl=0, max_mw=200)
    assert len(patterns) > 0
    # Must include vinyl fluoride and vinylidene fluoride
    canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in ["FC=C", "FC=CF", "FC(F)=C"]}
    for expected in canon:
        assert expected in patterns, f"Missing {expected}"


def test_enumerate_halogen_patterns_excludes_parent():
    """The unsubstituted parent hydrocarbon should not appear (no halogens)."""
    patterns = _enumerate_halogen_patterns("CC=C", max_cl=0, max_mw=200)
    propene = Chem.MolToSmiles(Chem.MolFromSmiles("CC=C"))
    assert propene not in patterns


def test_enumerate_halogen_patterns_with_chlorine():
    """With max_cl=1, should produce HCFOs (exactly one Cl)."""
    patterns = _enumerate_halogen_patterns("CC=C", max_cl=1, max_mw=200)
    has_cl = sum(1 for s in patterns if "Cl" in s)
    has_f_only = sum(1 for s in patterns if "F" in s and "Cl" not in s)
    assert has_cl > 0, "Should produce HCFO candidates"
    assert has_f_only > 0, "Should also produce HFO candidates"


def test_enumerate_halogen_patterns_hcfo1233zd():
    """The industrial blowing agent HCFO-1233zd should be generated from propene."""
    patterns = _enumerate_halogen_patterns("CC=C", max_cl=1, max_mw=200)
    hcfo_1233zd = Chem.MolToSmiles(Chem.MolFromSmiles("ClC=CC(F)(F)F"))
    assert hcfo_1233zd in patterns, (
        f"HCFO-1233zd ({hcfo_1233zd}) should be generated from propene backbone"
    )


def test_enumerate_halogen_patterns_hfo1234ze():
    """HFO-1234ze should be generated from propene."""
    patterns = _enumerate_halogen_patterns("CC=C", max_cl=0, max_mw=200)
    hfo_1234ze = Chem.MolToSmiles(Chem.MolFromSmiles("FC=CC(F)(F)F"))
    assert hfo_1234ze in patterns, (
        f"HFO-1234ze ({hfo_1234ze}) should be generated from propene backbone"
    )


def test_enumerate_halogen_patterns_mw_filter():
    """High-MW molecules should be excluded."""
    all_patterns = _enumerate_halogen_patterns("CCCC=C", max_cl=0, max_mw=999)
    filtered = _enumerate_halogen_patterns("CCCC=C", max_cl=0, max_mw=120)
    assert len(filtered) < len(all_patterns)


def test_enumerate_halogen_patterns_max_cl_zero():
    """max_cl=0 should produce only F-substituted molecules (no Cl)."""
    patterns = _enumerate_halogen_patterns("CC=C", max_cl=0, max_mw=200)
    for smi in patterns:
        assert "Cl" not in smi, f"Unexpected Cl in HFO-only mode: {smi}"


def test_generate_systematic_candidates_nonempty():
    """Systematic generation should produce a large number of candidates."""
    candidates = generate_systematic_candidates(max_cl=1, max_mw=200)
    assert len(candidates) > 1000, (
        f"Expected >1000 systematic candidates, got {len(candidates)}"
    )


def test_generate_systematic_candidates_all_valid():
    """Every generated candidate should be a valid RDKit Mol."""
    candidates = generate_systematic_candidates(
        backbones=["C=C", "CC=C"], max_cl=0, max_mw=200,
    )
    for smi, mol in candidates:
        assert mol is not None, f"Invalid mol for {smi}"
        assert Chem.MolToSmiles(mol) == smi


def test_generate_systematic_candidates_all_have_double_bond():
    """Every generated candidate must retain a C=C double bond."""
    candidates = generate_systematic_candidates(
        backbones=["C=C", "CC=C", "CCC=C"], max_cl=0, max_mw=200,
    )
    cc_double = Chem.MolFromSmarts("C=C")
    for smi, mol in candidates:
        assert mol.HasSubstructMatch(cc_double), (
            f"Missing C=C double bond in {smi}"
        )


def test_generate_systematic_includes_stereoisomers():
    """E/Z stereoisomers should be enumerated for 2-butene derivatives."""
    candidates = generate_systematic_candidates(
        backbones=["CC=CC"], max_cl=0, max_mw=200,
    )
    smiles_set = {smi for smi, _ in candidates}
    has_stereo = any("/" in s or "\\" in s for s in smiles_set)
    assert has_stereo, "Should enumerate E/Z stereoisomers"


def _strip_stereo(smi: str) -> str:
    """Canonical SMILES with stereochemistry removed."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def test_generate_hfo_candidates_systematic_mode():
    """The 'systematic' scaffold_set should use combinatorial enumeration."""
    candidates = generate_hfo_candidates(scaffold_set="systematic", max_mw=150)
    assert len(candidates) > 100
    flat_smiles = {_strip_stereo(s) for s, _ in candidates}
    hfo_1234ze = _strip_stereo("FC=CC(F)(F)F")
    assert hfo_1234ze in flat_smiles, (
        "HFO-1234ze (or its stereoisomers) should be in systematic candidates"
    )


def test_generate_hfo_candidates_backward_compatible():
    """Old scaffold sets should still work unchanged."""
    candidates = generate_hfo_candidates(scaffold_set="hfo")
    assert len(candidates) > 0


def test_systematic_no_duplicates():
    """Generated candidates should have unique SMILES."""
    candidates = generate_systematic_candidates(
        backbones=["CC=C"], max_cl=1, max_mw=200,
    )
    smiles = [s for s, _ in candidates]
    assert len(smiles) == len(set(smiles)), "Duplicate SMILES found"


def test_systematic_at_least_one_f():
    """Every HFO/HCFO candidate must contain at least one fluorine."""
    candidates = generate_systematic_candidates(
        backbones=["CC=C"], max_cl=1, max_mw=200,
    )
    for smi, mol in candidates:
        has_f = any(a.GetAtomicNum() == 9 for a in mol.GetAtoms())
        has_cl = any(a.GetAtomicNum() == 17 for a in mol.GetAtoms())
        assert has_f or has_cl, f"No halogens in {smi}"
