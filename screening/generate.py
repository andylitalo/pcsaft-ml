"""Generate HFO blowing agent candidate SMILES with stereoisomer enumeration."""

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

# Base HFO structures: fluorinated propenes and butenes
# These are core scaffolds; substituent variations are generated below.
BASE_HFOS = [
    # Fluorinated propenes (HFO-1234 family)
    "FC=CC(F)(F)F",       # 1,3,3,3-tetrafluoroprop-1-ene (HFO-1234ze)
    "FC(=CF)C(F)F",       # 2,3,3,3-tetrafluoroprop-1-ene (HFO-1234yf)
    "F/C=C\\C(F)F",       # 1,1,3-trifluoroprop-1-ene
    "FC=CF",              # 1,2-difluoroethene
    "FC=CC(F)F",          # 1,3,3-trifluoroprop-1-ene
    "FC(F)=CC(F)F",       # 1,1,3,3-tetrafluoroprop-1-ene
    "FC=CC(F)(F)C(F)(F)F",  # fluorinated butene variant
    # Fluorinated butenes
    "FC=CCC(F)(F)F",      # fluorinated 1-butene
    "FCC=CC(F)F",         # fluorinated 2-butene
    "FC(F)=CCC(F)F",      # 1,1,4,4-tetrafluoro-2-butene
    "FC=CC(F)C(F)F",      # branching pattern
    # Cyclopropane derivatives (small ring fluorinated)
    "FC1CC1F",            # 1,2-difluorocyclopropane
    "FC1CC1(F)F",         # 1,1,2-trifluorocyclopropane
]


def _enumerate_stereoisomers(mol: Chem.Mol) -> list[Chem.Mol]:
    """Enumerate all stereoisomers of a molecule."""
    opts = StereoEnumerationOptions(unique=True, onlyUnassigned=False)
    return list(EnumerateStereoisomers(mol, options=opts))


def _generate_fluorine_variants(smi: str) -> list[str]:
    """Generate variants by adding/removing fluorine at available positions.

    Simple approach: for each H on a carbon, try replacing with F.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []

    variants = set()
    variants.add(Chem.MolToSmiles(mol))

    rw = Chem.RWMol(mol)
    Chem.SanitizeMol(rw)

    # Find carbons with implicit H that could be fluorinated
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # Carbon
            num_h = atom.GetTotalNumHs()
            if num_h > 0:
                # Create variant with one H replaced by F
                ed = Chem.RWMol(mol)
                ed.GetAtomWithIdx(atom.GetIdx()).SetNumExplicitHs(num_h - 1)
                f_idx = ed.AddAtom(Chem.Atom(9))  # Fluorine
                ed.AddBond(atom.GetIdx(), f_idx, Chem.BondType.SINGLE)
                try:
                    Chem.SanitizeMol(ed)
                    variants.add(Chem.MolToSmiles(ed))
                except Exception:
                    pass

    return list(variants)


def generate_hfo_candidates(
    base_smiles: list[str] | None = None,
) -> list[tuple[str, Chem.Mol]]:
    """Generate HFO candidate molecules with stereoisomer enumeration.

    Parameters
    ----------
    base_smiles : list[str] | None
        Starting SMILES. If None, uses built-in HFO scaffolds.

    Returns
    -------
    list[tuple[str, Mol]]
        List of (canonical_SMILES, RDKit Mol) tuples.
    """
    if base_smiles is None:
        base_smiles = BASE_HFOS

    seen = set()
    candidates = []

    for smi in base_smiles:
        # Generate fluorine variants from this scaffold
        variants = _generate_fluorine_variants(smi)

        for var_smi in variants:
            mol = Chem.MolFromSmiles(var_smi)
            if mol is None:
                continue

            # Enumerate stereoisomers
            stereo_isomers = _enumerate_stereoisomers(mol)
            if not stereo_isomers:
                stereo_isomers = [mol]

            for iso in stereo_isomers:
                can_smi = Chem.MolToSmiles(iso)
                if can_smi not in seen:
                    seen.add(can_smi)
                    candidates.append((can_smi, iso))

    print(f"Generated {len(candidates)} unique HFO candidates")
    return candidates
