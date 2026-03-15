"""Generate blowing agent candidate SMILES with stereoisomer enumeration.

Supports multiple scaffold families:
- HFOs (hydrofluoroolefins): fluorinated propenes and butenes
- HCFOs (hydrochlorofluoroolefins): Cl-substituted HFO variants
- HFEs (hydrofluoroethers): ether linkages with fluorinated chains
- Unsaturated hydrocarbons: C3-C5 alkenes and cycloalkenes
- Cyclic fluorinated: C3-C6 rings with F substitution patterns
- Systematic: exhaustive combinatorial F/Cl substitution on alkene backbones
"""

import logging
from itertools import combinations, product

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

logger = logging.getLogger(__name__)

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

# HCFOs: hydrochlorofluoroolefins (Cl-substituted variants)
BASE_HCFOS = [
    "ClC=CC(F)(F)F",     # 1-chloro-3,3,3-trifluoroprop-1-ene (HCFO-1233zd)
    "FC(Cl)=CC(F)F",     # chloro-difluoropropene
    "ClC=CF",            # 1-chloro-2-fluoroethene
    "FC(=CCl)C(F)F",     # chloro-trifluoropropene variant
    "ClC(F)=CC(F)(F)F",  # 1-chloro-1,3,3,3-tetrafluoroprop-1-ene
    "ClC=CC(F)F",        # 1-chloro-3,3-difluoroprop-1-ene
    "FC(Cl)=CF",         # 1-chloro-1,2-difluoroethene
    "ClC(F)=CCF",        # chloro-fluoro-propene
    "ClC=CCC(F)(F)F",    # chloro-fluorobutene
    "FC(Cl)=CCC(F)F",    # chloro-difluorobutene
]

# HFEs: hydrofluoroethers (ether linkages with fluorinated C2-C4 chains)
BASE_HFES = [
    "COCF",              # fluoromethyl methyl ether
    "COC(F)F",           # difluoromethyl methyl ether
    "COC(F)(F)F",        # trifluoromethyl methyl ether
    "CCOC(F)F",          # difluoromethyl ethyl ether
    "CCOC(F)(F)F",       # trifluoromethyl ethyl ether
    "FC(F)OC(F)F",       # bis(difluoromethyl) ether
    "FC(F)OC(F)(F)F",    # difluoromethyl trifluoromethyl ether
    "COCC(F)(F)F",       # 3,3,3-trifluoropropyl methyl ether
    "COCF(F)CF",         # fluorinated propyl ether
    "CCOCC(F)F",         # 2,2-difluoroethyl ethyl ether
    "FC(F)(F)OCCF",      # trifluoromethyl fluoroethyl ether
    "FC(F)OCCC",         # difluoromethyl propyl ether
    "FCOC(F)C(F)F",      # polyfluorinated ether
    "CC(F)OC(F)F",       # fluoroisopropyl difluoromethyl ether
    "FCOCF",             # bis(fluoromethyl) ether
    "CCCOC(F)F",         # difluoromethyl propyl ether
    "CCCOC(F)(F)F",      # trifluoromethyl propyl ether
    "FC(F)OCC",          # difluoromethyl ethyl ether (alt)
    "COC(F)(F)CF",       # methyl 2,2-difluoroethyl ether
    "CCCOCC(F)(F)F",     # 3,3,3-trifluoropropy propyl ether
    "FC(F)(F)OC(F)(F)F", # bis(trifluoromethyl) ether
    "COC(C)(F)F",        # methyl 1,1-difluoroethyl ether
    "CC(F)(F)OCC",       # 1,1-difluoroethyl ethyl ether
]

# Unsaturated hydrocarbons: C3-C5 alkenes and cycloalkenes
BASE_UNSATURATED = [
    "C=CC",              # propene
    "CC=CC",             # 2-butene
    "C=CCC",             # 1-butene
    "CC(=C)C",           # isobutylene
    "C=CCCC",            # 1-pentene
    "CC=CCC",            # 2-pentene
    "CCC=CC",            # 3-pentene (same as 2-pentene reversed)
    "CC(=C)CC",          # 2-methyl-1-butene
    "C1CC=CC1",          # cyclopentene
    "C1CC=C1",           # cyclobutene
    "C1C=C1",            # cyclopropene
    "C=C(C)CC",          # 2-methyl-1-butene variant
    "C/C=C\\C",          # cis-2-butene
    "C/C=C/C",           # trans-2-butene
    "C1=CCCCC1",         # cyclohexene
    "CC1CC=CC1",         # methylcyclopentene
    "C=C",               # ethylene
    "CC=C",              # propene (alt)
    "C(=C)CC",           # 1-butene (alt)
    "CC(C)=CC",          # 2-methyl-2-butene
    "C=CC(C)C",          # 3-methyl-1-butene
    "C1C=CCC1",          # cyclopentene (alt)
    "C=CC=C",            # 1,3-butadiene
    "CC=CC=C",           # 1,3-pentadiene
    "C1CC=CCC1",         # cyclohexene (alt)
]

# Cyclic fluorinated: C3-C6 rings with F substitution patterns
BASE_CYCLIC_FLUORINATED = [
    "FC1CCC1",           # fluorocyclobutane
    "FC1CCCC1",          # fluorocyclopentane
    "FC1CC(F)CC1",       # 1,3-difluorocyclopentane
    "FC1(F)CCC1",        # 1,1-difluorocyclobutane
    "FC1CC(F)C1",        # 1,3-difluorocyclobutane
    "FC1(F)CCCC1",       # 1,1-difluorocyclopentane
    "FC1CC(F)CC1F",      # 1,2,4-trifluorocyclopentane
    "FC1CCC(F)C1F",      # 1,2,4-trifluorocyclopentane (alt)
    "FC1CCCCC1",         # fluorocyclohexane
    "FC1(F)CCCCC1",      # 1,1-difluorocyclohexane
    "FC1CC(F)CCC1",      # 1,3-difluorocyclohexane
    "FC1CCC(F)CC1",      # 1,4-difluorocyclohexane
    "FC1CC1",            # fluorocyclopropane
    "FC1(F)CC1",         # 1,1-difluorocyclopropane
    "FC1C(F)C1F",        # 1,2,3-trifluorocyclopropane
    "FC1(F)C(F)C1F",     # 1,1,2,3-tetrafluorocyclopropane
    "FC1(F)CC1F",        # 1,1,2-trifluorocyclopropane (dup check)
    "FC1(F)C(F)(F)C1",   # tetrafluorocyclopropane
]

# ---------------------------------------------------------------------------
# Systematic enumeration: bare alkene backbones for exhaustive F/Cl placement
# ---------------------------------------------------------------------------
# Every C-H position on these backbones will be combinatorially assigned
# H, F, or (at most one) Cl. The C=C double bond is preserved in all variants
# to ensure short atmospheric lifetime (low GWP).

ALKENE_BACKBONES: list[str] = [
    # Acyclic — C2
    "C=C",              # ethene
    # Acyclic — C3
    "CC=C",             # propene
    # Acyclic — C4
    "CCC=C",            # 1-butene
    "CC=CC",            # 2-butene
    "CC(=C)C",          # isobutylene (2-methylpropene)
    # Acyclic — C5
    "CCCC=C",           # 1-pentene
    "CCC=CC",           # 2-pentene
    "CCC(=C)C",         # 2-methyl-1-butene
    "CC(C)C=C",         # 3-methyl-1-butene
    "CC(C)=CC",         # 2-methyl-2-butene
    # Endocyclic alkenes
    "C1=CC1",           # cyclopropene
    "C1=CCC1",          # cyclobutene
    "C1=CCCC1",         # cyclopentene
    "C1=CCCCC1",        # cyclohexene
    # Exocyclic alkenes
    "C=C1CC1",          # methylenecyclopropane
    "C=C1CCC1",         # methylenecyclobutane
]


def _enumerate_halogen_patterns(
    backbone_smi: str,
    *,
    max_cl: int = 1,
    max_mw: float = 200.0,
) -> set[str]:
    """Exhaustively enumerate F/Cl substitution patterns on a backbone.

    For each H bonded to a carbon atom, independently assign H (keep),
    F, or Cl, subject to the constraint that at most *max_cl* Cl atoms
    appear and at least one F is present (to exclude plain hydrocarbons).

    Parameters
    ----------
    backbone_smi : str
        SMILES of the unsubstituted alkene backbone.
    max_cl : int
        Maximum number of Cl atoms per molecule (default 1 for HCFOs).
    max_mw : float
        Reject molecules with MW above this threshold. Blowing agents
        need to be volatile; 200 Da covers all commercial HFOs/HCFOs.

    Returns
    -------
    set[str]
        Canonical SMILES of all valid substitution patterns.
    """
    mol = Chem.MolFromSmiles(backbone_smi)
    if mol is None:
        return set()

    mol = Chem.AddHs(mol)

    h_on_c_indices: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 6:
                h_on_c_indices.append(atom.GetIdx())
                break

    n = len(h_on_c_indices)
    if n == 0:
        return set()

    results: set[str] = set()

    for n_cl in range(min(max_cl, n) + 1):
        for cl_sites in combinations(range(n), n_cl):
            cl_set = set(cl_sites)
            remaining = [i for i in range(n) if i not in cl_set]

            for f_pattern in product((1, 9), repeat=len(remaining)):
                if n_cl == 0 and 9 not in f_pattern:
                    continue  # skip parent hydrocarbon (no halogens)

                ed = Chem.RWMol(Chem.Mol(mol))

                for si in cl_sites:
                    ed.GetAtomWithIdx(h_on_c_indices[si]).SetAtomicNum(17)
                for si, z in zip(remaining, f_pattern):
                    if z == 9:
                        ed.GetAtomWithIdx(h_on_c_indices[si]).SetAtomicNum(9)

                try:
                    Chem.SanitizeMol(ed)
                    if Descriptors.ExactMolWt(ed) > max_mw:
                        continue
                    ed_clean = Chem.RemoveHs(ed)
                    results.add(Chem.MolToSmiles(ed_clean))
                except Exception:
                    pass

    return results


def generate_systematic_candidates(
    backbones: list[str] | None = None,
    *,
    max_cl: int = 1,
    max_mw: float = 200.0,
) -> list[tuple[str, Chem.Mol]]:
    """Systematically enumerate HFO/HCFO candidates from alkene backbones.

    For each backbone, exhaustively places F (and optionally Cl) at every
    C-H site, then enumerates E/Z and R/S stereoisomers.

    Parameters
    ----------
    backbones : list[str] | None
        Alkene backbone SMILES. Defaults to ALKENE_BACKBONES.
    max_cl : int
        Maximum Cl atoms per molecule (0 = HFO-only, 1 = include HCFOs).
    max_mw : float
        Molecular weight ceiling in Da.

    Returns
    -------
    list[tuple[str, Mol]]
        Deduplicated (canonical_SMILES, Mol) pairs with stereoisomers.
    """
    if backbones is None:
        backbones = ALKENE_BACKBONES

    all_smiles: set[str] = set()
    for bb in backbones:
        patterns = _enumerate_halogen_patterns(
            bb, max_cl=max_cl, max_mw=max_mw,
        )
        logger.info("Backbone %s → %d unique patterns", bb, len(patterns))
        all_smiles.update(patterns)

    logger.info(
        "Combinatorial enumeration: %d unique SMILES from %d backbones",
        len(all_smiles), len(backbones),
    )

    seen: set[str] = set()
    candidates: list[tuple[str, Chem.Mol]] = []

    for smi in sorted(all_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for iso in _enumerate_stereoisomers(mol) or [mol]:
            can = Chem.MolToSmiles(iso)
            if can not in seen:
                seen.add(can)
                candidates.append((can, iso))

    logger.info(
        "After stereoisomer enumeration: %d unique candidates", len(candidates),
    )
    print(
        f"Systematic enumeration: {len(candidates)} unique candidates "
        f"from {len(backbones)} backbones (max_cl={max_cl}, max_mw={max_mw})"
    )
    return candidates


# Scaffold set mapping
SCAFFOLD_SETS = {
    "hfo": [BASE_HFOS],
    "broad": [BASE_HFOS, BASE_HCFOS, BASE_HFES, BASE_UNSATURATED, BASE_CYCLIC_FLUORINATED],
    "all": [BASE_HFOS, BASE_HCFOS, BASE_HFES, BASE_UNSATURATED, BASE_CYCLIC_FLUORINATED],
}


def _enumerate_stereoisomers(mol: Chem.Mol) -> list[Chem.Mol]:
    """Enumerate all stereoisomers of a molecule."""
    opts = StereoEnumerationOptions(unique=True, onlyUnassigned=False)
    return list(EnumerateStereoisomers(mol, options=opts))


def _generate_halogen_variants(
    smi: str, halogens: list[int] | None = None
) -> list[str]:
    """Generate variants by replacing H atoms on carbons with halogens.

    For each H on a carbon, generates a variant with that H replaced by
    each halogen in *halogens*. Also returns the unmodified parent.

    Parameters
    ----------
    smi : str
        Input SMILES.
    halogens : list[int] | None
        Atomic numbers to substitute. Default: [9] (F only).
        Use [9, 17] for F and Cl.
    """
    if halogens is None:
        halogens = [9]  # Fluorine only by default

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []

    variants = set()
    variants.add(Chem.MolToSmiles(mol))

    # Find carbons with implicit H that could be halogenated
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # Carbon
            num_h = atom.GetTotalNumHs()
            if num_h > 0:
                for halogen_z in halogens:
                    ed = Chem.RWMol(mol)
                    ed.GetAtomWithIdx(atom.GetIdx()).SetNumExplicitHs(num_h - 1)
                    h_idx = ed.AddAtom(Chem.Atom(halogen_z))
                    ed.AddBond(atom.GetIdx(), h_idx, Chem.BondType.SINGLE)
                    try:
                        Chem.SanitizeMol(ed)
                        variants.add(Chem.MolToSmiles(ed))
                    except Exception:
                        pass

    return list(variants)


def _generate_fluorine_variants(smi: str) -> list[str]:
    """Generate variants by adding fluorine at available positions.

    Simple approach: for each H on a carbon, try replacing with F.
    """
    return _generate_halogen_variants(smi, halogens=[9])


def generate_hfo_candidates(
    base_smiles: list[str] | None = None,
    scaffold_set: str = "hfo",
    *,
    max_cl: int = 1,
    max_mw: float = 200.0,
) -> list[tuple[str, Chem.Mol]]:
    """Generate candidate molecules with stereoisomer enumeration.

    Parameters
    ----------
    base_smiles : list[str] | None
        Starting SMILES. If None, uses scaffolds from *scaffold_set*.
    scaffold_set : str
        Which scaffold families to use: "hfo" (original), "broad" (all
        families), "all" (same as broad), or "systematic" (exhaustive
        combinatorial F/Cl placement on alkene backbones).
    max_cl : int
        For "systematic" mode: max Cl atoms per molecule.
    max_mw : float
        For "systematic" mode: MW ceiling in Da.

    Returns
    -------
    list[tuple[str, Mol]]
        List of (canonical_SMILES, RDKit Mol) tuples.
    """
    if scaffold_set == "systematic" and base_smiles is None:
        return generate_systematic_candidates(
            max_cl=max_cl, max_mw=max_mw,
        )

    if base_smiles is None:
        scaffold_lists = SCAFFOLD_SETS.get(scaffold_set, [BASE_HFOS])
        base_smiles = []
        for slist in scaffold_lists:
            base_smiles.extend(slist)

    # For broad/all sets, use Cl+F substitution for more diversity
    use_chlorine = scaffold_set in ("broad", "all")

    seen = set()
    candidates = []

    for smi in base_smiles:
        # For broad/all scaffolds, generate F+Cl variants for all molecules
        if use_chlorine:
            variants = _generate_halogen_variants(smi, halogens=[9, 17])
        else:
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

    logger.info(
        "Generated %d unique candidates (scaffold_set=%s)",
        len(candidates),
        scaffold_set,
    )
    print(f"Generated {len(candidates)} unique candidates (scaffold_set={scaffold_set})")
    return candidates
