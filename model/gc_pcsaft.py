"""Simplified group-contribution PC-SAFT parameter estimation.

This implements a simplified GC-PC-SAFT method based on functional group
additivity for estimating PC-SAFT parameters (m, sigma, epsilon_k).

References:
- Gross & Sadowski, Ind. Eng. Chem. Res. 40, 1244-1260 (2001)
- Gross & Sadowski, Ind. Eng. Chem. Res. 41, 5510-5515 (2002)
- Sauer et al., Ind. Eng. Chem. Res. 53, 14854-14864 (2014)

IMPORTANT: This is a simplified approximation, not a full GC-PC-SAFT
implementation. It uses group additivity with published parameters for
common functional groups. Predictions should be treated as rough baselines
for evaluating ML model performance, not as production-quality estimates.

Limitations:
- Ring strain effects are approximated, not rigorously handled
- Cross-interaction terms between groups are neglected
- Branching corrections are simplified
- Associating groups (OH, NH) only affect m*sigma^3 and m*epsilon_k, not
  the full association scheme parameters (epsilon_AB, kappa_AB)
"""

import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)

# Group contribution parameters: (delta_m, delta_m_sigma3, delta_m_epsk)
# where:
#   m = sum(delta_m)
#   m * sigma^3 = sum(delta_m_sigma3)  ->  sigma = (sum(m_sigma3) / m)^(1/3)
#   m * epsilon_k = sum(delta_m_epsk)  ->  epsilon_k = sum(m_epsk) / m
#
# Values derived from Sauer et al. 2014 and Gross & Sadowski 2001/2002
# for common organic functional groups.
GROUP_PARAMS = {
    # (SMARTS, delta_m, delta_m_sigma3, delta_m_epsk)
    # Aliphatic carbon groups
    "CH3": ("[CH3]", 0.6148, 23.50, 162.0),
    "CH2": ("[CH2]", 0.4868, 17.12, 131.0),
    "CH": ("[CH1;!$([CH1]=*)]", 0.2700, 10.50, 80.0),
    "C": ("[C;X4;H0;!$([C]=*)]", 0.1500, 5.50, 40.0),
    # Aromatic groups
    "ACH": ("[cH]", 0.3763, 13.00, 120.0),
    "AC": ("[c;H0]", 0.2800, 9.50, 90.0),
    # Double bond groups
    "CH2=CH": ("[CH2]=[CH]", 0.9000, 34.00, 250.0),
    "CH=CH": ("[CH]=[CH]", 0.7800, 28.00, 220.0),
    "C=C": ("[C;H0]=[C;H0]", 0.5500, 19.00, 160.0),
    "CH2=C": ("[CH2]=[C;H0]", 0.8000, 30.00, 230.0),
    # Halogen groups
    "F": ("[F]", 0.2500, 7.50, 70.0),
    "Cl": ("[Cl]", 0.4500, 17.00, 140.0),
    "Br": ("[Br]", 0.5500, 24.00, 170.0),
    # Oxygen groups
    "OH": ("[OX2H]", 0.4000, 8.00, 180.0),
    "CO_ether": ("[OX2;!$([OX2H])]", 0.2500, 7.00, 100.0),
    "CHO": ("[CH]=O", 0.5000, 14.00, 190.0),
    "CO_ketone": ("[CX3](=O)[#6]", 0.4000, 12.00, 165.0),
    "COOH": ("[CX3](=O)[OX2H]", 0.7500, 18.00, 280.0),
    "COO": ("[CX3](=O)[OX2;!$([OX2H])]", 0.6500, 16.00, 200.0),
    # Nitrogen groups
    "NH2": ("[NX3H2]", 0.4500, 10.00, 160.0),
    "NH": ("[NX3H1]", 0.3500, 8.00, 130.0),
    "N": ("[NX3H0]", 0.2500, 6.00, 100.0),
    "CN": ("[CX2]#[NX1]", 0.6000, 14.00, 220.0),
}

# Ring correction: cyclic molecules have reduced sigma due to compact geometry
RING_CORRECTION_SIGMA3 = -3.0  # per ring, subtracted from m*sigma^3
RING_CORRECTION_EPSK = 15.0    # per ring, added to m*epsilon_k (ring strain)


def _count_groups(mol: Chem.Mol) -> dict[str, int]:
    """Count functional group occurrences using SMARTS matching.

    Uses a hierarchical approach: more specific groups are matched first,
    then atoms already matched are excluded from simpler patterns.
    """
    counts = {}
    matched_atoms = set()

    # Match complex groups first (multi-atom), then simple ones
    complex_groups = [
        "CH2=CH", "CH=CH", "C=C", "CH2=C",
        "COOH", "COO", "CHO", "CO_ketone",
        "CN",
    ]
    simple_groups = [
        "ACH", "AC",
        "CH3", "CH2", "CH", "C",
        "F", "Cl", "Br",
        "OH", "CO_ether",
        "NH2", "NH", "N",
    ]

    for group_name in complex_groups + simple_groups:
        smarts_str = GROUP_PARAMS[group_name][0]
        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        count = 0
        for match in matches:
            # Only count if the primary atom hasn't been assigned yet
            # (for single-atom groups, check the first atom)
            primary = match[0]
            if primary not in matched_atoms:
                count += 1
                matched_atoms.update(match)
        if count > 0:
            counts[group_name] = count

    return counts


def predict_gc_pcsaft_single(smiles: str) -> dict[str, float] | None:
    """Predict PC-SAFT parameters for a single molecule using group contribution.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    dict or None
        Dictionary with keys {m, sigma, epsilon_k} or None if prediction fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Invalid SMILES for GC-PC-SAFT: %s", smiles)
        return None

    counts = _count_groups(mol)
    if not counts:
        logger.warning("No recognized groups for GC-PC-SAFT: %s", smiles)
        return None

    # Sum group contributions
    total_m = 0.0
    total_m_sigma3 = 0.0
    total_m_epsk = 0.0

    for group_name, count in counts.items():
        _, delta_m, delta_ms3, delta_mek = GROUP_PARAMS[group_name]
        total_m += count * delta_m
        total_m_sigma3 += count * delta_ms3
        total_m_epsk += count * delta_mek

    if total_m <= 0:
        logger.warning("Non-positive m for GC-PC-SAFT: %s (m=%.3f)", smiles, total_m)
        return None

    # Apply ring corrections
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    if n_rings > 0:
        total_m_sigma3 += n_rings * RING_CORRECTION_SIGMA3
        total_m_epsk += n_rings * RING_CORRECTION_EPSK

    # Ensure physical constraints
    total_m_sigma3 = max(total_m_sigma3, 1.0)
    total_m_epsk = max(total_m_epsk, total_m * 50.0)

    sigma = (total_m_sigma3 / total_m) ** (1.0 / 3.0)
    epsilon_k = total_m_epsk / total_m

    return {"m": total_m, "sigma": sigma, "epsilon_k": epsilon_k}


def predict_gc_pcsaft(smiles_list: list[str]) -> pd.DataFrame:
    """Predict PC-SAFT parameters for a list of molecules using group contribution.

    This is a simplified GC-PC-SAFT baseline. Parameters are estimated by
    summing group contributions for common functional groups (CH3, CH2, F, Cl,
    aromatic rings, etc.) with ring corrections.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: smiles, m, sigma, epsilon_k.
        Rows where prediction fails have NaN values.
    """
    rows = []
    for smi in smiles_list:
        result = predict_gc_pcsaft_single(smi)
        if result is None:
            rows.append({"smiles": smi, "m": np.nan, "sigma": np.nan, "epsilon_k": np.nan})
        else:
            rows.append({"smiles": smi, **result})

    return pd.DataFrame(rows)
