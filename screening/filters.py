"""Three-stage screening filters for blowing agent candidates.

1. Synthesizability: SA Score ≤ threshold
2. Patent freedom: PubChem PUG REST patent check
3. PC-SAFT similarity: distance to cyclopentane target parameters
4. Fluorination safety filters (HFO screening):
   - Fluorine mass fraction
   - CF3 terminal group detection
   - Reactive fluorination site detection
"""

import logging
import time
from collections.abc import Callable

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer

logger = logging.getLogger(__name__)

CYCLOPENTANE_M = 2.3655
CYCLOPENTANE_SIGMA = 3.7114
CYCLOPENTANE_EPS_K = 288.84

ASSOCIATION_SMARTS = [
    "[OX2H]",      # hydroxyl (alcohols, phenols)
    "[NX3H2]",     # primary amine
    "[NX3H1]",     # secondary amine
    "[CX3](=O)[OX2H]",  # carboxylic acid
]


def _get_default_predictor() -> Callable:
    """Lazy import of model.predict to avoid hard coupling at module load time."""
    from model.predict import predict_pcsaft
    return predict_pcsaft


def is_associating(smiles: str) -> bool:
    """Check if a molecule has hydrogen-bonding association sites.

    PC-SAFT requires 5 parameters for associating fluids (OH, NH, COOH, etc.)
    but this project only predicts the 3 non-associating parameters. Molecules
    returning True should be treated with extra caution.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for smarts in ASSOCIATION_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return True
    return False


def filter_synthesizability(
    candidates: list[tuple[str, Chem.Mol]], threshold: float = 4.5
) -> list[tuple[str, Chem.Mol, float]]:
    """Filter candidates by synthetic accessibility score.

    Parameters
    ----------
    candidates : list[tuple[str, Mol]]
        (SMILES, Mol) tuples.
    threshold : float
        Maximum SA score (1=easy, 10=hard). Default 4.5.

    Returns
    -------
    list[tuple[str, Mol, float]]
        Passing candidates with (SMILES, Mol, SA_score).
    """
    passed = []
    for smi, mol in candidates:
        try:
            score = sascorer.calculateScore(mol)
        except Exception:
            continue
        if score <= threshold:
            passed.append((smi, mol, score))

    logger.info("SA Score filter: %d/%d passed (≤ %s)", len(passed), len(candidates), threshold)
    return passed


def filter_patents(
    candidates: list[tuple[str, Chem.Mol, float]], delay: float = 0.3
) -> list[tuple[str, Chem.Mol, float]]:
    """Filter out candidates with PubChem patent references.

    Uses PubChem PUG REST API. A 404 response means the compound is not
    in PubChem (likely novel). Non-404 responses with patent data mean
    the compound has known patents.

    Parameters
    ----------
    candidates : list[tuple[str, Mol, float]]
        (SMILES, Mol, SA_score) tuples.
    delay : float
        Seconds between API calls to respect rate limits.

    Returns
    -------
    list[tuple[str, Mol, float]]
        Candidates without patent references.
    """
    passed = []
    for smi, mol, sa_score in candidates:
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            f"{requests.utils.quote(smi)}/xrefs/PatentID/JSON"
        )
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 404:
                # Not in PubChem — likely novel
                passed.append((smi, mol, sa_score))
            elif resp.status_code == 200:
                data = resp.json()
                # Check if there are actual patent IDs
                patents = (
                    data.get("InformationList", {})
                    .get("Information", [{}])[0]
                    .get("PatentID", [])
                )
                if not patents:
                    passed.append((smi, mol, sa_score))
            # else: API error, skip conservatively
        except requests.RequestException:
            # Network error — include candidate (benefit of the doubt)
            passed.append((smi, mol, sa_score))

        time.sleep(delay)

    logger.info("Patent filter: %d/%d passed (no patents found)", len(passed), len(candidates))
    return passed


def filter_pcsaft_similarity(
    candidates: list[tuple[str, Chem.Mol, float]],
    predict_fn: Callable | None = None,
    target_m: float = CYCLOPENTANE_M,
    target_sigma: float = CYCLOPENTANE_SIGMA,
    target_eps: float = CYCLOPENTANE_EPS_K,
    weights: dict[str, float] | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """Predict PC-SAFT parameters and rank by weighted similarity to target.

    Parameters
    ----------
    candidates : list[tuple[str, Mol, float]]
        (SMILES, Mol, SA_score) tuples.
    predict_fn : callable, optional
        Function that takes a list of SMILES and returns a DataFrame with
        columns [smiles, m, sigma, epsilon_k]. If None, uses model.predict.
    target_m, target_sigma, target_eps : float
        Target PC-SAFT parameters (default: cyclopentane).
    weights : dict, optional
        Weights for distance metric. Default: {"m": 1.0, "sigma": 1.0, "epsilon_k": 3.0}.
        Higher weight on epsilon_k reflects its dominance in vapor pressure.
    threshold : float | None
        If set, only return candidates with distance ≤ threshold.

    Returns
    -------
    list[dict]
        Ranked candidates with keys: smiles, sa_score, m, sigma, epsilon_k,
        distance, is_associating.
    """
    if predict_fn is None:
        predict_fn = _get_default_predictor()

    if weights is None:
        weights = {"m": 1.0, "sigma": 1.0, "epsilon_k": 3.0}

    smiles_list = [smi for smi, _, _ in candidates]
    sa_scores = {smi: sa for smi, _, sa in candidates}

    predictions = predict_fn(smiles_list)

    wm = weights.get("m", 1.0)
    ws = weights.get("sigma", 1.0)
    we = weights.get("epsilon_k", 3.0)

    results = []
    for _, row in predictions.iterrows():
        dm = wm * ((row["m"] - target_m) / target_m) ** 2
        ds = ws * ((row["sigma"] - target_sigma) / target_sigma) ** 2
        de = we * ((row["epsilon_k"] - target_eps) / target_eps) ** 2
        distance = np.sqrt(dm + ds + de)

        if threshold is not None and distance > threshold:
            continue

        results.append(
            {
                "smiles": row["smiles"],
                "sa_score": sa_scores.get(row["smiles"], float("nan")),
                "m": row["m"],
                "sigma": row["sigma"],
                "epsilon_k": row["epsilon_k"],
                "distance": distance,
                "is_associating": is_associating(row["smiles"]),
            }
        )

    results.sort(key=lambda x: x["distance"])
    logger.info(
        "PC-SAFT ranking: %d candidates ranked by cyclopentane similarity", len(results)
    )
    return results


# ============================================================================
# Fluorination Safety Filters (Step 32)
# ============================================================================

def fluorine_mass_fraction(smiles: str) -> float:
    """Compute mass fraction of fluorine in a molecule.

    Args:
        smiles: SMILES string

    Returns:
        float: Fluorine mass fraction (0.0 to 1.0), or 0.0 if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    mw = Descriptors.ExactMolWt(mol)
    n_fluorine = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    return (n_fluorine * 18.998) / mw if mw > 0 else 0.0


def passes_fluorine_mass_fraction(smiles: str, threshold: float = 0.65) -> bool:
    """Check if molecule has sufficient fluorine mass fraction for non-flammability.

    The 65 wt% threshold is a heuristic based on ASHRAE 34 A1 classification
    correlation. This is NOT a flammability guarantee or regulatory certification.

    Args:
        smiles: SMILES string
        threshold: Minimum fluorine mass fraction (default 0.65 = 65 wt%)

    Returns:
        bool: True if fluorine mass fraction >= threshold
    """
    return fluorine_mass_fraction(smiles) >= threshold


# CF3 terminal group SMARTS: sp3 carbon with 3 fluorines
CF3_SMARTS = Chem.MolFromSmarts("[CX4](F)(F)F")


def count_cf3_groups(smiles: str) -> int:
    """Count terminal CF3 groups in a molecule.

    Note: This SMARTS pattern matches any sp3 carbon with 3 fluorines, not
    necessarily "terminal" in the strict sense. The name reflects the intended
    use case (detecting -CF3 groups), but the pattern may also match internal
    CF3 on branched carbons.

    Args:
        smiles: SMILES string

    Returns:
        int: Number of CF3 groups, or 0 if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    if CF3_SMARTS is None:
        return 0
    matches = mol.GetSubstructMatches(CF3_SMARTS)
    return len(matches)


def has_cf3_group(smiles: str) -> bool:
    """Check if molecule has at least one CF3 terminal group.

    Args:
        smiles: SMILES string

    Returns:
        bool: True if at least one CF3 group is present
    """
    return count_cf3_groups(smiles) > 0


# Reactive fluorination site SMARTS patterns
# These are heuristic patterns for thermally or chemically unstable motifs

# Isolated tertiary fluorine: sp3 carbon with exactly one F and no other F neighbors
# Pattern: carbon with 4 bonds (sp3), exactly one F, and no CF2/CF3 groups
ISOLATED_TERT_F = Chem.MolFromSmarts("[CH0;X4;!$([CH0](F)(F))](F)")

# Allylic -CHF- adjacent to C=C (risk of HF elimination)
ALLYLIC_CHF = Chem.MolFromSmarts("[C]=[C][CH1]F")

# Allylic -CH2F adjacent to C=C (risk of HF elimination)
ALLYLIC_CH2F = Chem.MolFromSmarts("[C]=[C][CH2]F")

REACTIVE_PATTERNS = [
    (ISOLATED_TERT_F, "isolated_tertiary_fluorine"),
    (ALLYLIC_CHF, "allylic_CHF"),
    (ALLYLIC_CH2F, "allylic_CH2F"),
]


def has_reactive_fluorine(smiles: str) -> tuple[bool, list[str]]:
    """Check if molecule has reactive fluorination sites.

    Reactive sites include:
    - Isolated tertiary fluorine (single F on sp3 carbon, no neighboring F)
    - Allylic -CHF- adjacent to C=C bond
    - Allylic -CH2F adjacent to C=C bond

    These patterns are heuristic indicators of potential thermal or chemical
    instability. They are NOT formal stability certifications.

    Args:
        smiles: SMILES string

    Returns:
        tuple: (has_reactive, list_of_pattern_names)
            - has_reactive: True if any reactive pattern is found
            - list_of_pattern_names: Names of matched patterns
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, []
    found = []
    for pattern, name in REACTIVE_PATTERNS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            found.append(name)
    return len(found) > 0, found
