"""Three-stage screening filters for blowing agent candidates.

1. Synthesizability: SA Score ≤ threshold
2. Patent freedom: PubChem PUG REST patent check
3. PC-SAFT similarity: distance to cyclopentane target parameters
"""

import time

import numpy as np
import requests
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

from model.predict import predict_pcsaft

# Cyclopentane PC-SAFT reference parameters (Gross & Sadowski, 2001)
CYCLOPENTANE_M = 2.3655
CYCLOPENTANE_SIGMA = 3.7114
CYCLOPENTANE_EPS_K = 288.84


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

    print(f"SA Score filter: {len(passed)}/{len(candidates)} passed (≤ {threshold})")
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

    print(f"Patent filter: {len(passed)}/{len(candidates)} passed (no patents found)")
    return passed


def filter_pcsaft_similarity(
    candidates: list[tuple[str, Chem.Mol, float]],
    target_m: float = CYCLOPENTANE_M,
    target_sigma: float = CYCLOPENTANE_SIGMA,
    target_eps: float = CYCLOPENTANE_EPS_K,
    threshold: float | None = None,
) -> list[dict]:
    """Predict PC-SAFT parameters and rank by similarity to target.

    Computes normalized Euclidean distance where each parameter is
    divided by the target value before computing distance.

    Parameters
    ----------
    candidates : list[tuple[str, Mol, float]]
        (SMILES, Mol, SA_score) tuples.
    target_m, target_sigma, target_eps : float
        Target PC-SAFT parameters (default: cyclopentane).
    threshold : float | None
        If set, only return candidates with distance ≤ threshold.

    Returns
    -------
    list[dict]
        Ranked candidates with keys: smiles, sa_score, m, sigma, epsilon_k, distance.
    """
    smiles_list = [smi for smi, _, _ in candidates]
    sa_scores = {smi: sa for smi, _, sa in candidates}

    predictions = predict_pcsaft(smiles_list)

    results = []
    for _, row in predictions.iterrows():
        # Normalized Euclidean distance
        dm = (row["m"] - target_m) / target_m
        ds = (row["sigma"] - target_sigma) / target_sigma
        de = (row["epsilon_k"] - target_eps) / target_eps
        distance = np.sqrt(dm**2 + ds**2 + de**2)

        if threshold is not None and distance > threshold:
            continue

        results.append(
            {
                "smiles": row["smiles"],
                "sa_score": sa_scores[row["smiles"]],
                "m": row["m"],
                "sigma": row["sigma"],
                "epsilon_k": row["epsilon_k"],
                "distance": distance,
            }
        )

    results.sort(key=lambda x: x["distance"])
    print(f"PC-SAFT ranking: {len(results)} candidates ranked by cyclopentane similarity")
    return results
