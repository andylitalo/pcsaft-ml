"""Application domain presets for PC-SAFT screening.

Each preset defines target properties, filtering criteria, and ranking metrics
for different application domains (blowing agents, refrigerants, solvents, etc.).
"""

PRESETS = {
    "blowing_agents": {
        "name": "Blowing Agents",
        "description": (
            "Physical blowing agents for polyurethane foams "
            "(cyclopentane-like profiles)"
        ),
        "default_reference": "Cyclopentane",
        "filters": {
            "m_range": (1.8, 3.5),
            "sigma_range": (3.0, 4.5),
            "epsilon_k_range": (200, 400),
            "max_mw": 200,
            "require_in_domain": True,
            "min_tanimoto": 0.3,
        },
        "ranking_metric": "distance_to_reference",  # Euclidean distance in normalized param space
        "soft_preferences": {
            "low_gwp": True,  # Prefer low Global Warming Potential (heuristic: avoid fluorinated)
            "low_toxicity": True,  # Prefer non-toxic (heuristic: avoid aromatics, halogens)
        },
        "caution": (
            "Illustrative heuristic for exploratory screening; "
            "thresholds are not universally validated for blowing agent applications. "
            "Always validate thermodynamic properties (vapor pressure, boiling point) "
            "and safety (flammability, toxicity) before experimental use."
        ),
    },
    "refrigerants": {
        "name": "Refrigerants",
        "description": (
            "HFC/HFO-like refrigerants with low GWP "
            "and favorable thermodynamic properties"
        ),
        "default_reference": "R-134a",  # 1,1,1,2-Tetrafluoroethane
        "filters": {
            "m_range": (1.5, 3.0),
            "sigma_range": (2.5, 4.0),
            "epsilon_k_range": (150, 350),
            "max_mw": 150,
            "require_in_domain": True,
            "min_tanimoto": 0.25,
        },
        "ranking_metric": "distance_to_reference",
        "soft_preferences": {
            "low_gwp": True,
            "prefer_fluorinated": True,  # Heuristic: refrigerants often fluorinated
        },
        "caution": (
            "Illustrative heuristic for exploratory screening; "
            "thresholds are not universally validated for refrigerant applications. "
            "Critical properties (Tc, Pc, vapor pressure curve), GWP, ODP, "
            "flammability, and toxicity must be validated via detailed simulations "
            "and experimental testing before deployment."
        ),
    },
    "solvents": {
        "name": "Industrial Solvents",
        "description": "Green solvents with favorable solvation properties and low toxicity",
        "default_reference": "Ethyl acetate",
        "filters": {
            "m_range": (1.5, 4.0),
            "sigma_range": (3.0, 5.0),
            "epsilon_k_range": (150, 450),
            "max_mw": 250,
            "require_in_domain": True,
            "min_tanimoto": 0.2,
        },
        "ranking_metric": "epsilon_k",  # Rank by dispersion energy (solvation power proxy)
        "soft_preferences": {
            "low_toxicity": True,
            "biodegradable": True,  # Heuristic: prefer esters, simple alcohols
        },
        "caution": (
            "Illustrative heuristic for exploratory screening; "
            "thresholds are not universally validated for solvent applications. "
            "Solvation properties (Hansen solubility parameters, dielectric constant), "
            "environmental impact (biodegradability, bioaccumulation), "
            "and safety (flammability, toxicity) must be validated experimentally."
        ),
    },
    "general": {
        "name": "General Exploration",
        "description": "No application-specific filters; rank by prediction confidence only",
        "default_reference": "Cyclopentane",
        "filters": {
            "require_in_domain": False,  # Allow out-of-domain molecules
            "min_tanimoto": 0.0,  # No similarity cutoff
        },
        "ranking_metric": "tanimoto",  # Rank by similarity to training data (confidence proxy)
        "soft_preferences": {},
        "caution": (
            "General exploratory mode with no application-specific filters. "
            "Results are ranked by similarity to training data (Tanimoto score) "
            "as a proxy for prediction confidence. All predictions should be "
            "validated experimentally or via detailed simulations."
        ),
    },
}


def get_preset(name: str) -> dict:
    """Get a preset by name.

    Parameters
    ----------
    name : str
        Preset name (e.g., "blowing_agents", "refrigerants").

    Returns
    -------
    dict
        Preset configuration.

    Raises
    ------
    KeyError
        If preset name is not found.
    """
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset {name!r}. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name]


def list_presets() -> list[str]:
    """Return list of available preset names.

    Returns
    -------
    list[str]
        List of preset names.
    """
    return list(PRESETS.keys())
