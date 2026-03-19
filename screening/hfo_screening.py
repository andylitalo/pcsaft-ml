"""Boiling point computation and HFO screening filters.

This module provides functions for:
1. Computing boiling points from PC-SAFT parameters via vapor pressure root finding
2. Computing parameter distance to HFO reference compounds
3. Applying HFO-specific screening filters
4. Validating against experimental boiling points

Reference compound: HFO-1336mzz(Z) - a commercial low-GWP refrigerant.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer
from scipy.optimize import brentq

from model.thermodynamic import compute_properties
from screening.filters import (
    count_cf3_groups,
    has_reactive_fluorine,
    passes_fluorine_mass_fraction,
)

# HFO-1336mzz(Z) reference (Konnova 2014)
HFO_1336MZZ = {
    "name": "HFO-1336mzz(Z)",
    "smiles": "F/C(=C\\C(F)(F)F)C(F)(F)F",
    "m": 2.7894,
    "sigma": 3.4521,
    "epsilon_k": 178.44,
    "source": "Konnova 2014",
    "boiling_point_K": 306.55,
}

# Experimental boiling points for fluorinated compounds (NIST, manufacturers)
KNOWN_BOILING_POINTS = {
    "FCF": 221.5,  # R-32
    "FCC(F)(F)F": 247.1,  # R-134a
    "FC(F)C(F)(F)F": 225.1,  # R-125
    "CC(F)(F)F": 225.9,  # R-143a
    "CC(F)F": 249.1,  # R-152a
    "FC(F)C(F)(F)C(F)(F)F": 256.8,  # R-227ea
    "FC(F)F": 191.1,  # R-23
    "C=C(F)C(F)(F)F": 243.7,  # HFO-1234yf
    "F/C=C/C(F)(F)F": 254.2,  # HFO-1234ze(E)
    "F/C(=C\\C(F)(F)F)C(F)(F)F": 306.55,  # HFO-1336mzz(Z)
    "CF": 194.8,  # fluoromethane
    "CCF": 235.5,  # fluoroethane
    "FC(F)CC(F)(F)F": 288.5,  # R-245fa
    "CC(F)C(F)C(F)(F)F": 313.3,  # R-365mfc
    "FC(F)Cl": 232.3,  # R-22
}


def compute_boiling_point(m, sigma, epsilon_k, T_min=200.0, T_max=500.0):
    """Compute boiling point (T where P_vapor = 1 atm) from PC-SAFT parameters.

    Uses scipy.optimize.brentq to find the root of:
        vapor_pressure(T) - 101325 Pa = 0

    Args:
        m: Number of segments (dimensionless)
        sigma: Segment diameter (Angstrom)
        epsilon_k: Dispersion energy (K)
        T_min: Lower bound for temperature search (K), default 200K
        T_max: Upper bound for temperature search (K), default 500K

    Returns:
        float: Boiling point in Kelvin, or NaN if calculation fails
    """
    def objective(T):
        """Objective function: P_vapor - P_atm."""
        props = compute_properties(m, sigma, epsilon_k, T)
        vp = props.get("vapor_pressure_Pa", np.nan)
        if np.isnan(vp):
            return np.nan
        return vp - 101325.0

    try:
        # Evaluate at endpoints
        f_min = objective(T_min)
        f_max = objective(T_max)

        # If either endpoint fails, try to find a working bracket
        if np.isnan(f_min) or np.isnan(f_max):
            # Try a few intermediate points to find a working range
            test_temps = [220, 250, 280, 298, 320, 350, 400, 450]
            valid_points = []
            for T in test_temps:
                f = objective(T)
                if not np.isnan(f):
                    valid_points.append((T, f))

            if len(valid_points) < 2:
                return np.nan

            # Look for a sign change
            for i in range(len(valid_points) - 1):
                T1, f1 = valid_points[i]
                T2, f2 = valid_points[i + 1]
                if f1 * f2 < 0:
                    # Found a bracket
                    T_b = brentq(objective, T1, T2, xtol=0.1, maxiter=100)
                    return float(T_b)

            return np.nan

        # If both are same sign, no root in bracket
        if f_min * f_max > 0:
            return np.nan

        T_b = brentq(objective, T_min, T_max, xtol=0.1, maxiter=100)
        return float(T_b)

    except (ValueError, RuntimeError):
        return np.nan


def batch_boiling_points(candidates_df, n_jobs=1):
    """Compute boiling points for all candidates in a DataFrame.

    Args:
        candidates_df: DataFrame with columns m, sigma, epsilon_k
        n_jobs: Number of parallel workers. Use -1 for all cores.
            Default 1 (sequential).

    Returns:
        DataFrame: Original df with added column boiling_point_K
    """
    total = len(candidates_df)

    if n_jobs != 1:
        from joblib import Parallel, delayed

        rows = [row for _, row in candidates_df.iterrows()]
        boiling_points = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(compute_boiling_point)(row["m"], row["sigma"], row["epsilon_k"])
            for row in rows
        )
    else:
        boiling_points = []
        for idx, (_, row) in enumerate(candidates_df.iterrows()):
            if idx % 100 == 0:
                print(f"Computing boiling points: {idx}/{total}...")
            T_b = compute_boiling_point(row["m"], row["sigma"], row["epsilon_k"])
            boiling_points.append(T_b)

    df = candidates_df.copy()
    df["boiling_point_K"] = boiling_points
    print(f"Completed: {total}/{total} boiling points computed")

    return df


def hfo_parameter_distance(m, sigma, epsilon_k, reference=None, weights=None):
    """Compute weighted Euclidean distance to reference HFO in normalized parameter space.

    Distance metric:
        d = sqrt(w_m*((m-ref_m)/ref_m)^2 + w_s*((s-ref_s)/ref_s)^2 + w_e*((e-ref_e)/ref_e)^2)

    Args:
        m: Number of segments
        sigma: Segment diameter (Angstrom)
        epsilon_k: Dispersion energy (K)
        reference: dict with keys m, sigma, epsilon_k (default: HFO_1336MZZ)
        weights: dict with keys m, sigma, epsilon_k (default: all 1.0)

    Returns:
        float: Normalized weighted distance
    """
    if reference is None:
        reference = HFO_1336MZZ

    if weights is None:
        weights = {"m": 1.0, "sigma": 1.0, "epsilon_k": 1.0}

    wm = weights.get("m", 1.0)
    ws = weights.get("sigma", 1.0)
    we = weights.get("epsilon_k", 1.0)

    ref_m = reference["m"]
    ref_s = reference["sigma"]
    ref_e = reference["epsilon_k"]

    dm = wm * ((m - ref_m) / ref_m) ** 2
    ds = ws * ((sigma - ref_s) / ref_s) ** 2
    de = we * ((epsilon_k - ref_e) / ref_e) ** 2

    return float(np.sqrt(dm + ds + de))


def apply_hfo_filters(df):
    """Apply HFO-specific screening filters to candidate DataFrame.

    Filters:
    1. Boiling point in range [288, 323] K (15-50°C)
    2. Has C=C bond (olefin)
    3. No chlorine (Cl-free)
    4. At least 2 fluorine atoms
    5. SA score ≤ 4.5
    6. Fluorine mass fraction >= 65%
    7. No reactive fluorination sites

    Args:
        df: DataFrame with columns: smiles, boiling_point_K, sa_score (optional)

    Returns:
        tuple: (filtered_df, filter_stats_dict)
            filtered_df has added column cf3_count for ranking
            filter_stats_dict has keys: boiling_point, has_double_bond,
                                        no_chlorine, fluorine_count, sa_score,
                                        fluorine_mass_fraction, no_reactive_sites, total
    """
    initial_count = len(df)

    # Track per-filter pass counts
    stats = {}

    # Filter 1: Boiling point range
    mask_bp = (df["boiling_point_K"] >= 288) & (df["boiling_point_K"] <= 323)
    stats["boiling_point"] = mask_bp.sum()

    # Filter 2: Has C=C bond
    pattern_cc = Chem.MolFromSmarts("C=C")
    mask_cc = df["smiles"].apply(
        lambda s: Chem.MolFromSmiles(s).HasSubstructMatch(pattern_cc)
        if Chem.MolFromSmiles(s) is not None else False
    )
    stats["has_double_bond"] = mask_cc.sum()

    # Filter 3: No chlorine
    pattern_cl = Chem.MolFromSmarts("[Cl]")
    mask_no_cl = df["smiles"].apply(
        lambda s: not Chem.MolFromSmiles(s).HasSubstructMatch(pattern_cl)
        if Chem.MolFromSmiles(s) is not None else False
    )
    stats["no_chlorine"] = mask_no_cl.sum()

    # Filter 4: At least 2 fluorine atoms
    def count_fluorines(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F")

    mask_f = df["smiles"].apply(lambda s: count_fluorines(s) >= 2)
    stats["fluorine_count"] = mask_f.sum()

    # Filter 5: SA score ≤ 4.5 (if column exists)
    if "sa_score" in df.columns:
        mask_sa = df["sa_score"] <= 4.5
        stats["sa_score"] = mask_sa.sum()
    else:
        # Compute SA scores on the fly
        def compute_sa(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10.0  # Fail high
            try:
                return sascorer.calculateScore(mol)
            except Exception:
                return 10.0

        sa_scores = df["smiles"].apply(compute_sa)
        mask_sa = sa_scores <= 4.5
        stats["sa_score"] = mask_sa.sum()

    # Filter 6: Fluorine mass fraction >= 65%
    mask_f_mass = df["smiles"].apply(lambda s: passes_fluorine_mass_fraction(s, threshold=0.65))
    stats["fluorine_mass_fraction"] = mask_f_mass.sum()

    # Filter 7: No reactive fluorination sites
    def is_safe(smiles):
        has_reactive, _ = has_reactive_fluorine(smiles)
        return not has_reactive

    mask_no_reactive = df["smiles"].apply(is_safe)
    stats["no_reactive_sites"] = mask_no_reactive.sum()

    # Combine all filters
    final_mask = mask_bp & mask_cc & mask_no_cl & mask_f & mask_sa & mask_f_mass & mask_no_reactive
    stats["total"] = final_mask.sum()

    filtered = df[final_mask].copy()

    # Add CF3 count column for ranking (not a hard filter)
    filtered["cf3_count"] = filtered["smiles"].apply(count_cf3_groups)

    print("\nHFO Filter Results:")
    print(f"  Initial candidates: {initial_count}")
    print(f"  Boiling point [288-323K]: {stats['boiling_point']}")
    print(f"  Has C=C bond: {stats['has_double_bond']}")
    print(f"  No chlorine: {stats['no_chlorine']}")
    print(f"  F count >= 2: {stats['fluorine_count']}")
    print(f"  SA score <= 4.5: {stats['sa_score']}")
    print(f"  Fluorine mass fraction >= 65%: {stats['fluorine_mass_fraction']}")
    print(f"  No reactive fluorination sites: {stats['no_reactive_sites']}")
    print(f"  Final passing all filters: {stats['total']}")

    return filtered, stats


def validate_boiling_points():
    """Validate boiling point computation against literature PC-SAFT parameters.

    Loads fluorinated_pcsaft.csv, matches to KNOWN_BOILING_POINTS by SMILES,
    computes PC-SAFT boiling points from literature params, compares to experimental.

    Returns:
        pd.DataFrame: Columns: smiles, name, m, sigma, epsilon_k,
                      T_b_experimental_K, T_b_predicted_K, error_K
    """
    import os
    from pathlib import Path

    # Load fluorinated PC-SAFT parameters
    data_path = Path(__file__).parent.parent / "model" / "data" / "fluorinated_pcsaft.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df_lit = pd.read_csv(data_path)

    # Match to known boiling points
    results = []
    for _, row in df_lit.iterrows():
        smiles = row["smiles"]
        if smiles in KNOWN_BOILING_POINTS:
            T_b_exp = KNOWN_BOILING_POINTS[smiles]
            T_b_pred = compute_boiling_point(row["m"], row["sigma"], row["epsilon_k"])

            results.append({
                "smiles": smiles,
                "name": row.get("name", ""),
                "m": row["m"],
                "sigma": row["sigma"],
                "epsilon_k": row["epsilon_k"],
                "T_b_experimental_K": T_b_exp,
                "T_b_predicted_K": T_b_pred,
                "error_K": T_b_pred - T_b_exp if not np.isnan(T_b_pred) else np.nan,
            })

    return pd.DataFrame(results)


def validate_boiling_points_rf():
    """End-to-end validation: SMILES → RF → PC-SAFT → T_b vs experimental.

    Same as validate_boiling_points() but predicts PC-SAFT params via RF first.

    Returns:
        pd.DataFrame: Columns: smiles, name, m, sigma, epsilon_k,
                      T_b_experimental_K, T_b_predicted_K, error_K
    """
    from model.predict import predict_pcsaft

    # Get SMILES with known boiling points
    smiles_list = list(KNOWN_BOILING_POINTS.keys())

    # Predict PC-SAFT parameters
    predictions = predict_pcsaft(smiles_list)

    # Compute boiling points and compare
    results = []
    for _, row in predictions.iterrows():
        smiles = row["smiles"]
        T_b_exp = KNOWN_BOILING_POINTS[smiles]
        T_b_pred = compute_boiling_point(row["m"], row["sigma"], row["epsilon_k"])

        results.append({
            "smiles": smiles,
            "name": "",  # No names in RF prediction
            "m": row["m"],
            "sigma": row["sigma"],
            "epsilon_k": row["epsilon_k"],
            "T_b_experimental_K": T_b_exp,
            "T_b_predicted_K": T_b_pred,
            "error_K": T_b_pred - T_b_exp if not np.isnan(T_b_pred) else np.nan,
        })

    return pd.DataFrame(results)


def apply_cyclopentane_filters(df):
    """Apply broad screening filters for cyclopentane-centric ranking.

    Unlike apply_hfo_filters(), this does NOT require:
    - C=C double bond
    - No chlorine
    - Minimum fluorine count
    - Fluorine mass fraction threshold
    - Reactive fluorination site check

    Only applies:
    1. Boiling point in range [288, 323] K
    2. SA score <= 4.5

    Args:
        df: DataFrame with columns: smiles, boiling_point_K, sa_score (optional)

    Returns:
        tuple: (filtered_df, filter_stats_dict)
    """
    initial_count = len(df)
    stats = {}

    mask_bp = (df["boiling_point_K"] >= 288) & (df["boiling_point_K"] <= 323)
    stats["boiling_point"] = int(mask_bp.sum())

    if "sa_score" in df.columns:
        mask_sa = df["sa_score"] <= 4.5
    else:
        def compute_sa(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10.0
            try:
                return sascorer.calculateScore(mol)
            except Exception:
                return 10.0

        sa_scores = df["smiles"].apply(compute_sa)
        df = df.copy()
        df["sa_score"] = sa_scores
        mask_sa = sa_scores <= 4.5
    stats["sa_score"] = int(mask_sa.sum())

    final_mask = mask_bp & mask_sa
    stats["total"] = int(final_mask.sum())

    filtered = df[final_mask].copy()

    print("\nCyclopentane-centric Filter Results:")
    print(f"  Initial candidates: {initial_count}")
    print(f"  Boiling point [288-323K]: {stats['boiling_point']}")
    print(f"  SA score <= 4.5: {stats['sa_score']}")
    print(f"  Final passing all filters: {stats['total']}")

    return filtered, stats
