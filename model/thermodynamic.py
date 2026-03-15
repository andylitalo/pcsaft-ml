"""Thermodynamic property computation from PC-SAFT parameters via teqp."""

import numpy as np
import pandas as pd
import teqp

# Cyclopentane reference
CYCLOPENTANE = {"m": 2.3655, "sigma": 3.7114, "epsilon_k": 288.84}
T_REF = 298.15  # K
R_GAS = 8.314462  # J/(mol·K)


def make_pcsaft_model(m, sigma, epsilon_k):
    """Create a teqp PCSAFT model.

    Args:
        m: Number of segments (dimensionless)
        sigma: Segment diameter (Angstrom)
        epsilon_k: Dispersion energy (K)

    Returns:
        teqp.PCSAFTEOS: PC-SAFT equation of state model
    """
    c = teqp.SAFTCoeffs()
    c.m = float(m)
    c.sigma_Angstrom = float(sigma)
    c.epsilon_over_k = float(epsilon_k)
    return teqp.PCSAFTEOS([c])


def compute_properties(m, sigma, epsilon_k, T=T_REF):
    """Compute vapor pressure and liquid density at temperature T.

    Args:
        m: Number of segments (dimensionless)
        sigma: Segment diameter (Angstrom)
        epsilon_k: Dispersion energy (K)
        T: Temperature (K), default is T_REF (298.15 K)

    Returns:
        dict: Contains vapor_pressure_Pa and liquid_density_mol_m3
              Returns NaN values if calculation fails.
    """
    try:
        model = make_pcsaft_model(m, sigma, epsilon_k)
        # Use good initial guesses: ~10000 mol/m³ for liquid, ~1 mol/m³ for vapor
        rho_liq, rho_vap = model.pure_VLE_T(T, 10000.0, 1.0, 100)

        # Validate: liquid should be denser than vapor
        if rho_liq <= rho_vap or rho_vap <= 0 or rho_liq <= 0:
            return {"vapor_pressure_Pa": np.nan, "liquid_density_mol_m3": np.nan}

        z = np.array([1.0])
        Ar01 = model.get_Ar01(T, rho_vap, z)
        p_sat = rho_vap * R_GAS * T * (1 + Ar01)

        if p_sat <= 0:
            return {"vapor_pressure_Pa": np.nan, "liquid_density_mol_m3": np.nan}

        return {
            "vapor_pressure_Pa": float(p_sat),
            "liquid_density_mol_m3": float(rho_liq),
        }
    except Exception:
        return {"vapor_pressure_Pa": np.nan, "liquid_density_mol_m3": np.nan}


def validate_candidates(candidates_df, reference=CYCLOPENTANE, T=T_REF):
    """Add thermodynamic property columns to screening results.

    Args:
        candidates_df: DataFrame with columns m, sigma, epsilon_k
        reference: dict with reference compound PC-SAFT parameters
        T: Temperature (K)

    Returns:
        tuple: (validated_df, ref_props)
            validated_df: Original df with added columns:
                - vapor_pressure_Pa
                - liquid_density_mol_m3
                - vp_ratio_to_ref
                - rho_ratio_to_ref
                - property_distance
            ref_props: dict with reference properties
    """
    # Compute reference properties
    ref_props = compute_properties(
        reference["m"], reference["sigma"], reference["epsilon_k"], T
    )

    results = []
    for _, row in candidates_df.iterrows():
        props = compute_properties(row["m"], row["sigma"], row["epsilon_k"], T)
        props["vp_ratio_to_ref"] = (
            props["vapor_pressure_Pa"] / ref_props["vapor_pressure_Pa"]
            if ref_props["vapor_pressure_Pa"] > 0
            else np.nan
        )
        props["rho_ratio_to_ref"] = (
            props["liquid_density_mol_m3"] / ref_props["liquid_density_mol_m3"]
            if ref_props["liquid_density_mol_m3"] > 0
            else np.nan
        )
        results.append(props)

    props_df = pd.DataFrame(results)
    out = pd.concat([candidates_df.reset_index(drop=True), props_df], axis=1)

    # Compute property distance (normalized Euclidean distance)
    ref_vp = ref_props["vapor_pressure_Pa"]
    ref_rho = ref_props["liquid_density_mol_m3"]
    out["property_distance"] = np.sqrt(
        ((out["vapor_pressure_Pa"] - ref_vp) / ref_vp) ** 2
        + ((out["liquid_density_mol_m3"] - ref_rho) / ref_rho) ** 2
    )

    return out, ref_props


# n-Hexane reference for Henry's constant computations
HEXANE = {"m": 3.0576, "sigma": 3.7983, "epsilon_k": 236.77}


def compute_henrys_constant(solute_params, solvent_params=HEXANE, T=T_REF, k_ij=0.0):
    """Compute Henry's constant for solute at infinite dilution in solvent via PC-SAFT.

    Method (from Step 21):
    1. Pure solvent VLE -> rho_L, P_sat via pure_VLE_T
    2. Create binary PC-SAFT model [solute, solvent]
    3. Set rho_vec = [eps*rho_L, (1-eps)*rho_L] with eps = 1e-10
    4. phi_inf = exp(get_fugacity_coefficients(T, rho_vec)[0])
    5. H = phi_inf * P_sat

    For binary models with k_ij, use teqp's kmat parameter.

    Args:
        solute_params: dict with keys m, sigma, epsilon_k
        solvent_params: dict with keys m, sigma, epsilon_k (default: hexane)
        T: Temperature (K)
        k_ij: Binary interaction parameter (default: 0.0)

    Returns:
        dict with keys: H_Pa, phi_inf, P_sat_solvent_Pa, ln_phi_inf
        Returns None if computation fails.
    """
    try:
        # Step 1: Compute pure solvent VLE
        solvent_model = make_pcsaft_model(
            solvent_params["m"], solvent_params["sigma"], solvent_params["epsilon_k"]
        )
        rho_liq, rho_vap = solvent_model.pure_VLE_T(T, 10000.0, 1.0, 100)

        # Validate VLE
        if rho_liq <= rho_vap or rho_vap <= 0 or rho_liq <= 0:
            return None

        # Compute saturation pressure
        z = np.array([1.0])
        Ar01 = solvent_model.get_Ar01(T, rho_vap, z)
        P_sat = rho_vap * R_GAS * T * (1 + Ar01)

        if P_sat <= 0:
            return None

        # Step 2: Create binary PC-SAFT model [solute, solvent]
        c_solute = teqp.SAFTCoeffs()
        c_solute.m = float(solute_params["m"])
        c_solute.sigma_Angstrom = float(solute_params["sigma"])
        c_solute.epsilon_over_k = float(solute_params["epsilon_k"])

        c_solvent = teqp.SAFTCoeffs()
        c_solvent.m = float(solvent_params["m"])
        c_solvent.sigma_Angstrom = float(solvent_params["sigma"])
        c_solvent.epsilon_over_k = float(solvent_params["epsilon_k"])

        # Create binary model with k_ij matrix
        kmat = [[0.0, k_ij], [k_ij, 0.0]]
        binary_model = teqp.PCSAFTEOS([c_solute, c_solvent], kmat=kmat)

        # Step 3: Set infinite dilution composition (eps = 1e-10)
        eps = 1e-10
        rho_vec = np.array([eps * rho_liq, (1.0 - eps) * rho_liq])

        # Step 4: Compute fugacity coefficient at infinite dilution
        ln_phi = binary_model.get_fugacity_coefficients(T, rho_vec)
        phi_inf = np.exp(ln_phi[0])

        # Step 5: Henry's constant
        H = phi_inf * P_sat

        return {
            "H_Pa": float(H),
            "phi_inf": float(phi_inf),
            "P_sat_solvent_Pa": float(P_sat),
            "ln_phi_inf": float(ln_phi[0]),
        }
    except Exception:
        return None


def batch_henrys_constant(
    candidates_df, solvent_params=HEXANE, T=T_REF, k_ij=0.0, reference=CYCLOPENTANE
):
    """Compute Henry's constant for all candidates in a DataFrame.

    Expects columns: m, sigma, epsilon_k.
    Adds columns: H_Pa, H_ratio (relative to cyclopentane in same solvent).

    Args:
        candidates_df: DataFrame with columns m, sigma, epsilon_k
        solvent_params: dict with solvent PC-SAFT parameters
        T: Temperature (K)
        k_ij: Binary interaction parameter
        reference: dict with reference compound PC-SAFT parameters

    Returns:
        DataFrame with added columns: H_Pa, phi_inf, P_sat_solvent_Pa, H_ratio
    """
    # Compute reference Henry's constant
    ref_result = compute_henrys_constant(reference, solvent_params, T, k_ij)
    if ref_result is None:
        raise ValueError("Failed to compute reference Henry's constant")

    H_ref = ref_result["H_Pa"]

    results = []
    total = len(candidates_df)
    for idx, row in candidates_df.iterrows():
        if idx % 50 == 0:
            print(f"Processing {idx}/{total}...")

        solute_params = {
            "m": row["m"],
            "sigma": row["sigma"],
            "epsilon_k": row["epsilon_k"],
        }

        result = compute_henrys_constant(solute_params, solvent_params, T, k_ij)

        if result is None:
            results.append(
                {
                    "H_Pa": np.nan,
                    "phi_inf": np.nan,
                    "P_sat_solvent_Pa": np.nan,
                    "H_ratio": np.nan,
                }
            )
        else:
            results.append(
                {
                    "H_Pa": result["H_Pa"],
                    "phi_inf": result["phi_inf"],
                    "P_sat_solvent_Pa": result["P_sat_solvent_Pa"],
                    "H_ratio": result["H_Pa"] / H_ref,
                }
            )

    results_df = pd.DataFrame(results)
    return pd.concat([candidates_df.reset_index(drop=True), results_df], axis=1)
