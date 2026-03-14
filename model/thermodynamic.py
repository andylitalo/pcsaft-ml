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
