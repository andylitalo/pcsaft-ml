"""Thermodynamic property computation from PC-SAFT parameters via teqp.

This module provides optional thermodynamic property calculations that require
the teqp library. If teqp is not installed, these functions will raise ImportError.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Module-level flag to check if teqp is available
_TEQP_AVAILABLE = None


def _check_teqp():
    """Check if teqp is available and cache the result."""
    global _TEQP_AVAILABLE
    if _TEQP_AVAILABLE is None:
        try:
            import teqp  # noqa: F401

            _TEQP_AVAILABLE = True
        except ImportError:
            _TEQP_AVAILABLE = False
    return _TEQP_AVAILABLE


# Constants
T_REF = 298.15  # K
R_GAS = 8.314462  # J/(mol·K)
P_ATM = 101325.0  # Pa

# Reference compounds
CYCLOPENTANE = {"m": 2.3655, "sigma": 3.7114, "epsilon_k": 288.84}
HEXANE = {"m": 3.0576, "sigma": 3.7983, "epsilon_k": 236.77}


def compute_boiling_point(m, sigma, epsilon_k, T_min=200.0, T_max=500.0):  # noqa: N803
    """Compute boiling point (T where P_vapor = 1 atm) from PC-SAFT parameters.

    Requires teqp to be installed.

    Parameters
    ----------
    m : float
        Number of segments (dimensionless).
    sigma : float
        Segment diameter (Angstrom).
    epsilon_k : float
        Dispersion energy (K).
    T_min : float
        Lower bound for temperature search (K), default 200K.
    T_max : float
        Upper bound for temperature search (K), default 500K.

    Returns
    -------
    float
        Boiling point in Kelvin, or NaN if calculation fails.

    Raises
    ------
    ImportError
        If teqp is not installed.
    """
    if not _check_teqp():
        raise ImportError(
            "teqp is required for thermodynamic calculations. "
            "Install it with: pip install teqp"
        )

    import teqp
    from scipy.optimize import brentq

    def make_pcsaft_model(m, sigma, epsilon_k):
        """Create a teqp PCSAFT model."""
        c = teqp.SAFTCoeffs()
        c.m = float(m)
        c.sigma_Angstrom = float(sigma)
        c.epsilon_over_k = float(epsilon_k)
        return teqp.PCSAFTEOS([c])

    def compute_vapor_pressure(m, sigma, epsilon_k, T):  # noqa: N803
        """Compute vapor pressure at temperature T."""
        try:
            model = make_pcsaft_model(m, sigma, epsilon_k)
            rho_liq, rho_vap = model.pure_VLE_T(T, 10000.0, 1.0, 100)

            # Validate VLE
            if rho_liq <= rho_vap or rho_vap <= 0 or rho_liq <= 0:
                return np.nan

            z = np.array([1.0])
            Ar01 = model.get_Ar01(T, rho_vap, z)  # noqa: N806
            p_sat = rho_vap * R_GAS * T * (1 + Ar01)

            if p_sat <= 0:
                return np.nan

            return float(p_sat)
        except Exception:
            return np.nan

    def objective(T):  # noqa: N803
        """Objective function: P_vapor - P_atm."""
        vp = compute_vapor_pressure(m, sigma, epsilon_k, T)
        if np.isnan(vp):
            return np.nan
        return vp - P_ATM

    try:
        # Evaluate at endpoints
        f_min = objective(T_min)
        f_max = objective(T_max)

        # If either endpoint fails, try to find a working bracket
        if np.isnan(f_min) or np.isnan(f_max):
            # Try a few intermediate points to find a working range
            test_temps = [220, 250, 280, 298, 320, 350, 400, 450]
            valid_points = []
            for T in test_temps:  # noqa: N806
                f = objective(T)
                if not np.isnan(f):
                    valid_points.append((T, f))

            if len(valid_points) < 2:
                return np.nan

            # Look for a sign change
            for i in range(len(valid_points) - 1):
                T1, f1 = valid_points[i]  # noqa: N806
                T2, f2 = valid_points[i + 1]  # noqa: N806
                if f1 * f2 < 0:
                    # Found a bracket
                    T_b = brentq(objective, T1, T2, xtol=0.1, maxiter=100)  # noqa: N806
                    return float(T_b)

            return np.nan

        # Check if we have a sign change
        if f_min * f_max > 0:
            # No sign change, boiling point outside range
            return np.nan

        # Use Brent's method to find the root
        T_b = brentq(objective, T_min, T_max, xtol=0.1, maxiter=100)  # noqa: N806
        return float(T_b)

    except Exception as e:
        logger.warning("Boiling point calculation failed: %s", e)
        return np.nan


def compute_henrys_constant(solute_params, solvent_params=None, T=T_REF, k_ij=0.0):  # noqa: N803
    """Compute Henry's constant for solute at infinite dilution in solvent via PC-SAFT.

    Requires teqp to be installed.

    Method:
    1. Pure solvent VLE -> rho_L, P_sat via pure_VLE_T
    2. Create binary PC-SAFT model [solute, solvent]
    3. Set rho_vec = [eps*rho_L, (1-eps)*rho_L] with eps = 1e-10
    4. phi_inf = exp(get_fugacity_coefficients(T, rho_vec)[0])
    5. H = phi_inf * P_sat

    Parameters
    ----------
    solute_params : dict
        Dictionary with keys: m, sigma, epsilon_k.
    solvent_params : dict | None
        Solvent parameters (default: hexane). Dictionary with keys: m, sigma, epsilon_k.
    T : float
        Temperature (K), default is T_REF (298.15 K).
    k_ij : float
        Binary interaction parameter (default: 0.0).

    Returns
    -------
    dict | None
        Dictionary with keys: H_Pa, phi_inf, P_sat_solvent_Pa, ln_phi_inf.
        Returns None if computation fails.

    Raises
    ------
    ImportError
        If teqp is not installed.
    """
    if not _check_teqp():
        raise ImportError(
            "teqp is required for thermodynamic calculations. "
            "Install it with: pip install teqp"
        )

    import teqp

    if solvent_params is None:
        solvent_params = HEXANE

    def make_pcsaft_model(m, sigma, epsilon_k):
        """Create a teqp PCSAFT model."""
        c = teqp.SAFTCoeffs()
        c.m = float(m)
        c.sigma_Angstrom = float(sigma)
        c.epsilon_over_k = float(epsilon_k)
        return teqp.PCSAFTEOS([c])

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
        Ar01 = solvent_model.get_Ar01(T, rho_vap, z)  # noqa: N806
        P_sat = rho_vap * R_GAS * T * (1 + Ar01)  # noqa: N806

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
        H = phi_inf * P_sat  # noqa: N806

        return {
            "H_Pa": float(H),
            "phi_inf": float(phi_inf),
            "P_sat_solvent_Pa": float(P_sat),
            "ln_phi_inf": float(ln_phi[0]),
        }
    except Exception as e:
        logger.warning("Henry's constant calculation failed: %s", e)
        return None
