"""Uncertainty quantification for PC-SAFT predictions.

Provides:
- bootstrap_metric_ci: Bootstrap confidence intervals for evaluation metrics
- binomial_ci: Wilson score interval for proportions
- get_rf_tree_predictions: Per-tree joint (m, sigma, eps/k) from RF models
- propagate_trees_vp: VP ratio distribution across RF trees
- propagate_trees_henrys: Henry's constant ratio distribution across RF trees
- combined_uncertainty: Merge tree variance + k_ij sensitivity into CIs

The per-tree approach propagates each RF tree's *joint* (m, sigma, epsilon_k)
prediction through teqp, preserving parameter covariance that would be lost
by independent Gaussian sampling.
"""

from __future__ import annotations

import logging

import numpy as np
import teqp

from model.thermodynamic import (
    CYCLOPENTANE,
    HEXANE,
    R_GAS,
    T_REF,
    compute_properties,
    make_pcsaft_model,
)

logger = logging.getLogger(__name__)

TARGETS = ["m", "sigma", "epsilon_k"]


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for metrics
# ---------------------------------------------------------------------------


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_fn,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for a metric function.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values (NaN-safe: masked before resampling).
    metric_fn : callable
        ``metric_fn(y_true, y_pred) -> float``
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (0.95 = 95% CI).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``point``, ``ci_lo``, ``ci_hi``, ``std``, ``n_boot``.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 2:
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "std": np.nan, "n_boot": 0}

    point = float(metric_fn(y_true, y_pred))

    boot_values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            boot_values[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            boot_values[i] = np.nan

    valid = boot_values[np.isfinite(boot_values)]
    if len(valid) == 0:
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan,
                "std": np.nan, "n_boot": 0}

    alpha = (1 - ci) / 2
    return {
        "point": point,
        "ci_lo": float(np.percentile(valid, 100 * alpha)),
        "ci_hi": float(np.percentile(valid, 100 * (1 - alpha))),
        "std": float(np.std(valid)),
        "n_boot": len(valid),
    }


def binomial_ci(k: int, n: int, ci: float = 0.95) -> dict[str, float]:
    """Wilson score interval for a binomial proportion k/n.

    Returns dict with ``point``, ``ci_lo``, ``ci_hi``.
    """
    if n == 0:
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    from scipy.stats import norm as _norm

    p = k / n
    z = _norm.ppf(1 - (1 - ci) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom

    return {
        "point": float(p),
        "ci_lo": float(max(0.0, center - spread)),
        "ci_hi": float(min(1.0, center + spread)),
    }


# ---------------------------------------------------------------------------
# RF per-tree prediction extraction
# ---------------------------------------------------------------------------


def get_rf_tree_predictions(
    rf_models: dict,
    X: np.ndarray,
) -> np.ndarray:
    """Extract per-tree joint (m, sigma, epsilon_k) from separate RF models.

    Trees are paired by index across the three target models.  Since each
    target is trained independently, pairing is approximate but preserves
    the marginal variance structure.

    Parameters
    ----------
    rf_models : dict
        ``{target_name: sklearn.RandomForestRegressor}`` for each of
        ``m``, ``sigma``, ``epsilon_k``.
    X : np.ndarray
        Feature matrix ``(n_molecules, n_features)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_trees, n_molecules, 3)`` — axis 2 order is
        ``[m, sigma, epsilon_k]``.
    """
    tree_counts = [len(rf_models[t].estimators_) for t in TARGETS]
    n_trees = min(tree_counts)
    n_mol = X.shape[0]

    joint = np.empty((n_trees, n_mol, 3))
    for t_idx, target in enumerate(TARGETS):
        for i in range(n_trees):
            joint[i, :, t_idx] = rf_models[target].estimators_[i].predict(X)
    return joint


# ---------------------------------------------------------------------------
# Thermodynamic propagation helpers
# ---------------------------------------------------------------------------


def _prepare_solvent_vle(solvent: dict, T: float):
    """Pre-compute solvent VLE (reusable across all solutes and k_ij)."""
    model = make_pcsaft_model(solvent["m"], solvent["sigma"], solvent["epsilon_k"])
    rho_liq, rho_vap = model.pure_VLE_T(T, 10000.0, 1.0, 100)
    if rho_liq <= rho_vap or rho_vap <= 0 or rho_liq <= 0:
        raise ValueError("Solvent VLE failed")
    z = np.array([1.0])
    Ar01 = model.get_Ar01(T, rho_vap, z)
    P_sat = rho_vap * R_GAS * T * (1 + Ar01)
    if P_sat <= 0:
        raise ValueError(f"Solvent P_sat <= 0 ({P_sat})")
    return float(rho_liq), float(P_sat)


def _henrys_fast(
    m: float,
    sigma: float,
    epsilon_k: float,
    solvent: dict,
    solvent_rho_liq: float,
    solvent_Psat: float,
    T: float,
    k_ij: float,
) -> float:
    """Henry's constant using pre-computed solvent VLE (avoids redundant work)."""
    try:
        c_solute = teqp.SAFTCoeffs()
        c_solute.m = float(m)
        c_solute.sigma_Angstrom = float(sigma)
        c_solute.epsilon_over_k = float(epsilon_k)

        c_solvent = teqp.SAFTCoeffs()
        c_solvent.m = float(solvent["m"])
        c_solvent.sigma_Angstrom = float(solvent["sigma"])
        c_solvent.epsilon_over_k = float(solvent["epsilon_k"])

        kmat = [[0.0, k_ij], [k_ij, 0.0]]
        binary_model = teqp.PCSAFTEOS([c_solute, c_solvent], kmat=kmat)

        eps = 1e-10
        rho_vec = np.array([eps * solvent_rho_liq, (1.0 - eps) * solvent_rho_liq])
        ln_phi = binary_model.get_fugacity_coefficients(T, rho_vec)
        return float(np.exp(ln_phi[0]) * solvent_Psat)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Per-tree thermodynamic propagation
# ---------------------------------------------------------------------------


def propagate_trees_vp(
    tree_preds: np.ndarray,
    reference: dict = CYCLOPENTANE,
    T: float = T_REF,
) -> np.ndarray:
    """Compute VP ratio for each tree's (m, sigma, eps/k).

    Returns ``(n_trees, n_mol)`` array of VP/VP_ref.
    """
    n_trees, n_mol, _ = tree_preds.shape
    ref_props = compute_properties(
        reference["m"], reference["sigma"], reference["epsilon_k"], T
    )
    ref_vp = ref_props["vapor_pressure_Pa"]
    if not np.isfinite(ref_vp) or ref_vp <= 0:
        logger.warning("Reference VP computation failed")
        return np.full((n_trees, n_mol), np.nan)

    vp_ratios = np.full((n_trees, n_mol), np.nan)
    for i in range(n_trees):
        for j in range(n_mol):
            props = compute_properties(
                tree_preds[i, j, 0], tree_preds[i, j, 1],
                tree_preds[i, j, 2], T,
            )
            vp = props["vapor_pressure_Pa"]
            if np.isfinite(vp) and vp > 0:
                vp_ratios[i, j] = vp / ref_vp
        if (i + 1) % 20 == 0:
            logger.info("  VP trees: %d/%d", i + 1, n_trees)
    return vp_ratios


def propagate_trees_henrys(
    tree_preds: np.ndarray,
    reference: dict = CYCLOPENTANE,
    solvent: dict = HEXANE,
    T: float = T_REF,
    k_ij: float = 0.0,
) -> np.ndarray:
    """Compute H/H_ref for each tree's (m, sigma, eps/k).

    Uses cached solvent VLE to avoid redundant computation.
    Returns ``(n_trees, n_mol)`` array.
    """
    n_trees, n_mol, _ = tree_preds.shape

    solvent_rho_liq, solvent_Psat = _prepare_solvent_vle(solvent, T)

    ref_H = _henrys_fast(
        reference["m"], reference["sigma"], reference["epsilon_k"],
        solvent, solvent_rho_liq, solvent_Psat, T, k_ij,
    )
    if not np.isfinite(ref_H) or ref_H <= 0:
        logger.warning("Reference Henry's constant computation failed (k_ij=%s)", k_ij)
        return np.full((n_trees, n_mol), np.nan)

    H_ratios = np.full((n_trees, n_mol), np.nan)
    for i in range(n_trees):
        for j in range(n_mol):
            H = _henrys_fast(
                tree_preds[i, j, 0], tree_preds[i, j, 1],
                tree_preds[i, j, 2],
                solvent, solvent_rho_liq, solvent_Psat, T, k_ij,
            )
            if np.isfinite(H):
                H_ratios[i, j] = H / ref_H
        if (i + 1) % 20 == 0:
            logger.info(
                "  Henry's trees (k_ij=%.3f): %d/%d", k_ij, i + 1, n_trees,
            )
    return H_ratios


# ---------------------------------------------------------------------------
# Combined uncertainty (tree variance + k_ij sensitivity)
# ---------------------------------------------------------------------------


def combined_uncertainty(
    tree_preds: np.ndarray,
    k_ij_values: tuple[float, ...] = (-0.05, -0.025, 0.0, 0.025, 0.05),
    reference: dict = CYCLOPENTANE,
    solvent: dict = HEXANE,
    T: float = T_REF,
    ci: float = 0.95,
) -> dict[str, np.ndarray]:
    """Full per-molecule CIs: tree variance + k_ij sensitivity.

    For each molecule, creates an ``(n_kij * n_trees)`` distribution of
    H_ratio values and computes percentile-based CIs.

    Parameters
    ----------
    tree_preds : np.ndarray
        Shape ``(n_trees, n_mol, 3)`` from :func:`get_rf_tree_predictions`.
    k_ij_values : tuple
        k_ij values to sweep.
    reference, solvent : dict
        PC-SAFT parameters for reference and solvent.
    T : float
        Temperature (K).
    ci : float
        Confidence level.

    Returns
    -------
    dict of ``(n_mol,)`` arrays with keys:
        ``H_ratio_mean``, ``H_ratio_lo``, ``H_ratio_hi`` (combined),
        ``H_ratio_tree_lo``, ``H_ratio_tree_hi`` (tree-only, k_ij=0),
        ``VP_ratio_mean``, ``VP_ratio_lo``, ``VP_ratio_hi``.
    """
    n_trees, n_mol, _ = tree_preds.shape
    alpha = (1 - ci) / 2
    pct_lo = 100 * alpha
    pct_hi = 100 * (1 - alpha)

    # VP (independent of k_ij)
    logger.info("Propagating VP through %d trees x %d molecules...", n_trees, n_mol)
    vp_dist = propagate_trees_vp(tree_preds, reference, T)

    # Henry's for each k_ij
    H_by_kij: dict[float, np.ndarray] = {}
    for k_ij in k_ij_values:
        logger.info(
            "Propagating Henry's (k_ij=%.3f) through %d trees x %d molecules...",
            k_ij, n_trees, n_mol,
        )
        H_by_kij[k_ij] = propagate_trees_henrys(
            tree_preds, reference, solvent, T, k_ij,
        )

    # Stack: (n_kij, n_trees, n_mol) → flatten to (n_kij*n_trees, n_mol)
    all_H = np.stack([H_by_kij[k] for k in k_ij_values], axis=0)
    all_H_flat = all_H.reshape(-1, n_mol)

    # Tree-only distribution (k_ij=0)
    H_kij0 = H_by_kij[0.0]  # (n_trees, n_mol)

    return {
        "H_ratio_mean": np.nanmean(all_H_flat, axis=0),
        "H_ratio_lo": np.nanpercentile(all_H_flat, pct_lo, axis=0),
        "H_ratio_hi": np.nanpercentile(all_H_flat, pct_hi, axis=0),
        "H_ratio_tree_mean": np.nanmean(H_kij0, axis=0),
        "H_ratio_tree_lo": np.nanpercentile(H_kij0, pct_lo, axis=0),
        "H_ratio_tree_hi": np.nanpercentile(H_kij0, pct_hi, axis=0),
        "VP_ratio_mean": np.nanmean(vp_dist, axis=0),
        "VP_ratio_lo": np.nanpercentile(vp_dist, pct_lo, axis=0),
        "VP_ratio_hi": np.nanpercentile(vp_dist, pct_hi, axis=0),
    }
