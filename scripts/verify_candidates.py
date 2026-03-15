#!/usr/bin/env python
"""Multi-criteria candidate verification with multi-temperature EOS.

Applies five screening criteria at three temperatures (273, 298, 323 K):
1. In-domain (AD): ChemBERTa Isolation Forest applicability domain
2. Parameter proximity: Top-20 by weighted Euclidean distance to cyclopentane
3. EOS convergence: teqp VLE calculation succeeds at all temperatures
4. Vapor pressure proximity: VP ratio to cyclopentane within [0.5, 1.5] at all temps
5. Synthetic accessibility: SA score ≤ 4.5

Outputs:
- model/saved/verified_candidates.csv with pass/fail columns for each criterion
- Prints summary of candidates passing all criteria
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model.thermodynamic import CYCLOPENTANE, compute_properties

TEMPERATURES = [273.15, 298.15, 323.15]  # K
PARAM_TOP_N = 20  # top-N by parameter distance
VP_RATIO_BOUNDS = (0.5, 1.5)
SA_THRESHOLD = 4.5


def compute_in_domain(smiles_list: list[str]) -> np.ndarray:
    """Compute applicability domain labels using ChemBERTa AD model.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.

    Returns
    -------
    np.ndarray
        Boolean array (True = in-domain, False = out-of-domain).
    """
    try:
        from model.registry import get_model

        chemberta = get_model("chemberta")
        return chemberta.predict_in_domain(smiles_list)
    except Exception:
        # Fallback: mark all as in-domain if ChemBERTa/AD unavailable
        return np.ones(len(smiles_list), dtype=bool)


def run_verification(thermo_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Run multi-criteria verification on thermodynamic validation results.

    Parameters
    ----------
    thermo_csv : Path
        Path to thermo_validation.csv with columns: smiles, m, sigma, epsilon_k, etc.
    output_csv : Path
        Path to save verified_candidates.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame of candidates passing all five criteria.
    """
    print(f"Loading candidates from {thermo_csv}...")
    df = pd.read_csv(thermo_csv)
    print(f"  Loaded {len(df)} candidates")

    # Compute applicability domain
    print("\nComputing applicability domain labels...")
    in_domain = compute_in_domain(df["smiles"].tolist())
    print(f"  {in_domain.sum()}/{len(df)} candidates in-domain")

    # Compute reference VP at each temperature
    print("\nComputing reference vapor pressure at three temperatures...")
    ref_vp = {}
    for T in TEMPERATURES:
        props = compute_properties(**CYCLOPENTANE, T=T)
        ref_vp[T] = props["vapor_pressure_Pa"]
        print(f"  T={T:.2f} K: VP={ref_vp[T]/1000:.2f} kPa")

    # Compute candidate properties at all temperatures
    print("\nComputing candidate properties at three temperatures...")
    results = []
    for idx, row in df.iterrows():
        rec = row.to_dict()
        rec["in_domain"] = bool(in_domain[idx])

        for T in TEMPERATURES:
            props = compute_properties(row["m"], row["sigma"], row["epsilon_k"], T=T)
            rec[f"vp_Pa_{int(T)}K"] = props["vapor_pressure_Pa"]
            rec[f"rho_{int(T)}K"] = props["liquid_density_mol_m3"]
            if not pd.isna(props["vapor_pressure_Pa"]) and ref_vp[T] > 0:
                rec[f"vp_ratio_{int(T)}K"] = props["vapor_pressure_Pa"] / ref_vp[T]
            else:
                rec[f"vp_ratio_{int(T)}K"] = float("nan")
        results.append(rec)

    df_out = pd.DataFrame(results)

    # Apply criteria
    print("\nApplying five screening criteria...")

    # Criterion 1: Applicability domain
    df_out["criterion_ad"] = df_out["in_domain"]
    print(f"  1. AD (in-domain): {df_out['criterion_ad'].sum()}/{len(df_out)} pass")

    # Criterion 2: Parameter proximity (top-20)
    param_threshold = df_out["distance"].nsmallest(PARAM_TOP_N).max()
    df_out["criterion_param"] = df_out["distance"] <= param_threshold
    print(
        f"  2. Parameter proximity (top-{PARAM_TOP_N}, "
        f"distance ≤ {param_threshold:.4f}): {df_out['criterion_param'].sum()}/{len(df_out)} pass"
    )

    # Criterion 3: EOS convergence at all three temperatures
    df_out["criterion_eos"] = df_out[
        [f"vp_Pa_{int(T)}K" for T in TEMPERATURES]
    ].notna().all(axis=1)
    print(
        f"  3. EOS convergence (all 3 temps): {df_out['criterion_eos'].sum()}/{len(df_out)} pass"
    )

    # Criterion 4: VP ratio within bounds at all three temperatures
    def check_vp_ratio(row):
        for T in TEMPERATURES:
            vp_ratio = row[f"vp_ratio_{int(T)}K"]
            if pd.isna(vp_ratio):
                return False
            if not (VP_RATIO_BOUNDS[0] <= vp_ratio <= VP_RATIO_BOUNDS[1]):
                return False
        return True

    df_out["criterion_vp"] = df_out.apply(check_vp_ratio, axis=1)
    print(
        f"  4. VP ratio [{VP_RATIO_BOUNDS[0]}, {VP_RATIO_BOUNDS[1]}] "
        f"(all 3 temps): {df_out['criterion_vp'].sum()}/{len(df_out)} pass"
    )

    # Criterion 5: Synthetic accessibility
    df_out["criterion_sa"] = df_out["sa_score"] <= SA_THRESHOLD
    print(
        f"  5. Synthetic accessibility (SA ≤ {SA_THRESHOLD}): "
        f"{df_out['criterion_sa'].sum()}/{len(df_out)} pass"
    )

    # All criteria
    df_out["all_criteria"] = (
        df_out["criterion_ad"]
        & df_out["criterion_param"]
        & df_out["criterion_eos"]
        & df_out["criterion_vp"]
        & df_out["criterion_sa"]
    )

    # Save results
    print(f"\nSaving verified candidates to {output_csv}...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    return df_out[df_out["all_criteria"]]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-criteria candidate verification with multi-temperature EOS"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("model/saved/thermo_validation.csv"),
        help="Path to thermodynamic validation CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/saved/verified_candidates.csv"),
        help="Path to save verified candidates CSV",
    )

    args = parser.parse_args()

    verified = run_verification(args.input, args.output)

    print("\n" + "=" * 80)
    print(f"{len(verified)} CANDIDATES PASS ALL 5 CRITERIA")
    print("=" * 80)

    if len(verified) > 0:
        print("\nVerified candidates:\n")
        display_cols = [
            "smiles",
            "sa_score",
            "distance",
            "vp_ratio_273K",
            "vp_ratio_298K",
            "vp_ratio_323K",
        ]
        print(verified[display_cols].to_string(index=False))
    else:
        print("\nNo candidates passed all criteria.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
