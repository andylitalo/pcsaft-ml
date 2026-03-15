#!/usr/bin/env python
"""Run NIST WebBook experimental validation.

This script validates predicted PC-SAFT parameters against experimental
vapor pressure and liquid density data from NIST WebBook. It compares:
1. RF-predicted parameters → teqp → VP/density vs experimental
2. SPT-PCSAFT parameters → teqp → VP/density vs experimental (where available)
3. Esper reference parameters → teqp → VP/density vs experimental (where available)

This answers: "do the predicted parameters produce accurate thermodynamic
properties?" rather than "are the predicted parameters close to reference?"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.data.load import DATA_DIR
from model.predict import predict_pcsaft
from model.thermodynamic import T_REF, compute_properties


def compute_mare(predicted, experimental, name=""):
    """Compute mean absolute relative error, excluding NaN and gas-phase compounds.

    Args:
        predicted: Array of predicted values
        experimental: Array of experimental values
        name: Name of the property for logging

    Returns:
        float: MARE as a percentage
    """
    pred = np.array(predicted)
    exp = np.array(experimental)

    # Filter out NaN and invalid values
    valid = ~(np.isnan(pred) | np.isnan(exp) | (exp <= 0))

    if not valid.any():
        print(f"  Warning: No valid {name} values for MARE calculation")
        return np.nan

    pred_valid = pred[valid]
    exp_valid = exp[valid]

    are = np.abs((pred_valid - exp_valid) / exp_valid)
    mare = np.mean(are) * 100

    return mare


def within_tolerance(predicted, experimental, tolerance=0.20):
    """Compute fraction of predictions within ±tolerance of experiment.

    Args:
        predicted: Array of predicted values
        experimental: Array of experimental values
        tolerance: Relative tolerance (default 0.20 for ±20%)

    Returns:
        float: Fraction within tolerance (0 to 1)
    """
    pred = np.array(predicted)
    exp = np.array(experimental)

    valid = ~(np.isnan(pred) | np.isnan(exp) | (exp <= 0))

    if not valid.any():
        return np.nan

    pred_valid = pred[valid]
    exp_valid = exp[valid]

    relative_error = np.abs((pred_valid - exp_valid) / exp_valid)
    within = (relative_error <= tolerance).sum()

    return within / len(pred_valid)


def plot_parity(
    exp_vp, pred_vp, exp_rho, pred_rho, output_dir, label="RF"
):
    """Create parity plots for VP and density.

    Args:
        exp_vp: Experimental vapor pressure (Pa)
        pred_vp: Predicted vapor pressure (Pa)
        exp_rho: Experimental liquid density (kg/m³)
        pred_rho: Predicted liquid density (kg/m³)
        output_dir: Directory to save figures
        label: Label for the prediction method
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Vapor pressure parity plot (convert to kPa)
    exp_vp_kpa = np.array(exp_vp) / 1000
    pred_vp_kpa = np.array(pred_vp) / 1000

    # Filter valid values
    valid_vp = ~(np.isnan(exp_vp_kpa) | np.isnan(pred_vp_kpa) | (exp_vp_kpa <= 0))
    exp_vp_valid = exp_vp_kpa[valid_vp]
    pred_vp_valid = pred_vp_kpa[valid_vp]

    if len(exp_vp_valid) > 0:
        ax1.scatter(
            exp_vp_valid, pred_vp_valid,
            alpha=0.6, s=60, edgecolors='black', linewidths=0.5
        )

        # Diagonal line
        vp_min = min(exp_vp_valid.min(), pred_vp_valid.min())
        vp_max = max(exp_vp_valid.max(), pred_vp_valid.max())
        ax1.plot([vp_min, vp_max], [vp_min, vp_max], 'r--', lw=2, label='Ideal')

        # ±20% bounds
        ax1.plot([vp_min, vp_max], [vp_min*0.8, vp_max*0.8], 'k:', lw=1, alpha=0.5, label='±20%')
        ax1.plot([vp_min, vp_max], [vp_min*1.2, vp_max*1.2], 'k:', lw=1, alpha=0.5)

        # Compute statistics
        mare_vp = compute_mare(pred_vp_valid, exp_vp_valid)
        within_20_vp = within_tolerance(pred_vp_valid, exp_vp_valid, 0.20) * 100

        textstr = (
            f'MARE: {mare_vp:.1f}%\n'
            f'Within ±20%: {within_20_vp:.1f}%\n'
            f'n = {len(exp_vp_valid)}'
        )
        ax1.text(
            0.05, 0.95, textstr,
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    ax1.set_xlabel('Experimental VP (kPa)', fontsize=14)
    ax1.set_ylabel(f'Predicted VP ({label}) (kPa)', fontsize=14)
    ax1.set_title(f'Vapor Pressure at {T_REF} K', fontsize=16, pad=20)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.tick_params(labelsize=11)

    # Liquid density parity plot
    exp_rho_arr = np.array(exp_rho)
    pred_rho_arr = np.array(pred_rho)

    valid_rho = ~(np.isnan(exp_rho_arr) | np.isnan(pred_rho_arr) | (exp_rho_arr <= 0))
    exp_rho_valid = exp_rho_arr[valid_rho]
    pred_rho_valid = pred_rho_arr[valid_rho]

    if len(exp_rho_valid) > 0:
        ax2.scatter(
            exp_rho_valid, pred_rho_valid,
            alpha=0.6, s=60, edgecolors='black', linewidths=0.5, color='green'
        )

        # Diagonal line
        rho_min = min(exp_rho_valid.min(), pred_rho_valid.min())
        rho_max = max(exp_rho_valid.max(), pred_rho_valid.max())
        ax2.plot([rho_min, rho_max], [rho_min, rho_max], 'r--', lw=2, label='Ideal')

        # ±20% bounds
        ax2.plot(
            [rho_min, rho_max],
            [rho_min*0.8, rho_max*0.8],
            'k:', lw=1, alpha=0.5, label='±20%'
        )
        ax2.plot([rho_min, rho_max], [rho_min*1.2, rho_max*1.2], 'k:', lw=1, alpha=0.5)

        # Compute statistics
        mare_rho = compute_mare(pred_rho_valid, exp_rho_valid)
        within_20_rho = within_tolerance(pred_rho_valid, exp_rho_valid, 0.20) * 100

        textstr = (
            f'MARE: {mare_rho:.1f}%\n'
            f'Within ±20%: {within_20_rho:.1f}%\n'
            f'n = {len(exp_rho_valid)}'
        )
        ax2.text(
            0.05, 0.95, textstr,
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    ax2.set_xlabel('Experimental Density (kg/m³)', fontsize=14)
    ax2.set_ylabel(f'Predicted Density ({label}) (kg/m³)', fontsize=14)
    ax2.set_title(f'Liquid Density at {T_REF} K', fontsize=16, pad=20)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)

    plt.tight_layout()
    filename = f"parity_{label.lower().replace(' ', '_')}.png"
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()

    print(f"  ✓ {filename}")


def plot_method_comparison(results_df, output_dir):
    """Create comparison plots for RF vs SPT vs Esper predictions.

    Args:
        results_df: DataFrame with all prediction methods
        output_dir: Directory to save figures
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    methods = [
        ('rf', 'RF Predicted', 'blue'),
        ('spt', 'SPT-PCSAFT', 'orange'),
        ('esper', 'Esper Reference', 'green')
    ]

    for idx, (method, label, color) in enumerate(methods):
        vp_col = f'vp_{method}_Pa'
        rho_col = f'rho_{method}_kg_m3'

        if vp_col not in results_df.columns:
            continue

        # Vapor pressure subplot
        ax_vp = axes[0, idx]
        exp_vp = results_df['vp_exp_Pa'].values / 1000
        pred_vp = results_df[vp_col].values / 1000

        valid_vp = ~(np.isnan(exp_vp) | np.isnan(pred_vp) | (exp_vp <= 0))
        if valid_vp.any():
            ax_vp.scatter(
                exp_vp[valid_vp], pred_vp[valid_vp],
                alpha=0.6, s=50, edgecolors='black', linewidths=0.5, color=color
            )

            vp_min = min(exp_vp[valid_vp].min(), pred_vp[valid_vp].min())
            vp_max = max(exp_vp[valid_vp].max(), pred_vp[valid_vp].max())
            ax_vp.plot([vp_min, vp_max], [vp_min, vp_max], 'r--', lw=2)

            mare = compute_mare(pred_vp[valid_vp], exp_vp[valid_vp])
            ax_vp.text(
                0.05, 0.95, f'MARE: {mare:.1f}%',
                transform=ax_vp.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        ax_vp.set_xlabel('Exp VP (kPa)', fontsize=12)
        ax_vp.set_ylabel('Pred VP (kPa)', fontsize=12)
        ax_vp.set_title(f'{label}: Vapor Pressure', fontsize=13)
        ax_vp.grid(True, alpha=0.3)
        ax_vp.set_xscale('log')
        ax_vp.set_yscale('log')

        # Density subplot
        ax_rho = axes[1, idx]
        exp_rho = results_df['rho_exp_kg_m3'].values
        pred_rho = results_df[rho_col].values

        valid_rho = ~(np.isnan(exp_rho) | np.isnan(pred_rho) | (exp_rho <= 0))
        if valid_rho.any():
            ax_rho.scatter(
                exp_rho[valid_rho], pred_rho[valid_rho],
                alpha=0.6, s=50, edgecolors='black', linewidths=0.5, color=color
            )

            rho_min = min(exp_rho[valid_rho].min(), pred_rho[valid_rho].min())
            rho_max = max(exp_rho[valid_rho].max(), pred_rho[valid_rho].max())
            ax_rho.plot([rho_min, rho_max], [rho_min, rho_max], 'r--', lw=2)

            mare = compute_mare(pred_rho[valid_rho], exp_rho[valid_rho])
            ax_rho.text(
                0.05, 0.95, f'MARE: {mare:.1f}%',
                transform=ax_rho.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        ax_rho.set_xlabel('Exp Density (kg/m³)', fontsize=12)
        ax_rho.set_ylabel('Pred Density (kg/m³)', fontsize=12)
        ax_rho.set_title(f'{label}: Liquid Density', fontsize=13)
        ax_rho.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=300)
    plt.close()

    print("  ✓ method_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description="Validate predicted PC-SAFT parameters against NIST experimental data"
    )
    parser.add_argument(
        "--nist-data",
        type=Path,
        default=DATA_DIR / "nist_validation.csv",
        help="Path to NIST validation dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/saved/nist_validation_results.csv"),
        help="Path for output CSV with all predictions",
    )

    args = parser.parse_args()

    # Load NIST validation data
    print(f"Loading NIST validation data from {args.nist_data}...")
    nist_df = pd.read_csv(args.nist_data)
    print(f"Loaded {len(nist_df)} molecules")

    # Filter out compounds without experimental data
    valid_df = nist_df.dropna(subset=['vp_exp_Pa', 'rho_exp_kg_m3'])
    print(f"  {len(valid_df)} molecules have both VP and density data")

    # Predict PC-SAFT parameters using RF
    print("\nPredicting PC-SAFT parameters using RF model...")
    smiles_list = valid_df['smiles'].tolist()
    rf_predictions = predict_pcsaft(smiles_list)

    # Compute thermodynamic properties from RF predictions
    print(f"Computing thermodynamic properties at {T_REF} K...")
    vp_rf = []
    rho_rf = []

    for _, row in rf_predictions.iterrows():
        props = compute_properties(row['m'], row['sigma'], row['epsilon_k'], T_REF)

        # Convert density from mol/m³ to kg/m³ (need molecular weight)
        # For now, store mol/m³ and convert later with MW from RDKit
        vp_rf.append(props['vapor_pressure_Pa'])
        rho_mol_m3 = props['liquid_density_mol_m3']

        # Get molecular weight from SMILES
        from rdkit import Chem
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            mw = Chem.Descriptors.MolWt(mol) / 1000  # kg/mol
            rho_kg_m3 = rho_mol_m3 * mw
        else:
            rho_kg_m3 = np.nan

        rho_rf.append(rho_kg_m3)

    # Add RF results to dataframe
    results_df = valid_df.copy()
    results_df['m_rf'] = rf_predictions['m'].values
    results_df['sigma_rf'] = rf_predictions['sigma'].values
    results_df['epsilon_k_rf'] = rf_predictions['epsilon_k'].values
    results_df['vp_rf_Pa'] = vp_rf
    results_df['rho_rf_kg_m3'] = rho_rf

    # Load SPT-PCSAFT parameters if available
    spt_csv = DATA_DIR / "spt_pcsaft.csv"
    if spt_csv.exists():
        print("\nLoading SPT-PCSAFT parameters...")
        spt_df = pd.read_csv(spt_csv, skiprows=1)  # Skip license header

        # Rename SPT columns to avoid conflicts
        spt_df = spt_df.rename(columns={
            'm': 'm_spt',
            'sigma': 'sigma_spt',
            'epsilon_k': 'epsilon_k_spt'
        })

        # Merge with validation set on SMILES
        results_df = results_df.merge(
            spt_df[['SMILES0', 'm_spt', 'sigma_spt', 'epsilon_k_spt']],
            left_on='smiles',
            right_on='SMILES0',
            how='left'
        )

        # Compute properties from SPT parameters
        vp_spt = []
        rho_spt = []

        for idx, row in results_df.iterrows():
            if pd.notna(row.get('m_spt')):
                props = compute_properties(
                    row['m_spt'], row['sigma_spt'], row['epsilon_k_spt'], T_REF
                )

                from rdkit import Chem
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol is not None:
                    mw = Chem.Descriptors.MolWt(mol) / 1000
                    rho_kg_m3 = props['liquid_density_mol_m3'] * mw
                else:
                    rho_kg_m3 = np.nan

                vp_spt.append(props['vapor_pressure_Pa'])
                rho_spt.append(rho_kg_m3)
            else:
                vp_spt.append(np.nan)
                rho_spt.append(np.nan)

        results_df['vp_spt_Pa'] = vp_spt
        results_df['rho_spt_kg_m3'] = rho_spt

        n_spt_overlap = results_df['m_spt'].notna().sum()
        print(f"  {n_spt_overlap} molecules found in SPT-PCSAFT dataset")

    # Load Esper reference parameters if available
    esper_csv = DATA_DIR / "esper_pcsaft.csv"
    if esper_csv.exists():
        print("\nLoading Esper reference parameters...")
        esper_df = pd.read_csv(esper_csv)

        # Rename Esper columns to avoid conflicts
        esper_df = esper_df.rename(columns={
            'm': 'm_esper',
            'sigma': 'sigma_esper',
            'epsilon_k': 'epsilon_k_esper'
        })

        results_df = results_df.merge(
            esper_df[['smiles', 'm_esper', 'sigma_esper', 'epsilon_k_esper']],
            on='smiles',
            how='left'
        )

        # Compute properties from Esper parameters
        vp_esper = []
        rho_esper = []

        for idx, row in results_df.iterrows():
            if pd.notna(row.get('m_esper')):
                props = compute_properties(
                    row['m_esper'], row['sigma_esper'], row['epsilon_k_esper'], T_REF
                )

                from rdkit import Chem
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol is not None:
                    mw = Chem.Descriptors.MolWt(mol) / 1000
                    rho_kg_m3 = props['liquid_density_mol_m3'] * mw
                else:
                    rho_kg_m3 = np.nan

                vp_esper.append(props['vapor_pressure_Pa'])
                rho_esper.append(rho_kg_m3)
            else:
                vp_esper.append(np.nan)
                rho_esper.append(np.nan)

        results_df['vp_esper_Pa'] = vp_esper
        results_df['rho_esper_kg_m3'] = rho_esper

        n_esper_overlap = results_df['m_esper'].notna().sum()
        print(f"  {n_esper_overlap} molecules found in Esper dataset")

    # Compute validation metrics
    print("\n" + "="*70)
    print("VALIDATION METRICS")
    print("="*70)

    print("\nRF Predicted Parameters:")
    mare_vp_rf = compute_mare(results_df['vp_rf_Pa'], results_df['vp_exp_Pa'], "VP")
    mare_rho_rf = compute_mare(
        results_df['rho_rf_kg_m3'], results_df['rho_exp_kg_m3'], "density"
    )
    within20_vp_rf = within_tolerance(
        results_df['vp_rf_Pa'], results_df['vp_exp_Pa']
    ) * 100
    within20_rho_rf = within_tolerance(
        results_df['rho_rf_kg_m3'], results_df['rho_exp_kg_m3']
    ) * 100

    print(f"  MARE(VP):      {mare_vp_rf:.1f}%")
    print(f"  MARE(density): {mare_rho_rf:.1f}%")
    print(f"  VP within ±20%:      {within20_vp_rf:.1f}%")
    print(f"  Density within ±20%: {within20_rho_rf:.1f}%")

    if 'vp_spt_Pa' in results_df.columns:
        print("\nSPT-PCSAFT Parameters:")
        mare_vp_spt = compute_mare(
            results_df['vp_spt_Pa'], results_df['vp_exp_Pa'], "VP"
        )
        mare_rho_spt = compute_mare(
            results_df['rho_spt_kg_m3'], results_df['rho_exp_kg_m3'], "density"
        )
        within20_vp_spt = within_tolerance(
            results_df['vp_spt_Pa'], results_df['vp_exp_Pa']
        ) * 100
        within20_rho_spt = within_tolerance(
            results_df['rho_spt_kg_m3'], results_df['rho_exp_kg_m3']
        ) * 100

        print(f"  MARE(VP):      {mare_vp_spt:.1f}%")
        print(f"  MARE(density): {mare_rho_spt:.1f}%")
        print(f"  VP within ±20%:      {within20_vp_spt:.1f}%")
        print(f"  Density within ±20%: {within20_rho_spt:.1f}%")

    if 'vp_esper_Pa' in results_df.columns:
        print("\nEsper Reference Parameters:")
        mare_vp_esper = compute_mare(
            results_df['vp_esper_Pa'], results_df['vp_exp_Pa'], "VP"
        )
        mare_rho_esper = compute_mare(
            results_df['rho_esper_kg_m3'], results_df['rho_exp_kg_m3'], "density"
        )
        within20_vp_esper = within_tolerance(
            results_df['vp_esper_Pa'], results_df['vp_exp_Pa']
        ) * 100
        within20_rho_esper = within_tolerance(
            results_df['rho_esper_kg_m3'], results_df['rho_exp_kg_m3']
        ) * 100

        print(f"  MARE(VP):      {mare_vp_esper:.1f}%")
        print(f"  MARE(density): {mare_rho_esper:.1f}%")
        print(f"  VP within ±20%:      {within20_vp_esper:.1f}%")
        print(f"  Density within ±20%: {within20_rho_esper:.1f}%")

    # Save results
    print(f"\nSaving results to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    # Generate figures
    fig_dir = Path("figures/17_nist_validation")
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures in {fig_dir}...")

    plot_parity(
        results_df['vp_exp_Pa'], results_df['vp_rf_Pa'],
        results_df['rho_exp_kg_m3'], results_df['rho_rf_kg_m3'],
        fig_dir, label="RF"
    )

    if 'vp_spt_Pa' in results_df.columns:
        plot_parity(
            results_df['vp_exp_Pa'], results_df['vp_spt_Pa'],
            results_df['rho_exp_kg_m3'], results_df['rho_spt_kg_m3'],
            fig_dir, label="SPT"
        )

    if 'vp_esper_Pa' in results_df.columns:
        plot_parity(
            results_df['vp_exp_Pa'], results_df['vp_esper_Pa'],
            results_df['rho_exp_kg_m3'], results_df['rho_esper_kg_m3'],
            fig_dir, label="Esper"
        )

    if 'vp_spt_Pa' in results_df.columns or 'vp_esper_Pa' in results_df.columns:
        plot_method_comparison(results_df, fig_dir)

    print("\n" + "="*70)
    print("NIST experimental validation complete!")
    print(f"  Results: {args.output}")
    print(f"  Figures: {fig_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
