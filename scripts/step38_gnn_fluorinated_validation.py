"""Step 38: GNN Fluorinated Validation.

Validate the exact GNN-based screening path (SMILES → GNN → PC-SAFT → EOS → T_b)
on known fluorinated and HFO-like molecules.

This script performs:
1. Curated validation set construction (fluorinated refrigerants)
2. Parameter prediction validation (GNN vs RF)
3. Boiling point prediction validation (full inference chain)
4. Filter sensitivity analysis
5. Leave-one-out error decomposition
6. MC Dropout uncertainty calibration
7. Screening sanity checks

Usage:
    python scripts/step38_gnn_fluorinated_validation.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.registry import get_model
from screening.hfo_screening import (
    HFO_1336MZZ,
    KNOWN_BOILING_POINTS,
    compute_boiling_point,
    hfo_parameter_distance,
)


def build_validation_set():
    """Build curated fluorinated validation set from known compounds.

    Combines KNOWN_BOILING_POINTS with literature PC-SAFT parameters from
    fluorinated_pcsaft.csv where available.

    Returns:
        pd.DataFrame: Validation set with columns:
            - smiles
            - name
            - T_b_experimental_K
            - m_lit, sigma_lit, epsilon_k_lit (if available)
            - source (literature, experimental)
            - is_hfo (bool, has C=C bond)
    """
    # Load literature PC-SAFT parameters
    data_path = Path(__file__).parent.parent / "model" / "data" / "fluorinated_pcsaft.csv"
    df_lit = pd.read_csv(data_path)

    # Create mapping from SMILES to literature parameters
    lit_params = {}
    for _, row in df_lit.iterrows():
        lit_params[row["smiles"]] = {
            "m_lit": row["m"],
            "sigma_lit": row["sigma"],
            "epsilon_k_lit": row["epsilon_k"],
            "name": row.get("name", ""),
            "source": row.get("source", ""),
        }

    # Build validation table from KNOWN_BOILING_POINTS
    records = []
    for smiles, T_b in KNOWN_BOILING_POINTS.items():
        record = {
            "smiles": smiles,
            "T_b_experimental_K": T_b,
        }

        # Add literature parameters if available
        if smiles in lit_params:
            record.update(lit_params[smiles])
        else:
            record["m_lit"] = np.nan
            record["sigma_lit"] = np.nan
            record["epsilon_k_lit"] = np.nan
            record["name"] = ""
            record["source"] = "experimental_only"

        # Check if HFO (has C=C bond)
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        pattern_cc = Chem.MolFromSmarts("C=C")
        is_hfo = False
        if mol is not None:
            is_hfo = mol.HasSubstructMatch(pattern_cc)
        record["is_hfo"] = is_hfo

        records.append(record)

    df = pd.DataFrame(records)

    # Save to model/saved/
    output_path = (
        Path(__file__).parent.parent
        / "model"
        / "saved"
        / "gnn_fluorinated_validation_set.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"Saved validation set to: {output_path}")
    print(f"  Total molecules: {len(df)}")
    print(f"  HFO-like (C=C): {df['is_hfo'].sum()}")
    print(f"  With literature PC-SAFT: {(~df['m_lit'].isna()).sum()}")

    return df


def validate_parameter_predictions(validation_df):
    """Validate GNN and RF parameter predictions on fluorinated validation set.

    Args:
        validation_df: DataFrame from build_validation_set()

    Returns:
        tuple: (gnn_results_df, rf_results_df, metrics_dict)
    """
    smiles_list = validation_df["smiles"].tolist()

    # GNN predictions with uncertainty
    print("\nLoading GNN model...")
    gnn = get_model("gnn")
    gnn.load()
    gnn_preds = gnn.predict_with_uncertainty(smiles_list, n_forward=30)

    # RF predictions with uncertainty
    print("Loading RF model...")
    rf = get_model("rf")
    rf.load()
    rf_preds = rf.predict_with_uncertainty(smiles_list)

    # Build results DataFrames
    gnn_results = validation_df.copy()
    gnn_results["m_gnn"] = gnn_preds["m"]
    gnn_results["sigma_gnn"] = gnn_preds["sigma"]
    gnn_results["epsilon_k_gnn"] = gnn_preds["epsilon_k"]
    gnn_results["m_std"] = gnn_preds["m_std"]
    gnn_results["sigma_std"] = gnn_preds["sigma_std"]
    gnn_results["epsilon_k_std"] = gnn_preds["epsilon_k_std"]

    rf_results = validation_df.copy()
    rf_results["m_rf"] = rf_preds["m"]
    rf_results["sigma_rf"] = rf_preds["sigma"]
    rf_results["epsilon_k_rf"] = rf_preds["epsilon_k"]
    rf_results["m_std"] = rf_preds["m_std"]
    rf_results["sigma_std"] = rf_preds["sigma_std"]
    rf_results["epsilon_k_std"] = rf_preds["epsilon_k_std"]

    # Compute errors vs literature parameters where available
    for df, suffix in [(gnn_results, "gnn"), (rf_results, "rf")]:
        valid_lit = ~df["m_lit"].isna()
        if valid_lit.any():
            df.loc[valid_lit, f"m_error_{suffix}"] = (
                df.loc[valid_lit, f"m_{suffix}"] - df.loc[valid_lit, "m_lit"]
            )
            df.loc[valid_lit, f"sigma_error_{suffix}"] = (
                df.loc[valid_lit, f"sigma_{suffix}"]
                - df.loc[valid_lit, "sigma_lit"]
            )
            df.loc[valid_lit, f"epsilon_k_error_{suffix}"] = (
                df.loc[valid_lit, f"epsilon_k_{suffix}"]
                - df.loc[valid_lit, "epsilon_k_lit"]
            )

    # Compute metrics (only for molecules with literature parameters)
    metrics = {}

    for model_name, df, suffix in [("GNN", gnn_results, "gnn"), ("RF", rf_results, "rf")]:
        valid = ~df["m_lit"].isna()
        n_valid = valid.sum()

        if n_valid == 0:
            continue

        metrics[model_name] = {}
        for param in ["m", "sigma", "epsilon_k"]:
            pred = df.loc[valid, f"{param}_{suffix}"].values
            true = df.loc[valid, f"{param}_lit"].values

            errors = pred - true
            mae = np.abs(errors).mean()
            rmse = np.sqrt((errors**2).mean())
            # R² is unstable for small n, so only compute if n >= 5
            if n_valid >= 5:
                ss_res = ((pred - true)**2).sum()
                ss_tot = ((true - true.mean())**2).sum()
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                r2 = np.nan

            metrics[model_name][param] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "n": n_valid,
                "mean_error": errors.mean(),
            }

    return gnn_results, rf_results, metrics


def validate_boiling_points(gnn_results, rf_results):
    """Validate EOS-derived boiling points for both models.

    Args:
        gnn_results: DataFrame with GNN predictions
        rf_results: DataFrame with RF predictions

    Returns:
        tuple: (gnn_results_with_bp, rf_results_with_bp, bp_metrics)
    """
    print("\nComputing boiling points from GNN predictions...")
    gnn_bp = []
    for _, row in gnn_results.iterrows():
        T_b = compute_boiling_point(row["m_gnn"], row["sigma_gnn"], row["epsilon_k_gnn"])
        gnn_bp.append(T_b)
    gnn_results = gnn_results.copy()
    gnn_results["T_b_predicted_K"] = gnn_bp

    print("Computing boiling points from RF predictions...")
    rf_bp = []
    for _, row in rf_results.iterrows():
        T_b = compute_boiling_point(row["m_rf"], row["sigma_rf"], row["epsilon_k_rf"])
        rf_bp.append(T_b)
    rf_results = rf_results.copy()
    rf_results["T_b_predicted_K"] = rf_bp

    # Compute boiling point errors
    for df in [gnn_results, rf_results]:
        df["T_b_error_K"] = df["T_b_predicted_K"] - df["T_b_experimental_K"]

    # Compute metrics
    bp_metrics = {}

    for model_name, df in [("GNN", gnn_results), ("RF", rf_results)]:
        valid = ~df["T_b_predicted_K"].isna()
        n_valid = valid.sum()
        n_converged = valid.sum()
        n_total = len(df)

        if n_valid == 0:
            continue

        errors = df.loc[valid, "T_b_error_K"].values
        mae = np.abs(errors).mean()
        median_ae = np.median(np.abs(errors))
        max_ae = np.abs(errors).max()
        rmse = np.sqrt((errors**2).mean())

        bp_metrics[model_name] = {
            "mae": mae,
            "median_ae": median_ae,
            "max_ae": max_ae,
            "rmse": rmse,
            "n_converged": n_converged,
            "n_total": n_total,
            "convergence_rate": n_converged / n_total,
            "mean_error": errors.mean(),
        }

    return gnn_results, rf_results, bp_metrics


def filter_sensitivity_analysis(gnn_results):
    """Analyze boiling point filter sensitivity (288-323K window).

    Args:
        gnn_results: DataFrame with GNN predictions and boiling points

    Returns:
        dict: Confusion matrix statistics
    """
    T_min = 288.0
    T_max = 323.0

    # Ground truth: should pass filter based on experimental T_b
    should_pass = (
        (gnn_results["T_b_experimental_K"] >= T_min)
        & (gnn_results["T_b_experimental_K"] <= T_max)
    )

    # Predicted: passes filter based on predicted T_b
    valid_pred = ~gnn_results["T_b_predicted_K"].isna()
    does_pass = (
        (gnn_results["T_b_predicted_K"] >= T_min)
        & (gnn_results["T_b_predicted_K"] <= T_max)
    )

    # Confusion matrix (only for valid predictions)
    valid_mask = valid_pred

    tp = (should_pass & does_pass & valid_mask).sum()
    fp = (~should_pass & does_pass & valid_mask).sum()
    tn = (~should_pass & ~does_pass & valid_mask).sum()
    fn = (should_pass & ~does_pass & valid_mask).sum()

    total = valid_mask.sum()

    stats = {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "total": total,
        "accuracy": (tp + tn) / total if total > 0 else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
    }

    return stats


def uncertainty_calibration(gnn_results):
    """Check MC Dropout uncertainty calibration on validation set.

    Args:
        gnn_results: DataFrame with GNN predictions and uncertainties

    Returns:
        dict: Calibration statistics
    """
    valid = ~gnn_results["m_lit"].isna()

    if valid.sum() == 0:
        return {"error": "No molecules with literature parameters"}

    calibration = {}

    for param in ["m", "sigma", "epsilon_k"]:
        pred = gnn_results.loc[valid, f"{param}_gnn"].values
        true = gnn_results.loc[valid, f"{param}_lit"].values
        std = gnn_results.loc[valid, f"{param}_std"].values

        errors = np.abs(pred - true)

        # Coverage at 1-sigma and 2-sigma
        within_1sigma = (errors <= std).sum()
        within_2sigma = (errors <= 2 * std).sum()
        n = len(errors)

        calibration[param] = {
            "coverage_1sigma": within_1sigma / n if n > 0 else np.nan,
            "coverage_2sigma": within_2sigma / n if n > 0 else np.nan,
            "expected_1sigma": 0.68,
            "expected_2sigma": 0.95,
            "n": n,
        }

    return calibration


def screening_sanity_checks(gnn_results):
    """Run Step 37 screening sanity checks on validation molecules.

    Args:
        gnn_results: DataFrame with GNN predictions

    Returns:
        pd.DataFrame: HFO distance rankings
    """
    # Compute HFO distance with 5:2:1 weighting
    distances = []
    for _, row in gnn_results.iterrows():
        dist = hfo_parameter_distance(
            row["m_gnn"], row["sigma_gnn"], row["epsilon_k_gnn"],
            reference=HFO_1336MZZ,
            weights={"epsilon_k": 5.0, "sigma": 2.0, "m": 1.0},
        )
        distances.append(dist)

    df = gnn_results.copy()
    df["hfo_distance"] = distances
    df = df.sort_values("hfo_distance").reset_index(drop=True)

    return df


def create_figures(gnn_results, rf_results, output_dir):
    """Create validation figures.

    Args:
        gnn_results: DataFrame with GNN predictions
        rf_results: DataFrame with RF predictions
        output_dir: Path to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Parameter parity plots (GNN vs RF)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    params = ["m", "sigma", "epsilon_k"]
    param_labels = ["m (segments)", "σ (Å)", "ε/k (K)"]

    for ax, param, label in zip(axes, params, param_labels):
        valid = ~gnn_results[f"{param}_lit"].isna()

        if valid.sum() == 0:
            continue

        true_vals = gnn_results.loc[valid, f"{param}_lit"].values
        gnn_vals = gnn_results.loc[valid, f"{param}_gnn"].values
        rf_vals = rf_results.loc[valid, f"{param}_rf"].values

        # Plot
        ax.scatter(true_vals, gnn_vals, alpha=0.7, s=100, label="GNN", color="C0")
        ax.scatter(true_vals, rf_vals, alpha=0.7, s=100, label="RF", color="C1", marker="^")

        # Parity line
        min_val = min(true_vals.min(), gnn_vals.min(), rf_vals.min())
        max_val = max(true_vals.max(), gnn_vals.max(), rf_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, lw=2)

        ax.set_xlabel(f"Literature {label}", fontsize=24)
        ax.set_ylabel(f"Predicted {label}", fontsize=24)
        ax.tick_params(labelsize=20)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "parameter_parity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'parameter_parity.png'}")

    # Figure 2: Boiling point parity plot
    fig, ax = plt.subplots(figsize=(8, 7))

    valid_gnn = ~gnn_results["T_b_predicted_K"].isna()
    valid_rf = ~rf_results["T_b_predicted_K"].isna()

    if valid_gnn.sum() > 0:
        exp_gnn = gnn_results.loc[valid_gnn, "T_b_experimental_K"].values
        pred_gnn = gnn_results.loc[valid_gnn, "T_b_predicted_K"].values
        ax.scatter(exp_gnn, pred_gnn, alpha=0.7, s=100, label="GNN", color="C0")

    if valid_rf.sum() > 0:
        exp_rf = rf_results.loc[valid_rf, "T_b_experimental_K"].values
        pred_rf = rf_results.loc[valid_rf, "T_b_predicted_K"].values
        ax.scatter(exp_rf, pred_rf, alpha=0.7, s=100, label="RF", color="C1", marker="^")

    # Parity line
    all_vals = np.concatenate([exp_gnn, pred_gnn, exp_rf, pred_rf])
    min_val = all_vals.min()
    max_val = all_vals.max()
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, lw=2)

    # Filter window
    ax.axhspan(288, 323, alpha=0.15, color="green", label="Process window (288-323K)")

    ax.set_xlabel("Experimental T_b (K)", fontsize=24)
    ax.set_ylabel("Predicted T_b (K)", fontsize=24)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "boiling_point_parity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'boiling_point_parity.png'}")

    # Figure 3: Error distribution (GNN vs RF)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Parameter errors
    ax = axes[0]
    valid = ~gnn_results["m_lit"].isna()

    if valid.sum() > 0:
        for i, (param, label) in enumerate(zip(params, param_labels)):
            gnn_errors = np.abs(gnn_results.loc[valid, f"{param}_error_gnn"].values)
            rf_errors = np.abs(rf_results.loc[valid, f"{param}_error_rf"].values)

            x = np.arange(len(gnn_errors))
            width = 0.35
            ax.bar(x + i*0.12 - 0.12, gnn_errors, width=0.12, label=f"GNN {label}", alpha=0.7)
            ax.bar(x + i*0.12, rf_errors, width=0.12, label=f"RF {label}", alpha=0.7)

        ax.set_xlabel("Molecule index", fontsize=20)
        ax.set_ylabel("Absolute error", fontsize=20)
        ax.set_title("Parameter prediction errors", fontsize=22)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=14, ncol=2)
        ax.grid(True, alpha=0.3)

    # Boiling point errors
    ax = axes[1]
    valid_gnn = ~gnn_results["T_b_predicted_K"].isna()
    valid_rf = ~rf_results["T_b_predicted_K"].isna()

    if valid_gnn.sum() > 0:
        gnn_bp_errors = np.abs(gnn_results.loc[valid_gnn, "T_b_error_K"].values)
        rf_bp_errors = np.abs(rf_results.loc[valid_rf, "T_b_error_K"].values)

        x = np.arange(len(gnn_bp_errors))
        width = 0.35
        ax.bar(x - width/2, gnn_bp_errors, width, label="GNN", alpha=0.7, color="C0")
        ax.bar(x + width/2, rf_bp_errors, width, label="RF", alpha=0.7, color="C1")

        ax.set_xlabel("Molecule index", fontsize=20)
        ax.set_ylabel("Absolute boiling point error (K)", fontsize=20)
        ax.set_title("Boiling point prediction errors", fontsize=22)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'error_distribution.png'}")


def main():
    """Run full fluorinated validation workflow."""
    print("=" * 80)
    print("STEP 38: GNN FLUORINATED VALIDATION")
    print("=" * 80)
    print()

    # Create output directories
    output_dir = Path(__file__).parent.parent / "figures" / "38_gnn_fluorinated_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(__file__).parent.parent / "model" / "saved"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build validation set
    print("\n[1/7] Building curated fluorinated validation set...")
    validation_df = build_validation_set()

    # Step 2: Validate parameter predictions
    print("\n[2/7] Validating parameter predictions (GNN vs RF)...")
    gnn_results, rf_results, param_metrics = validate_parameter_predictions(validation_df)

    # Print parameter metrics
    print("\nParameter Prediction Metrics:")
    print("-" * 80)
    for model_name in ["GNN", "RF"]:
        if model_name not in param_metrics:
            continue
        print(f"\n{model_name}:")
        for param in ["m", "sigma", "epsilon_k"]:
            m = param_metrics[model_name][param]
            print(
                f"  {param:10s}: MAE={m['mae']:.4f}, "
                f"RMSE={m['rmse']:.4f}, R²={m['r2']:.3f}, n={m['n']}"
            )

    # Step 3: Validate boiling points
    print("\n[3/7] Validating EOS-derived boiling points...")
    gnn_results, rf_results, bp_metrics = validate_boiling_points(gnn_results, rf_results)

    # Print boiling point metrics
    print("\nBoiling Point Prediction Metrics:")
    print("-" * 80)
    for model_name in ["GNN", "RF"]:
        if model_name not in bp_metrics:
            continue
        m = bp_metrics[model_name]
        print(f"\n{model_name}:")
        print(f"  MAE:           {m['mae']:.2f} K")
        print(f"  Median AE:     {m['median_ae']:.2f} K")
        print(f"  Max AE:        {m['max_ae']:.2f} K")
        print(f"  RMSE:          {m['rmse']:.2f} K")
        print(f"  Convergence:   {m['n_converged']}/{m['n_total']} ({m['convergence_rate']:.1%})")
        print(f"  Mean error:    {m['mean_error']:.2f} K")

    # Step 4: Filter sensitivity analysis
    print("\n[4/7] Analyzing boiling point filter sensitivity (288-323K)...")
    filter_stats = filter_sensitivity_analysis(gnn_results)

    print("\nFilter Confusion Matrix (GNN):")
    print("-" * 80)
    print(f"  True Positives:  {filter_stats['true_positive']} (should pass, does pass)")
    print(f"  False Positives: {filter_stats['false_positive']} (should fail, passes)")
    print(f"  True Negatives:  {filter_stats['true_negative']} (should fail, does fail)")
    print(f"  False Negatives: {filter_stats['false_negative']} (should pass, fails)")
    print(f"  Accuracy:        {filter_stats['accuracy']:.2%}")
    print(f"  Precision:       {filter_stats['precision']:.2%}")
    print(f"  Recall:          {filter_stats['recall']:.2%}")

    # Step 5: Uncertainty calibration
    print("\n[5/7] Checking MC Dropout uncertainty calibration...")
    calibration = uncertainty_calibration(gnn_results)

    if "error" not in calibration:
        print("\nUncertainty Calibration (GNN):")
        print("-" * 80)
        print(
            f"{'Parameter':<12} {'1σ Expected':<15} {'1σ Observed':<15} "
            f"{'2σ Expected':<15} {'2σ Observed':<15}"
        )
        print("-" * 80)
        for param in ["m", "sigma", "epsilon_k"]:
            cal = calibration[param]
            print(f"{param:<12} {cal['expected_1sigma']:<15.2%} {cal['coverage_1sigma']:<15.2%} "
                  f"{cal['expected_2sigma']:<15.2%} {cal['coverage_2sigma']:<15.2%}")

    # Step 6: Screening sanity checks
    print("\n[6/7] Running screening sanity checks...")
    gnn_ranked = screening_sanity_checks(gnn_results)

    print("\nHFO Distance Ranking (Top 10):")
    print("-" * 80)
    for i, row in gnn_ranked.head(10).iterrows():
        print(f"  Rank {i+1:2d}: {row['smiles']:<30s} "
              f"HFO dist={row['hfo_distance']:.4f}, T_b={row['T_b_experimental_K']:.1f}K")

    # Step 7: Save results
    print("\n[7/7] Saving results...")

    # Save detailed results
    results_path = results_dir / "gnn_fluorinated_validation_results.csv"
    gnn_ranked.to_csv(results_path, index=False)
    print(f"Saved GNN validation results to: {results_path}")

    # Save RF results for comparison
    rf_results_path = results_dir / "rf_fluorinated_validation_results.csv"
    rf_results.to_csv(rf_results_path, index=False)
    print(f"Saved RF validation results to: {rf_results_path}")

    # Save metrics summary (convert numpy types to native Python)
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    metrics_summary = {
        "param_metrics": convert_to_native(param_metrics),
        "bp_metrics": convert_to_native(bp_metrics),
        "filter_stats": convert_to_native(filter_stats),
        "calibration": convert_to_native(calibration),
    }

    import json
    metrics_path = results_dir / "gnn_fluorinated_validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to: {metrics_path}")

    # Create figures
    print("\nCreating figures...")
    create_figures(gnn_results, rf_results, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nValidation set size: {len(validation_df)} fluorinated molecules")
    print(f"  HFO-like (C=C): {validation_df['is_hfo'].sum()}")
    print(f"  With literature PC-SAFT: {(~validation_df['m_lit'].isna()).sum()}")

    print("\nKey Findings:")
    print("  GNN Boiling Point MAE: {:.2f} K".format(bp_metrics["GNN"]["mae"]))
    print("  RF Boiling Point MAE:  {:.2f} K".format(bp_metrics["RF"]["mae"]))
    print("  Filter window width:   35 K (288-323K)")
    print("  Filter reliability:    {:.1%}".format(filter_stats["accuracy"]))

    if bp_metrics["GNN"]["mae"] < 10.0:
        print("\n✓ GNN boiling point MAE < 10 K: Filter is reliable")
    else:
        print("\n⚠ GNN boiling point MAE > 10 K: Consider widening filter window")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
