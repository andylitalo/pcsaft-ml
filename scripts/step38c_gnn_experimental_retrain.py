"""Step 38c: GNN Retrain on Experimental (Esper-only) Data.

The unified GNN (Step 31) catastrophically fails on fluorinated compounds
(133 K boiling point MAE vs RF's 8 K) because SPT-PCSAFT data dominates
the training corpus and introduces systematic bias.

This script retrains the GNN on Esper-only data (~1,801 molecules) and
validates whether the GNN architecture itself can match RF performance
when given clean, experimentally fitted parameters.

Stages:
    1. Back up existing unified GNN
    2. Retrain GNN on Esper-only data
    3. Evaluate on Esper test set
    4. Validate on 15-compound fluorinated set (boiling point MAE)
    5. MC Dropout uncertainty calibration
    6. Run HFO screening with retrained GNN
    7. Generate figures
    8. Save results and report data

Usage:
    python scripts/step38c_gnn_experimental_retrain.py
"""

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.data.load import TARGETS  # noqa: E402
from model.gnn.train_gnn import train_gnn  # noqa: E402
from model.registry import get_model  # noqa: E402
from screening.hfo_screening import (  # noqa: E402
    HFO_1336MZZ,
    KNOWN_BOILING_POINTS,
    apply_hfo_filters,
    batch_boiling_points,
    compute_boiling_point,
    hfo_parameter_distance,
)

SAVED_DIR = PROJECT_ROOT / "model" / "saved"
FIGURES_DIR = PROJECT_ROOT / "figures" / "38c_gnn_experimental_retrain"
RESULTS_DIR = PROJECT_ROOT / "screening" / "results"


def backup_unified_gnn():
    """Back up existing unified GNN model and metrics."""
    print("\n[1/8] Backing up existing unified GNN...")

    gnn_path = SAVED_DIR / "gnn_pcsaft.pt"
    metrics_path = SAVED_DIR / "gnn_metrics.json"
    backup_gnn = SAVED_DIR / "gnn_pcsaft_unified.pt"
    backup_metrics = SAVED_DIR / "gnn_metrics_unified.json"

    if gnn_path.exists() and not backup_gnn.exists():
        shutil.copy2(gnn_path, backup_gnn)
        print(f"  Backed up {gnn_path.name} -> {backup_gnn.name}")
    elif backup_gnn.exists():
        print(f"  Backup {backup_gnn.name} already exists, skipping")
    else:
        print("  WARNING: No existing GNN model found to back up")

    if metrics_path.exists() and not backup_metrics.exists():
        shutil.copy2(metrics_path, backup_metrics)
        print(f"  Backed up {metrics_path.name} -> {backup_metrics.name}")
    elif backup_metrics.exists():
        print(f"  Backup {backup_metrics.name} already exists, skipping")

    # Also back up the test_set.csv since retraining on Esper will produce
    # a different test split
    test_set_path = SAVED_DIR / "test_set.csv"
    backup_test = SAVED_DIR / "test_set_unified.csv"
    if test_set_path.exists() and not backup_test.exists():
        shutil.copy2(test_set_path, backup_test)
        print(f"  Backed up {test_set_path.name} -> {backup_test.name}")


def retrain_gnn_esper():
    """Retrain GNN on Esper-only data with overfitting-aware hyperparameters.

    Returns:
        dict: Metrics on the Esper test set
        list: Per-epoch training history (train_loss, val_loss)
    """
    print("\n[2/8] Retraining GNN on Esper-only data...")
    print("  Dataset: Esper (~1,801 molecules)")
    print("  Note: 12x smaller than unified corpus; using regularization")

    # Delete the test_set.csv so train_gnn creates a fresh Esper-only split
    # (the existing one has 'source' column from unified corpus)
    test_set_path = SAVED_DIR / "test_set.csv"
    if test_set_path.exists():
        existing_test = pd.read_csv(test_set_path)
        if "source" in existing_test.columns:
            test_set_path.unlink()
            print("  Removed unified test_set.csv to create Esper-only split")

    # Train with adjusted hyperparameters for smaller dataset:
    # - Smaller hidden dim (128 vs 256) to reduce overfitting
    # - Fewer layers (3 vs 4)
    # - Higher dropout (0.2 vs 0.1)
    # - More patience (25 vs 15) to let the smaller dataset converge
    # - More epochs (200) since convergence is slower on small data
    start_time = time.time()
    metrics = train_gnn(
        source="esper",
        epochs=200,
        lr=1e-3,
        batch_size=32,       # Smaller batch for 1,801 molecules
        hidden_dim=128,      # Smaller network for small data
        num_layers=3,        # Fewer layers to reduce overfitting
        dropout=0.2,         # Higher dropout for regularization
        patience=25,         # More patience
        weight_decay=1e-4,   # Stronger weight decay
    )
    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")

    return metrics


def load_training_history():
    """Load training history from the saved checkpoint (losses per epoch).

    The train_gnn function prints epoch info but doesn't store history.
    We reconstruct what we can from the final metrics.
    """
    # train_gnn doesn't save per-epoch history, so we return the final metrics
    metrics_path = SAVED_DIR / "gnn_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


def validate_fluorinated(model_name="gnn"):
    """Validate retrained GNN on the 15-compound fluorinated set.

    Args:
        model_name: Registry model name to validate

    Returns:
        tuple: (results_df, bp_metrics_dict, param_metrics_dict)
    """
    print(f"\n[3/8] Validating {model_name.upper()} on fluorinated set...")

    # Load validation set
    val_path = SAVED_DIR / "gnn_fluorinated_validation_set.csv"
    if not val_path.exists():
        print("  Validation set not found, building...")
        val_df = build_validation_set_inline()
    else:
        val_df = pd.read_csv(val_path)
    print(f"  Validation set: {len(val_df)} molecules")

    smiles_list = val_df["smiles"].tolist()

    # Predict with the retrained model
    model = get_model(model_name)
    model.load()
    preds_unc = model.predict_with_uncertainty(smiles_list, n_forward=30)

    results = val_df.copy()
    for param in TARGETS:
        results[f"{param}_pred"] = preds_unc[param]
        results[f"{param}_std"] = preds_unc[f"{param}_std"]

    # Compute boiling points
    bp_predicted = []
    for _, row in results.iterrows():
        T_b = compute_boiling_point(
            row["m_pred"], row["sigma_pred"], row["epsilon_k_pred"]
        )
        bp_predicted.append(T_b)
    results["T_b_predicted_K"] = bp_predicted
    results["T_b_error_K"] = results["T_b_predicted_K"] - results["T_b_experimental_K"]

    # Boiling point metrics
    valid_bp = ~results["T_b_predicted_K"].isna()
    bp_errors = results.loc[valid_bp, "T_b_error_K"].values
    bp_metrics = {
        "mae": float(np.abs(bp_errors).mean()) if len(bp_errors) > 0 else np.nan,
        "median_ae": float(np.median(np.abs(bp_errors))) if len(bp_errors) > 0 else np.nan,
        "max_ae": float(np.abs(bp_errors).max()) if len(bp_errors) > 0 else np.nan,
        "rmse": float(np.sqrt((bp_errors**2).mean())) if len(bp_errors) > 0 else np.nan,
        "n_converged": int(valid_bp.sum()),
        "n_total": len(results),
        "convergence_rate": float(valid_bp.sum() / len(results)),
        "mean_error": float(bp_errors.mean()) if len(bp_errors) > 0 else np.nan,
    }

    # Parameter metrics (vs literature)
    param_metrics = {}
    valid_lit = ~results["m_lit"].isna()
    n_lit = valid_lit.sum()
    if n_lit > 0:
        for param in TARGETS:
            pred = results.loc[valid_lit, f"{param}_pred"].values
            true = results.loc[valid_lit, f"{param}_lit"].values
            errors = pred - true
            mae = float(np.abs(errors).mean())
            rmse = float(np.sqrt((errors**2).mean()))
            if n_lit >= 5:
                ss_res = float(((pred - true)**2).sum())
                ss_tot = float(((true - true.mean())**2).sum())
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                r2 = np.nan
            param_metrics[param] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "n": int(n_lit),
                "mean_error": float(errors.mean()),
            }

    print(f"\n  Boiling Point MAE: {bp_metrics['mae']:.2f} K")
    print(f"  Convergence rate: {bp_metrics['n_converged']}/{bp_metrics['n_total']}")

    if param_metrics:
        print("\n  Parameter Metrics (vs literature):")
        for param in TARGETS:
            m = param_metrics[param]
            print(f"    {param:12s}: MAE={m['mae']:.4f}, R2={m['r2']:.3f}")

    return results, bp_metrics, param_metrics


def build_validation_set_inline():
    """Build validation set from known boiling points (inline version)."""
    from rdkit import Chem

    data_path = PROJECT_ROOT / "model" / "data" / "fluorinated_pcsaft.csv"
    df_lit = pd.read_csv(data_path)

    lit_params = {}
    for _, row in df_lit.iterrows():
        lit_params[row["smiles"]] = {
            "m_lit": row["m"],
            "sigma_lit": row["sigma"],
            "epsilon_k_lit": row["epsilon_k"],
            "name": row.get("name", ""),
            "source": row.get("source", ""),
        }

    records = []
    for smiles, T_b in KNOWN_BOILING_POINTS.items():
        record = {"smiles": smiles, "T_b_experimental_K": T_b}
        if smiles in lit_params:
            record.update(lit_params[smiles])
        else:
            record["m_lit"] = np.nan
            record["sigma_lit"] = np.nan
            record["epsilon_k_lit"] = np.nan
            record["name"] = ""
            record["source"] = "experimental_only"

        mol = Chem.MolFromSmiles(smiles)
        pattern_cc = Chem.MolFromSmarts("C=C")
        record["is_hfo"] = bool(mol and mol.HasSubstructMatch(pattern_cc))
        records.append(record)

    return pd.DataFrame(records)


def uncertainty_calibration(results_df):
    """Compute MC Dropout uncertainty calibration on the fluorinated set.

    Args:
        results_df: DataFrame with *_pred and *_std columns

    Returns:
        dict: Calibration statistics per parameter
    """
    print("\n[4/8] Checking MC Dropout uncertainty calibration...")

    valid = ~results_df["m_lit"].isna()
    if valid.sum() == 0:
        print("  No molecules with literature parameters")
        return {"error": "No molecules with literature parameters"}

    calibration = {}
    for param in TARGETS:
        pred = results_df.loc[valid, f"{param}_pred"].values
        true = results_df.loc[valid, f"{param}_lit"].values
        std = results_df.loc[valid, f"{param}_std"].values

        errors = np.abs(pred - true)
        n = len(errors)

        within_1sigma = int((errors <= std).sum())
        within_2sigma = int((errors <= 2 * std).sum())

        calibration[param] = {
            "coverage_1sigma": within_1sigma / n if n > 0 else np.nan,
            "coverage_2sigma": within_2sigma / n if n > 0 else np.nan,
            "expected_1sigma": 0.68,
            "expected_2sigma": 0.95,
            "n": n,
            "mean_std": float(std.mean()),
            "mean_error": float(errors.mean()),
        }

    print("\n  Uncertainty Calibration:")
    print(f"  {'Parameter':<12} {'1sig Obs':<12} {'1sig Exp':<12} "
          f"{'2sig Obs':<12} {'2sig Exp':<12}")
    print("  " + "-" * 60)
    for param in TARGETS:
        cal = calibration[param]
        print(f"  {param:<12} {cal['coverage_1sigma']:<12.2%} "
              f"{cal['expected_1sigma']:<12.2%} "
              f"{cal['coverage_2sigma']:<12.2%} "
              f"{cal['expected_2sigma']:<12.2%}")

    return calibration


def run_hfo_screening():
    """Run HFO screening with the retrained Esper-only GNN.

    Returns:
        tuple: (ranked_df, filter_funnel_df)
    """
    print("\n[5/8] Running HFO screening with Esper-only GNN...")

    from rdkit import Chem
    from rdkit.Contrib.SA_Score import sascorer

    from screening.generate import generate_systematic_candidates

    # Generate candidates
    print("  Generating systematic candidates...")
    start = time.time()
    candidates = generate_systematic_candidates(max_cl=1, max_mw=200.0)
    smiles_list = [smi for smi, _ in candidates]
    df = pd.DataFrame({"smiles": smiles_list})
    print(f"  Generated {len(df)} candidates in {time.time() - start:.1f}s")

    # Predict PC-SAFT parameters
    print("  Predicting PC-SAFT parameters via Esper-only GNN...")
    start = time.time()
    gnn = get_model("gnn")
    gnn.load()
    predictions = gnn.predict(smiles_list)
    df["m"] = predictions["m"]
    df["sigma"] = predictions["sigma"]
    df["epsilon_k"] = predictions["epsilon_k"]
    print(f"  Predicted {len(df)} parameter sets in {time.time() - start:.1f}s")

    # Compute boiling points
    print("  Computing boiling points...")
    start = time.time()
    df = batch_boiling_points(df)
    n_valid = (~df["boiling_point_K"].isna()).sum()
    print(f"  {n_valid}/{len(df)} valid boiling points in {time.time() - start:.1f}s")

    # SA scores
    print("  Computing SA scores...")
    sa_scores = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            sa_scores.append(10.0)
        else:
            try:
                sa_scores.append(sascorer.calculateScore(mol))
            except Exception:
                sa_scores.append(10.0)
    df["sa_score"] = sa_scores

    # Apply filters
    print("  Applying 7 HFO-centric filters...")
    filtered_df, filter_stats = apply_hfo_filters(df)

    # Build filter funnel
    filter_funnel = pd.DataFrame([
        {"stage": "Initial candidates", "count": len(df)},
        {"stage": "Boiling point [288-323K]", "count": filter_stats["boiling_point"]},
        {"stage": "Has C=C bond", "count": filter_stats["has_double_bond"]},
        {"stage": "No chlorine", "count": filter_stats["no_chlorine"]},
        {"stage": "F count >= 2", "count": filter_stats["fluorine_count"]},
        {"stage": "SA score <= 4.5", "count": filter_stats["sa_score"]},
        {
            "stage": "Fluorine mass fraction >= 65%",
            "count": filter_stats["fluorine_mass_fraction"],
        },
        {"stage": "No reactive fluorination sites", "count": filter_stats["no_reactive_sites"]},
        {"stage": "Final (all filters)", "count": filter_stats["total"]},
    ])

    # Rank by HFO distance (5:2:1 weighting)
    if len(filtered_df) > 0:
        distances = []
        for _, row in filtered_df.iterrows():
            dist = hfo_parameter_distance(
                row["m"], row["sigma"], row["epsilon_k"],
                reference=HFO_1336MZZ,
                weights={"epsilon_k": 5.0, "sigma": 2.0, "m": 1.0},
            )
            distances.append(dist)

        filtered_df = filtered_df.copy()
        filtered_df["hfo_distance"] = distances
        filtered_df = filtered_df.sort_values("hfo_distance").reset_index(drop=True)

        # Count fluorines
        def count_fluorines(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F")

        filtered_df["n_fluorine"] = filtered_df["smiles"].apply(count_fluorines)

    print(f"\n  Final candidates passing all filters: {len(filtered_df)}")

    return filtered_df, filter_funnel


def generate_figures(
    esper_metrics,
    fluorinated_results,
    bp_metrics,
    calibration,
    filter_funnel,
):
    """Generate all Step 38c figures.

    Args:
        esper_metrics: dict of Esper test set metrics
        fluorinated_results: DataFrame of fluorinated validation results
        bp_metrics: dict with GNN_esper, GNN_unified, RF boiling point metrics
        calibration: dict of uncertainty calibration
        filter_funnel: DataFrame of filter funnel
    """
    print("\n[6/8] Generating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 1: Parity plot — GNN-Esper vs unified GNN vs RF
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    param_labels = {
        "m": "m (segments)",
        "sigma": r"$\sigma$ (A)",
        "epsilon_k": r"$\varepsilon$/k (K)",
    }

    valid = ~fluorinated_results["m_lit"].isna()
    if valid.sum() > 0:
        for ax, param in zip(axes, TARGETS):
            true = fluorinated_results.loc[valid, f"{param}_lit"].values
            pred = fluorinated_results.loc[valid, f"{param}_pred"].values

            ax.scatter(true, pred, s=100, alpha=0.7, color="C0", zorder=3,
                       label="GNN (Esper)")

            # Error bars from MC dropout
            std = fluorinated_results.loc[valid, f"{param}_std"].values
            ax.errorbar(true, pred, yerr=std, fmt="none", ecolor="C0",
                        alpha=0.3, capsize=3)

            # Parity line
            all_vals = np.concatenate([true, pred])
            lo, hi = all_vals.min() * 0.95, all_vals.max() * 1.05
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, lw=2, label="Parity")

            ax.set_xlabel(f"Literature {param_labels[param]}", fontsize=20)
            ax.set_ylabel(f"Predicted {param_labels[param]}", fontsize=20)
            ax.tick_params(labelsize=16)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)

    fig.suptitle("GNN (Esper-only) vs Literature: Fluorinated Compounds", fontsize=22, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "parity_gnn_esper_vs_unified.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: parity_gnn_esper_vs_unified.png")

    # -------------------------------------------------------------------------
    # Figure 2: Boiling point validation comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))

    valid_bp = ~fluorinated_results["T_b_predicted_K"].isna()
    if valid_bp.sum() > 0:
        exp = fluorinated_results.loc[valid_bp, "T_b_experimental_K"].values
        pred = fluorinated_results.loc[valid_bp, "T_b_predicted_K"].values

        ax.scatter(exp, pred, s=120, alpha=0.7, color="C0", zorder=3,
                   label=f"GNN (Esper): MAE={bp_metrics['GNN_esper']['mae']:.1f} K")

        # Parity line
        all_vals = np.concatenate([exp, pred])
        lo, hi = all_vals.min() - 10, all_vals.max() + 10
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, lw=2, label="Parity")

        # Process window
        ax.axhspan(288, 323, alpha=0.15, color="green", label="Process window (288-323 K)")

        # Reference lines for other models
        ax.set_title("Boiling Point Validation: Fluorinated Compounds", fontsize=22)
        ax.set_xlabel("Experimental T$_b$ (K)", fontsize=20)
        ax.set_ylabel("Predicted T$_b$ (K)", fontsize=20)
        ax.tick_params(labelsize=16)

        # Add text box with comparison
        textstr = (
            f"GNN (Esper): {bp_metrics['GNN_esper']['mae']:.1f} K MAE\n"
            f"GNN (unified): {bp_metrics['GNN_unified']['mae']:.1f} K MAE\n"
            f"RF: {bp_metrics['RF']['mae']:.1f} K MAE"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=16,
                verticalalignment="top", bbox=props)

        ax.legend(fontsize=14, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "boiling_point_validation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: boiling_point_validation.png")

    # -------------------------------------------------------------------------
    # Figure 3: Uncertainty calibration
    # -------------------------------------------------------------------------
    if "error" not in calibration:
        fig, ax = plt.subplots(figsize=(8, 6))

        params = TARGETS
        x = np.arange(len(params))
        width = 0.25

        obs_1sig = [calibration[p]["coverage_1sigma"] for p in params]
        obs_2sig = [calibration[p]["coverage_2sigma"] for p in params]

        ax.bar(x - width/2, obs_1sig, width, label="Observed 1sig",
               color="C0", alpha=0.8)
        ax.bar(x + width/2, obs_2sig, width, label="Observed 2sig",
               color="C1", alpha=0.8)

        ax.axhline(y=0.68, color="C0", linestyle="--", alpha=0.6,
                   label="Expected 1sig (68%)")
        ax.axhline(y=0.95, color="C1", linestyle="--", alpha=0.6,
                   label="Expected 2sig (95%)")

        ax.set_xticks(x)
        ax.set_xticklabels(["m", r"$\sigma$", r"$\varepsilon$/k"], fontsize=18)
        ax.set_ylabel("Coverage Fraction", fontsize=20)
        ax.set_title("MC Dropout Uncertainty Calibration (GNN Esper)", fontsize=20)
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=14, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "uncertainty_calibration.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: uncertainty_calibration.png")

    # -------------------------------------------------------------------------
    # Figure 4: Screening funnel
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = filter_funnel["stage"].tolist()
    counts = filter_funnel["count"].tolist()

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(stages)))
    bars = ax.barh(range(len(stages)), counts, color=colors, alpha=0.8)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Candidates", fontsize=18)
    ax.set_title("HFO Screening Funnel (GNN Esper-only)", fontsize=20)
    ax.tick_params(labelsize=14)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                f"{count:,}", ha="left", va="center", fontsize=12)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "screening_funnel_gnn_esper.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: screening_funnel_gnn_esper.png")


def save_results(
    esper_metrics,
    fluorinated_results,
    bp_metrics,
    param_metrics,
    calibration,
    ranked_df,
    filter_funnel,
):
    """Save all results to disk."""
    print("\n[7/8] Saving results...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save ranked screening results
    if ranked_df is not None and len(ranked_df) > 0:
        ranked_path = RESULTS_DIR / "hfo_gnn_esper_ranked.csv"
        ranked_df.to_csv(ranked_path, index=False)
        print(f"  Saved: {ranked_path}")

    # Save filter funnel
    funnel_path = RESULTS_DIR / "hfo_gnn_esper_filter_funnel.csv"
    filter_funnel.to_csv(funnel_path, index=False)
    print(f"  Saved: {funnel_path}")

    # Save fluorinated validation results
    val_path = SAVED_DIR / "gnn_esper_fluorinated_validation_results.csv"
    fluorinated_results.to_csv(val_path, index=False)
    print(f"  Saved: {val_path}")

    # Save comprehensive metrics JSON
    def to_native(obj):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        else:
            return obj

    metrics_summary = {
        "esper_test_metrics": to_native(esper_metrics),
        "fluorinated_bp_metrics": to_native(bp_metrics),
        "fluorinated_param_metrics": to_native(param_metrics),
        "uncertainty_calibration": to_native(calibration),
        "screening_summary": {
            "model": "gnn_esper",
            "n_candidates_generated": int(filter_funnel["count"].iloc[0]),
            "n_passing_all_filters": int(filter_funnel["count"].iloc[-1]),
        },
    }

    metrics_path = SAVED_DIR / "gnn_esper_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Saved: {metrics_path}")


def print_summary(esper_metrics, bp_metrics, param_metrics, calibration, n_screening):
    """Print final summary comparison."""
    print("\n" + "=" * 80)
    print("STEP 38c SUMMARY: GNN RETRAIN ON EXPERIMENTAL DATA ONLY")
    print("=" * 80)

    # Esper test set metrics
    print("\n1. Esper Test-Set Metrics (GNN Esper-only):")
    print(f"   {'Parameter':<12} {'R2':<10} {'MAE':<10} {'RMSE':<10}")
    print("   " + "-" * 42)
    for param in TARGETS:
        m = esper_metrics[param]
        print(f"   {param:<12} {m['r2']:<10.4f} {m['mae']:<10.4f} {m['rmse']:<10.4f}")

    # Fluorinated validation comparison
    print("\n2. Fluorinated Boiling Point MAE (key metric):")
    print(f"   GNN (Esper-only):  {bp_metrics['GNN_esper']['mae']:.2f} K")
    print(f"   GNN (unified):     {bp_metrics['GNN_unified']['mae']:.2f} K")
    print(f"   RF:                {bp_metrics['RF']['mae']:.2f} K")

    improvement = bp_metrics["GNN_unified"]["mae"] - bp_metrics["GNN_esper"]["mae"]
    print(f"   Improvement over unified GNN: {improvement:.1f} K")

    # Fluorinated parameter metrics
    if param_metrics:
        print("\n3. Fluorinated Parameter Metrics (vs literature):")
        print(f"   {'Param':<12} {'MAE (Esper)':<14} {'MAE (Unified)':<14}")
        print("   " + "-" * 40)

        # Load unified metrics for comparison
        unified_path = SAVED_DIR / "gnn_fluorinated_validation_metrics.json"
        unified_pm = {}
        if unified_path.exists():
            with open(unified_path) as f:
                um = json.load(f)
            if "param_metrics" in um and "GNN" in um["param_metrics"]:
                unified_pm = um["param_metrics"]["GNN"]

        for param in TARGETS:
            esper_mae = param_metrics[param]["mae"]
            unified_mae = unified_pm.get(param, {}).get("mae", np.nan)
            print(f"   {param:<12} {esper_mae:<14.4f} {unified_mae:<14.4f}")

    # Uncertainty calibration
    if "error" not in calibration:
        print("\n4. Uncertainty Calibration (1sigma coverage):")
        for param in TARGETS:
            cal = calibration[param]
            print(f"   {param}: {cal['coverage_1sigma']:.0%} observed vs 68% expected")

    # Screening
    print(f"\n5. HFO Screening: {n_screening} candidates passing all filters")

    # Verdict
    print("\n" + "-" * 80)
    esper_mae = bp_metrics["GNN_esper"]["mae"]
    if esper_mae < 15.0:
        print("VERDICT: GNN architecture CAN match RF when given clean data.")
        print(f"  Boiling point MAE {esper_mae:.1f} K is within usable range (<15 K).")
    elif esper_mae < 50.0:
        print("VERDICT: GNN architecture PARTIALLY recovers when given clean data.")
        print(f"  Boiling point MAE {esper_mae:.1f} K is better than unified but lags RF.")
    else:
        print("VERDICT: GNN architecture does NOT match RF even with clean data.")
        print(f"  Boiling point MAE {esper_mae:.1f} K remains high.")
    print("=" * 80)


def main():
    """Run full Step 38c workflow."""
    print("=" * 80)
    print("STEP 38c: GNN RETRAIN ON EXPERIMENTAL (ESPER-ONLY) DATA")
    print("=" * 80)
    print("\nObjective: Test whether GNN architecture can match RF when trained")
    print("on clean, experimentally fitted PC-SAFT parameters (no SPT data).")

    # Stage 1: Backup
    backup_unified_gnn()

    # Stage 2: Retrain
    esper_metrics = retrain_gnn_esper()

    # Stage 3: Fluorinated validation
    fluorinated_results, bp_metrics_esper, param_metrics = validate_fluorinated("gnn")

    # Load comparison metrics from Step 38 (unified GNN, RF)
    unified_metrics_path = SAVED_DIR / "gnn_fluorinated_validation_metrics.json"
    if unified_metrics_path.exists():
        with open(unified_metrics_path) as f:
            step38_metrics = json.load(f)
        bp_unified = step38_metrics.get("bp_metrics", {}).get("GNN", {})
        bp_rf = step38_metrics.get("bp_metrics", {}).get("RF", {})
    else:
        bp_unified = {"mae": 133.72, "n_converged": 6, "n_total": 15}
        bp_rf = {"mae": 8.17, "n_converged": 6, "n_total": 15}

    bp_metrics = {
        "GNN_esper": bp_metrics_esper,
        "GNN_unified": bp_unified,
        "RF": bp_rf,
    }

    # Stage 4: Uncertainty calibration
    calibration = uncertainty_calibration(fluorinated_results)

    # Stage 5: HFO screening
    ranked_df, filter_funnel = run_hfo_screening()

    # Stage 6: Figures
    generate_figures(esper_metrics, fluorinated_results, bp_metrics,
                     calibration, filter_funnel)

    # Stage 7: Save results
    save_results(esper_metrics, fluorinated_results, bp_metrics,
                 param_metrics, calibration, ranked_df, filter_funnel)

    # Stage 8: Print summary
    n_screening = len(ranked_df) if ranked_df is not None else 0
    print_summary(esper_metrics, bp_metrics, param_metrics, calibration, n_screening)

    print("\nDone. See figures/ and screening/results/ for outputs.")


if __name__ == "__main__":
    main()
