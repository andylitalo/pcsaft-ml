"""Train RF with GC-PC-SAFT features and compare to baseline.

This script implements the ablation experiment for Step 18:
1. Baseline: Morgan + RDKit features (no GC)
2. Augmented: Morgan + RDKit + GC-PC-SAFT features
3. GC-only: GC-PC-SAFT features only (for reference)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from model.data.descriptors import build_features  # noqa: E402
from model.data.load import load_data  # noqa: E402

# Parameters
RANDOM_STATE = 42
N_ESTIMATORS = 100
TEST_SIZE = 0.2


def train_rf_for_target(X_train, y_train, X_test, y_test, target_name):
    """Train RF for a single target and return test metrics."""
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Compute metrics
    return {
        "target": target_name,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def run_ablation_experiment(df, targets):
    """Run ablation experiment: baseline vs +GC vs GC-only."""
    smiles = df["smiles"].tolist()

    # Stratified split on epsilon_k
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=pd.qcut(df["epsilon_k"], q=5, duplicates="drop"),
    )

    results = []

    # Experiment 1: Baseline (Morgan + RDKit, no GC)
    print("\n" + "=" * 70)
    print("Experiment 1: Baseline (Morgan + RDKit)")
    print("=" * 70)
    X_baseline = build_features(
        smiles, use_morgan=True, use_rdkit=True, use_gc_pcsaft=False
    )
    print(f"Feature shape: {X_baseline.shape}")

    for target in targets:
        y = df[target].values
        metrics = train_rf_for_target(
            X_baseline[train_idx],
            y[train_idx],
            X_baseline[test_idx],
            y[test_idx],
            target,
        )
        metrics["experiment"] = "baseline"
        results.append(metrics)
        print(
            f"  {target:10s}: R²={metrics['r2']:.3f}, "
            f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}"
        )

    # Experiment 2: Augmented (Morgan + RDKit + GC)
    print("\n" + "=" * 70)
    print("Experiment 2: Augmented (Morgan + RDKit + GC-PC-SAFT)")
    print("=" * 70)
    X_augmented = build_features(
        smiles, use_morgan=True, use_rdkit=True, use_gc_pcsaft=True
    )
    print(f"Feature shape: {X_augmented.shape} (+3 GC features)")

    for target in targets:
        y = df[target].values
        metrics = train_rf_for_target(
            X_augmented[train_idx],
            y[train_idx],
            X_augmented[test_idx],
            y[test_idx],
            target,
        )
        metrics["experiment"] = "augmented"
        results.append(metrics)
        print(
            f"  {target:10s}: R²={metrics['r2']:.3f}, "
            f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}"
        )

    # Experiment 3: GC-only (for reference)
    print("\n" + "=" * 70)
    print("Experiment 3: GC-PC-SAFT only (reference)")
    print("=" * 70)
    X_gc_only = build_features(
        smiles, use_morgan=False, use_rdkit=False, use_gc_pcsaft=True
    )
    print(f"Feature shape: {X_gc_only.shape}")

    for target in targets:
        y = df[target].values
        metrics = train_rf_for_target(
            X_gc_only[train_idx],
            y[train_idx],
            X_gc_only[test_idx],
            y[test_idx],
            target,
        )
        metrics["experiment"] = "gc_only"
        results.append(metrics)
        print(
            f"  {target:10s}: R²={metrics['r2']:.3f}, "
            f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}"
        )

    return pd.DataFrame(results)


def print_ablation_table(df_results):
    """Print formatted ablation comparison table."""
    print("\n" + "=" * 70)
    print("ABLATION TABLE: R² COMPARISON")
    print("=" * 70)
    print(f"{'Feature Set':<40} {'R² (m)':<10} {'R² (σ)':<10} {'R² (ε/k)':<10}")
    print("-" * 70)

    for exp_name, exp_label in [
        ("baseline", "Morgan + RDKit (baseline)"),
        ("augmented", "Morgan + RDKit + GC-PC-SAFT"),
        ("gc_only", "GC-PC-SAFT only (reference)"),
    ]:
        exp_data = df_results[df_results["experiment"] == exp_name]
        r2_m = exp_data[exp_data["target"] == "m"]["r2"].values[0]
        r2_sigma = exp_data[exp_data["target"] == "sigma"]["r2"].values[0]
        r2_epsk = exp_data[exp_data["target"] == "epsilon_k"]["r2"].values[0]
        print(f"{exp_label:<40} {r2_m:<10.3f} {r2_sigma:<10.3f} {r2_epsk:<10.3f}")

    print("\n" + "=" * 70)
    print("ABLATION TABLE: MAE COMPARISON")
    print("=" * 70)
    print(
        f"{'Feature Set':<40} {'MAE (m)':<10} "
        f"{'MAE (σ)':<10} {'MAE (ε/k)':<10}"
    )
    print("-" * 70)

    for exp_name, exp_label in [
        ("baseline", "Morgan + RDKit (baseline)"),
        ("augmented", "Morgan + RDKit + GC-PC-SAFT"),
        ("gc_only", "GC-PC-SAFT only (reference)"),
    ]:
        exp_data = df_results[df_results["experiment"] == exp_name]
        mae_m = exp_data[exp_data["target"] == "m"]["mae"].values[0]
        mae_sigma = exp_data[exp_data["target"] == "sigma"]["mae"].values[0]
        mae_epsk = exp_data[exp_data["target"] == "epsilon_k"]["mae"].values[0]
        print(f"{exp_label:<40} {mae_m:<10.3f} {mae_sigma:<10.3f} {mae_epsk:<10.3f}")

    # Delta analysis
    baseline = df_results[df_results["experiment"] == "baseline"]
    augmented = df_results[df_results["experiment"] == "augmented"]

    print("\n" + "=" * 70)
    print("DELTA: Augmented vs Baseline")
    print("=" * 70)
    print(f"{'Parameter':<15} {'ΔR²':<10} {'ΔMAE':<10}")
    print("-" * 70)

    for target in ["m", "sigma", "epsilon_k"]:
        r2_base = baseline[baseline["target"] == target]["r2"].values[0]
        r2_aug = augmented[augmented["target"] == target]["r2"].values[0]
        mae_base = baseline[baseline["target"] == target]["mae"].values[0]
        mae_aug = augmented[augmented["target"] == target]["mae"].values[0]

        delta_r2 = r2_aug - r2_base
        delta_mae = mae_aug - mae_base

        print(f"{target:<15} {delta_r2:+.4f}     {delta_mae:+.4f}")


def main():
    """Run the GC-PC-SAFT feature ablation experiment."""
    print("=" * 70)
    print("Step 18: GC-PC-SAFT as Input Feature Experiment")
    print("=" * 70)

    # Load data
    print("\nLoading Esper dataset...")
    df = load_data(source="esper")
    print(f"Loaded {len(df)} molecules")

    # Define targets
    targets = ["m", "sigma", "epsilon_k"]

    # Run experiment
    df_results = run_ablation_experiment(df, targets)

    # Print tables
    print_ablation_table(df_results)

    # Save results
    output_dir = project_root / "figures" / "18_gc_as_feature"
    output_dir.mkdir(exist_ok=True, parents=True)

    results_path = output_dir / "ablation_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")

    # Save summary as JSON
    summary = {}
    for exp in ["baseline", "augmented", "gc_only"]:
        summary[exp] = {}
        exp_data = df_results[df_results["experiment"] == exp]
        for target in targets:
            target_data = exp_data[exp_data["target"] == target].iloc[0]
            summary[exp][target] = {
                "r2": float(target_data["r2"]),
                "mae": float(target_data["mae"]),
                "rmse": float(target_data["rmse"]),
            }

    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
