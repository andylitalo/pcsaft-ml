"""Evaluate GNN, RF, and ensemble on the unified test set.

This script:
1. Loads the unified test set from model/saved/test_set.csv
2. Evaluates RF, GNN, and ensemble models
3. Computes metrics overall and by source (Esper, ML-SAFT, SPT-PCSAFT)
4. Identifies fluorinated molecules and computes subset metrics
5. Saves results to CSV and generates comparison tables
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.data.load import TARGETS
from model.registry import get_model


def is_fluorinated(smiles: str) -> bool:
    """Check if a SMILES contains fluorine."""
    return "F" in smiles


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², MAE, RMSE for predictions."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "n": mask.sum()}
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    return {
        "r2": r2_score(y_true_clean, y_pred_clean),
        "mae": mean_absolute_error(y_true_clean, y_pred_clean),
        "rmse": np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        "n": len(y_true_clean),
    }


def evaluate_model(model_name: str, test_df: pd.DataFrame) -> dict:
    """Evaluate a single model on the test set."""
    print(f"\nEvaluating {model_name}...")
    model = get_model(model_name)
    model.load()

    predictions = model.predict(test_df["smiles"].tolist())

    results = {}
    for target in TARGETS:
        y_true = test_df[target].values
        y_pred = predictions[target]
        results[target] = compute_metrics(y_true, y_pred)

    return results


def evaluate_ensemble(test_df: pd.DataFrame, mode: str = "equal") -> dict:
    """Evaluate ensemble with equal weighting or inverse-variance weighting."""
    print(f"\nEvaluating ensemble ({mode} weighting)...")

    # Load models
    rf_model = get_model("rf")
    rf_model.load()
    gnn_model = get_model("gnn")
    gnn_model.load()

    smiles = test_df["smiles"].tolist()

    if mode == "equal":
        # Equal weighting
        rf_preds = rf_model.predict(smiles)
        gnn_preds = gnn_model.predict(smiles)

        combined = {}
        for target in TARGETS:
            combined[target] = 0.5 * rf_preds[target] + 0.5 * gnn_preds[target]
    else:
        # Inverse-variance weighting
        rf_preds = rf_model.predict_with_uncertainty(smiles)
        gnn_preds = gnn_model.predict_with_uncertainty(smiles)

        combined = {}
        for target in TARGETS:
            rf_mean = rf_preds[target]
            rf_std = rf_preds[f"{target}_std"]
            gnn_mean = gnn_preds[target]
            gnn_std = gnn_preds[f"{target}_std"]

            # Inverse-variance weights
            rf_weight = 1.0 / (rf_std**2 + 1e-6)
            gnn_weight = 1.0 / (gnn_std**2 + 1e-6)
            total_weight = rf_weight + gnn_weight

            combined[target] = (rf_weight * rf_mean + gnn_weight * gnn_mean) / total_weight

    results = {}
    for target in TARGETS:
        y_true = test_df[target].values
        y_pred = combined[target]
        results[target] = compute_metrics(y_true, y_pred)

    return results


def main():
    # Load test set
    test_set_path = "model/saved/test_set.csv"
    test_df = pd.read_csv(test_set_path)
    print(f"Test set size: {len(test_df)}")

    # Check if source column exists
    if "source" in test_df.columns:
        print("\nSource distribution:")
        print(test_df["source"].value_counts())

    # Add fluorinated flag
    test_df["is_fluorinated"] = test_df["smiles"].apply(is_fluorinated)
    n_fluorinated = test_df["is_fluorinated"].sum()
    print(f"\nFluorinated molecules: {n_fluorinated} ({100*n_fluorinated/len(test_df):.1f}%)")

    # Evaluate models
    results = {}
    for model_name in ["rf", "gnn"]:
        results[model_name] = evaluate_model(model_name, test_df)

    results["ensemble_equal"] = evaluate_ensemble(test_df, mode="equal")
    results["ensemble_invvar"] = evaluate_ensemble(test_df, mode="invvar")

    # Print overall results
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Target':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'N':>6}")
    print("-"*80)
    for model_name, model_results in results.items():
        for target in TARGETS:
            metrics = model_results[target]
            print(f"{model_name:<20} {target:<12} {metrics['r2']:>8.4f} {metrics['mae']:>8.4f} "
                  f"{metrics['rmse']:>8.4f} {metrics['n']:>6}")

    # Source-specific results
    if "source" in test_df.columns:
        print("\n" + "="*80)
        print("RESULTS BY SOURCE")
        print("="*80)
        for source in test_df["source"].unique():
            source_df = test_df[test_df["source"] == source]
            print(f"\n{source.upper()} (n={len(source_df)})")
            print(f"{'Model':<20} {'Target':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
            print("-"*70)

            for model_name in ["rf", "gnn"]:
                model = get_model(model_name)
                model.load()
                preds = model.predict(source_df["smiles"].tolist())
                for target in TARGETS:
                    y_true = source_df[target].values
                    y_pred = preds[target]
                    metrics = compute_metrics(y_true, y_pred)
                    print(f"{model_name:<20} {target:<12} {metrics['r2']:>8.4f} "
                          f"{metrics['mae']:>8.4f} {metrics['rmse']:>8.4f}")

    # Fluorinated subset
    if n_fluorinated > 0:
        print("\n" + "="*80)
        print(f"FLUORINATED SUBSET (n={n_fluorinated})")
        print("="*80)
        fluor_df = test_df[test_df["is_fluorinated"]]
        print(f"{'Model':<20} {'Target':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
        print("-"*70)

        for model_name in ["rf", "gnn"]:
            model = get_model(model_name)
            model.load()
            preds = model.predict(fluor_df["smiles"].tolist())
            for target in TARGETS:
                y_true = fluor_df[target].values
                y_pred = preds[target]
                metrics = compute_metrics(y_true, y_pred)
                print(f"{model_name:<20} {target:<12} {metrics['r2']:>8.4f} "
                      f"{metrics['mae']:>8.4f} {metrics['rmse']:>8.4f}")

    # Save results
    rows = []
    for model_name, model_results in results.items():
        for target in TARGETS:
            metrics = model_results[target]
            rows.append({
                "model": model_name,
                "target": target,
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "n": metrics["n"],
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv("model/saved/unified_evaluation_results.csv", index=False)
    print("\nResults saved to model/saved/unified_evaluation_results.csv")


if __name__ == "__main__":
    main()
