"""Analyze RF vs GNN performance on Esper fluorinated vs non-fluorinated subsets."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.data.load import TARGETS
from model.registry import get_model


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


def main():
    # Load test set
    test_df = pd.read_csv("model/saved/test_set.csv")

    # Get Esper subset
    esper_df = test_df[test_df["source"] == "esper"]
    print(f"Esper test set: {len(esper_df)} molecules")

    # Separate fluorinated and non-fluorinated
    esper_df["is_fluorinated"] = esper_df["smiles"].str.contains("F")
    esper_fluor = esper_df[esper_df["is_fluorinated"]]
    esper_nonfluor = esper_df[~esper_df["is_fluorinated"]]

    print(f"  Fluorinated: {len(esper_fluor)}")
    print(f"  Non-fluorinated: {len(esper_nonfluor)}")

    # Load models
    rf_model = get_model("rf")
    rf_model.load()
    gnn_model = get_model("gnn")
    gnn_model.load()

    print("\n" + "="*80)
    print("ESPER FLUORINATED (n={})".format(len(esper_fluor)))
    print("="*80)
    print(f"{'Model':<15} {'Target':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("-"*70)

    for model_name, model in [("RF", rf_model), ("GNN", gnn_model)]:
        preds = model.predict(esper_fluor["smiles"].tolist())
        for target in TARGETS:
            y_true = esper_fluor[target].values
            y_pred = preds[target]
            metrics = compute_metrics(y_true, y_pred)
            print(f"{model_name:<15} {target:<12} {metrics['r2']:>8.4f} "
                  f"{metrics['mae']:>8.4f} {metrics['rmse']:>8.4f}")

    print("\n" + "="*80)
    print("ESPER NON-FLUORINATED (n={})".format(len(esper_nonfluor)))
    print("="*80)
    print(f"{'Model':<15} {'Target':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("-"*70)

    for model_name, model in [("RF", rf_model), ("GNN", gnn_model)]:
        preds = model.predict(esper_nonfluor["smiles"].tolist())
        for target in TARGETS:
            y_true = esper_nonfluor[target].values
            y_pred = preds[target]
            metrics = compute_metrics(y_true, y_pred)
            print(f"{model_name:<15} {target:<12} {metrics['r2']:>8.4f} "
                  f"{metrics['mae']:>8.4f} {metrics['rmse']:>8.4f}")

    # Also check parameter distributions
    print("\n" + "="*80)
    print("PARAMETER DISTRIBUTIONS")
    print("="*80)
    print("\nEsper fluorinated:")
    for target in TARGETS:
        vals = esper_fluor[target].values
        print(f"  {target}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"range=[{vals.min():.2f}, {vals.max():.2f}]")

    print("\nEsper non-fluorinated:")
    for target in TARGETS:
        vals = esper_nonfluor[target].values
        print(f"  {target}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"range=[{vals.min():.2f}, {vals.max():.2f}]")

    # Compare with SPT-PCSAFT
    spt_df = test_df[test_df["source"] == "spt_pcsaft"]
    print("\nSPT-PCSAFT (all):")
    for target in TARGETS:
        vals = spt_df[target].values
        print(f"  {target}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"range=[{vals.min():.2f}, {vals.max():.2f}]")


if __name__ == "__main__":
    main()
