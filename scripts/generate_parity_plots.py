"""Generate parity plots for RF vs GNN on unified test set."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.data.load import TARGETS
from model.registry import get_model

# Set font sizes for presentation
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
})


def main():
    # Load test set
    test_df = pd.read_csv("model/saved/test_set.csv")
    print(f"Test set: {len(test_df)} molecules")

    # Load models
    rf_model = get_model("rf")
    rf_model.load()
    gnn_model = get_model("gnn")
    gnn_model.load()

    # Get predictions
    rf_preds = rf_model.predict(test_df["smiles"].tolist())
    gnn_preds = gnn_model.predict(test_df["smiles"].tolist())

    # Create figure with 3 rows x 2 cols (RF and GNN side by side)
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))

    target_labels = {
        "m": "m (segments)",
        "sigma": "σ (Å)",
        "epsilon_k": "ε/k (K)",
    }

    for i, target in enumerate(TARGETS):
        y_true = test_df[target].values

        # RF parity plot
        ax_rf = axes[i, 0]
        y_pred_rf = rf_preds[target]
        mask_rf = np.isfinite(y_true) & np.isfinite(y_pred_rf)

        ax_rf.scatter(y_true[mask_rf], y_pred_rf[mask_rf], alpha=0.3, s=20)

        # Add diagonal
        lims = [
            min(y_true[mask_rf].min(), y_pred_rf[mask_rf].min()),
            max(y_true[mask_rf].max(), y_pred_rf[mask_rf].max()),
        ]
        ax_rf.plot(lims, lims, 'k--', lw=2, alpha=0.5, label='Perfect prediction')

        ax_rf.set_xlabel(f"True {target_labels[target]}")
        ax_rf.set_ylabel(f"Predicted {target_labels[target]}")
        ax_rf.set_title(f"RF: {target_labels[target]}")
        ax_rf.legend()
        ax_rf.grid(alpha=0.3)

        # Compute R²
        from sklearn.metrics import r2_score
        r2_rf = r2_score(y_true[mask_rf], y_pred_rf[mask_rf])
        bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax_rf.text(0.05, 0.95, f"R² = {r2_rf:.3f}", transform=ax_rf.transAxes,
                   verticalalignment='top', bbox=bbox_props)

        # GNN parity plot
        ax_gnn = axes[i, 1]
        y_pred_gnn = gnn_preds[target]
        mask_gnn = np.isfinite(y_true) & np.isfinite(y_pred_gnn)

        ax_gnn.scatter(y_true[mask_gnn], y_pred_gnn[mask_gnn], alpha=0.3, s=20, color='C1')

        ax_gnn.plot(lims, lims, 'k--', lw=2, alpha=0.5, label='Perfect prediction')

        ax_gnn.set_xlabel(f"True {target_labels[target]}")
        ax_gnn.set_ylabel(f"Predicted {target_labels[target]}")
        ax_gnn.set_title(f"GNN: {target_labels[target]}")
        ax_gnn.legend()
        ax_gnn.grid(alpha=0.3)

        r2_gnn = r2_score(y_true[mask_gnn], y_pred_gnn[mask_gnn])
        ax_gnn.text(0.05, 0.95, f"R² = {r2_gnn:.3f}", transform=ax_gnn.transAxes,
                    verticalalignment='top', bbox=bbox_props)

    plt.tight_layout()
    output_path = "figures/31_unified_gnn_retrain/parity_rf_vs_gnn.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved parity plots to {output_path}")
    plt.close()

    # Also create source-colored parity plots for GNN
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for i, target in enumerate(TARGETS):
        ax = axes[i]
        y_true = test_df[target].values
        y_pred = gnn_preds[target]

        # Plot by source
        for source, color, label in [
            ("esper", "red", "Esper"),
            ("mlsaft", "blue", "ML-SAFT"),
            ("spt_pcsaft", "green", "SPT-PCSAFT"),
        ]:
            mask = (test_df["source"] == source) & np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() > 0:
                ax.scatter(y_true[mask], y_pred[mask], alpha=0.4, s=30,
                          c=color, label=label)

        # Add diagonal
        lims = [y_true.min(), y_true.max()]
        ax.plot(lims, lims, 'k--', lw=2, alpha=0.5)

        ax.set_xlabel(f"True {target_labels[target]}")
        ax.set_ylabel(f"Predicted {target_labels[target]}")
        ax.set_title(f"GNN: {target_labels[target]} (by source)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = "figures/31_unified_gnn_retrain/parity_gnn_by_source.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved source-colored parity plots to {output_path}")


if __name__ == "__main__":
    main()
