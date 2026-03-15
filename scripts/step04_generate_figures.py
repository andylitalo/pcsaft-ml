"""Generate Step 04 figures: ChemBERTa parity, 4-way comparison, loss curves, screening."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures" / "04_chemberta_finetuning"
SAVED_DIR = PROJECT_ROOT / "model" / "saved"
SCREENING_DIR = PROJECT_ROOT / "screening" / "results"

TARGETS = ["m", "sigma", "epsilon_k"]
UNITS = {"m": "segments", "sigma": "\u00c5", "epsilon_k": "K"}
TARGET_LABELS = {"m": "m", "sigma": "\u03c3", "epsilon_k": "\u03b5/k"}


def _load_metrics():
    """Load the comparison_metrics.csv."""
    return pd.read_csv(SAVED_DIR / "comparison_metrics.csv")


def plot_chemberta_parity():
    """Parity plots for ChemBERTa predictions on test set."""
    from model.registry import get_model

    test_df = pd.read_csv(SAVED_DIR / "test_set.csv")
    smiles_list = test_df["smiles"].tolist()

    model = get_model("chemberta")
    model.load()
    preds = model.predict(smiles_list)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for j, target in enumerate(TARGETS):
        ax = axes[j]
        y_true = test_df[target].values
        y_pred = preds[target]
        mask = np.isfinite(y_true) & np.isfinite(y_pred)

        ax.scatter(
            y_true[mask], y_pred[mask],
            alpha=0.5, s=25, c="#e7298a", edgecolors="k",
            linewidths=0.3, zorder=3,
        )

        # Parity line
        all_vals = np.concatenate([y_true[mask], y_pred[mask]])
        lo, hi = all_vals.min(), all_vals.max()
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", lw=1, alpha=0.5, zorder=2,
        )

        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        r2 = r2_score(y_true[mask], y_pred[mask])

        label = TARGET_LABELS[target]
        ax.set_xlabel(f"True {label} ({UNITS[target]})", fontsize=13)
        ax.set_ylabel(f"Predicted {label} ({UNITS[target]})", fontsize=13)
        ax.set_title(
            f"ChemBERTa | {label}: R\u00b2={r2:.3f}, MAE={mae:.3f}",
            fontsize=14,
        )
        ax.tick_params(labelsize=11)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("ChemBERTa Fine-Tuned for PC-SAFT Parameters", fontsize=16, y=1.02)
    fig.tight_layout()
    path = FIGURES_DIR / "chemberta_parity_plots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_four_way_r2_comparison():
    """Bar chart comparing R2 across all four models."""
    metrics = _load_metrics()

    models = ["gc_pcsaft", "rf", "nn", "chemberta"]
    model_labels = ["GC-PC-SAFT", "Random Forest", "PCSAFTNet (NN)", "ChemBERTa"]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TARGETS))
    width = 0.18

    for i, (mname, mlabel, color) in enumerate(zip(models, model_labels, colors)):
        sub = metrics[metrics["model"] == mname]
        r2_vals = []
        for target in TARGETS:
            row = sub[sub["target"] == target]
            r2_vals.append(row["r2"].values[0] if len(row) > 0 else 0)

        offset = (i - 1.5) * width
        ax.bar(x + offset, r2_vals, width, label=mlabel, color=color,
               edgecolor="k", linewidth=0.5)

    ax.set_xlabel("PC-SAFT Parameter", fontsize=13)
    ax.set_ylabel("R\u00b2 (test set)", fontsize=13)
    ax.set_title("Four-Way Model Comparison: R\u00b2 on Esper Test Set", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TARGET_LABELS[t]} ({UNITS[t]})" for t in TARGETS], fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(-1.2, ax.get_ylim()[0]), max(0.8, ax.get_ylim()[1]))

    fig.tight_layout()
    path = FIGURES_DIR / "four_way_r2_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_chemberta_loss_curves():
    """Training and validation loss curves from ChemBERTa fine-tuning."""
    history_path = SAVED_DIR / "chemberta" / "training_history.json"
    if not history_path.exists():
        print("  (training_history.json not found; skipping)")
        return

    history = json.loads(history_path.read_text())

    fig, ax = plt.subplots(figsize=(8, 5))

    # Training losses (logged at step level, group by epoch)
    if history.get("train_losses"):
        train_epochs = [e["epoch"] for e in history["train_losses"]]
        train_loss = [e["loss"] for e in history["train_losses"]]
        ax.plot(train_epochs, train_loss, label="Train loss", lw=1.5,
                color="#d95f02", alpha=0.7)

    # Eval losses (logged at epoch level)
    if history.get("eval_losses"):
        eval_epochs = [e["epoch"] for e in history["eval_losses"]]
        eval_loss = [e["eval_loss"] for e in history["eval_losses"]]
        ax.plot(eval_epochs, eval_loss, label="Eval loss", lw=2, color="#e7298a",
                marker="o", markersize=5)

        # Mark best epoch
        best_idx = np.argmin(eval_loss)
        ax.scatter(
            [eval_epochs[best_idx]], [eval_loss[best_idx]],
            marker="*", s=200, c="gold", edgecolors="k", zorder=5,
            label=f"Best epoch ({eval_epochs[best_idx]:.0f})",
        )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss (MSE, normalized targets)", fontsize=13)
    ax.set_title("ChemBERTa Fine-Tuning Loss Curves", fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "chemberta_loss_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_screening_results():
    """Histogram of screening candidate distances to cyclopentane."""
    # Try broad results first, then default
    for fname in ["ranked_candidates_broad.csv", "ranked_candidates.csv"]:
        fpath = SCREENING_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            break
    else:
        print("  (No screening results found; skipping)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distance histogram
    ax = axes[0]
    ax.hist(df["distance"], bins=40, edgecolor="k", linewidth=0.5,
            alpha=0.7, color="#e7298a")
    ax.axvline(df["distance"].median(), color="r", linestyle="--", lw=1.5,
               label=f"Median={df['distance'].median():.3f}")
    ax.set_xlabel("Weighted Distance to Cyclopentane", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"Screening: {len(df)} Candidates Ranked", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    # Scatter: predicted m vs epsilon_k, colored by distance
    ax = axes[1]
    sc = ax.scatter(
        df["m"], df["epsilon_k"], c=df["distance"],
        cmap="RdYlGn_r", alpha=0.6, s=25, edgecolors="k", linewidths=0.3,
    )
    plt.colorbar(sc, ax=ax, label="Distance to Cyclopentane")
    # Mark cyclopentane
    ax.scatter([2.3655], [288.84], marker="*", s=200, c="blue", edgecolors="k",
               zorder=5, label="Cyclopentane")
    ax.set_xlabel("Predicted m (segments)", fontsize=13)
    ax.set_ylabel("Predicted \u03b5/k (K)", fontsize=13)
    ax.set_title("Candidate Parameter Space", fontsize=14)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    path = FIGURES_DIR / "screening_results.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_chemberta_parity()
    plot_four_way_r2_comparison()
    plot_chemberta_loss_curves()
    plot_screening_results()
