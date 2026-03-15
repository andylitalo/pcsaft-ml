"""Generate visualizations for ChemBERTa Applicability Domain analysis."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHEMBERTA_DIR = Path(__file__).resolve().parent / "model" / "saved" / "chemberta"
FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "04b_chemberta_ad"


def main():
    # Load training embeddings
    embeddings_path = CHEMBERTA_DIR / "train_cls_embeddings.npy"
    if not embeddings_path.exists():
        logger.error("Training embeddings not found at %s", embeddings_path)
        return

    train_embeddings = np.load(embeddings_path)
    logger.info("Loaded %d training embeddings", len(train_embeddings))

    # Load AD model
    from model.hf.ad import load_chemberta_ad_model

    ad_model = load_chemberta_ad_model()

    # Get AD scores and labels for training set
    train_scores = ad_model.decision_function(train_embeddings)
    train_labels = ad_model.predict(train_embeddings) == 1  # True = in-domain

    # Load test embeddings
    from model.data.load import load_data, split_data
    from model.registry import get_model

    df = load_data("auto")
    _, test_df = split_data(df, stratify_bins=5)
    test_smiles = test_df["smiles"].tolist()

    model = get_model("chemberta")
    model.load()
    test_embeddings = model.embed(test_smiles)
    test_scores = ad_model.decision_function(test_embeddings)
    test_labels = ad_model.predict(test_embeddings) == 1

    logger.info("Test set: %d molecules", len(test_smiles))
    logger.info("Test in-domain: %d (%.1f%%)", test_labels.sum(), 100 * test_labels.mean())

    # --- Figure 1: PCA of embeddings colored by train/test and AD ---
    logger.info("Generating PCA plot...")
    pca = PCA(n_components=2, random_state=42)
    all_embeddings = np.vstack([train_embeddings, test_embeddings])
    all_pca = pca.fit_transform(all_embeddings)

    train_pca = all_pca[: len(train_embeddings)]
    test_pca = all_pca[len(train_embeddings) :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: train/test split
    ax = axes[0]
    ax.scatter(
        train_pca[:, 0],
        train_pca[:, 1],
        alpha=0.5,
        s=20,
        c="#1f77b4",
        label="Train",
        edgecolors="none",
    )
    ax.scatter(
        test_pca[:, 0],
        test_pca[:, 1],
        alpha=0.5,
        s=20,
        c="#ff7f0e",
        label="Test",
        edgecolors="none",
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("ChemBERTa CLS Embeddings (Train vs Test)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Right: AD labels
    ax = axes[1]
    train_in = train_pca[train_labels]
    train_ood = train_pca[~train_labels]
    test_in = test_pca[test_labels]
    test_ood = test_pca[~test_labels]

    ax.scatter(
        train_in[:, 0],
        train_in[:, 1],
        alpha=0.5,
        s=20,
        c="#1f77b4",
        label=f"Train in-domain ({train_labels.sum()})",
        edgecolors="none",
    )
    if len(train_ood) > 0:
        ax.scatter(
            train_ood[:, 0],
            train_ood[:, 1],
            alpha=0.6,
            s=30,
            c="#e74c3c",
            marker="^",
            label=f"Train OOD ({(~train_labels).sum()})",
            edgecolors="k",
            linewidths=0.5,
        )
    ax.scatter(
        test_in[:, 0],
        test_in[:, 1],
        alpha=0.5,
        s=20,
        c="#2ca02c",
        label=f"Test in-domain ({test_labels.sum()})",
        edgecolors="none",
    )
    if len(test_ood) > 0:
        ax.scatter(
            test_ood[:, 0],
            test_ood[:, 1],
            alpha=0.6,
            s=30,
            c="#d62728",
            marker="^",
            label=f"Test OOD ({(~test_labels).sum()})",
            edgecolors="k",
            linewidths=0.5,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("ChemBERTa AD Labels (Embedding Space)", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    pca_path = FIGURES_DIR / "chemberta_embedding_pca.png"
    plt.savefig(pca_path, dpi=150, bbox_inches="tight")
    logger.info("Saved PCA plot to %s", pca_path)
    plt.close()

    # --- Figure 2: Histogram of AD scores ---
    logger.info("Generating AD score histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        train_scores,
        bins=50,
        alpha=0.6,
        color="#1f77b4",
        label=f"Train (n={len(train_scores)})",
        edgecolor="k",
        linewidth=0.5,
    )
    ax.hist(
        test_scores,
        bins=50,
        alpha=0.6,
        color="#ff7f0e",
        label=f"Test (n={len(test_scores)})",
        edgecolor="k",
        linewidth=0.5,
    )

    # Show threshold (decision boundary is at 0 for IsolationForest)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="AD threshold (score=0)")

    ax.set_xlabel("AD Score (IsolationForest decision function)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("ChemBERTa AD Score Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # Add text box with stats
    train_ood_pct = 100 * (~train_labels).mean()
    test_ood_pct = 100 * (~test_labels).mean()
    stats_text = f"Train OOD: {train_ood_pct:.1f}%\nTest OOD: {test_ood_pct:.1f}%"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    hist_path = FIGURES_DIR / "chemberta_ad_score_histogram.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    logger.info("Saved AD score histogram to %s", hist_path)
    plt.close()

    # --- Figure 3: Per-target metrics split by AD ---
    logger.info("Generating per-target AD metrics plot...")
    from sklearn.metrics import mean_absolute_error, r2_score

    from model.data.load import TARGETS

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, target in enumerate(TARGETS):
        ax = axes[i]
        y_true = test_df[target].values
        preds = model.predict(test_smiles)
        y_pred = preds[target]

        # In-domain metrics
        in_mask = test_labels & np.isfinite(y_true) & np.isfinite(y_pred)
        ood_mask = ~test_labels & np.isfinite(y_true) & np.isfinite(y_pred)

        if in_mask.sum() >= 2:
            r2_in = r2_score(y_true[in_mask], y_pred[in_mask])
            mae_in = mean_absolute_error(y_true[in_mask], y_pred[in_mask])
        else:
            r2_in = mae_in = np.nan

        if ood_mask.sum() >= 2:
            r2_ood = r2_score(y_true[ood_mask], y_pred[ood_mask])
            mae_ood = mean_absolute_error(y_true[ood_mask], y_pred[ood_mask])
        else:
            r2_ood = mae_ood = np.nan

        # Plot parity
        if in_mask.any():
            ax.scatter(
                y_true[in_mask],
                y_pred[in_mask],
                alpha=0.6,
                s=30,
                c="#1f77b4",
                edgecolors="k",
                linewidths=0.3,
                label=f"In-domain (n={in_mask.sum()})",
            )
        if ood_mask.any():
            ax.scatter(
                y_true[ood_mask],
                y_pred[ood_mask],
                alpha=0.6,
                s=30,
                c="#e74c3c",
                marker="^",
                edgecolors="k",
                linewidths=0.3,
                label=f"OOD (n={ood_mask.sum()})",
            )

        # Parity line
        all_vals = np.concatenate([y_true[in_mask | ood_mask], y_pred[in_mask | ood_mask]])
        lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", lw=1, alpha=0.5)

        ax.set_xlabel(f"True {target}", fontsize=12)
        ax.set_ylabel(f"Predicted {target}", fontsize=12)
        title = (
            f"{target}\n"
            f"In: R²={r2_in:.3f}, MAE={mae_in:.3f}\n"
            f"OOD: R²={r2_ood:.3f}, MAE={mae_ood:.3f}"
        )
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    parity_path = FIGURES_DIR / "chemberta_ad_parity_by_target.png"
    plt.savefig(parity_path, dpi=150, bbox_inches="tight")
    logger.info("Saved per-target parity plot to %s", parity_path)
    plt.close()

    logger.info("Figure generation complete!")


if __name__ == "__main__":
    main()
