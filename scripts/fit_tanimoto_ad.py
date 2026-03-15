"""Fit TanimotoAD model on training data and generate AD comparison figures."""
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from model.ad_tanimoto import TanimotoAD
from model.ad_williams import plot_williams
from model.data.load import load_data, split_data
from model.nn.ad import load_ad_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parent.parent / "model" / "saved"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "14_ad"


def fit_and_save_tanimoto_ad():
    """Fit TanimotoAD on training SMILES and save to model/saved/."""
    logger.info("Loading data...")
    df = load_data(source="auto")
    train_df, test_df = split_data(df, test_size=0.2, random_state=42)

    logger.info("Fitting TanimotoAD on %d training molecules...", len(train_df))
    tanimoto_ad = TanimotoAD(radius=2, n_bits=2048)
    tanimoto_ad.fit(train_df["smiles"].tolist())

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVED_DIR / "tanimoto_ad.joblib"
    joblib.dump(tanimoto_ad, save_path)
    logger.info("Saved TanimotoAD model to %s", save_path)

    return tanimoto_ad, train_df, test_df


def generate_ad_comparison(tanimoto_ad, train_df, test_df):
    """Generate AD method comparison scatter (IF score vs Tanimoto similarity)."""
    logger.info("Loading Isolation Forest AD model...")
    try:
        if_ad_model = load_ad_model()
    except FileNotFoundError:
        logger.warning("Isolation Forest AD model not found. Skipping AD comparison figure.")
        return

    logger.info("Loading saved feature names for consistency...")
    feature_names_path = SAVED_DIR / "feature_names.joblib"
    if not feature_names_path.exists():
        logger.warning("Feature names not found. Skipping AD comparison figure.")
        return
    saved_feature_names = joblib.load(feature_names_path)

    # Extract RDKit descriptor names from saved features (exclude Morgan FP names)
    rdkit_names = [n for n in saved_feature_names if not n.startswith("morgan_")]

    logger.info("Computing features for test set...")
    from model.data.descriptors import build_features_with_names

    X_train, _ = build_features_with_names(
        train_df["smiles"].tolist(), rdkit_names=rdkit_names
    )
    X_test, _ = build_features_with_names(test_df["smiles"].tolist(), rdkit_names=rdkit_names)

    # Remove molecules with invalid features
    valid_mask = np.isfinite(X_test).all(axis=1)
    test_smiles = test_df["smiles"].iloc[valid_mask].tolist()
    X_test_valid = X_test[valid_mask]

    logger.info("Computing AD metrics for %d test molecules...", len(test_smiles))

    # IF scores (decision_function returns anomaly scores; higher = more anomalous)
    if_scores = if_ad_model.decision_function(X_test_valid)

    # Tanimoto similarities
    tanimoto_scores = np.array([tanimoto_ad.tanimoto_nn(s) for s in test_smiles])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by agreement: both say in-domain, both say OOD, or disagreement
    if_in_domain = if_ad_model.predict(X_test_valid) == 1
    tanimoto_in_domain = tanimoto_scores >= TanimotoAD.IN_DOMAIN_THRESHOLD

    both_in = if_in_domain & tanimoto_in_domain
    both_out = (~if_in_domain) & (~tanimoto_in_domain)
    disagree = if_in_domain != tanimoto_in_domain

    ax.scatter(
        tanimoto_scores[both_in],
        if_scores[both_in],
        alpha=0.5,
        s=20,
        label="Both in-domain",
        color="green",
    )
    ax.scatter(
        tanimoto_scores[both_out],
        if_scores[both_out],
        alpha=0.5,
        s=20,
        label="Both out-of-domain",
        color="red",
    )
    ax.scatter(
        tanimoto_scores[disagree],
        if_scores[disagree],
        alpha=0.5,
        s=20,
        label="Disagreement",
        color="orange",
    )

    ax.axvline(
        TanimotoAD.IN_DOMAIN_THRESHOLD,
        color="blue",
        linestyle="--",
        label=f"Tanimoto threshold ({TanimotoAD.IN_DOMAIN_THRESHOLD})",
    )
    ax.axhline(0, color="purple", linestyle="--", label="IF threshold (0)")

    ax.set_xlabel("Tanimoto Similarity to Training Set")
    ax.set_ylabel("Isolation Forest Anomaly Score")
    ax.set_title("AD Method Comparison (Test Set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "ad_method_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved AD comparison plot to %s", output_path)

    # Print agreement statistics
    agreement_pct = (both_in.sum() + both_out.sum()) / len(test_smiles) * 100
    logger.info(
        "AD method agreement: %.1f%% (%d/%d molecules)",
        agreement_pct,
        both_in.sum() + both_out.sum(),
        len(test_smiles),
    )
    logger.info("  Both in-domain: %d", both_in.sum())
    logger.info("  Both out-of-domain: %d", both_out.sum())
    logger.info("  Disagreement: %d", disagree.sum())


def generate_williams_plots(train_df, test_df):
    """Generate Williams plots for all three targets."""
    logger.info("Loading RF models for Williams plots...")
    rf_models = {}
    for target in ["m", "sigma", "epsilon_k"]:
        model_path = SAVED_DIR / f"rf_{target}.joblib"
        if not model_path.exists():
            logger.warning("RF model not found: %s. Skipping Williams plots.", model_path)
            return
        rf_models[target] = joblib.load(model_path)

    logger.info("Loading saved feature names for consistency...")
    feature_names_path = SAVED_DIR / "feature_names.joblib"
    if not feature_names_path.exists():
        logger.warning("Feature names not found. Skipping Williams plots.")
        return
    saved_feature_names = joblib.load(feature_names_path)

    # Extract RDKit descriptor names
    rdkit_names = [n for n in saved_feature_names if not n.startswith("morgan_")]

    logger.info("Computing features...")
    from model.data.descriptors import build_features_with_names

    X_train, _ = build_features_with_names(
        train_df["smiles"].tolist(), rdkit_names=rdkit_names
    )
    X_test, _ = build_features_with_names(test_df["smiles"].tolist(), rdkit_names=rdkit_names)

    # Remove invalid features
    test_valid = np.isfinite(X_test).all(axis=1)
    X_test = X_test[test_valid]

    test_df_valid = test_df.iloc[test_valid]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for target in ["m", "sigma", "epsilon_k"]:
        logger.info("Generating Williams plot for %s...", target)
        y_test = test_df_valid[target].values
        y_pred = rf_models[target].predict(X_test)

        output_path = FIGURES_DIR / f"williams_{target}.png"
        plot_williams(
            X_train, X_test, y_test, y_pred, target_name=target, output_path=str(output_path)
        )
        logger.info("Saved Williams plot to %s", output_path)


def main():
    """Fit TanimotoAD and generate all AD-related figures."""
    tanimoto_ad, train_df, test_df = fit_and_save_tanimoto_ad()
    generate_ad_comparison(tanimoto_ad, train_df, test_df)
    generate_williams_plots(train_df, test_df)
    logger.info("Done!")


if __name__ == "__main__":
    main()
