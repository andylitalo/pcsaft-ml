"""Train PC-SAFT prediction models (Random Forest or Neural Network).

Usage:
    python -m model.train [--model rf|nn] [--data ...] [--features ...] [--tune]
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from model.data.descriptors import build_features_with_names
from model.data.load import TARGETS, load_data, split_data

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent / "saved"


def _parse_feature_flags(features: str) -> tuple[bool, bool]:
    """Convert feature string to (use_morgan, use_rdkit) booleans."""
    if features == "morgan":
        return True, False
    elif features == "rdkit":
        return False, True
    elif features == "combined":
        return True, True
    else:
        raise ValueError(f"Unknown features option: {features!r}")


def train(
    source: str = "auto",
    tune: bool = False,
    features: str = "combined",
    stratify_bins: int = 5,
) -> dict[str, RandomForestRegressor]:
    """Train one Random Forest per PC-SAFT target parameter.

    Parameters
    ----------
    source : str
        Data source: "esper", "fallback", "combined", or "auto".
    tune : bool
        If True, run GridSearchCV for hyperparameter tuning.
    features : str
        Feature set: "rdkit", "morgan", or "combined".
    stratify_bins : int
        Number of epsilon_k bins for stratified splitting. Set to 0 to disable.

    Returns
    -------
    dict[str, RandomForestRegressor]
        Mapping from target name to trained model.
    """
    use_morgan, use_rdkit = _parse_feature_flags(features)

    print(f"Loading data (source={source})...")
    df = load_data(source)
    print(f"  Loaded {len(df)} molecules")

    train_df, test_df = split_data(df, stratify_bins=stratify_bins)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Compute features on all data together to ensure consistent RDKit
    # descriptor columns (clean_descriptors is data-dependent), then split
    print(f"Computing features (mode={features})...")
    all_smiles = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    X_all, feature_names = build_features_with_names(
        all_smiles,
        use_morgan=use_morgan,
        use_rdkit=use_rdkit,
    )
    n_train = len(train_df)
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]

    # build_features returns no NaN, but handle edge cases for safety
    train_mask = np.isfinite(X_train).all(axis=1)
    test_mask = np.isfinite(X_test).all(axis=1)
    X_train = X_train[train_mask]
    X_test = X_test[test_mask]
    train_df = train_df.iloc[train_mask]
    test_df = test_df.iloc[test_mask]

    print(f"  Features: {len(feature_names)}")
    print(f"  Usable train: {len(X_train)}, Usable test: {len(X_test)}")

    # Fit and save StandardScaler on the RDKit portion (for later NN use)
    scaler = None
    if use_rdkit:
        # RDKit features are the last portion of the feature vector
        morgan_cols = sum(1 for n in feature_names if n.startswith("morgan_"))
        rdkit_start = morgan_cols
        if rdkit_start < X_train.shape[1]:
            scaler = StandardScaler()
            scaler.fit(X_train[:, rdkit_start:])

    # Train models
    models = {}
    for target in TARGETS:
        print(f"\nTraining model for '{target}'...")
        y_train = train_df[target].values
        y_test = test_df[target].values

        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 20, 30],
                "min_samples_split": [2, 5],
            }
            gs = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            print(f"  Best params: {gs.best_params_}")
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

        # Quick score
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test) if len(X_test) > 0 else float("nan")
        print(f"  R² train={train_score:.4f}, test={test_score:.4f}")

        models[target] = model

    # Save models and feature names
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    for target, model in models.items():
        path = SAVED_DIR / f"rf_{target}.joblib"
        joblib.dump(model, path)
        print(f"  Saved {path}")

    joblib.dump(feature_names, SAVED_DIR / "feature_names.joblib")
    print(f"  Saved feature names ({len(feature_names)} features)")

    # Save feature config for downstream modules
    feature_config = {
        "features": features,
        "use_morgan": use_morgan,
        "use_rdkit": use_rdkit,
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "n_features": len(feature_names),
    }
    feature_config_path = SAVED_DIR / "feature_config.json"
    feature_config_path.write_text(json.dumps(feature_config, indent=2) + "\n")
    print("  Saved feature_config.json")

    # Save StandardScaler if fitted
    if scaler is not None:
        joblib.dump(scaler, SAVED_DIR / "rdkit_scaler.joblib")
        print("  Saved rdkit_scaler.joblib")

    # Save test set for evaluation
    test_df.to_csv(SAVED_DIR / "test_set.csv", index=False)
    print(f"  Saved test set ({len(test_df)} molecules)")

    # Write MANIFEST.json for artifact tracking
    artifacts = []
    for target in TARGETS:
        artifacts.append({
            "filename": f"rf_{target}.joblib",
            "type": "model",
            "model": "rf",
            "target": target,
        })
    artifacts.append({
        "filename": "feature_names.joblib",
        "type": "feature_names",
        "n_features": len(feature_names),
    })
    artifacts.append({
        "filename": "feature_config.json",
        "type": "feature_config",
        "features": features,
    })
    artifacts.append({
        "filename": "test_set.csv",
        "type": "test_set",
        "n_molecules": len(test_df),
    })
    if scaler is not None:
        artifacts.append({
            "filename": "rdkit_scaler.joblib",
            "type": "scaler",
            "scaler": "StandardScaler",
        })
    manifest = {
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "features": features,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    manifest_path = SAVED_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  Saved MANIFEST.json ({len(artifacts)} artifacts)")

    return models


def train_neural_network(
    source: str = "auto",
    features: str = "combined",
    stratify_bins: int = 5,
) -> None:
    """Train the PCSAFTNet multi-task neural network.

    Uses the same data loading and feature pipeline as the RF trainer,
    then delegates to ``model.nn.trainer.train_nn``.

    Parameters
    ----------
    source : str
        Data source: "esper", "fallback", "combined", or "auto".
    features : str
        Feature set: "rdkit", "morgan", or "combined".
    stratify_bins : int
        Number of epsilon_k bins for stratified splitting.
    """
    from model.nn.ad import train_ad_model
    from model.nn.trainer import train_nn

    use_morgan, use_rdkit = _parse_feature_flags(features)

    print(f"Loading data (source={source})...")
    df = load_data(source)
    print(f"  Loaded {len(df)} molecules")

    train_df, test_df = split_data(df, stratify_bins=stratify_bins)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Compute features on all data together for consistent columns
    print(f"Computing features (mode={features})...")
    all_smiles = train_df["smiles"].tolist() + test_df["smiles"].tolist()
    X_all, feature_names = build_features_with_names(
        all_smiles,
        use_morgan=use_morgan,
        use_rdkit=use_rdkit,
    )
    n_train = len(train_df)
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]

    # Handle non-finite values
    train_mask = np.isfinite(X_train).all(axis=1)
    test_mask = np.isfinite(X_test).all(axis=1)
    X_train = X_train[train_mask]
    X_test = X_test[test_mask]
    train_df = train_df.iloc[train_mask]
    test_df = test_df.iloc[test_mask]

    print(f"  Features: {len(feature_names)}")
    print(f"  Usable train: {len(X_train)}, Usable test: {len(X_test)}")

    # Save feature config and names (same as RF pipeline)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    feature_config = {
        "features": features,
        "use_morgan": use_morgan,
        "use_rdkit": use_rdkit,
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "n_features": len(feature_names),
    }
    feature_config_path = SAVED_DIR / "feature_config.json"
    feature_config_path.write_text(json.dumps(feature_config, indent=2) + "\n")
    print("  Saved feature_config.json")

    joblib.dump(feature_names, SAVED_DIR / "feature_names.joblib")
    print(f"  Saved feature names ({len(feature_names)} features)")

    # Save test set for evaluation
    test_df.to_csv(SAVED_DIR / "test_set.csv", index=False)
    print(f"  Saved test set ({len(test_df)} molecules)")

    # Prepare targets
    y_train = {target: train_df[target].values for target in TARGETS}

    # Train NN
    print("\nTraining PCSAFTNet multi-task neural network...")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_nn(X_train, y_train, feature_config)

    # Train AD model
    print("\nTraining Applicability Domain model (Isolation Forest)...")
    train_ad_model(X_train)

    print("\nNN training complete. Artifacts saved to model/saved/")


def main():
    parser = argparse.ArgumentParser(description="Train PC-SAFT prediction models")
    parser.add_argument(
        "--model",
        choices=["rf", "nn", "chemberta"],
        default="rf",
        help="Model type: rf (Random Forest), nn (Neural Network), or chemberta. Default: rf",
    )
    parser.add_argument(
        "--data",
        choices=["esper", "fallback", "combined", "auto"],
        default="auto",
        help="Data source (default: auto)",
    )
    parser.add_argument(
        "--features",
        choices=["rdkit", "morgan", "combined"],
        default="combined",
        help="Feature set (default: combined)",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Run GridSearchCV hyperparameter tuning"
    )
    args = parser.parse_args()

    if args.model == "chemberta":
        from model.hf.train_chemberta import train_chemberta

        train_chemberta(source=args.data)
    elif args.model == "nn":
        train_neural_network(
            source=args.data,
            features=args.features,
        )
    else:
        train(source=args.data, features=args.features, tune=args.tune)


if __name__ == "__main__":
    main()
