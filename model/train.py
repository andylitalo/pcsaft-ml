"""Train Random Forest models to predict PC-SAFT parameters.

Usage:
    python -m model.train [--data esper|fallback] [--tune]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from model.data.descriptors import clean_descriptors, compute_descriptors
from model.data.load import TARGETS, load_data, split_data

SAVED_DIR = Path(__file__).parent / "saved"


def train(
    source: str = "auto", tune: bool = False
) -> dict[str, RandomForestRegressor]:
    """Train one Random Forest per PC-SAFT target parameter.

    Parameters
    ----------
    source : str
        Data source: "esper", "fallback", or "auto".
    tune : bool
        If True, run GridSearchCV for hyperparameter tuning.

    Returns
    -------
    dict[str, RandomForestRegressor]
        Mapping from target name to trained model.
    """
    print(f"Loading data (source={source})...")
    df = load_data(source)
    print(f"  Loaded {len(df)} molecules")

    train_df, test_df = split_data(df)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Compute descriptors
    print("Computing descriptors...")
    X_train_raw = compute_descriptors(train_df["smiles"].tolist())
    X_test_raw = compute_descriptors(test_df["smiles"].tolist())

    # Clean: fit on training data, apply same columns to test
    X_train = clean_descriptors(X_train_raw)
    feature_names = X_train.columns.tolist()
    X_test = X_test_raw[feature_names]

    # Drop rows with NaN descriptors
    train_mask = X_train.notna().all(axis=1)
    test_mask = X_test.notna().all(axis=1)
    X_train = X_train[train_mask]
    X_test = X_test[test_mask]
    train_df = train_df.iloc[train_mask.values]
    test_df = test_df.iloc[test_mask.values]

    print(f"  Features: {len(feature_names)}")
    print(f"  Usable train: {len(X_train)}, Usable test: {len(X_test)}")

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
        "filename": "test_set.csv",
        "type": "test_set",
        "n_molecules": len(test_df),
    })
    manifest = {
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    manifest_path = SAVED_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  Saved MANIFEST.json ({len(artifacts)} artifacts)")

    return models


def main():
    parser = argparse.ArgumentParser(description="Train PC-SAFT prediction models")
    parser.add_argument(
        "--data",
        choices=["esper", "fallback", "auto"],
        default="auto",
        help="Data source (default: auto)",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Run GridSearchCV hyperparameter tuning"
    )
    args = parser.parse_args()
    train(source=args.data, tune=args.tune)


if __name__ == "__main__":
    main()
