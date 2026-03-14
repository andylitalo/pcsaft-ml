"""Retrain the Random Forest model on merged data."""

from kfp import dsl


def retrain_model_logic(
    merged_csv: str,
    model_artifact: str,
    metrics_artifact: str,
) -> dict:
    """Core logic for retraining RF model.

    Parameters
    ----------
    merged_csv : str
        Path to merged training data CSV.
    model_artifact : str
        Path to save trained model artifact.
    metrics_artifact : str
        Path to save metrics JSON.

    Returns
    -------
    dict
        Training metrics.
    """
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    TARGETS = ["m", "sigma", "epsilon_k"]

    # Load data
    df = pd.read_csv(merged_csv)
    df = df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])

    # Build features (RDKit descriptors + Morgan fingerprints)
    def compute_descriptors(smiles_list):
        names = [name for name, _ in Descriptors.descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append([np.nan] * len(names))
            else:
                rows.append(list(calc.CalcDescriptors(mol)))
        return pd.DataFrame(rows, columns=names)

    def compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(np.zeros(n_bits, dtype=np.float32))
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(np.array(fp, dtype=np.float32))
        return np.stack(fps)

    def clean_descriptors(df):
        df = df.dropna(axis=1, how="all")
        variances = df.var(numeric_only=True)
        zero_var = variances[variances == 0].index.tolist()
        df = df.drop(columns=zero_var, errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan)
        if len(df.columns) > 1 and len(df) > 2:
            corr = df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            df = df.drop(columns=to_drop, errors="ignore")
        return df

    # Build features
    morgan_fps = compute_morgan_fingerprints(df["smiles"].tolist())
    desc_df = compute_descriptors(df["smiles"].tolist())
    desc_df = clean_descriptors(desc_df)
    desc_arr = desc_df.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    X = np.hstack([morgan_fps, desc_arr])

    # Split data
    train_idx, test_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42
    )
    X_train = X[train_idx]
    X_test = X[test_idx]

    # Train models
    models = {}
    metrics = {}

    for target in TARGETS:
        y = df[target].values
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        models[target] = model
        metrics[target] = {
            "r2_train": float(train_score),
            "r2_test": float(test_score),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

    # Save artifacts
    joblib.dump(models, model_artifact)
    with open(metrics_artifact, "w") as f:
        json.dump(metrics, f, indent=2)

    # Return aggregated metrics
    avg_r2_test = sum(m["r2_test"] for m in metrics.values()) / len(metrics)
    return {"avg_r2_test": avg_r2_test, "metrics": metrics}


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.1.0",
        "numpy==1.26.0",
        "rdkit==2023.9.1",
        "scikit-learn==1.3.0",
        "joblib==1.3.0",
    ],
)
def retrain_model(
    merged_csv: str,
    model_artifact: dsl.OutputPath(str),
    metrics_artifact: dsl.OutputPath(str),
) -> float:
    """Retrain the Random Forest model on merged data.

    Uses RDKit descriptors + Morgan fingerprints as features.
    Trains one RF model per PC-SAFT parameter (m, sigma, epsilon_k).

    Parameters
    ----------
    merged_csv : str
        Path to merged training data CSV.
    model_artifact : OutputPath(str)
        Path to save trained model artifact (joblib dict of RF models).
    metrics_artifact : OutputPath(str)
        Path to save training metrics JSON.

    Returns
    -------
    float
        Average test R² across all three parameters.
    """
    result = retrain_model_logic(merged_csv, model_artifact, metrics_artifact)
    return result["avg_r2_test"]
