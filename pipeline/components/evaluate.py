"""Evaluate the retrained model on held-out test set."""

from kfp import dsl


def evaluate_model_logic(
    model_artifact: str,
    test_csv: str,
    eval_metrics_artifact: str,
) -> dict:
    """Core logic for evaluating model on test set.

    Parameters
    ----------
    model_artifact : str
        Path to trained model artifact.
    test_csv : str
        Path to test set CSV.
    eval_metrics_artifact : str
        Path to save evaluation metrics JSON.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    TARGETS = ["m", "sigma", "epsilon_k"]

    # Load model and test data
    models = joblib.load(model_artifact)
    df = pd.read_csv(test_csv)
    df = df.dropna(subset=["smiles", "m", "sigma", "epsilon_k"])

    # Build features (same as training)
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

    morgan_fps = compute_morgan_fingerprints(df["smiles"].tolist())
    desc_df = compute_descriptors(df["smiles"].tolist())
    desc_df = clean_descriptors(desc_df)
    desc_arr = desc_df.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    X = np.hstack([morgan_fps, desc_arr])

    # Evaluate each target
    metrics = {}
    for target in TARGETS:
        y_true = df[target].values
        y_pred = models[target].predict(X)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        metrics[target] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "n_samples": len(y_true),
        }

    # Save metrics
    with open(eval_metrics_artifact, "w") as f:
        json.dump(metrics, f, indent=2)

    avg_r2 = sum(m["r2"] for m in metrics.values()) / len(metrics)
    return {"avg_r2": avg_r2, "metrics": metrics}


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
def evaluate_model(
    model_artifact: str,
    test_csv: str,
    eval_metrics_artifact: dsl.OutputPath(str),
) -> float:
    """Evaluate the retrained model on held-out test set.

    Computes MAE, RMSE, R² on test set for each parameter.

    Parameters
    ----------
    model_artifact : str
        Path to trained model artifact.
    test_csv : str
        Path to test set CSV.
    eval_metrics_artifact : OutputPath(str)
        Path to save evaluation metrics JSON.

    Returns
    -------
    float
        Average test R² across all three parameters.
    """
    result = evaluate_model_logic(model_artifact, test_csv, eval_metrics_artifact)
    return result["avg_r2"]
