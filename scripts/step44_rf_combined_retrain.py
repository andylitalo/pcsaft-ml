"""Step 44: RF Combined Retrain (Esper + ML-SAFT) with uncertainty-aware comparison.

Controlled two-model RF experiment:
  - RF_esper: trained on Esper-only data
  - RF_combined: trained on Esper + ML-SAFT (InChI-deduplicated, Esper-priority)

Evaluated on:
  1. Esper benchmark test set (canonical holdout)
  2. Combined benchmark test set
  3. Esper subset of combined test
  4. ML-SAFT-only subset of combined test
  5. Optional fluorinated external validation set

Uncertainty metrics: tree-disagreement stds, 1-sigma/2-sigma coverage, calibration.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.data.descriptors import build_features, build_features_with_names
from model.data.load import TARGETS, load_data, split_data

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures" / "44_rf_combined_retrain"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SAVED_DIR = PROJECT_ROOT / "model" / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
RF_PARAMS = {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1}

# Font sizing conventions (from CLAUDE.md)
FONTSIZE_TITLE = 16
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_LEGEND = 11
DPI = 150

TARGET_DISPLAY = {
    "m": "m (segments)",
    "sigma": r"$\sigma$ ($\AA$)",
    "epsilon_k": r"$\varepsilon$/k (K)",
}


# ---------------------------------------------------------------------------
# Utility: compute InChI for a SMILES
# ---------------------------------------------------------------------------
def smiles_to_inchi(smi: str) -> str | None:
    """Convert SMILES to InChI string, returning None on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return MolToInchi(mol)


def add_inchi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an '_inchi' column to the DataFrame from the 'smiles' column."""
    df = df.copy()
    df["_inchi"] = [smiles_to_inchi(s) for s in df["smiles"]]
    return df


# ---------------------------------------------------------------------------
# Section 44.1: Verify dataset composition
# ---------------------------------------------------------------------------
def verify_datasets() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and verify dataset counts. Returns (df_esper, df_combined, provenance)."""
    df_esper = load_data("esper")
    df_combined = load_data("combined")

    # Load raw ML-SAFT to get raw count
    df_mlsaft_raw = load_data("mlsaft")

    # Compute overlap by InChI
    esper_inchis = set()
    for smi in df_esper["smiles"]:
        inchi = smiles_to_inchi(smi)
        if inchi:
            esper_inchis.add(inchi)

    mlsaft_inchis = set()
    for smi in df_mlsaft_raw["smiles"]:
        inchi = smiles_to_inchi(smi)
        if inchi:
            mlsaft_inchis.add(inchi)

    overlap_count = len(esper_inchis & mlsaft_inchis)
    net_new = len(df_combined) - len(df_esper)

    provenance = {
        "raw_esper": len(df_esper),
        "raw_mlsaft": len(df_mlsaft_raw),
        "deduplicated_combined": len(df_combined),
        "overlap_by_inchi": overlap_count,
        "net_new_mlsaft": net_new,
    }

    logger.info("=" * 60)
    logger.info("SECTION 44.1: Dataset Composition Verification")
    logger.info("=" * 60)
    logger.info("Esper: %d molecules", provenance["raw_esper"])
    logger.info("ML-SAFT (raw): %d molecules", provenance["raw_mlsaft"])
    logger.info("Combined (deduplicated): %d molecules", provenance["deduplicated_combined"])
    logger.info("Overlap by InChI: %d", provenance["overlap_by_inchi"])
    logger.info("Net new ML-SAFT molecules: %d", provenance["net_new_mlsaft"])

    return df_esper, df_combined, provenance


# ---------------------------------------------------------------------------
# Section 44.2: Freeze the split strategy
# ---------------------------------------------------------------------------
def create_splits(
    df_esper: pd.DataFrame, df_combined: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create canonical splits for both datasets.

    Critical: the Esper holdout (esper_test) must NOT appear in the combined
    training set, otherwise evaluating RF_combined on the Esper benchmark
    would suffer from data leakage. We remove Esper test molecules from
    the combined data before constructing the combined train/test split.

    Split strategy:
      1. esper_train, esper_test = standard 80/20 on Esper (canonical holdout)
      2. combined_train = combined minus esper_test molecules
      3. combined_test  = independent 80/20 split of combined (for combined
         benchmark and ML-SAFT-only slice analysis)
    """
    # Canonical Esper split
    esper_train, esper_test = split_data(
        df_esper, test_size=0.2, random_state=RANDOM_STATE, stratify_bins=5
    )

    # Build InChI set for Esper test molecules
    esper_test_inchis = set()
    for smi in esper_test["smiles"]:
        inchi = smiles_to_inchi(smi)
        if inchi:
            esper_test_inchis.add(inchi)

    # Strategy: use the independent 80/20 combined split to determine which
    # molecules are held out for the combined benchmark and ML-SAFT-only
    # slice analysis. Then build the combined training set by excluding both
    # the esper_test and the combined_test molecules, so there is zero
    # leakage for either benchmark.
    _, combined_test = split_data(
        df_combined, test_size=0.2, random_state=RANDOM_STATE, stratify_bins=5
    )

    # Build combined_train as: combined - esper_test - combined_test
    esper_test_smiles = set(esper_test["smiles"].values)
    combined_test_smiles = set(combined_test["smiles"].values)
    holdout_smiles = esper_test_smiles | combined_test_smiles

    combined_train = df_combined[
        ~df_combined["smiles"].isin(holdout_smiles)
    ].copy()

    # Verify no leakage
    train_smiles = set(combined_train["smiles"].values)
    leaked_esper = train_smiles & esper_test_smiles
    leaked_combined = train_smiles & combined_test_smiles
    if leaked_esper or leaked_combined:
        logger.error(
            "LEAKAGE detected: %d from esper_test, %d from combined_test",
            len(leaked_esper),
            len(leaked_combined),
        )

    logger.info("=" * 60)
    logger.info("SECTION 44.2: Split Strategy")
    logger.info("=" * 60)
    logger.info("Esper train: %d, Esper test: %d", len(esper_train), len(esper_test))
    logger.info(
        "Combined train (excl. Esper holdout): %d, Combined test: %d",
        len(combined_train),
        len(combined_test),
    )
    logger.info(
        "Molecules held out from combined train: %d (esper_test + combined_test)",
        len(df_combined) - len(combined_train),
    )

    return esper_train, esper_test, combined_train, combined_test


def slice_combined_test(
    combined_test: pd.DataFrame, esper_inchi_set: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice combined test into Esper-subset and ML-SAFT-only subset by InChI."""
    combined_test = add_inchi_column(combined_test)
    esper_mask = combined_test["_inchi"].isin(esper_inchi_set)
    esper_subset = combined_test[esper_mask].drop(columns=["_inchi"])
    mlsaft_subset = combined_test[~esper_mask].drop(columns=["_inchi"])

    logger.info(
        "Combined test sliced: Esper-subset=%d, ML-SAFT-only=%d",
        len(esper_subset),
        len(mlsaft_subset),
    )
    return esper_subset, mlsaft_subset


# ---------------------------------------------------------------------------
# Section 44.3: Train the paired RF models
# ---------------------------------------------------------------------------
def train_paired_rfs(
    esper_train: pd.DataFrame,
    combined_train: pd.DataFrame,
) -> dict:
    """Train Esper-only and Combined RF for each target.

    Returns
    -------
    dict with structure:
        {
            "esper": {target: model},
            "combined": {target: model},
            "esper_rdkit_names": [...],
            "combined_rdkit_names": [...],
            "X_train_esper": ndarray,
            "X_train_combined": ndarray,
        }
    """
    logger.info("=" * 60)
    logger.info("SECTION 44.3: Training Paired RF Models")
    logger.info("=" * 60)

    # Build features
    logger.info("Building features for Esper train (%d molecules)...", len(esper_train))
    X_train_esper, esper_feat_names = build_features_with_names(
        esper_train["smiles"].tolist()
    )
    esper_rdkit_names = [n for n in esper_feat_names if not n.startswith("morgan_")]

    logger.info("Building features for combined train (%d molecules)...", len(combined_train))
    X_train_combined, combined_feat_names = build_features_with_names(
        combined_train["smiles"].tolist()
    )
    combined_rdkit_names = [n for n in combined_feat_names if not n.startswith("morgan_")]

    logger.info(
        "Feature dims: Esper train %s, Combined train %s",
        X_train_esper.shape,
        X_train_combined.shape,
    )

    result = {
        "esper": {},
        "combined": {},
        "esper_rdkit_names": esper_rdkit_names,
        "combined_rdkit_names": combined_rdkit_names,
        "X_train_esper": X_train_esper,
        "X_train_combined": X_train_combined,
    }

    for target in TARGETS:
        y_esper = esper_train[target].values
        y_combined = combined_train[target].values

        rf_esper = RandomForestRegressor(**RF_PARAMS)
        rf_esper.fit(X_train_esper, y_esper)
        result["esper"][target] = rf_esper

        rf_combined = RandomForestRegressor(**RF_PARAMS)
        rf_combined.fit(X_train_combined, y_combined)
        result["combined"][target] = rf_combined

        logger.info("Trained %s: RF_esper and RF_combined", target)

    return result


# ---------------------------------------------------------------------------
# Section 44.4 / 44.5: Prediction and metrics computation
# ---------------------------------------------------------------------------
def predict_with_uncertainty(
    model: RandomForestRegressor, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Predict and compute tree-disagreement standard deviation."""
    pred = model.predict(X)
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    std = tree_preds.std(axis=0)
    return pred, std


def compute_coverage(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict:
    """Compute 1-sigma and 2-sigma empirical coverage."""
    abs_errors = np.abs(y_true - y_pred)
    cov_1sigma = float(np.mean(abs_errors <= y_std))
    cov_2sigma = float(np.mean(abs_errors <= 2 * y_std))
    return {"coverage_1sigma": cov_1sigma, "coverage_2sigma": cov_2sigma}


def compute_slice_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray
) -> dict:
    """Compute point metrics + uncertainty metrics for a single target/slice."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mean_std = float(np.mean(y_std))
    mean_abs_error = float(np.mean(np.abs(y_true - y_pred)))
    std_error_ratio = mean_std / mean_abs_error if mean_abs_error > 0 else float("inf")
    cov = compute_coverage(y_true, y_pred, y_std)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_std": mean_std,
        "mean_abs_error": mean_abs_error,
        "std_error_ratio": float(std_error_ratio),
        "coverage_1sigma": cov["coverage_1sigma"],
        "coverage_2sigma": cov["coverage_2sigma"],
        "n_samples": len(y_true),
    }


def _eval_single_slice(
    rf_models: dict,
    model_name: str,
    slice_df: pd.DataFrame,
    slice_name: str,
    train_n: int,
    fluor_target_map: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate one model on one slice for all targets.

    Returns (slice_metrics_dict, prediction_rows_list).
    """
    rdkit_key = f"{model_name}_rdkit_names"
    rdkit_names = rf_models[rdkit_key]
    smiles_list = slice_df["smiles"].tolist()
    X = build_features(smiles_list, rdkit_names=rdkit_names)

    metrics_dict = {}
    pred_rows = []

    for target in TARGETS:
        model = rf_models[model_name][target]
        pred, std = predict_with_uncertainty(model, X)

        # Ground truth
        if fluor_target_map and slice_name == "fluorinated_external":
            y_col = fluor_target_map[target]
            if y_col not in slice_df.columns:
                logger.warning("Column %s not in fluorinated set; skipping", y_col)
                continue
            y_true = slice_df[y_col].values
        else:
            y_true = slice_df[target].values

        metrics = compute_slice_metrics(y_true, pred, std)
        metrics["train_samples"] = train_n
        metrics_dict[target] = (metrics, pred, std, y_true)

        logger.info(
            "%s | %s | %s: R2=%.4f MAE=%.4f 1sig=%.3f 2sig=%.3f (n=%d)",
            model_name,
            slice_name,
            target,
            metrics["r2"],
            metrics["mae"],
            metrics["coverage_1sigma"],
            metrics["coverage_2sigma"],
            len(y_true),
        )

    # Build prediction rows
    for i, smi in enumerate(smiles_list):
        row = {
            "smiles": smi,
            "inchi": smiles_to_inchi(smi) or "",
            "eval_set": slice_name,
            "model_name": f"rf_{model_name}",
        }
        if slice_name == "fluorinated_external":
            row["eval_source"] = "fluorinated"
        elif slice_name == "esper_benchmark":
            row["eval_source"] = "esper"
        elif slice_name == "combined_mlsaft_only":
            row["eval_source"] = "mlsaft_only"
        elif slice_name == "combined_esper_subset":
            row["eval_source"] = "esper"
        else:
            row["eval_source"] = "combined"

        for target in TARGETS:
            if target not in metrics_dict:
                continue
            _, pred, std, y_true = metrics_dict[target]
            row[f"{target}_true"] = float(y_true[i])
            row[f"{target}_pred"] = float(pred[i])
            row[f"{target}_std"] = float(std[i])

        pred_rows.append(row)

    return metrics_dict, pred_rows


def evaluate_all_slices(
    rf_models: dict,
    esper_train: pd.DataFrame,
    esper_test: pd.DataFrame,
    combined_train: pd.DataFrame,
    combined_test: pd.DataFrame,
    combined_test_esper_subset: pd.DataFrame,
    combined_test_mlsaft_subset: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate both RFs on all required slices.

    For slices derived from the combined test set, RF_esper is evaluated
    only on molecules NOT in esper_train (to avoid train/test leakage).

    Returns (all_metrics, predictions_df).
    """
    logger.info("=" * 60)
    logger.info("SECTION 44.4/44.5: Evaluation and Uncertainty Comparison")
    logger.info("=" * 60)

    # Optional fluorinated validation
    fluor_path = SAVED_DIR / "gnn_fluorinated_validation_set.csv"
    fluor_df = None
    fluor_target_map = {"m": "m_lit", "sigma": "sigma_lit", "epsilon_k": "epsilon_k_lit"}
    if fluor_path.exists():
        fluor_df = pd.read_csv(fluor_path)
        logger.info("Fluorinated validation set loaded: %d compounds", len(fluor_df))

    # Build esper_train SMILES set for leakage filtering
    esper_train_smiles = set(esper_train["smiles"].values)

    all_metrics = {}
    prediction_rows = []

    # Define evaluation plan: (slice_name, slice_df)
    eval_plan = [
        ("esper_benchmark", esper_test),
        ("combined_benchmark", combined_test),
        ("combined_esper_subset", combined_test_esper_subset),
        ("combined_mlsaft_only", combined_test_mlsaft_subset),
    ]
    if fluor_df is not None:
        eval_plan.append(("fluorinated_external", fluor_df))

    for model_name in ["esper", "combined"]:
        train_n = len(esper_train) if model_name == "esper" else len(combined_train)

        for slice_name, slice_df in eval_plan:
            if len(slice_df) == 0:
                logger.warning(
                    "Skipping empty slice: %s for %s", slice_name, model_name
                )
                continue

            # For RF_esper on combined-derived slices, filter out molecules
            # that are in esper_train to prevent leakage
            eval_df = slice_df
            if model_name == "esper" and slice_name in (
                "combined_benchmark",
                "combined_esper_subset",
                "combined_mlsaft_only",
            ):
                eval_df = slice_df[
                    ~slice_df["smiles"].isin(esper_train_smiles)
                ].copy()
                if len(eval_df) < len(slice_df):
                    logger.info(
                        "RF_esper on %s: filtered %d -> %d molecules (removed "
                        "esper_train overlaps)",
                        slice_name,
                        len(slice_df),
                        len(eval_df),
                    )
                if len(eval_df) < 5:
                    logger.warning(
                        "Too few molecules (%d) for %s/%s after filtering; skipping",
                        len(eval_df),
                        model_name,
                        slice_name,
                    )
                    continue

            ftm = fluor_target_map if slice_name == "fluorinated_external" else None
            metrics_dict, pred_rows = _eval_single_slice(
                rf_models, model_name, eval_df, slice_name, train_n, ftm
            )

            for target, (m, _, _, _) in metrics_dict.items():
                all_metrics[(model_name, slice_name, target)] = m

            prediction_rows.extend(pred_rows)

    predictions_df = pd.DataFrame(prediction_rows)
    return all_metrics, predictions_df


# ---------------------------------------------------------------------------
# Section 44.6: Figures
# ---------------------------------------------------------------------------
def plot_parity_rf_esper_vs_combined(
    rf_models: dict,
    esper_test: pd.DataFrame,
) -> None:
    """Figure 1: 2x3 parity plot (top: Esper RF, bottom: Combined RF) on Esper benchmark."""
    esper_rdkit_names = rf_models["esper_rdkit_names"]
    combined_rdkit_names = rf_models["combined_rdkit_names"]

    smiles_list = esper_test["smiles"].tolist()
    X_esper = build_features(smiles_list, rdkit_names=esper_rdkit_names)
    X_combined = build_features(smiles_list, rdkit_names=combined_rdkit_names)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col_idx, target in enumerate(TARGETS):
        y_true = esper_test[target].values

        for row_idx, (model_name, X, color) in enumerate([
            ("esper", X_esper, "steelblue"),
            ("combined", X_combined, "darkorange"),
        ]):
            ax = axes[row_idx, col_idx]
            model = rf_models[model_name][target]
            pred, std = predict_with_uncertainty(model, X)
            r2 = r2_score(y_true, pred)
            mae = mean_absolute_error(y_true, pred)

            ax.errorbar(
                y_true, pred, yerr=std,
                fmt="o", alpha=0.3, markersize=3, elinewidth=0.5,
                color=color, ecolor="lightgray",
            )

            lo = min(y_true.min(), pred.min())
            hi = max(y_true.max(), pred.max())
            margin = 0.05 * (hi - lo)
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", lw=1)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

            ax.set_xlabel(f"True {TARGET_DISPLAY[target]}", fontsize=FONTSIZE_LABEL)
            ax.set_ylabel(f"Predicted {TARGET_DISPLAY[target]}", fontsize=FONTSIZE_LABEL)
            variant_label = f"RF_{model_name}"
            ax.set_title(
                f"{variant_label}: {TARGET_DISPLAY[target]}", fontsize=FONTSIZE_TITLE
            )
            ax.tick_params(labelsize=FONTSIZE_TICK)

            textbox = f"R$^2$ = {r2:.3f}\nMAE = {mae:.3f}"
            ax.text(
                0.05, 0.95, textbox, transform=ax.transAxes,
                fontsize=FONTSIZE_LEGEND, verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

    fig.tight_layout()
    path = FIGURES_DIR / "parity_rf_esper_vs_combined.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_slice_metrics_comparison(all_metrics: dict) -> None:
    """Figure 2: Dot plot comparing MAE/RMSE/R2 across slices for both RFs."""
    metric_keys = ["mae", "rmse", "r2"]
    metric_labels = ["MAE", "RMSE", "R$^2$"]
    slice_order = [
        "esper_benchmark",
        "combined_benchmark",
        "combined_esper_subset",
        "combined_mlsaft_only",
    ]
    slice_display = {
        "esper_benchmark": "Esper\nbenchmark",
        "combined_benchmark": "Combined\nbenchmark",
        "combined_esper_subset": "Combined\n(Esper subset)",
        "combined_mlsaft_only": "Combined\n(ML-SAFT only)",
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    for row_idx, target in enumerate(TARGETS):
        for col_idx, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[row_idx, col_idx]

            for model_idx, (model_name, color, marker) in enumerate([
                ("esper", "steelblue", "o"),
                ("combined", "darkorange", "s"),
            ]):
                vals = []
                positions = []
                for s_idx, sn in enumerate(slice_order):
                    key = (model_name, sn, target)
                    if key in all_metrics:
                        vals.append(all_metrics[key][mk])
                        positions.append(s_idx + model_idx * 0.2 - 0.1)

                ax.scatter(
                    positions, vals, color=color, marker=marker, s=80, zorder=5,
                    label=f"RF_{model_name}" if row_idx == 0 and col_idx == 0 else None,
                )
                # Connect dots with lines for readability
                if len(vals) > 1:
                    ax.plot(positions, vals, color=color, alpha=0.3, lw=1)

            ax.set_xticks(range(len(slice_order)))
            ax.set_xticklabels(
                [slice_display.get(s, s) for s in slice_order],
                fontsize=FONTSIZE_TICK - 1,
            )
            ax.set_ylabel(ml, fontsize=FONTSIZE_LABEL)
            title_suffix = " **" if target == "epsilon_k" else ""
            ax.set_title(
                f"{TARGET_DISPLAY[target]} - {ml}{title_suffix}",
                fontsize=FONTSIZE_TITLE - 1,
            )
            ax.tick_params(labelsize=FONTSIZE_TICK)
            ax.grid(axis="y", alpha=0.3)

    # Add legend to top-left subplot
    axes[0, 0].legend(fontsize=FONTSIZE_LEGEND, loc="best")

    fig.tight_layout()
    path = FIGURES_DIR / "slice_metrics_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_uncertainty_calibration(
    rf_models: dict,
    esper_test: pd.DataFrame,
    combined_test_mlsaft_subset: pd.DataFrame,
) -> None:
    """Figure 3: Uncertainty calibration (mean predicted std vs mean absolute error)."""
    esper_rdkit_names = rf_models["esper_rdkit_names"]
    combined_rdkit_names = rf_models["combined_rdkit_names"]

    # Prepare data for two slices
    slices_to_plot = [
        ("Esper benchmark", esper_test),
    ]
    if len(combined_test_mlsaft_subset) >= 10:
        slices_to_plot.append(("ML-SAFT-only", combined_test_mlsaft_subset))

    n_slices = len(slices_to_plot)
    fig, axes = plt.subplots(n_slices, 3, figsize=(15, 5 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for s_idx, (slice_label, slice_df) in enumerate(slices_to_plot):
        smiles_list = slice_df["smiles"].tolist()
        X_esper = build_features(smiles_list, rdkit_names=esper_rdkit_names)
        X_combined = build_features(smiles_list, rdkit_names=combined_rdkit_names)

        for col_idx, target in enumerate(TARGETS):
            ax = axes[s_idx, col_idx]
            y_true = slice_df[target].values

            for model_name, X, color, marker in [
                ("esper", X_esper, "steelblue", "o"),
                ("combined", X_combined, "darkorange", "s"),
            ]:
                model = rf_models[model_name][target]
                pred, std = predict_with_uncertainty(model, X)
                errors = np.abs(y_true - pred)

                n_bins = min(5, max(3, len(y_true) // 10))
                try:
                    bins = pd.qcut(std, q=n_bins, duplicates="drop")
                    bin_centers = []
                    bin_errors = []
                    for bin_label in sorted(bins.unique()):
                        mask = bins == bin_label
                        bin_centers.append(np.mean(std[mask]))
                        bin_errors.append(np.mean(errors[mask]))

                    ax.plot(
                        bin_centers, bin_errors, f"-{marker}",
                        color=color, label=f"RF_{model_name}", markersize=6,
                    )
                except ValueError:
                    logger.warning(
                        "Could not bin uncertainty for %s/%s/%s",
                        slice_label, target, model_name,
                    )

            # Perfect calibration line
            all_vals = []
            for mn in ["esper", "combined"]:
                X_sel = X_esper if mn == "esper" else X_combined
                _, s = predict_with_uncertainty(rf_models[mn][target], X_sel)
                all_vals.extend(s.tolist())
            if all_vals:
                lo_v, hi_v = min(all_vals), max(all_vals)
                ax.plot(
                    [lo_v, hi_v], [lo_v, hi_v], "k--", lw=1, alpha=0.5,
                    label="Perfect calibration",
                )

            ax.set_xlabel("Mean predicted std", fontsize=FONTSIZE_LABEL)
            ax.set_ylabel("Mean absolute error", fontsize=FONTSIZE_LABEL)
            ax.set_title(
                f"{slice_label}: {TARGET_DISPLAY[target]}",
                fontsize=FONTSIZE_TITLE - 1,
            )
            ax.legend(fontsize=FONTSIZE_LEGEND - 1)
            ax.tick_params(labelsize=FONTSIZE_TICK)

    fig.tight_layout()
    path = FIGURES_DIR / "uncertainty_calibration_rf_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_coverage_comparison(all_metrics: dict) -> None:
    """Figure 4: 1-sigma and 2-sigma coverage by target and slice for both RFs."""
    slice_order = [
        "esper_benchmark",
        "combined_benchmark",
        "combined_esper_subset",
        "combined_mlsaft_only",
    ]
    slice_display = {
        "esper_benchmark": "Esper bench.",
        "combined_benchmark": "Combined bench.",
        "combined_esper_subset": "Comb. (Esper)",
        "combined_mlsaft_only": "Comb. (ML-SAFT)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col_idx, target in enumerate(TARGETS):
        for row_idx, (cov_key, cov_label, ideal) in enumerate([
            ("coverage_1sigma", "1-sigma coverage", 0.683),
            ("coverage_2sigma", "2-sigma coverage", 0.954),
        ]):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(slice_order))
            width = 0.35

            vals_esper = []
            vals_combined = []
            labels = []
            for sn in slice_order:
                labels.append(slice_display.get(sn, sn))
                key_e = ("esper", sn, target)
                key_c = ("combined", sn, target)
                vals_esper.append(
                    all_metrics[key_e][cov_key] if key_e in all_metrics else 0
                )
                vals_combined.append(
                    all_metrics[key_c][cov_key] if key_c in all_metrics else 0
                )

            ax.bar(
                x - width / 2, vals_esper, width,
                label="RF_esper", color="steelblue", alpha=0.7, edgecolor="black",
            )
            ax.bar(
                x + width / 2, vals_combined, width,
                label="RF_combined", color="darkorange", alpha=0.7, edgecolor="black",
            )
            ax.axhline(
                ideal, color="red", linestyle="--", lw=1, alpha=0.6,
                label=f"Ideal ({ideal:.1%})",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=FONTSIZE_TICK - 2, rotation=15)
            ax.set_ylabel(cov_label, fontsize=FONTSIZE_LABEL)
            ax.set_title(
                f"{TARGET_DISPLAY[target]}", fontsize=FONTSIZE_TITLE - 1
            )
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=FONTSIZE_TICK)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=FONTSIZE_LEGEND - 1, loc="lower right")

    fig.tight_layout()
    path = FIGURES_DIR / "coverage_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_combined_test_source_breakdown(
    combined_test: pd.DataFrame, esper_inchi_set: set[str]
) -> None:
    """Figure 5: Composition of the combined test set by source."""
    combined_test_inchi = add_inchi_column(combined_test)
    esper_mask = combined_test_inchi["_inchi"].isin(esper_inchi_set)
    n_esper = int(esper_mask.sum())
    n_mlsaft = len(combined_test) - n_esper

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bars = ax.bar(
        ["Esper-derived", "ML-SAFT-only"],
        [n_esper, n_mlsaft],
        color=["steelblue", "darkorange"],
        edgecolor="black",
        alpha=0.8,
    )

    # Add count labels on bars
    for bar, count in zip(bars, [n_esper, n_mlsaft]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(count), ha="center", va="bottom", fontsize=FONTSIZE_LABEL,
            fontweight="bold",
        )

    total = n_esper + n_mlsaft
    ax.set_ylabel("Number of molecules", fontsize=FONTSIZE_LABEL)
    ax.set_title(
        f"Combined Test Set Composition (n={total})", fontsize=FONTSIZE_TITLE
    )
    ax.tick_params(labelsize=FONTSIZE_TICK)

    # Add percentage annotations
    ax.text(
        0, n_esper / 2, f"{100 * n_esper / total:.1f}%",
        ha="center", va="center", fontsize=FONTSIZE_LABEL,
        color="white", fontweight="bold",
    )
    ax.text(
        1, n_mlsaft / 2, f"{100 * n_mlsaft / total:.1f}%",
        ha="center", va="center", fontsize=FONTSIZE_LABEL,
        color="white", fontweight="bold",
    )

    fig.tight_layout()
    path = FIGURES_DIR / "combined_test_source_breakdown.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Section 44.7: Supporting overlap diagnostics
# ---------------------------------------------------------------------------
def overlap_diagnostics(
    df_esper: pd.DataFrame, all_metrics: dict
) -> dict | None:
    """If combined RF degrades on Esper benchmark, compute supporting diagnostics."""
    # Check if combined RF degrades on Esper benchmark for any target
    any_degradation = False
    for target in TARGETS:
        key_e = ("esper", "esper_benchmark", target)
        key_c = ("combined", "esper_benchmark", target)
        if key_e in all_metrics and key_c in all_metrics:
            if all_metrics[key_c]["r2"] < all_metrics[key_e]["r2"] - 0.02:
                any_degradation = True
                break

    if not any_degradation:
        logger.info("No meaningful degradation detected; skipping overlap diagnostics.")
        return None

    logger.info("=" * 60)
    logger.info("SECTION 44.7: Supporting Overlap Diagnostics")
    logger.info("=" * 60)

    # Load raw datasets
    df_mlsaft = load_data("mlsaft")

    # Build InChI maps
    esper_map = {}
    for _, row in df_esper.iterrows():
        inchi = smiles_to_inchi(row["smiles"])
        if inchi:
            esper_map[inchi] = {t: row[t] for t in TARGETS}

    mlsaft_map = {}
    for _, row in df_mlsaft.iterrows():
        inchi = smiles_to_inchi(row["smiles"])
        if inchi:
            mlsaft_map[inchi] = {t: row[t] for t in TARGETS}

    overlap_inchis = set(esper_map.keys()) & set(mlsaft_map.keys())
    logger.info("Overlap molecules by InChI: %d", len(overlap_inchis))

    deltas = {t: [] for t in TARGETS}
    for inchi in overlap_inchis:
        for t in TARGETS:
            delta = mlsaft_map[inchi][t] - esper_map[inchi][t]
            deltas[t].append(delta)

    diagnostics = {"overlap_count": len(overlap_inchis)}
    for t in TARGETS:
        arr = np.array(deltas[t])
        diagnostics[f"{t}_mean_delta"] = float(np.mean(arr))
        diagnostics[f"{t}_std_delta"] = float(np.std(arr))
        diagnostics[f"{t}_mean_abs_delta"] = float(np.mean(np.abs(arr)))
        logger.info(
            "Overlap %s: mean delta=%.4f, std=%.4f, mean |delta|=%.4f",
            t, diagnostics[f"{t}_mean_delta"],
            diagnostics[f"{t}_std_delta"],
            diagnostics[f"{t}_mean_abs_delta"],
        )

    return diagnostics


# ---------------------------------------------------------------------------
# Section 44.8: Save artifacts
# ---------------------------------------------------------------------------
def save_predictions(predictions_df: pd.DataFrame) -> None:
    """Save per-molecule predictions CSV."""
    path = SAVED_DIR / "step44_rf_combined_predictions.csv"
    predictions_df.to_csv(path, index=False)
    logger.info("Saved predictions: %s (%d rows)", path, len(predictions_df))


def save_metrics(all_metrics: dict, provenance: dict, overlap_diag: dict | None) -> None:
    """Save aggregate metrics JSON."""
    # Convert tuple keys to string keys for JSON
    metrics_out = {}
    for (model_name, slice_name, target), m in all_metrics.items():
        key = f"{model_name}__{slice_name}__{target}"
        metrics_out[key] = m

    output = {
        "provenance": provenance,
        "metrics": metrics_out,
    }
    if overlap_diag:
        output["overlap_diagnostics"] = overlap_diag

    path = SAVED_DIR / "step44_rf_combined_metrics.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved metrics: %s", path)


# ---------------------------------------------------------------------------
# Section 44.9: Report generation
# ---------------------------------------------------------------------------
def generate_report(
    all_metrics: dict,
    provenance: dict,
    esper_train: pd.DataFrame,
    esper_test: pd.DataFrame,
    combined_train: pd.DataFrame,
    combined_test: pd.DataFrame,
    combined_test_esper_subset: pd.DataFrame,
    combined_test_mlsaft_subset: pd.DataFrame,
    overlap_diag: dict | None,
) -> None:
    """Generate the Step 44 report."""
    lines = []
    lines.append("# Results Report: RF Combined Retrain (Esper + ML-SAFT)")
    lines.append("")

    # --- Model Performance ---
    lines.append("## Model Performance")
    lines.append("")
    lines.append(
        "Two Random Forest models were trained under identical conditions "
        "(100 trees, Morgan FP + RDKit descriptors, random_state=42) "
        "and evaluated across multiple benchmark slices. "
        "The only experimental variable is the training corpus:"
    )
    lines.append("")
    lines.append(
        f"- **RF_esper**: trained on {len(esper_train)} Esper-only molecules"
    )
    lines.append(
        f"- **RF_combined**: trained on {len(combined_train)} molecules "
        f"(Esper + ML-SAFT, InChI-deduplicated, Esper priority)"
    )
    lines.append("")

    # Provenance table
    lines.append("### Data Provenance")
    lines.append("")
    lines.append("| Quantity | Count |")
    lines.append("|---|---|")
    lines.append(f"| Raw Esper | {provenance['raw_esper']} |")
    lines.append(f"| Raw ML-SAFT | {provenance['raw_mlsaft']} |")
    lines.append(f"| Deduplicated combined | {provenance['deduplicated_combined']} |")
    lines.append(f"| Overlap (by InChI) | {provenance['overlap_by_inchi']} |")
    lines.append(f"| Net new ML-SAFT molecules | {provenance['net_new_mlsaft']} |")
    lines.append("")

    # --- Primary comparison: Esper benchmark ---
    lines.append("## Comparison to Baseline: Canonical Esper Holdout")
    lines.append("")
    lines.append(
        f"Primary apples-to-apples comparison. Both models evaluated on the "
        f"same {len(esper_test)} Esper holdout molecules."
    )
    lines.append("")
    lines.append(
        "| Target | Model | MAE | RMSE | R2 | Train n | Test n |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for target in TARGETS:
        for model_name in ["esper", "combined"]:
            key = (model_name, "esper_benchmark", target)
            m = all_metrics[key]
            lines.append(
                f"| {TARGET_DISPLAY[target]} | RF_{model_name} | "
                f"{m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} | "
                f"{m['train_samples']} | {m['n_samples']} |"
            )
    lines.append("")

    # Delta summary
    lines.append("### Esper Benchmark Delta (Combined - Esper)")
    lines.append("")
    lines.append("| Target | Delta MAE | Delta RMSE | Delta R2 |")
    lines.append("|---|---|---|---|")
    for target in TARGETS:
        key_e = ("esper", "esper_benchmark", target)
        key_c = ("combined", "esper_benchmark", target)
        me = all_metrics[key_e]
        mc = all_metrics[key_c]
        lines.append(
            f"| {TARGET_DISPLAY[target]} | "
            f"{mc['mae'] - me['mae']:+.4f} | "
            f"{mc['rmse'] - me['rmse']:+.4f} | "
            f"{mc['r2'] - me['r2']:+.4f} |"
        )
    lines.append("")

    # --- Uncertainty Quality ---
    lines.append("## Uncertainty Quality")
    lines.append("")
    lines.append(
        "RF tree-disagreement standard deviations used as uncertainty estimates. "
        "Coverage = fraction of test molecules where |error| <= k * predicted_std."
    )
    lines.append("")
    lines.append(
        "| Target | Model | Slice | Mean Std | Std/Error | "
        "1-sig Cov. | 2-sig Cov. |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for slice_name in ["esper_benchmark", "combined_mlsaft_only"]:
        slice_label = (
            "Esper bench."
            if slice_name == "esper_benchmark"
            else "ML-SAFT only"
        )
        for target in TARGETS:
            for model_name in ["esper", "combined"]:
                key = (model_name, slice_name, target)
                if key not in all_metrics:
                    continue
                m = all_metrics[key]
                lines.append(
                    f"| {TARGET_DISPLAY[target]} | RF_{model_name} | "
                    f"{slice_label} | {m['mean_std']:.4f} | "
                    f"{m['std_error_ratio']:.3f} | "
                    f"{m['coverage_1sigma']:.3f} | "
                    f"{m['coverage_2sigma']:.3f} |"
                )
    lines.append("")

    # Calibration interpretation
    lines.append("### Calibration Interpretation")
    lines.append("")
    lines.append(
        "For a well-calibrated Gaussian uncertainty estimate, 1-sigma coverage "
        "should be ~68.3% and 2-sigma coverage should be ~95.4%."
    )
    lines.append("")

    # Check calibration quality
    esper_cov1 = [
        all_metrics[("esper", "esper_benchmark", t)]["coverage_1sigma"]
        for t in TARGETS
    ]
    combined_cov1 = [
        all_metrics[("combined", "esper_benchmark", t)]["coverage_1sigma"]
        for t in TARGETS
    ]
    avg_esper_1sig = np.mean(esper_cov1)
    avg_combined_1sig = np.mean(combined_cov1)

    if abs(avg_esper_1sig - avg_combined_1sig) < 0.03:
        lines.append(
            "Both models show similar calibration behavior on the Esper benchmark. "
            f"Average 1-sigma coverage: RF_esper={avg_esper_1sig:.3f}, "
            f"RF_combined={avg_combined_1sig:.3f}."
        )
    elif avg_combined_1sig > avg_esper_1sig:
        lines.append(
            "RF_combined shows slightly higher 1-sigma coverage on the Esper benchmark "
            f"({avg_combined_1sig:.3f} vs {avg_esper_1sig:.3f}), suggesting its "
            "uncertainty estimates are more conservative (wider)."
        )
    else:
        lines.append(
            "RF_combined shows slightly lower 1-sigma coverage on the Esper benchmark "
            f"({avg_combined_1sig:.3f} vs {avg_esper_1sig:.3f}), suggesting its "
            "uncertainty estimates may be less calibrated."
        )
    lines.append("")

    # --- Slice Analysis ---
    lines.append("## Slice Analysis")
    lines.append("")

    # Combined benchmark
    lines.append("### Combined Benchmark Test Set")
    lines.append("")
    # Get actual sample sizes for each model on this slice
    comb_n_esper = all_metrics.get(
        ("esper", "combined_benchmark", "m"), {}
    ).get("n_samples", "N/A")
    comb_n_combined = all_metrics.get(
        ("combined", "combined_benchmark", "m"), {}
    ).get("n_samples", len(combined_test))
    lines.append(
        f"Combined test set has {len(combined_test)} molecules total. "
        f"RF_esper is evaluated on a leak-free subset (n={comb_n_esper}; "
        "molecules also in esper_train are excluded). "
        f"RF_combined is evaluated on all {comb_n_combined} molecules."
    )
    lines.append("")
    lines.append("| Target | Model | MAE | RMSE | R2 | n |")
    lines.append("|---|---|---|---|---|---|")
    for target in TARGETS:
        for model_name in ["esper", "combined"]:
            key = (model_name, "combined_benchmark", target)
            m = all_metrics[key]
            lines.append(
                f"| {TARGET_DISPLAY[target]} | RF_{model_name} | "
                f"{m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} | "
                f"{m['n_samples']} |"
            )
    lines.append("")

    # Esper subset of combined
    lines.append("### Esper Subset of Combined Test")
    lines.append("")
    lines.append(
        f"Esper-origin molecules within the combined test set "
        f"({len(combined_test_esper_subset)} molecules)."
    )
    lines.append("")
    lines.append("| Target | Model | MAE | RMSE | R2 |")
    lines.append("|---|---|---|---|---|")
    for target in TARGETS:
        for model_name in ["esper", "combined"]:
            key = (model_name, "combined_esper_subset", target)
            m = all_metrics[key]
            lines.append(
                f"| {TARGET_DISPLAY[target]} | RF_{model_name} | "
                f"{m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} |"
            )
    lines.append("")

    # ML-SAFT-only subset
    lines.append("### ML-SAFT-Only Subset of Combined Test")
    lines.append("")
    n_mlsaft = len(combined_test_mlsaft_subset)
    if n_mlsaft >= 5:
        lines.append(
            f"ML-SAFT-only molecules in the combined test set ({n_mlsaft} molecules). "
            "Small sample; treat as directional evidence."
        )
        lines.append("")
        lines.append("| Target | Model | MAE | RMSE | R2 |")
        lines.append("|---|---|---|---|---|")
        for target in TARGETS:
            for model_name in ["esper", "combined"]:
                key = (model_name, "combined_mlsaft_only", target)
                if key in all_metrics:
                    m = all_metrics[key]
                    lines.append(
                        f"| {TARGET_DISPLAY[target]} | RF_{model_name} | "
                        f"{m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} |"
                    )
        lines.append("")
    else:
        lines.append(
            f"Only {n_mlsaft} ML-SAFT-only molecules in the combined test set; "
            "too few for meaningful evaluation."
        )
        lines.append("")

    # Fluorinated external
    fluor_path = SAVED_DIR / "gnn_fluorinated_validation_set.csv"
    if fluor_path.exists():
        lines.append("### Fluorinated External Validation (15 compounds)")
        lines.append("")
        lines.append("| Target | Model | MAE | RMSE | R2 |")
        lines.append("|---|---|---|---|---|")
        for target in TARGETS:
            for model_name in ["esper", "combined"]:
                key = (model_name, "fluorinated_external", target)
                if key in all_metrics:
                    m = all_metrics[key]
                    lines.append(
                        f"| {TARGET_DISPLAY[target]} | RF_{model_name} | "
                        f"{m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} |"
                    )
        lines.append("")

    # --- Supporting Diagnostics ---
    if overlap_diag:
        lines.append("## Supporting Diagnostics: Overlap Variability")
        lines.append("")
        lines.append(
            f"Computed because combined RF showed degradation on Esper benchmark. "
            f"{overlap_diag['overlap_count']} overlapping molecules (by InChI) "
            f"between Esper and ML-SAFT."
        )
        lines.append("")
        lines.append("| Target | Mean Delta (ML-SAFT - Esper) | Std Delta | Mean |Delta| |")
        lines.append("|---|---|---|---|")
        for t in TARGETS:
            lines.append(
                f"| {TARGET_DISPLAY[t]} | "
                f"{overlap_diag[f'{t}_mean_delta']:.4f} | "
                f"{overlap_diag[f'{t}_std_delta']:.4f} | "
                f"{overlap_diag[f'{t}_mean_abs_delta']:.4f} |"
            )
        lines.append("")

    # --- Key Findings ---
    lines.append("## Key Findings")
    lines.append("")

    # Analyze results for findings
    esper_r2_deltas = {}
    for target in TARGETS:
        key_e = ("esper", "esper_benchmark", target)
        key_c = ("combined", "esper_benchmark", target)
        esper_r2_deltas[target] = all_metrics[key_c]["r2"] - all_metrics[key_e]["r2"]

    max_r2_change = max(abs(v) for v in esper_r2_deltas.values())
    if max_r2_change < 0.03:
        lines.append(
            "1. **Esper benchmark performance is essentially unchanged.** "
            "The largest R2 delta between RF_esper and RF_combined on the canonical "
            f"Esper holdout is {max_r2_change:.4f}. Adding ~{provenance['net_new_mlsaft']} "
            "ML-SAFT molecules does not materially change generalization on the Esper "
            "test distribution."
        )
    elif all(v >= 0 for v in esper_r2_deltas.values()):
        lines.append(
            "1. **Combined RF improves on Esper benchmark.** "
            "R2 improved across all targets, with the largest gain on "
            f"{max(esper_r2_deltas, key=esper_r2_deltas.get)} "
            f"(+{max(esper_r2_deltas.values()):.4f})."
        )
    else:
        lines.append(
            "1. **Mixed results on Esper benchmark.** "
            "R2 changes vary by target: "
            + ", ".join(f"{t}: {esper_r2_deltas[t]:+.4f}" for t in TARGETS)
            + "."
        )
    lines.append("")

    # Uncertainty finding
    esper_cov1_vals = {}
    combined_cov1_vals = {}
    for target in TARGETS:
        esper_cov1_vals[target] = all_metrics[
            ("esper", "esper_benchmark", target)
        ]["coverage_1sigma"]
        combined_cov1_vals[target] = all_metrics[
            ("combined", "esper_benchmark", target)
        ]["coverage_1sigma"]

    avg_e = np.mean(list(esper_cov1_vals.values()))
    avg_c = np.mean(list(combined_cov1_vals.values()))
    lines.append(
        f"2. **Uncertainty calibration comparison.** "
        f"Average 1-sigma coverage on Esper benchmark: "
        f"RF_esper={avg_e:.3f}, RF_combined={avg_c:.3f}. "
    )
    if abs(avg_e - avg_c) < 0.03:
        lines.append(
            "   Both models are similarly calibrated. The addition of ML-SAFT data "
            "does not degrade uncertainty quality."
        )
    elif avg_c > avg_e:
        lines.append(
            "   RF_combined is slightly more conservative (higher coverage), "
            "likely due to increased tree disagreement from the mixed training data."
        )
    else:
        lines.append(
            "   RF_combined shows slightly lower coverage, suggesting the "
            "mixed training data introduces some miscalibration."
        )
    lines.append("")

    # ML-SAFT-only finding
    if n_mlsaft >= 5:
        mlsaft_r2 = {}
        for target in TARGETS:
            key_c = ("combined", "combined_mlsaft_only", target)
            if key_c in all_metrics:
                mlsaft_r2[target] = all_metrics[key_c]["r2"]
        if mlsaft_r2:
            avg_mlsaft_r2 = np.mean(list(mlsaft_r2.values()))
            lines.append(
                f"3. **ML-SAFT-only molecule performance.** "
                f"RF_combined evaluated on {n_mlsaft} ML-SAFT-only test molecules "
                f"achieves average R2={avg_mlsaft_r2:.3f}. "
            )
            if avg_mlsaft_r2 < 0.2:
                lines.append(
                    "   This is weak, indicating these molecules remain hard to predict. "
                    "The combined RF does not dramatically improve on novel ML-SAFT chemistry."
                )
            else:
                lines.append(
                    "   This is reasonable, suggesting the combined training data "
                    "provides some generalization to novel ML-SAFT chemistry."
                )
            lines.append("")

    lines.append(
        "4. **This step extends Step 01b with mandatory uncertainty metrics.** "
        "The 1-sigma and 2-sigma coverage comparison provides a more complete "
        "picture than point metrics alone."
    )
    lines.append("")

    # --- Figures ---
    lines.append("## Figures")
    lines.append("")
    lines.append("See `figures/44_rf_combined_retrain/` for:")
    lines.append(
        "- `parity_rf_esper_vs_combined.png` -- 2x3 parity plots "
        "(RF_esper top row, RF_combined bottom row)"
    )
    lines.append(
        "- `slice_metrics_comparison.png` -- Dot plot comparing MAE/RMSE/R2 "
        "across evaluation slices"
    )
    lines.append(
        "- `uncertainty_calibration_rf_comparison.png` -- Mean predicted std "
        "vs mean absolute error for Esper benchmark and ML-SAFT-only slice"
    )
    lines.append(
        "- `coverage_comparison.png` -- 1-sigma and 2-sigma coverage by target "
        "and slice for both RFs"
    )
    lines.append(
        "- `combined_test_source_breakdown.png` -- Composition of the combined "
        "test set (Esper-derived vs ML-SAFT-only)"
    )
    lines.append("")

    # --- Deviations ---
    lines.append("## Deviations")
    lines.append("")
    lines.append(
        "**Split strategy adjusted to prevent data leakage.** The step guide calls "
        "for independent 80/20 splits on both datasets. However, because Esper "
        "molecules appear in both datasets, a naive independent split causes "
        "Esper test molecules to appear in the combined training set (293/361 = "
        "81% leakage). To prevent this, the combined training set excludes both "
        "the Esper holdout and the combined test set. Similarly, when evaluating "
        "RF_esper on combined-derived slices, molecules from esper_train are "
        "filtered out. This results in different effective sample sizes for some "
        "slice comparisons but ensures all reported metrics are free of "
        "train/test contamination."
    )
    lines.append("")

    # --- Readiness Check ---
    lines.append("## Readiness Check")
    lines.append("")
    lines.append(
        "- [x] Esper-only RF and Combined RF retrained in same script with "
        "identical hyperparameters"
    )
    lines.append(
        "- [x] Both models use the same feature pipeline and split seed (42)"
    )
    lines.append(
        "- [x] Both models evaluated on the Esper benchmark test set"
    )
    lines.append(
        "- [x] Both models evaluated on the combined benchmark test set"
    )
    lines.append(
        "- [x] ML-SAFT-only molecules isolated and reported separately"
    )
    lines.append(
        "- [x] Point metrics (MAE, RMSE, R2) reported for every target on every slice"
    )
    lines.append(
        "- [x] Uncertainty metrics (mean std, 1-sigma/2-sigma coverage, calibration) "
        "reported for both RFs"
    )
    lines.append(
        "- [x] All 5 required figures saved to `figures/44_rf_combined_retrain/`"
    )
    lines.append(
        "- [x] Per-molecule predictions saved to "
        "`model/saved/step44_rf_combined_predictions.csv`"
    )
    lines.append(
        "- [x] Aggregate metrics saved to `model/saved/step44_rf_combined_metrics.json`"
    )
    lines.append("")

    # --- Recommendation ---
    lines.append("## Recommendation")
    lines.append("")

    # Decide recommendation based on results
    all_r2_small = max_r2_change < 0.03
    uncertainty_ok = abs(avg_e - avg_c) < 0.05

    if all_r2_small and uncertainty_ok:
        lines.append(
            "**Keep Esper-only RF for production; use combined RF for exploratory "
            "analyses only.** The combined RF shows no meaningful improvement on the "
            "canonical Esper benchmark and its uncertainty calibration is comparable. "
            f"The net addition of {provenance['net_new_mlsaft']} molecules "
            f"({100 * provenance['net_new_mlsaft'] / provenance['raw_esper']:.1f}% increase) "
            "is insufficient to shift RF generalization. The Esper dataset remains the "
            "cleaner, better-characterized training corpus for production use. "
            "The combined RF may be useful for exploratory screening beyond the Esper "
            "chemical space, but predictions on ML-SAFT-only molecules should be flagged "
            "as lower confidence."
        )
    elif max_r2_change < 0.03 and not uncertainty_ok:
        lines.append(
            "**Keep Esper-only RF for production.** The combined RF shows no accuracy "
            "improvement and introduces uncertainty miscalibration from mixing two "
            "fitting protocols."
        )
    elif all(v >= 0 for v in esper_r2_deltas.values()):
        lines.append(
            "**Adopt the combined RF.** Performance improves or is maintained across "
            "all targets on the Esper benchmark, with acceptable uncertainty behavior."
        )
    else:
        lines.append(
            "**Keep Esper-only RF for production.** The combined RF shows mixed "
            "results on the Esper benchmark with some target degradation."
        )
    lines.append("")

    report_path = PROJECT_ROOT / "docs" / "reports" / "44_rf_combined_retrain.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("Step 44: RF Combined Retrain (Esper + ML-SAFT)")
    logger.info("=" * 60)

    # 44.1: Verify datasets
    df_esper, df_combined, provenance = verify_datasets()

    # Build Esper InChI set for source membership tests
    esper_inchi_set: set[str] = set()
    for smi in df_esper["smiles"]:
        inchi = smiles_to_inchi(smi)
        if inchi:
            esper_inchi_set.add(inchi)

    # 44.2: Create splits
    esper_train, esper_test, combined_train, combined_test = create_splits(
        df_esper, df_combined
    )

    # Slice combined test by source
    combined_test_esper_subset, combined_test_mlsaft_subset = slice_combined_test(
        combined_test, esper_inchi_set
    )

    # 44.3: Train paired RF models
    rf_models = train_paired_rfs(esper_train, combined_train)

    # 44.4/44.5: Evaluate all slices with uncertainty
    all_metrics, predictions_df = evaluate_all_slices(
        rf_models,
        esper_train,
        esper_test,
        combined_train,
        combined_test,
        combined_test_esper_subset,
        combined_test_mlsaft_subset,
    )

    # 44.6: Figures
    logger.info("=" * 60)
    logger.info("SECTION 44.6: Generating Figures")
    logger.info("=" * 60)

    plot_parity_rf_esper_vs_combined(rf_models, esper_test)
    plot_slice_metrics_comparison(all_metrics)
    plot_uncertainty_calibration(rf_models, esper_test, combined_test_mlsaft_subset)
    plot_coverage_comparison(all_metrics)
    plot_combined_test_source_breakdown(combined_test, esper_inchi_set)

    # 44.7: Overlap diagnostics (only if needed)
    overlap_diag = overlap_diagnostics(df_esper, all_metrics)

    # 44.8: Save artifacts
    save_predictions(predictions_df)
    save_metrics(all_metrics, provenance, overlap_diag)

    # 44.9: Generate report
    generate_report(
        all_metrics,
        provenance,
        esper_train,
        esper_test,
        combined_train,
        combined_test,
        combined_test_esper_subset,
        combined_test_mlsaft_subset,
        overlap_diag,
    )

    logger.info("=" * 60)
    logger.info("Step 44 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
