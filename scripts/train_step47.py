"""Step 47: Repeated SVR benchmark against RF/XGBoost baselines.

Runs repeated stratified outer splits on the Esper dataset and aggregates
SVR performance across runs. The goal is to test whether the kernel-method
benchmark is stable, rather than relying on a single favorable split.

Usage:
    python scripts/train_step47.py [--source esper] [--outer-splits 10]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402, I001
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

import model.svm  # noqa: F401, E402
import model.linear.ridge_model  # noqa: F401, E402
from model.data.descriptors import build_features_with_names  # noqa: E402
from model.data.load import TARGETS, load_data, split_data  # noqa: E402

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "47_svm_benchmark"
SAVED_DIR = Path(__file__).resolve().parent.parent / "model" / "saved"

TARGET_LABELS = {"m": "m (segments)", "sigma": "\u03c3 (\u00c5)", "epsilon_k": "\u03b5/k (K)"}


def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R2 for valid (non-NaN) entries."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
    }


def train_svm(X_train, y_train, feature_names, cv_folds):
    """Train target-wise SVR models with GridSearchCV."""
    from model.svm.svm_model import SVMPCSAFT

    svm_model = SVMPCSAFT(target_names=TARGETS)
    svm_model.fit(
        X_train,
        y_train,
        feature_names=feature_names,
        cv_folds=cv_folds,
        verbose=0,
    )
    return svm_model


def train_ridge(X_train, y_train, feature_names, cv_folds):
    """Train target-wise Ridge models with GridSearchCV."""
    from model.linear.ridge_model import RidgePCSAFT

    ridge_model = RidgePCSAFT(target_names=TARGETS)
    ridge_model.fit(
        X_train,
        y_train,
        feature_names=feature_names,
        cv_folds=cv_folds,
        verbose=0,
    )
    return ridge_model


def evaluate_model(preds, y_true):
    """Evaluate predictions against test targets."""
    results = {}
    for i, target in enumerate(TARGETS):
        if isinstance(preds, dict):
            y_pred = preds[target]
        else:
            y_pred = preds[:, i]
        metrics = compute_metrics(y_true[:, i], y_pred)
        results[target] = metrics
    return results


def load_baseline_metrics():
    """Load baseline metrics from saved comparison CSV."""
    baselines = {}
    comparison_path = SAVED_DIR / "comparison_metrics.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        for model_name in ["rf", "gc_pcsaft", "nn", "chemberta", "xgboost", "chemprop", "gnn"]:
            sub = df[df["model"] == model_name]
            if len(sub) == 0:
                continue
            baselines[model_name] = {}
            for _, row in sub.iterrows():
                baselines[model_name][row["target"]] = {
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "r2": row["r2"],
                    "mae_lo": row.get("mae_lo", np.nan),
                    "mae_hi": row.get("mae_hi", np.nan),
                    "rmse_lo": row.get("rmse_lo", np.nan),
                    "rmse_hi": row.get("rmse_hi", np.nan),
                    "r2_lo": row.get("r2_lo", np.nan),
                    "r2_hi": row.get("r2_hi", np.nan),
                    "r2_boot_std": row.get("r2_boot_std", np.nan),
                    "protocol": "single_split_artifact",
                }
    return baselines


def summarize_runs(split_results, prefix="svm"):
    """Aggregate repeated-split metrics into means/stds/intervals."""
    summary = {}
    for target in TARGETS:
        summary[target] = {}
        for metric_name in ["mae", "rmse", "r2"]:
            values = np.array(
                [run[f"{prefix}_test_metrics"][target][metric_name] for run in split_results],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                summary[target][metric_name] = np.nan
                summary[target][f"{metric_name}_std"] = np.nan
                summary[target][f"{metric_name}_lo"] = np.nan
                summary[target][f"{metric_name}_hi"] = np.nan
                continue
            summary[target][metric_name] = float(finite.mean())
            summary[target][f"{metric_name}_std"] = float(finite.std(ddof=0))
            summary[target][f"{metric_name}_lo"] = float(np.percentile(finite, 2.5))
            summary[target][f"{metric_name}_hi"] = float(np.percentile(finite, 97.5))
    return summary


def flatten_split_results(split_results):
    """Convert nested per-split metrics into a tabular DataFrame."""
    rows = []
    for run in split_results:
        base = {
            "split_idx": run["split_idx"],
            "seed": run["seed"],
            "train_size": run["train_size"],
            "test_size": run["test_size"],
            "train_valid": run["train_valid"],
            "test_valid": run["test_valid"],
        }
        for target in TARGETS:
            for prefix in ["svm", "ridge"]:
                row = dict(base)
                row["model"] = prefix
                row["target"] = target
                row.update(run[f"{prefix}_test_metrics"][target])
                row["train_r2"] = run[f"{prefix}_train_metrics"][target]["r2"]
                row["train_mae"] = run[f"{prefix}_train_metrics"][target]["mae"]
                row["train_rmse"] = run[f"{prefix}_train_metrics"][target]["rmse"]
                row["best_params"] = json.dumps(
                    run[f"{prefix}_best_params"][target], sort_keys=True
                )
                row["cv_best_score"] = run[f"{prefix}_cv_best_scores"][target]
                rows.append(row)
    return pd.DataFrame(rows)


def plot_r2_heatmap(all_metrics, save_dir):
    """Create R2 heatmap (models x targets) including aggregate SVR."""
    model_names = list(all_metrics.keys())
    targets = TARGETS

    data = np.zeros((len(model_names), len(targets)))
    for i, mname in enumerate(model_names):
        for j, target in enumerate(targets):
            data[i, j] = all_metrics[mname].get(target, {}).get("r2", np.nan)

    fig, ax = plt.subplots(figsize=(8, max(4, len(model_names) * 0.6 + 1)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        xticklabels=[TARGET_LABELS[t] for t in targets],
        yticklabels=model_names,
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("R\u00b2 by Model and Target (incl. SVM)", fontsize=14)
    fig.tight_layout()
    path = save_dir / "r2_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def metric_error(metric):
    """Return a scalar uncertainty estimate for plotting if available."""
    lo = metric.get("r2_lo", np.nan)
    hi = metric.get("r2_hi", np.nan)
    if np.isfinite(lo) and np.isfinite(hi):
        return float((hi - lo) / 2.0)
    std = metric.get("r2_std", np.nan)
    if np.isfinite(std):
        return float(std)
    boot_std = metric.get("r2_boot_std", np.nan)
    if np.isfinite(boot_std):
        return float(boot_std)
    return 0.0


def plot_bottleneck_comparison(svm_metrics, ridge_metrics, baselines, save_dir):
    """Bar chart comparing Ridge, RF, XGBoost, and repeated-split SVR on epsilon/k R²."""
    models_to_compare = {}
    errors = {}
    
    ridge_r2 = ridge_metrics.get("epsilon_k", {}).get("r2", np.nan)
    if np.isfinite(ridge_r2):
        models_to_compare["Ridge (repeated)"] = ridge_r2
        errors["Ridge (repeated)"] = metric_error(ridge_metrics.get("epsilon_k", {}))

    for name in ["rf", "xgboost"]:
        if name in baselines:
            metric = baselines[name].get("epsilon_k", {})
            r2 = metric.get("r2", np.nan)
            if np.isfinite(r2):
                label = name.upper() if name == "rf" else "XGBoost"
                models_to_compare[label] = r2
                errors[label] = metric_error(metric)
    svm_r2 = svm_metrics.get("epsilon_k", {}).get("r2", np.nan)
    if np.isfinite(svm_r2):
        models_to_compare["SVR (RBF, repeated)"] = svm_r2
        errors["SVR (RBF, repeated)"] = metric_error(svm_metrics.get("epsilon_k", {}))

    if len(models_to_compare) < 2:
        return

    names = list(models_to_compare.keys())
    values = list(models_to_compare.values())
    yerr = [errors[name] for name in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        names,
        values,
        yerr=yerr,
        capsize=4,
        color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"][:len(names)],
        edgecolor="k",
        linewidth=0.5,
    )
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("R\u00b2 (test set)", fontsize=12)
    ax.set_title("\u03b5/k R\u00b2: repeated SVR & Ridge vs saved tree baselines", fontsize=14)
    ax.set_ylim(min(-0.2, min(values) - 0.1), max(values) + max(yerr + [0.05]) + 0.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = save_dir / "bottleneck_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def print_aggregate_summary(aggregate_metrics):
    """Print aggregate test metrics across repeated splits."""
    print("\nAGGREGATE SVR RESULTS ACROSS SPLITS")
    print("=" * 60)
    for target in TARGETS:
        m = aggregate_metrics[target]
        print(
            f"  {target:12s}: "
            f"R2={m['r2']:.4f} ± {m['r2_std']:.4f} "
            f"[{m['r2_lo']:.4f}, {m['r2_hi']:.4f}]  "
            f"MAE={m['mae']:.4f} ± {m['mae_std']:.4f}  "
            f"RMSE={m['rmse']:.4f} ± {m['rmse_std']:.4f}"
        )


def run_outer_split(df, split_idx, seed, inner_cv):
    """Train/evaluate one repeated outer split."""
    train_df, test_df = split_data(df, random_state=seed)

    smiles_train = train_df["smiles"].tolist()
    smiles_test = test_df["smiles"].tolist()
    y_train = train_df[TARGETS].values.astype(np.float32)
    y_test = test_df[TARGETS].values.astype(np.float32)

    X_train, feature_names = build_features_with_names(smiles_train)
    X_test, _ = build_features_with_names(
        smiles_test,
        rdkit_names=[n for n in feature_names if not n.startswith("morgan_")],
    )

    train_valid = np.isfinite(X_train).all(axis=1)
    test_valid = np.isfinite(X_test).all(axis=1)

    X_train_clean = X_train[train_valid]
    y_train_clean = y_train[train_valid]
    X_test_clean = X_test[test_valid]
    y_test_clean = y_test[test_valid]

    print("\n" + "=" * 60)
    print(f"OUTER SPLIT {split_idx + 1} | seed={seed}")
    print("=" * 60)
    print(
        f"Valid features: {train_valid.sum()}/{len(X_train)} train, "
        f"{test_valid.sum()}/{len(X_test)} test"
    )

    svm_model = train_svm(X_train_clean, y_train_clean, feature_names, cv_folds=inner_cv)
    ridge_model = train_ridge(X_train_clean, y_train_clean, feature_names, cv_folds=inner_cv)
    
    test_preds_svm = svm_model.predict(X_test_clean)
    train_preds_svm = svm_model.predict(X_train_clean)
    test_metrics_svm = evaluate_model(test_preds_svm, y_test_clean)
    train_metrics_svm = evaluate_model(train_preds_svm, y_train_clean)

    test_preds_ridge = ridge_model.predict(X_test_clean)
    train_preds_ridge = ridge_model.predict(X_train_clean)
    test_metrics_ridge = evaluate_model(test_preds_ridge, y_test_clean)
    train_metrics_ridge = evaluate_model(train_preds_ridge, y_train_clean)

    ek_svm = test_metrics_svm["epsilon_k"]
    ek_ridge = test_metrics_ridge["epsilon_k"]
    print(
        "  SVR epsilon_k: "
        f"R2={ek_svm['r2']:.4f}  MAE={ek_svm['mae']:.4f}  RMSE={ek_svm['rmse']:.4f}"
    )
    print(
        "  Ridge epsilon_k: "
        f"R2={ek_ridge['r2']:.4f}  MAE={ek_ridge['mae']:.4f}  RMSE={ek_ridge['rmse']:.4f}"
    )

    return {
        "split_idx": split_idx,
        "seed": seed,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_valid": int(train_valid.sum()),
        "test_valid": int(test_valid.sum()),
        "svm_test_metrics": test_metrics_svm,
        "svm_train_metrics": train_metrics_svm,
        "svm_best_params": svm_model.best_params_,
        "svm_cv_best_scores": svm_model.cv_best_scores_,
        "ridge_test_metrics": test_metrics_ridge,
        "ridge_train_metrics": train_metrics_ridge,
        "ridge_best_params": ridge_model.best_params_,
        "ridge_cv_best_scores": ridge_model.cv_best_scores_,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 47: repeated SVR benchmark")
    parser.add_argument("--source", default="esper", help="Data source")
    parser.add_argument("--outer-splits", type=int, default=10, help="Number of outer splits")
    parser.add_argument("--inner-cv", type=int, default=3, help="Inner CV folds for tuning")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for outer splits")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("Loading data...")
    df = load_data(source=args.source)
    print(f"Dataset: {len(df)} total molecules")

    seeds = [args.base_seed + idx for idx in range(args.outer_splits)]
    split_results = [
        run_outer_split(df, split_idx=idx, seed=seed, inner_cv=args.inner_cv)
        for idx, seed in enumerate(seeds)
    ]
    svm_metrics = summarize_runs(split_results, prefix="svm")
    ridge_metrics = summarize_runs(split_results, prefix="ridge")
    print("\nAGGREGATE SVR RESULTS ACROSS SPLITS")
    print_aggregate_summary(svm_metrics)
    print("\nAGGREGATE RIDGE RESULTS ACROSS SPLITS")
    print_aggregate_summary(ridge_metrics)

    baselines = load_baseline_metrics()
    all_metrics = dict(baselines)
    all_metrics["ridge"] = ridge_metrics
    all_metrics["svm"] = svm_metrics

    print("\n" + "=" * 80)
    print("FULL MODEL COMPARISON")
    print("=" * 80)
    for model_name, metrics in all_metrics.items():
        print(f"\n  {model_name}:")
        for target in TARGETS:
            m = metrics.get(target, {})
            print(f"    {target:12s}: R2={m.get('r2', float('nan')):.4f}  "
                  f"MAE={m.get('mae', float('nan')):.4f}  "
                  f"RMSE={m.get('rmse', float('nan')):.4f}")

    print("\n" + "=" * 60)
    print("REPRESENTATION BOTTLENECK TEST: epsilon/k R\u00b2")
    print("=" * 60)
    ridge_ek = ridge_metrics.get("epsilon_k", {}).get("r2", float("nan"))
    ridge_ek_std = ridge_metrics.get("epsilon_k", {}).get("r2_std", float("nan"))
    rf_ek = baselines.get("rf", {}).get("epsilon_k", {}).get("r2", float("nan"))
    xgb_ek = baselines.get("xgboost", {}).get("epsilon_k", {}).get("r2", float("nan"))
    svm_ek = svm_metrics.get("epsilon_k", {}).get("r2", float("nan"))
    svm_ek_std = svm_metrics.get("epsilon_k", {}).get("r2_std", float("nan"))
    print(f"  Ridge:   {ridge_ek:.4f} ± {ridge_ek_std:.4f}")
    print(f"  RF:      {rf_ek:.4f}")
    print(f"  XGBoost: {xgb_ek:.4f}")
    print(f"  SVR:     {svm_ek:.4f} ± {svm_ek_std:.4f}")
    print("  Note: RF/XGBoost values come from saved single-split artifacts unless regenerated.")

    if all(np.isfinite([rf_ek, xgb_ek, svm_ek])):
        spread = max(rf_ek, xgb_ek, svm_ek) - min(rf_ek, xgb_ek, svm_ek)
        print(f"  Spread (non-linear):  {spread:.4f}")
        if spread < 0.05:
            print(
                "  >> Three non-linear algorithm families are close on epsilon/k point estimates"
            )
            print(
                "  >> Treat this as suggestive until RF/XGBoost are rerun under the same protocol"
            )
        else:
            print(f"  >> Spread of {spread:.4f} suggests algorithmic differences still matter")

    print("\nGenerating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_r2_heatmap(all_metrics, FIGURES_DIR)
    plot_bottleneck_comparison(svm_metrics, ridge_metrics, baselines, FIGURES_DIR)

    split_df = flatten_split_results(split_results)
    split_csv_path = SAVED_DIR / "step47_split_metrics.csv"
    split_csv_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_csv_path, index=False)
    print(f"Per-split metrics saved to {split_csv_path}")

    summary = {
        "protocol": {
            "type": "repeated_outer_splits",
            "source": args.source,
            "outer_splits": args.outer_splits,
            "inner_cv": args.inner_cv,
            "split_seeds": seeds,
        },
        "comparison_notes": [
            "SVR and Ridge metrics are aggregated across repeated outer splits.",
            "RF/XGBoost baselines loaded from comparison_metrics.csv may reflect "
            "older single-split evaluations.",
            "Do not interpret best-run SVR performance as evidence; use aggregate metrics only.",
        ],
        "split_results": split_results,
        "svm_metrics": svm_metrics,
        "svm_best_params_by_split": {
            f"split_{run['split_idx']}": run["svm_best_params"] for run in split_results
        },
        "ridge_metrics": ridge_metrics,
        "ridge_best_params_by_split": {
            f"split_{run['split_idx']}": run["ridge_best_params"] for run in split_results
        },
        "all_metrics": {
            name: {t: m for t, m in metrics.items()} for name, metrics in all_metrics.items()
        },
        "bottleneck_test": {
            "ridge_epsilon_k_r2": ridge_ek,
            "ridge_epsilon_k_r2_std": ridge_ek_std,
            "rf_epsilon_k_r2": rf_ek,
            "xgboost_epsilon_k_r2": xgb_ek,
            "svm_epsilon_k_r2": svm_ek,
            "svm_epsilon_k_r2_std": svm_ek_std,
        },
    }
    summary_path = SAVED_DIR / "step47_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved to {summary_path}")

    print("\nStep 47 complete!")


if __name__ == "__main__":
    main()
