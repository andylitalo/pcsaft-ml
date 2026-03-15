"""Williams plot (leverage vs standardized residuals) for RF AD visualization."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def compute_leverage(
    X_train: np.ndarray, X_test: np.ndarray, pca_components: int | None = 50
) -> np.ndarray:
    """Compute hat matrix diagonal h_i = x_i^T (X^T X)^{-1} x_i.

    For high-dimensional feature spaces, PCA is applied before leverage computation
    to avoid memory issues and numerical instability.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape ``(n_train, n_features)``.
    X_test : np.ndarray
        Test feature matrix of shape ``(n_test, n_features)``.
    pca_components : int | None
        Number of PCA components to use. If None, use raw features.
        Default is 50 to handle high-dimensional descriptors.

    Returns
    -------
    np.ndarray
        Leverage values of shape ``(n_test,)``.
    """
    if pca_components is not None and X_train.shape[1] > pca_components:
        pca = PCA(n_components=pca_components)
        X_train_proj = pca.fit_transform(X_train)
        X_test_proj = pca.transform(X_test)
    else:
        X_train_proj = X_train
        X_test_proj = X_test

    XtX_inv = np.linalg.pinv(X_train_proj.T @ X_train_proj)
    leverage = np.array([x @ XtX_inv @ x for x in X_test_proj])
    return leverage


def plot_williams(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    output_path: str,
    pca_components: int | None = 50,
):
    """Generate Williams plot for a single target.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True test target values.
    y_pred : np.ndarray
        Predicted test target values.
    target_name : str
        Name of the target parameter (e.g., "m", "sigma", "epsilon_k").
    output_path : str
        Path to save the figure.
    pca_components : int | None
        Number of PCA components for leverage computation. Default is 50.
    """
    leverage = compute_leverage(X_train, X_test, pca_components=pca_components)
    residuals = y_pred - y_test
    std_residuals = residuals / residuals.std()

    # Use PCA-projected dimensionality for h* threshold
    n_features = pca_components if pca_components is not None else X_train.shape[1]
    h_star = 3 * n_features / X_train.shape[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(leverage, std_residuals, alpha=0.5, s=20)
    ax.axhline(3, color="red", linestyle="--", label="±3σ")
    ax.axhline(-3, color="red", linestyle="--")
    ax.axvline(h_star, color="orange", linestyle="--", label=f"h* = {h_star:.3f}")
    ax.set_xlabel("Leverage (h)")
    ax.set_ylabel("Standardized Residual")
    ax.set_title(f"Williams Plot — {target_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
