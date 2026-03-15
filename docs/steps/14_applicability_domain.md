# Step 14: Improved Applicability Domain

## Purpose

Phase 1 used an Isolation Forest trained on the 2,218-dimensional Morgan+RDKit feature space as the sole AD indicator. This has two weaknesses:

1. **Not chemically interpretable**: a researcher cannot understand *why* a molecule is flagged OOD from a binary forest decision
2. **Not validated against an independent definition**: it is unclear whether the Isolation Forest flags are meaningful or an artifact of the feature space dimensionality

This step adds two complementary AD methods:

- **Tanimoto nearest-neighbor similarity**: `max(Tanimoto(query, train_set))` using Morgan fingerprints; chemically intuitive, directly says "this molecule is/isn't similar to anything we've trained on"
- **Williams plot** (leverage-based AD): standardized residuals vs leverage (hat matrix diagonal); the QSAR community standard for AD visualization

Both methods are compared to the existing Isolation Forest, and the Tanimoto score is exposed in the API response and portal.

## What It Adds Over the Current State

| Current | After This Step |
|---------|----------------|
| Isolation Forest AD (binary) | Isolation Forest + Tanimoto sim + Williams plot |
| AD not chemically interpretable | Tanimoto NN gives "closest training molecule" info |
| `/predict` returns `in_domain: bool` | `/predict` also returns `tanimoto_nn: float` |
| Portal shows no OOD detail | Portal shows Tanimoto score alongside in_domain flag |

## Skills Demonstrated

- **Chemoinformatics**: Tanimoto similarity with Morgan fingerprints via RDKit
- **QSAR methodology**: Williams plot (leverage vs standardized residual)
- **API extension**: backward-compatible schema addition

## Dependencies

- Requires: RF model and training set feature matrix (already saved)
- Requires: `model/saved/rf_*.joblib` and `model/saved/feature_names.joblib`
- Packages: `rdkit` (already a dependency), `numpy`, `matplotlib`

## Implementation Guide

### 14.1 Tanimoto Similarity AD

Create `model/ad_tanimoto.py`:

```python
"""Tanimoto nearest-neighbor applicability domain."""
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles


class TanimotoAD:
    """AD model based on max Tanimoto similarity to training set."""

    IN_DOMAIN_THRESHOLD = 0.4   # max similarity >= 0.4 → in domain
    WARNING_THRESHOLD = 0.3     # max similarity in [0.3, 0.4) → warning

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits
        self.train_fps_ = None

    def fit(self, smiles_list: list[str]):
        self.train_fps_ = [
            AllChem.GetMorganFingerprintAsBitVect(
                MolFromSmiles(s), self.radius, nBits=self.n_bits
            )
            for s in smiles_list if MolFromSmiles(s) is not None
        ]
        return self

    def tanimoto_nn(self, smiles: str) -> float:
        """Return max Tanimoto similarity to any training molecule."""
        mol = MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        sims = DataStructs.BulkTanimotoSimilarity(fp, self.train_fps_)
        return float(max(sims)) if sims else 0.0

    def in_domain(self, smiles: str) -> bool:
        return self.tanimoto_nn(smiles) >= self.IN_DOMAIN_THRESHOLD
```

Fit the `TanimotoAD` model on training SMILES, save via joblib alongside the existing AD model.

### 14.2 Williams Plot (`model/ad_williams.py`)

```python
"""Williams plot (leverage vs standardized residuals) for RF AD visualization."""
import numpy as np
import matplotlib.pyplot as plt


def compute_leverage(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Hat matrix diagonal h_i = x_i^T (X^T X)^{-1} x_i."""
    XtX_inv = np.linalg.pinv(X_train.T @ X_train)
    return np.array([x @ XtX_inv @ x for x in X_test])


def plot_williams(X_train, X_test, y_test, y_pred, target_name: str, output_path: str):
    """Generate Williams plot for a single target."""
    leverage = compute_leverage(X_train, X_test)
    residuals = y_pred - y_test
    std_residuals = residuals / residuals.std()

    h_star = 3 * X_train.shape[1] / X_train.shape[0]  # leverage warning threshold

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
```

Note: computing the full hat matrix for 2,218-dimensional features can be memory-intensive. Use PCA to project to 50 components before computing leverage if the matrix inversion is slow.

### 14.3 Expose Tanimoto in API (`serving/`)

Update `serving/schemas.py` — add `tanimoto_nn: float | None = None` to `PredictionResult`.

Update `serving/app.py` — load `TanimotoAD` at startup alongside Isolation Forest; compute `tanimoto_nn` in the `/predict` handler.

### 14.4 Portal — Show Tanimoto Score

In `portal/components/predictor.py`, display Tanimoto NN alongside `in_domain`:

```python
tanimoto = result.get("tanimoto_nn")
if tanimoto is not None:
    st.metric("Tanimoto similarity to training set", f"{tanimoto:.3f}")
    if tanimoto < 0.3:
        st.error("Low similarity to training data — high prediction uncertainty")
    elif tanimoto < 0.4:
        st.warning("Moderate similarity — treat prediction with caution")
```

### 14.5 AD Comparison Figure

`figures/14_ad/ad_method_comparison.png`: scatter of Isolation Forest score vs Tanimoto similarity for all test set molecules; color by whether they agree or disagree. Quantify: what fraction of molecules are flagged by one method but not the other?

### 14.6 Generate Williams Plots

Run Williams plot for RF on all three targets; save to:
- `figures/14_ad/williams_m.png`
- `figures/14_ad/williams_sigma.png`
- `figures/14_ad/williams_epsilon_k.png`

## Acceptance Criteria (When to Move On)

- [ ] `model/ad_tanimoto.py` with `TanimotoAD` class; fitted model saved as `model/saved/tanimoto_ad.joblib`
- [ ] `model/ad_williams.py` with `plot_williams` and `compute_leverage` functions
- [ ] `/predict` response includes `tanimoto_nn: float`
- [ ] Portal displays Tanimoto score with color-coded warning level
- [ ] Williams plots (3 targets) saved to `figures/14_ad/`
- [ ] AD comparison scatter saved to `figures/14_ad/`
- [ ] Report written at `docs/reports/14_applicability_domain.md`
- [ ] Add ≥ 8 tests (TanimotoAD, Williams functions, API schema)
- [ ] All existing tests pass
- [ ] Commit: `step 14: add Tanimoto AD and Williams plot`
