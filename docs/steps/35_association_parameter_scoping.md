# Step 35: Association Parameter Milestone Scoping

## Objective

Scope out the requirements, data availability, and architecture for predicting associating PC-SAFT parameters (ε_AB, κ_AB) as a **separate, clearly gated milestone**. This step produces a research plan document, not code. It also adds explicit limitations notices to the portal UI for associating molecules.

## Motivation

The current pipeline predicts only three non-associating PC-SAFT parameters (m, σ, ε/k). Molecules with hydrogen-bonding sites (OH, NH, COOH, etc.) require two additional association parameters (ε_AB, κ_AB) for accurate thermodynamic modeling. The existing `is_associating()` utility in `screening/filters.py` flags these molecules but offers no prediction capability.

Previous steps (Reports 25, 28) mentioned this limitation but mixed it into the screening narrative rather than treating it as a distinct, properly scoped milestone. The Hosted UI Pivot plan recommended separating this into its own phase with explicit go/no-go criteria.

Key questions this step answers:
1. How many molecules with association parameters exist in accessible databases?
2. Does teqp (or an alternative EOS backend) support associating PC-SAFT?
3. What model architecture changes are needed for 5-parameter prediction?
4. What is a realistic timeline and dataset size threshold to attempt this?

## Dependencies

- `screening/filters.py` exists with `is_associating()` function
- `serving/model_loader.py` exists with association warning logic
- `portal/app.py` or `portal/components/predictor.py` exists
- No dependency on Steps 31-34 (can run in parallel with anything)

## Implementation Guide

### 35.1 Survey available training data

Research and document available sources of experimental association parameters:

**Databases to check**:
1. **Esper et al. (Figshare 6821654)**: Check if the dataset includes 2B or 4C association scheme labels and ε_AB/κ_AB values for any molecules
2. **NIST ThermoData Engine**: Check availability of association parameters
3. **Dortmund Data Bank (DDB)**: Known to have association parameters for some systems; check access terms
4. **Literature compilations**: Papers by Gross & Sadowski (2001, 2002), Kontogeorgis & Folas (2010) contain association parameter tables
5. **ML-SAFT (Felton et al.)**: Check if the MIT-licensed dataset includes association parameters

For each source, record:
- Number of molecules with ε_AB and κ_AB
- Association scheme labels (1A, 2B, 3B, 4C, etc.)
- License / access terms
- Data format and quality

**Expected finding**: Sparse data. Most PC-SAFT parameter compilations report association parameters for 200-500 molecules (primarily alcohols, carboxylic acids, amines, and water). This is far less than the 13,764 non-associating training set.

### 35.2 Assess EOS backend support

Check teqp's capabilities for associating PC-SAFT:

```python
import teqp

# Test if teqp supports association parameters
# Try creating a PC-SAFT model with association scheme
model = teqp.PCSAFTEOS(
    m=[2.0], sigma=[3.5], epsilon_k=[200.0],
    epsilon_AB=[2500.0], kappa_AB=[0.03],
    scheme=["2B"]
)
```

Document:
- Does teqp support association schemes (1A, 2B, 3B, 4C)?
- What is the API for specifying association parameters?
- Are there numerical stability issues with VLE calculations for associating systems?
- If teqp does not support association, what alternatives exist? (e.g., FeOs, CoolProp, SAFT-gamma-Mie via Clapeyron.jl)

### 35.3 Define model architecture requirements

If/when association prediction is attempted, document the architecture options:

**Option A: Separate model**
- Train a dedicated classifier for association scheme (none, 1A, 2B, 3B, 4C)
- Train a separate regressor for ε_AB and κ_AB (conditional on scheme != none)
- Pro: Simpler, can use different training data sizes
- Con: No shared learning between non-associating and associating parameters

**Option B: Extended multi-task model**
- Add two output heads to the GNN: ε_AB and κ_AB
- Add a classification head for association scheme
- Train with masked loss (ε_AB/κ_AB loss only for molecules with association data)
- Pro: Shared molecular representation
- Con: Severe class imbalance (~96% non-associating), potential for degrading non-associating predictions

**Option C: Transfer learning**
- Start from the trained non-associating GNN
- Fine-tune with association data only
- Pro: Leverages existing learned representations
- Con: Small fine-tuning dataset may cause overfitting

Recommend an architecture with rationale.

### 35.4 Define gating criteria

Write explicit go/no-go criteria for starting the association parameter phase:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Training data size | >= 300 molecules with ε_AB, κ_AB | Minimum for meaningful ML training with cross-validation |
| Association scheme diversity | >= 3 scheme types represented | Avoid overfitting to a single scheme (e.g., all 2B) |
| Data license | Permissive (CC-BY or MIT) | Must be redistributable |
| EOS backend support | teqp or alternative supports associating PC-SAFT | Cannot validate predictions without EOS |
| Non-associating model stable | Step 31 complete, GNN R² >= 0.73 | Don't risk degrading the primary model |

If any gate is not met, the milestone is **deferred** with a documented reason.

### 35.5 Write research plan

Produce `docs/association_parameter_plan.md` containing:

1. **Data survey results** (from 35.1)
2. **EOS backend assessment** (from 35.2)
3. **Architecture recommendation** (from 35.3)
4. **Go/no-go evaluation** against gating criteria (from 35.4)
5. **Timeline estimate**: If gates are met, estimate:
   - Data collection and cleaning: 1-2 days
   - Model training and evaluation: 1-2 days
   - Integration into serving/portal: 1 day
   - Total: ~1 week
6. **Risk register**: What could go wrong (data too sparse, EOS instability, model degradation)

### 35.6 Add limitations notice to portal

Update `portal/components/predictor.py` to show a visible warning when the user submits a molecule that `is_associating()` returns True for:

```python
from screening.filters import is_associating

if is_associating(smiles):
    st.warning(
        "This molecule has hydrogen-bonding sites (OH, NH, COOH, etc.). "
        "Current predictions cover only non-associating PC-SAFT parameters "
        "(m, σ, ε/k). Association parameters (ε_AB, κ_AB) are not predicted. "
        "Thermodynamic calculations for this molecule will be less accurate."
    )
```

Check if a similar warning already exists. If so, ensure it is prominent and uses `st.warning()` (not just a footnote).

### 35.7 Report

Write `docs/reports/35_association_parameter_scoping.md` summarizing:
- Data survey findings
- EOS backend assessment
- Architecture recommendation
- Go/no-go evaluation
- Portal limitation notice added

## Artifacts

| File | Description |
|------|-------------|
| `docs/association_parameter_plan.md` | Full research plan |
| `docs/reports/35_association_parameter_scoping.md` | Step report |
| `portal/components/predictor.py` | Updated with association warning |

## Success Criteria

- [ ] Data survey documents >= 3 potential sources with molecule counts
- [ ] EOS backend assessment documents teqp's association support (or lack thereof)
- [ ] Architecture options compared with recommendation
- [ ] Gating criteria defined with clear thresholds
- [ ] Research plan written with timeline estimate
- [ ] Portal shows warning for associating molecules
- [ ] Report documents all findings
- [ ] `ruff check .` passes

## When to Move On

- Research plan is complete with clear go/no-go evaluation
- Portal warning is visible and tested
- All gates evaluated (even if some are "not met" — that's a valid outcome)

## Budget

15 minutes. This is research and documentation, not implementation. The only code change is the portal warning (if not already present).
