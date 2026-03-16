# Future Work: Association Parameter Milestone Scoping

> **Status**: Long-term research milestone. Not part of the current numbered step sequence.
> Deferral is the expected outcome of the gating analysis below.
> Revisit only after the hosted non-associating product (Steps 31-34) is stable.

## Objective

Produce a formal go/no-go memo on whether predicting associating PC-SAFT parameters (ε_AB, κ_AB) should remain deferred. This is primarily a scope-protection and limitations-documentation exercise, not a commitment to build a five-parameter model next. It may refresh the portal warning for associating molecules if the existing notice is not prominent enough.

## Motivation

The current pipeline predicts only three non-associating PC-SAFT parameters (m, σ, ε/k). Molecules with hydrogen-bonding sites (OH, NH, COOH, etc.) require two additional association parameters (ε_AB, κ_AB) for accurate thermodynamic modeling. The existing `is_associating()` utility in `screening/filters.py` flags these molecules but offers no prediction capability.

Previous steps and `docs/limitations.md` already treat this as out of scope for the main product. The purpose of this milestone is to make that decision explicit and evidence-based so the hosted portal does not drift into half-supported associating chemistry.

Key questions this milestone answers:
1. How many molecules with association parameters exist in accessible databases?
2. Does teqp (or an alternative EOS backend) support associating PC-SAFT?
3. What model architecture changes are needed for 5-parameter prediction?
4. Is there enough evidence to justify changing scope, or should association remain deferred?

## Dependencies

- `screening/filters.py` exists with `is_associating()` function
- `serving/model_loader.py` exists with association warning logic
- `portal/app.py` or `portal/components/predictor.py` exists
- No dependency on numbered steps for the memo itself, but any decision to expand scope should not compete with the main hosted-product path until the Cloud Run deployment is stable

## Implementation Guide

### Survey available training data

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

### Assess EOS backend support

Check teqp's capabilities for associating PC-SAFT. Verify the actual Python API (the constructor syntax may differ from keyword arguments):

```python
import teqp

# Test if teqp supports association parameters
# Verify actual API against teqp docs before relying on this example
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

### Define model architecture requirements

If association prediction were attempted later, document the architecture options:

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

Recommend an architecture only as part of the memo. Do not frame this as the next scheduled implementation unless the gates below are convincingly met.

### Define gating criteria

Write explicit go/no-go criteria for starting the association parameter phase:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Training data size | >= 300 molecules with ε_AB, κ_AB | Minimum for meaningful ML training with cross-validation |
| Association scheme diversity | >= 3 scheme types represented | Avoid overfitting to a single scheme (e.g., all 2B) |
| Data license | Permissive (CC-BY or MIT) | Must be redistributable |
| EOS backend support | teqp or alternative supports associating PC-SAFT | Cannot validate predictions without EOS |
| Property-level usefulness | Clear plan to show improvement in VLE, density, Henry's constant, or other downstream thermodynamic targets | Parameter prediction alone is not enough |
| Non-associating product stable | Cloud Run deployment complete and stable | Don't dilute the primary deliverable |

If any gate is not met, the milestone is **deferred** with a documented reason. Deferral is an acceptable and likely outcome.

### Write decision memo / research plan

Produce `docs/association_parameter_plan.md` containing:

1. **Data survey results**
2. **EOS backend assessment**
3. **Architecture recommendation**
4. **Go/no-go evaluation** against gating criteria
5. **Recommendation**: one of `defer`, `revisit after hosted release`, or `proceed to a dedicated research phase`
6. **Timeline estimate**: If gates are met, estimate:
   - Data collection and cleaning: 1-2 days
   - Model training and evaluation: 1-2 days
   - Integration into serving/portal: 1 day
   - Total: ~1 week
7. **Risk register**: What could go wrong (data too sparse, EOS instability, model degradation)

### Refresh limitations notice in portal if needed

Check the existing associating-molecule notice first. If it is missing or too subtle, update `portal/components/predictor.py` to show a visible warning when the user submits a molecule that `is_associating()` returns True for:

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

Check if a similar warning already exists. If so, prefer updating wording/prominence rather than introducing a second overlapping warning.

## Artifacts

| File | Description |
|------|-------------|
| `docs/association_parameter_plan.md` | Full research plan / go/no-go memo |
| `portal/components/predictor.py` | Updated with association warning (if needed) |

## Success Criteria

- [ ] Data survey documents >= 3 potential sources with molecule counts
- [ ] EOS backend assessment documents teqp's association support (or lack thereof)
- [ ] Architecture options compared with recommendation
- [ ] Gating criteria defined with clear thresholds
- [ ] Memo makes an explicit `go` / `defer` recommendation
- [ ] Research plan written with timeline estimate only if the gates are plausibly met
- [ ] Portal shows a clear warning for associating molecules, if the existing warning was insufficient

## Budget

15 minutes. This is research and documentation, not implementation. The only code change is the portal warning (if not already present).
