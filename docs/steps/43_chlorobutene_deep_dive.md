# Step 43: Chlorobutene Thermodynamic Near-Hit Deep-Dive

## Objective

Audit whether the chlorobutene near-hits identified in Steps 20–21 — **1-chlorobut-1-ene**
(`C=C(Cl)CC`), **(Z)-2-chloro-2-butene** (`C/C=C(/C)Cl`), and **(E)-2-chloro-2-butene**
(`C/C=C(\C)Cl`) — represent robust thermodynamic signals or artifacts of model uncertainty,
domain extrapolation, and reference disagreement. This is a retrospective evidence-synthesis
step, not a model-training step.

The conclusion should reinforce the project's main narrative: these molecules remain
**practically non-viable** regardless of thermodynamic assessment, but the scientific
question of signal robustness deserves a clear answer.

## Motivation

Steps 20 and 21 identified these three chlorobutenes as the **only** molecules passing all
five criteria in the systematic thermodynamic screen (4,663 candidates). Step 21 reported
H/H_ref ~ 1.02–1.05, suggesting mixture-behavior proximity to cyclopentane. Step 23 used
them as part of the argument that the screening question was scientifically closed, noting
their practical non-viability (ODP, toxicity, reactivity, regulatory barriers).

However, this signal has not been independently audited:

- RF and SPT-PCSAFT disagree by ~24 K on epsilon_k for 1-chlorobut-1-ene (RF: 267.8 K,
  SPT: 243.7 K), which qualitatively changes the Henry's-law verdict
- The targets are **not** in the Esper training set, so nearest-neighbor and applicability
  domain evidence matters
- The NIST validation (Step 17) showed that halogenated and polar compounds can be
  failure modes for the RF model (dichloromethane VP error: 1,858%)
- The chlorinated-alkene regime has not been isolated and audited on its own

This step answers: **How confident should we be that chlorobutene is a real thermodynamic
near-hit rather than an artifact?**

## Evidence Hierarchy

Interpret evidence in this strict order:

1. **External data first**: experimental boiling point, vapor pressure, or literature
   PC-SAFT/EOS parameters
2. **Local chemistry benchmark second**: nearest chlorinated alkenes already in Esper
3. **Applicability domain and RF uncertainty third**: Tanimoto similarity plus tree-based
   uncertainty propagation
4. **RF vs SPT disagreement last**: useful diagnostic, but not primary validation

## Dependencies

- Step 20 completed: chlorobutene candidates identified in systematic screen
- Step 21 completed: Henry's constant analysis with H/H_ref values
- Step 23 completed: narrative report synthesizing the screening conclusion
- Step 17 completed: NIST validation providing halogenated-compound failure context
- `model/predict.py` with `predict_pcsaft_with_ci()` available
- `model/uncertainty.py` with `combined_uncertainty()` and `get_rf_tree_predictions()`
- `model/ad_tanimoto.py` with `TanimotoAD` class
- `model/data/esper_pcsaft.csv` with chlorinated alkene training molecules
- `model/data/spt_pcsaft.csv` with SPT parameters for the target molecules

## Implementation Guide

### 43.1 Literature and external-data check

Search for published data on the three chlorobutene near-hits, prioritizing:

- Experimental boiling point
- Experimental vapor pressure near 298 K
- Published PC-SAFT or other EOS parameters
- Any VLE, density, or thermophysical property data

**Known literature values** (from standard references):

| Molecule | CAS | T_b exp (K) | Source |
|----------|-----|-------------|--------|
| 1-chlorobut-1-ene (cis) | 4894-61-5 | ~337 K (64°C) | CRC Handbook |
| 2-chloro-2-butene (mixture/Z) | 2211-68-9 | ~341 K (68°C) | CRC Handbook |
| 2-chloro-2-butene (E) | 2211-67-8 | ~336 K (63°C) | CRC Handbook |

If boiling point data is found, compare the EOS predictions from both RF and SPT
parameters to experiment by computing T_b from each parameter set via the
`screening.hfo_screening.compute_boiling_point()` function. This is the strongest
available adjudication.

### 43.2 Benchmark nearest chlorinated alkenes in Esper

Use the structurally closest chlorinated alkenes already present in the Esper training set.
These provide the best available local performance benchmark:

| SMILES | Name | Esper m | Esper σ | Esper ε/k |
|--------|------|---------|---------|-----------|
| `C=CCl` | Vinyl chloride | 1.938 | 3.469 | 239.95 |
| `C=CCCl` | Allyl chloride (3-chloro-1-propene) | 2.440 | 3.491 | 254.13 |
| `C=C(C)Cl` | 2-Chloropropene | 1.913 | 3.820 | 273.17 |
| `C=C(Cl)Cl` | 1,1-Dichloroethene | 2.039 | 3.676 | 275.86 |
| `ClC=CCl` | 1,2-Dichloroethene | 2.212 | 3.560 | 285.87 |

For each neighbor, compute or tabulate:

1. **RF prediction vs Esper ground truth**: use `predict_pcsaft()` to predict each
   neighbor's parameters and compare against the known Esper values. Compute signed
   error for each parameter to determine bias direction.

2. **SPT-PCSAFT values**: look up each neighbor in `model/data/spt_pcsaft.csv` where
   available. Compare SPT vs Esper to characterize SPT bias in this chemical subspace.

3. **Bias direction for epsilon_k**: determine whether RF tends to over- or under-predict
   epsilon_k for chlorinated alkenes, and whether SPT shows a consistent offset relative
   to Esper. This is the critical diagnostic — if RF systematically over-predicts epsilon_k
   for chlorinated alkenes, the 267.8 K value for 1-chlorobut-1-ene is likely inflated,
   and the near-hit conclusion is weakened.

4. **Leave-one-out analysis**: if practical, retrain the RF with each chlorinated alkene
   held out and predict its parameters. If not practical within the time budget, use the
   residuals from the full model (which is a lower bound on LOO error since the molecule
   is in the training set).

Weight this local benchmark more heavily than any global metric.

### 43.3 Compute Tanimoto applicability-domain status

Use `TanimotoAD` from `model/ad_tanimoto.py`:

```python
from model.ad_tanimoto import TanimotoAD
import pandas as pd

esper = pd.read_csv("model/data/esper_pcsaft.csv")
ad = TanimotoAD(radius=2, n_bits=2048)
ad.fit(esper["smiles"].tolist())

targets = ["C=C(Cl)CC", "C/C=C(/C)Cl", "C/C=C(\\C)Cl"]
for smi in targets:
    sim = ad.tanimoto_nn(smi)
    status = "in_domain" if sim >= 0.4 else ("warning" if sim >= 0.3 else "ood")
    print(f"{smi}: Tanimoto={sim:.3f} ({status})")
```

Report for each target:

- Max Tanimoto similarity to the Esper training set
- Identity of the nearest training molecule (find the SMILES with max similarity)
- In-domain / warning / out-of-domain classification (thresholds: 0.4 / 0.3)

Interpret as an applicability-domain diagnostic, not a pass/fail proof of validity.
Cross-reference with the neighbor benchmark: if the nearest training molecule is one of
the chlorinated alkenes from Section 43.2, the local bias analysis directly applies to
the target.

### 43.4 Compute parameter-level uncertainty for the target molecules

Use `predict_pcsaft_with_ci()` from `model/predict.py`:

```python
from model.predict import predict_pcsaft_with_ci

targets = ["C=C(Cl)CC", "C/C=C(/C)Cl", "C/C=C(\\C)Cl"]
ci_df = predict_pcsaft_with_ci(targets)
print(ci_df.to_string(index=False))
```

Report `m`, `sigma`, and `epsilon_k` with RF tree-variance intervals (95% CI). Note
explicitly in the report that:

- These intervals are approximate and under-cover for epsilon_k (90.3% empirical coverage
  at the nominal 95% level, from Step 39 calibration)
- CI overlap with SPT should be treated as suggestive, not decisive
- The ε/k interval is the most consequential because Henry's constant is exponentially
  sensitive to cross-interaction energy

### 43.5 Propagate uncertainty through the EOS

Use `combined_uncertainty()` from `model/uncertainty.py` to propagate per-tree joint
`(m, sigma, epsilon_k)` through teqp:

```python
from model.registry import _compute_features, get_model
from model.uncertainty import get_rf_tree_predictions, combined_uncertainty
from model.thermodynamic import CYCLOPENTANE, HEXANE, T_REF

rf = get_model("rf")
rf.load()
rf_models = rf._models

targets = ["C=C(Cl)CC", "C/C=C(/C)Cl", "C/C=C(\\C)Cl"]
X = _compute_features(targets)
tree_preds = get_rf_tree_predictions(rf_models, X)

result = combined_uncertainty(
    tree_preds,
    k_ij_values=(-0.05, -0.025, 0.0, 0.025, 0.05),
    reference=CYCLOPENTANE,
    solvent=HEXANE,
    T=T_REF,
    ci=0.95,
)
```

Report for each target:

- `VP_ratio` 95% interval (tree-only, independent of k_ij)
- `H/H_ref` 95% interval: tree-only (k_ij=0) and combined (tree + k_ij sweep)

The interpretation goal is qualitative:

- Does the target remain broadly in-band (H/H_ref ∈ [0.5, 2.0]) for thermodynamic
  similarity even at the CI boundaries?
- Or do the propagated intervals become wide enough that the near-hit conclusion is
  unstable?

Avoid overstating this as proof of absolute accuracy.

### 43.6 RF vs SPT-PCSAFT comparison as supporting evidence

The SPT-PCSAFT dataset contains all three target molecules (lines 2308–2310). Tabulate
the disagreement:

| Parameter | RF | SPT | Delta | Implication for H ratio |
|-----------|-----|-----|-------|-------------------------|
| m | 2.60 | 2.83 | -0.23 | |
| σ (Å) | 3.704 | 3.605 | +0.099 | |
| ε/k (K) | 267.8 | 243.7 | +24.1 | RF: H~1.05; SPT: H~30–60 |

Use this section to diagnose disagreement:

- SPT predicts epsilon_k ~ 244 K, consistent with the Esper values for vinyl chloride
  (240 K) and allyl chloride (254 K). The chlorobutenes are one CH₂ group longer, and
  SPT places them at the bottom of the chlorinated-alkene epsilon_k range.
- RF predicts epsilon_k ~ 268 K, ~24 K higher. If RF systematically over-predicts
  epsilon_k for chlorinated alkenes (as tested in Section 43.2), this offset may
  be partially or fully explained by local bias.

SPT is not ground truth — it has known chemistry-dependent biases and its own
uncertainty — but the direction of disagreement is informative when combined with the
local Esper benchmark.

### 43.7 Narrow chlorinated-compound bias assessment

Use the NIST validation findings from `docs/reports/17_nist_experimental_validation.md`
to contextualize the chlorobutene case. Do **not** build a broad polarity-proxy analysis.

Key points to address:

1. **NIST validation on halogenated compounds**: Step 17 found that dichloromethane (a
   small polar chlorinated molecule) had VP error of 1,858%. The report concluded that
   "polar compounds are problematic" and specifically called out dichloromethane as a
   failure mode. However, dichloromethane is a dichloride, not a vinylic monochloride.

2. **Compare chlorinated alkene neighbors to those failure modes**: The Esper chlorinated
   alkenes (vinyl chloride, allyl chloride, 2-chloropropene) are vinylic monochlorides
   like the targets. Their RF residuals (from Section 43.2) indicate whether vinylic
   monochlorides share dichloromethane's failure mode or behave more like tolerable
   halogenated compounds (e.g., the chlorinated refrigerant R-22, which was included in
   the boiling point validation at Step 24).

3. **Classify the chlorobutene regime**: argue whether a C₄ vinylic monochloride is more
   likely to behave like:
   - A tolerable case (moderate polarity, RF captures the key features)
   - A severe dipolar failure mode (strong dipole from C=C–Cl polarization pushes the
     molecule out of the RF model's reliable regime)

This keeps the analysis tied to evidence the repo already supports.

### 43.8 Create the driver script

Create `scripts/step43_chlorobutene_deep_dive.py` that:

1. Runs the literature comparison (compute T_b from both RF and SPT parameters)
2. Benchmarks the five nearest Esper chlorinated alkenes (RF predictions vs ground truth)
3. Computes Tanimoto AD for the three targets
4. Computes parameter-level CIs via `predict_pcsaft_with_ci()`
5. Runs `combined_uncertainty()` for EOS-propagated intervals
6. Tabulates the RF vs SPT disagreement
7. Saves all results to `model/saved/chlorobutene_deep_dive.csv`
8. Generates figures to `figures/43_chlorobutene_deep_dive/`

Required figures:

1. **`neighbor_benchmark.png`**: Bar chart of RF prediction error (RF − Esper) for each
   parameter across the five chlorinated-alkene neighbors. Highlights whether RF has a
   systematic epsilon_k over-prediction bias.

2. **`parameter_ci_comparison.png`**: Dot plot showing RF point estimates with 95% CI
   bars for each target molecule, overlaid with SPT values and (where available) Esper
   neighbor range. Three panels: m, sigma, epsilon_k.

3. **`henry_ratio_ci.png`**: Forest plot of H/H_ref for each target: point estimate,
   tree-only 95% CI, and combined (tree + k_ij) 95% CI. Acceptance band [0.5, 2.0]
   shaded. Include the SPT-implied H/H_ref as a separate marker if computable.

4. **`ad_similarity_heatmap.png`** (optional): Tanimoto similarity matrix between
   targets and the top 5 nearest Esper neighbors.

### 43.9 Write Step 43 report

Write `docs/reports/43_chlorobutene_deep_dive.md` following the standard report template
from `CLAUDE.md`, adapted for an analysis step:

**Required sections**:

1. **Title**: `# Results Report: Chlorobutene Thermodynamic Near-Hit Deep-Dive`
2. **Assessment setup**: data sources, evidence hierarchy, target molecules
3. **Literature evidence**: boiling point comparison, any experimental property data
4. **Nearest-neighbor Esper benchmark**: RF bias characterization on local chlorinated
   alkenes, with signed residuals and bias direction
5. **Tanimoto AD status**: similarity scores, nearest-neighbor identity, domain
   classification
6. **Parameter-level uncertainty**: RF CIs with coverage caveats
7. **EOS-propagated uncertainty**: VP_ratio and H/H_ref intervals
8. **RF vs SPT diagnostic**: disagreement table with interpretation
9. **Chlorinated-compound bias discussion**: contextualization against NIST findings
10. **Comparison to prior claims**: revisit the specific claims in Steps 20, 21, and 23
11. **Key findings**: bullet-point summary
12. **Conclusion**: one of three outcomes:
    - **Robust thermodynamic near-hit**: external/local evidence supports the signal
    - **Ambiguous near-hit**: evidence is mixed, uncertainty too wide for confidence
    - **Likely artifact**: local benchmarking and/or external data argue against the match
13. **Figures**: reference all saved figures
14. **Limitations**: analysis scope, data quality, model limitations
15. **Readiness check**: checklist confirming all criteria are met

The conclusion must reinforce the project's main narrative. This step **resolves** the
last scientific ambiguity around the chlorinated near-hits — it does not reopen the
project's conclusion or reposition chlorobutene as a practical candidate.

## Key Files

| File | Purpose |
|------|---------|
| `model/predict.py` | `predict_pcsaft_with_ci()` |
| `model/uncertainty.py` | `combined_uncertainty()`, `get_rf_tree_predictions()` |
| `model/ad_tanimoto.py` | `TanimotoAD` |
| `model/registry.py` | `RFModel.predict_with_uncertainty()` |
| `model/data/esper_pcsaft.csv` | Training data with chlorinated alkene ground truth |
| `model/data/spt_pcsaft.csv` | SPT parameters for targets and neighbors |
| `scripts/propagate_uncertainty.py` | Reference for uncertainty propagation pattern |
| `docs/reports/17_nist_experimental_validation.md` | Halogenated-compound failure context |
| `docs/reports/20_systematic_hfo_hcfo_enumeration.md` | Chlorobutene discovery |
| `docs/reports/21_henrys_constant_analysis.md` | Chlorobutene H/H_ref = 1.02–1.05 |
| `docs/reports/23_ml_chemistry_narrative.md` | Project-level conclusion to stay consistent with |

## Artifacts

| File | Description |
|------|-------------|
| `scripts/step43_chlorobutene_deep_dive.py` | Single driver script for all analyses |
| `docs/reports/43_chlorobutene_deep_dive.md` | Full report with confidence assessment |
| `figures/43_chlorobutene_deep_dive/` | CI plots, AD analysis, neighbor benchmark figures |
| `model/saved/chlorobutene_deep_dive.csv` | Reproducible summary table for targets and neighbors |

## Success Criteria

- [ ] Step is clearly framed as **Step 43**, with no numbering conflict
- [ ] Chlorobutene is treated as a **thermodynamic near-hit under audit**, not a practical candidate
- [ ] Literature boiling point comparison is performed (RF-predicted T_b vs SPT-predicted T_b vs experimental)
- [ ] Nearest-neighbor chlorinated-alkene benchmarking is completed against ≥ 4 Esper molecules
- [ ] RF bias direction for epsilon_k in chlorinated alkenes is determined (over- or under-prediction)
- [ ] Tanimoto similarity and nearest-neighbor identity are reported for all three targets
- [ ] Parameter-level CIs are computed and epsilon_k coverage caveat is noted
- [ ] EOS-propagated `H/H_ref` / `VP_ratio` intervals are computed and interpreted cautiously
- [ ] RF vs SPT disagreement is discussed as supporting diagnostic evidence only
- [ ] Chlorinated-compound bias is contextualized against NIST validation findings
- [ ] Final report assigns one of three outcomes: robust near-hit, ambiguous near-hit, or likely artifact
- [ ] Report does not reopen the project's main conclusion (no practical drop-in exists)
- [ ] All figures saved to `figures/43_chlorobutene_deep_dive/`
- [ ] `model/saved/chlorobutene_deep_dive.csv` contains all computed data
- [ ] All existing tests pass
- [ ] `ruff check .` passes

## When to Move On

- The chlorobutene near-hit question has a clear, evidence-based answer
- The evidence hierarchy was followed (external data > local benchmark > AD/uncertainty > RF vs SPT)
- The report's conclusion is consistent with the project's main narrative in Step 23
- No new scientific ambiguities were introduced
- The analysis is reproducible via `scripts/step43_chlorobutene_deep_dive.py`

## Budget

20 minutes. This is a retrospective analysis step using existing infrastructure. No model
training is required. The main computational cost is the `combined_uncertainty()`
propagation for 3 molecules (~100 trees × 5 k_ij × 3 molecules = 1,500 EOS evaluations).
