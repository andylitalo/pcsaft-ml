# Why Association Parameters Are Out of Scope

## What Are Association Parameters?

The non-associating PC-SAFT model uses three pure-component parameters (m, σ, ε/k) to capture chain length, segment size, and dispersion attraction. Molecules that form hydrogen bonds — alcohols, carboxylic acids, amines, water — require two additional **association parameters**:

- **ε^AB/k** (association energy, K): depth of the site-site hydrogen-bonding potential well
- **κ^AB** (association volume, dimensionless): effective bonding volume governing the probability that two association sites are close enough to interact

Each molecule is also assigned an **association scheme** (Huang & Radosz nomenclature: 1A, 2B, 3B, 4C, etc.) describing the number and type of donor/acceptor sites. For example, alcohols are typically modeled as 2B (one donor, one acceptor), carboxylic acids as 1A (dimerizing), and water as 4C (two donors, two acceptors).

Without these parameters, the 3-parameter model conflates hydrogen-bonding energy into ε/k, making associating molecules' parameters physically incomparable to those of non-associating molecules. See the [PC-SAFT primer](pcsaft_primer.md) for background on the three parameters we do predict.

---

## Why They Are Out of Scope

### 1. The reference compound is non-associating

Cyclopentane has no hydrogen-bond donors or acceptors. Its thermodynamic behavior is fully described by the 3-parameter model. There is no scientific need for association parameters to characterize the screening target.

### 2. The candidate space is non-associating

Most HFOs, HCFOs, and HFCs in the screened olefin space contain only C, F, H, and Cl atoms. C–F bonds are poor hydrogen-bond donors; C–Cl bonds are weak donors at best. The candidate molecules most relevant to the blowing agent question do not require association parameters for accurate thermodynamic modeling.

### 3. Training data is too sparse

Only approximately 200–500 compounds in the open literature have experimentally fitted association parameters (ε^AB/k and κ^AB). The best sources — the Dortmund Data Bank (DDB) and DIPPR — are commercially licensed and not redistributable. The Esper dataset, which anchors this project's training, does not include association parameters at all. The available open data is dominated by 2B-scheme alcohols, with minimal representation of other schemes (1A, 3B, 4C), creating severe class imbalance.

For context, the non-associating model trains on 1,801 molecules with ~2,150 features. Fitting a model on <500 molecules with the same feature dimensionality, unbalanced across association schemes, is not viable for meaningful generalization.

### 4. The EOS backend does not support association

teqp, the equation-of-state backend used throughout the pipeline for property computation, VLE calculations, and Henry's constant evaluation, does not implement associating PC-SAFT. Supporting association would require switching to an alternative backend:

| Backend | Association support | Language | License |
|---------|-------------------|----------|---------|
| teqp | No | C++/Python | MIT |
| `pcsaft` (Clapeyron.jl) | Yes | Julia | MIT |
| FeOs | Yes | Rust/Python | (varies) |
| Custom differentiable PC-SAFT | Yes (if built) | Python/PyTorch | — |

Any switch would require rewriting the thermodynamic validation pipeline (`model/thermodynamic.py`), the screening filters, and the serving layer — substantial engineering effort for uncertain scientific payoff.

### 5. Five-parameter screening is strictly harder than three

The screening already fails to find a close 3-parameter match to cyclopentane among HFOs. Requiring simultaneous agreement on five parameters (m, σ, ε/k, ε^AB/k, κ^AB) reduces the probability of finding viable candidates rather than improving it. The additional constraint space makes the search harder without changing the conclusion: no pure HFO is a thermodynamic drop-in for cyclopentane.

### 6. The state of the art is not reproducible with open data

Winter et al. (2025, SPT-PCSAFT) achieved the best published results for ML-based PC-SAFT parameter prediction by training a SMILES transformer end-to-end on experimental VP and density data through a differentiable PC-SAFT implementation. This approach sidesteps the parameter non-uniqueness problem entirely. However, it requires commercially licensed experimental data (NIST TDE / DDB) and a differentiable EOS that supports association — a fundamentally different architecture that cannot be reproduced with open data alone.

---

## Limitations of the 3-Parameter Scope

The decision to exclude association parameters introduces known limitations:

- **Strongly polar or H-bonding compounds** (alcohols, acids, amines): their ε/k values in 3-parameter fits conflate dispersion with association energy, making them incomparable to non-associating compounds. Predictions for such molecules should not be trusted.
- **Fluorinated molecules**: strong C–F dipoles are partially conflated with dispersion energy in the 3-parameter model. The fit compensates for missing polarity by inflating ε/k, contributing to prediction error and parameter-space bias for fluorinated compounds.
- **Solvent choice for Henry's constant**: the pipeline uses n-hexane rather than a polyol as the model solvent specifically because polyols are associating and the 3-parameter model cannot represent them. See [why n-hexane](hexane_solvent_choice.md) for the full justification.

---

## Gating Criteria for Future Revisit

If association parameters were to be pursued in a future phase, all of the following gates must be met:

| Gate | Threshold | Current status |
|------|-----------|---------------|
| Training data size | >= 300 molecules with ε^AB, κ^AB | Uncertain — open data likely below threshold |
| Association scheme diversity | >= 3 scheme types represented | Unlikely — most open data is 2B (alcohols) |
| Data license | Permissive (CC-BY or MIT) | Fails — richest sources (DDB, DIPPR) are commercial |
| EOS backend support | teqp or alternative supports associating PC-SAFT | Fails — teqp does not support association |
| Property-level usefulness | Clear plan for improvement in VLE, density, or Henry's constant | Cannot test without backend |
| Non-associating product stable | Deployed and stable | Met |

If any gate is not met, the milestone is deferred. **Deferral is the expected outcome.**

---

## Architecture Options if Pursued

Three approaches have been scoped for a hypothetical future phase:

**Option A — Separate model**: Train a dedicated classifier for association scheme (none/1A/2B/3B/4C) and a separate regressor for ε^AB and κ^AB, conditional on the molecule being associating. Simpler to implement; no risk of degrading non-associating predictions. Cannot leverage shared molecular representations.

**Option B — Extended multi-task model**: Add ε^AB, κ^AB, and scheme classification heads to the existing GNN. Train with masked loss (association loss only for molecules with labeled data). Shares the molecular representation but introduces severe class imbalance (~96% non-associating) and risks degrading the primary predictions.

**Option C — Transfer learning**: Start from the trained non-associating GNN; fine-tune on association data only. Leverages learned representations but risks overfitting on a small fine-tuning dataset.

None of these options are scheduled for implementation.

---

## Bottom Line

The project's negative result — no fluorinated olefin can serve as a thermodynamic drop-in for cyclopentane — does not change under 5-parameter screening. The candidates that fail on ε/k in the 3-parameter model would fail by a wider margin if association parameters further constrained the search. The scope limitation is scientifically justified, practically necessary given data and tooling constraints, and does not weaken the screening conclusion.

---

## References

- Gross, J. & Sadowski, G. (2001). Perturbed-Chain SAFT: An Equation of State Based on a Perturbation Theory for Chain Molecules. *Ind. Eng. Chem. Res.*, 40(4), 1244–1260.
- Gross, J. & Sadowski, G. (2002). Application of the Perturbed-Chain SAFT Equation of State to Associating Systems. *Ind. Eng. Chem. Res.*, 41(22), 5510–5515.
- Huang, S.H. & Radosz, M. (1990). Equation of State for Small, Large, Polydisperse, and Associating Molecules. *Ind. Eng. Chem. Res.*, 29(11), 2284–2294.
- Winter, B. et al. (2025). SPT-PCSAFT: Understanding the language of molecules. *Digital Discovery*, 4, 1142–1157.
- Esper, T. et al. (2023). Systematic PC-SAFT parameter regression for 1,842 substances.
