# Screening Methodology for Blowing Agent Candidates: Supplementary Information

This document explains the screening methodology and filter cascade used to identify blowing agent candidates from enumerated F/Cl-substituted olefins. Candidates are benchmarked against cyclopentane (T_b = 322 K), a reference blowing agent for polyurethane foam.

---

## Why F/Cl-Substituted Olefins?

The candidate space — halogenated olefins with only F and Cl substituents on C2–C6 alkene backbones — is not arbitrary. It follows from three independent constraints: atmospheric chemistry, regulatory history, and modeling scope.

### The double bond: atmospheric degradation and low GWP

A blowing agent trapped in foam insulation leaks slowly over decades. Its climate impact depends on atmospheric lifetime: molecules that persist for years accumulate and warm the planet (high GWP); molecules that degrade in days do not. The C=C double bond provides a fast degradation pathway — OH radicals and ozone attack the π bond, giving atmospheric lifetimes on the order of days to weeks. Without the double bond, fully saturated halocarbons (HFCs like HFC-245fa, HFC-365mfc) persist for years and have GWP in the hundreds to thousands. This is exactly why the Kigali Amendment (2016) mandates an 80–85% phase-down of HFC production. The olefin requirement is not a chemical preference; it is an atmospheric chemistry constraint that the regulatory landscape enforces.

### Why fluorine?

Fluorine provides non-flammability. The C–F bond is the strongest single bond to carbon (~485 kJ/mol), making fluorinated molecules chemically inert and thermally stable during the foam exotherm. High fluorine content correlates with ASHRAE A1 (non-flammable) classification, which is a hard requirement for many building code and appliance applications. Fluorine also contributes zero ODP because F radicals do not catalyze ozone destruction. This combination — non-flammable, zero ODP, and (with a double bond) low GWP — is why HFOs are the current industry standard.

### Why chlorine?

Chlorine is included in the broader search because it maintains stronger dispersion interactions than fluorine. The Cl atom is larger and more polarizable than F, producing higher ε/k values that are closer to cyclopentane's. HCFOs (hydrochlorofluoroolefins) like HCFO-1233zd(E) are already commercial blowing agents — the short atmospheric lifetime from the double bond keeps their ODP very low (~0.00034), even though each Cl atom contributes nonzero ozone depletion potential. Including Cl in the enumeration tests whether the dispersion energy advantage of chlorination can produce thermodynamic near-hits that fluorine alone cannot.

### Why not bromine or iodine?

Bromine has roughly 45–60× the ozone depletion efficiency per atom compared to chlorine. Even with the short atmospheric lifetime conferred by a double bond, Br-containing olefins would have unacceptable ODP. Halons (Br-containing halocarbons) were among the first substances controlled under the Montreal Protocol for this reason. Iodine is excluded on stability and practicality grounds: C–I bonds are weak (~240 kJ/mol), making iodoalkenes thermally and photochemically unstable — they would decompose during the foam exotherm or degrade in storage. Iodinated compounds are also expensive and raise toxicological concerns.

**Note**: Step 53 (expanded screening) relaxes this restriction and includes Br and I for completeness, with applicability domain flagging for prediction confidence. The results confirm the regulatory concerns: the highest-ranked Br-containing candidate (FC(F)=C(Cl)Br, rank 7) achieves higher epsilon_k than typical HFOs but is an ODP nonstarter. Iodinated candidates (ranks 13-14) are flagged as thermally unstable.

### Why not other heteroatoms (O, N, S)?

Molecules containing –OH, –NH, –COOH, or –SH groups are hydrogen-bond donors or acceptors, making them **associating** molecules that require five PC-SAFT parameters (m, σ, ε/k, ε^AB/k, κ^AB) rather than three. The 3-parameter non-associating model used in this project cannot represent them accurately — see [why association is out of scope](association_parameters_out_of_scope.md). Ethers (C–O–C) can be non-associating, but they introduce additional concerns: reactivity with isocyanate during PU foam synthesis, different atmospheric degradation pathways, and distinct regulatory classifications. The F/Cl olefin space keeps the chemistry within the scope of both the model and the regulatory framework.

### Why C2–C6 backbones?

The backbone length is set by the boiling point window. Blowing agents must vaporize during the foam exotherm (upper bound ~322 K, matching cyclopentane) and remain gaseous in the final product (lower bound ~288 K). C1 cannot form an olefin. Heavily halogenated C2 molecules tend to be gases well below the target range. C7+ olefins with halogenation would be too heavy, with boiling points above the foam processing window. The C2–C6 range brackets the practical volatility envelope with margin on both ends.

### Why not entirely different molecular classes?

- **Saturated alkanes**: already explored — cyclopentane is one, and it defines the reference. Other alkanes (n-pentane, isopentane) are commercial blowing agents with similar limitations (flammable, VOC). Saturated halocarbons (HFCs) have high GWP as discussed above.
- **Aromatics**: too heavy for the boiling point window at low carbon number, and raise toxicological concerns (benzene is a Group 1 carcinogen; toluene is a regulated VOC).
- **Inorganic gases** (CO2, N2): already used as physical blowing agents, but they represent fundamentally different foam technology (supercritical CO2 blowing, water-blown CO2 generation). They cannot be meaningfully compared in the PC-SAFT parameter similarity framework because the question "which molecule has similar thermodynamic behavior to cyclopentane?" presupposes a condensable organic blowing agent.

The F/Cl olefin space is where the regulatory trajectory, atmospheric chemistry, and PC-SAFT modeling scope converge. It is also where industry R&D is actively focused, making the screening results directly relevant to ongoing formulation efforts.

---

## Candidate Enumeration

Candidates are generated by systematic enumeration of all fluorine- and chlorine-substituted C2–C6 olefins.

- **Full set**: 16 alkene backbones × all possible halogen substitution patterns = **10,700 candidates**.
- **HFO-specific subset**: 1,471 pure HFOs (hydrofluoroolefins) satisfying: no chlorine, ≥2 fluorine atoms, and fluorine mass fraction ≥ 65%.

The HFO subset targets Montreal Protocol–compliant, low-GWP alternatives; the full set supports broader screening including chlorinated options.

---

## Screening Cascade (7 Filters for HFO Screening)

Candidates in the HFO subset pass through seven sequential filters:

### 1. Boiling Point: 288–323 K (15–50°C)

The blowing agent must vaporize during the foam exotherm and remain gaseous in the final product. The upper bound aligns with cyclopentane’s boiling point (322 K). Boiling points are computed via PC-SAFT vapor–liquid equilibrium using teqp (`pure_VLE_T`).

### 2. C=C Double Bond

The molecule must contain a carbon–carbon double bond. This enables atmospheric degradation via ozone reactions, providing a low-GWP pathway.

### 3. No Chlorine

Chlorine atoms contribute nonzero ozone depletion potential (ODP). Montreal Protocol compliance requires zero-Cl for long-term, sustainable alternatives.

### 4. Fluorine Count ≥ 2

A minimum of two fluorine atoms is required for reduced flammability in the candidate space.

### 5. Synthetic Accessibility (SA) Score ≤ 4.5

Candidates must be practically synthesizable. SA scores are computed via RDKit. Scores above 4.5 indicate structures that are difficult or impractical to manufacture at scale.

### 6. Fluorine Mass Fraction ≥ 65%

ASHRAE 34 heuristic for non-flammability (A1 classification). Higher fluorine content correlates with reduced flammability.

### 7. No Reactive Fluorination Sites

Molecules with fluorine on strained or otherwise reactive positions are filtered out to avoid instability or unwanted chemistry during foam processing or in service.

---

## Thermodynamic Gates (Applied After Structural Filters)

Candidates passing the structural cascade are further screened by thermodynamic criteria:

### 1. EOS Convergence

PC-SAFT vapor–liquid equilibrium must converge at all three temperatures: 273 K, 298 K, and 323 K. Failure indicates numerical instability or incompatibility of the molecule with the equation of state.

### 2. Vapor Pressure Ratio

The ratio VP/VP_cyclopentane must lie in [0.5, 1.5] at 298 K (or at all three temperatures, depending on variant). This ensures similar volatility to the reference agent, which is critical for foam cell structure and processing.

### 3. Property Distance

A weighted parameter distance < 1.0 is required, with weights: ε/k = 5, σ = 2, m = 1. Dispersion energy (ε/k) is emphasized because it dominates mixture behavior through the Boltzmann factor.

### 4. Henry’s Constant Ratio

The ratio H/H_cyclopentane in n-hexane at 298 K must lie in [0.5, 2.0] (strict gate) or [0.1, 10.0] (loose gate). Values are computed via teqp. This measures dissolution similarity to the reference in a representative solvent.

---

## Cyclopentane-Centric Screening (Broader, Includes Cl)

A broader screening track allows chlorinated candidates and uses cyclopentane as the reference:

- Top candidates are ranked by weighted parameter distance.
- Additional filters: vapor pressure ratio, EOS convergence, and synthetic accessibility score.
- This track identified three chlorinated butenes that pass all cyclopentane-centric criteria.

---

## Why These Specific Ranges?

| Criterion | Rationale |
|-----------|-----------|
| **Boiling point 288–323 K** | Lower bound ensures the agent is a gas at typical foam service temperatures; upper bound matches the foam exotherm range where vaporization must occur. |
| **VP ratio [0.5, 1.5]** | Ensures similar vapor pressure behavior across the operating temperature range, critical for foam cell structure. |
| **Henry’s ratio [0.5, 2.0]** | A factor of 2 in dissolution behavior is compensable in formulation; larger differences require major reformulation. |
| **Parameter distance weights** | ε/k gets a 5× weight because it dominates mixture thermodynamics through the Boltzmann factor (exponential sensitivity). A 10% ε/k deficit can produce orders-of-magnitude differences in Henry’s constant. |

---

## Results Summary

- **Pure HFOs**: 0 of 1,471 pass the strict drop-in thermodynamic criteria. The primary bottleneck is ε/k being too low due to fluorination; the dispersion energy deficit cannot be compensated while satisfying the structural filters.
- **Chlorinated butenes**: 3 pass the broader cyclopentane-centric criteria, but these remain model-based near-hits rather than experimentally confirmed winners.
- For figure support, prefer `figures/25_hfo_centric_screening/parameter_space_comparison.png` and `figures/25_hfo_centric_screening/boiling_point_vs_hfo_distance.png`; the older `screening_funnel.png` is an HFO-specific screening snapshot and should not be used as evidence for the chlorobutene result.

---

## Expanded Screening: All Non-Associating C2-C6 Halogenated Hydrocarbons (Step 53)

Step 53 relaxed both scope restrictions from the original enumeration:

1. **Saturated backbones included**: 19 alkane backbones (C2-C6, including cyclic) added alongside the 16 alkene backbones, for 35 total.
2. **Br and I substituents included**: Up to 2 Br and 1 I atoms per molecule, in addition to F and Cl.

### Enumeration Scope

- **Total candidates**: 41,809 unique molecules (with stereoisomers) from exhaustive H/F/Cl/Br/I substitution on 35 C2-C6 backbones (MW <= 200 Da).
- **Olefin backbones**: 15,571 candidates (37%)
- **Saturated backbones**: 26,238 candidates (63%)
- **Unsubstituted parent hydrocarbons** included via `include_parent=True`, allowing the reference class itself (pentane, cyclopentane) to appear.

### Key Results

After broad cyclopentane-centric filtering (boiling point 288-323 K, SA score <= 4.5):

| Regulatory Class | Count | % |
|-----------------|-------|---|
| HFO-compliant | 797 | 49% |
| HFC-Kigali-phasedown | 398 | 24% |
| Br-ODP-nonstarter | 161 | 10% |
| HCFO-transitional | 134 | 8% |
| HCFC-Montreal-phaseout | 121 | 7% |
| flammable-hydrocarbon | 12 | 1% |
| I-unstable | 5 | <1% |

The top-ranked candidates by cyclopentane parameter distance are small cycloalkanes (methylenecyclopropane, methylcyclopropane, cyclobutane) and chlorinated compounds -- all with non-thermodynamic disqualifiers (flammability, ODP, regulatory restrictions).

### Comparison to Olefin-Only Screening

- **Zero overlap** between the expanded top-50 and the Step 38b HFO top-50.
- The first HFO-compliant candidate in the expanded ranking (2-fluorobutene, C=C(F)CC) appears at rank 15 with cyclopentane distance 0.420.
- Known commercial agents recovered: isopentane (rank 22), n-pentane (rank 37).
- The Step 38b conclusion is confirmed: no pure HFO achieves close thermodynamic proximity to cyclopentane.

### Conclusion

The expanded screening strengthens the negative result. The chemical space between "thermodynamically similar to cyclopentane" and "regulatory/safety viable" is empty. Molecules with high epsilon_k (needed for cyclopentane similarity) are necessarily non-fluorinated or lightly fluorinated, making them flammable, ozone-depleting, or unstable. The epsilon_k deficit introduced by fluorination is a fundamental thermodynamic constraint, not an artifact of the original enumeration scope.
