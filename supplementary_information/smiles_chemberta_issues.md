# Supplementary Information: SMILES-Based Representation (ChemBERTa) Issues for PC-SAFT Parameter Prediction

## What is ChemBERTa?

ChemBERTa is a BERT-style transformer pretrained on chemical data. For PC-SAFT parameter prediction, we fine-tuned `seyonec/ChemBERTa-zinc-base-v1` (~85M parameters) for regression on the three PC-SAFT segment parameters. The base model was pretrained on the ZINC dataset of SMILES strings (masked language modeling), and we fine-tuned it on the Esper PC-SAFT parameter labels.

Input to ChemBERTa is tokenized SMILES: a character-level (and subword-level) representation of the molecular structure encoded as a linear string. The model must learn to map SMILES token sequences to continuous property values through attention over the sequence.

## Performance

On the Esper test set, ChemBERTa achieved the following R²:

| Parameter | ChemBERTa R² | RF R² |
|-----------|--------------|-------|
| m (segments) | 0.53 | 0.62 |
| σ (Å) | 0.25 | 0.35 |
| ε/k (K) | 0.27 | 0.33 |

ChemBERTa was the second-best overall model in our comparison but underperformed Random Forest on all three parameters. The gap was largest for ε/k, the parameter most important for blowing agent screening (dispersion energy).

## Issues Identified

**1. Inference speed.** ChemBERTa required ~185 seconds to predict PC-SAFT parameters for 361 molecules on CPU, compared to ~2 seconds for Random Forest—a **92×** slowdown. For high-throughput screening of large virtual libraries, this latency is prohibitive.

**2. No clear applicability domain method.** Morgan fingerprint-based applicability domain (AD) methods—e.g., distance to training set in fingerprint space—do not transfer directly to SMILES-based representations. ChemBERTa operates in CLS embedding space; AD would require building distance thresholds or classifiers there. This adds complexity and is less interpretable than fingerprint-based AD.

**3. Lower accuracy on ε/k.** The dispersion energy parameter ε/k drives vapor pressure and phase behavior predictions, which are central to screening. ChemBERTa predicted ε/k less accurately than RF, undermining its utility for the primary use case.

**4. SMILES is a string, not a molecular representation.** SMILES encodes connectivity but not molecular structure in an explicit form. The model must infer structure from string patterns. Small textual changes (e.g., atom ordering, branching syntax) can correspond to large structural or property changes, and conversely, minor structural differences can produce very different SMILES. This non-linear mapping is harder to learn reliably with limited data.

## Connection to SPT-PCSAFT

SPT-PCSAFT also uses a SMILES transformer architecture to predict PC-SAFT parameters. The systematic bias observed in SPT predictions for fluorinated compounds suggests a shared limitation: SMILES-based representations may struggle for molecules whose properties depend strongly on electronic effects (e.g., C–F bond polarizability) that are not directly encoded in the SMILES string.

Both ChemBERTa and SPT-PCSAFT performed worse on ε/k than on m or σ. This is consistent with dispersion energy depending on electronic and polarizability properties that SMILES encodes only implicitly, requiring the model to learn them from string patterns rather than from explicit descriptors.

## Takeaway

SMILES-based models are a valid approach and can capture complex structure–property relationships when sufficient data is available. For PC-SAFT parameter prediction on small datasets (~1,800 molecules), feature engineering (e.g., RDKit descriptors, Morgan fingerprints) outperforms learned SMILES representations. Domain-specific features encode molecular properties explicitly; SMILES-based models must rediscover those relationships from tokens, which is data-intensive. With larger training sets, ChemBERTa and similar architectures may close or reverse this gap, but for the current Esper-scale setting, explicit feature engineering remains preferable.
