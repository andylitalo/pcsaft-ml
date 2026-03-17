---
This is an outline of a 20-minute presentation I will give to an AI for Chemistry startup to showcase a project in which I've applied ML to chemistry. It outlines the sections of the talk and includes bullet points explaining the decisions made and supporting them with results. To assist with answering deeper questions during the discussion afterwards, these results will hyperlink to markdown files describing the evidence that supports these results as well as figures in the supplementary_information/ directory. That way, a question about the justification or explanation of a result can be easily answered by clicking the hyperlinks.

- Q? indicates a question that is un-answered and should be either answered in-line or lead to an in-line change to the text (for example, if there is a Q? asking "is this point necessary?" and you find that it is not, remove that point)
- E! indicates a need for a deeper explanation in a linked document
---

# Title: Machine Learning Estimates Thermodynamic Properties of Molecules to Accelerate Search for Suitable Alternatives for a Polyurethane Blowing Agent

## Preamble

I've never applied ML to chemistry, but I have background in both and was excited to take this opportunity to do so to tackle an idea from my PhD.

# Introduction of the Problem

## Why do we care about alternative PU blowing agents?

### Polyurethane Insulating Foams are our best commodity insulation

- PU closed-cell, rigid insulating foams are the top commodity insulation (lowest thermal conductivity), exceeded only by aerogels that have yet to meet its price point
- PU is responsible for insulating our fridges, coolers, buildings, and even the space shuttle fuel tanks!

### Blowing agents are responsible for PU's low thermal conductivity

- Primary heat transfer is through gas conduction (Q? or convection?) and radiation
- Gas conduction (Q? or convection?) is inversely proportional (Q?) to MW of blowing agent
- Radiative heat transfer is inversely proportional to cell number density, which relies on enhanced nucleation by the blowing agent

In this talk, we will focus on radiative heat transfer because:
- it is a significant contribution
- it is least well understood how to reduce it (i.e., how to increase cell number density)
- Dow thought it was important enough to fund my PhD on it...

### Absent CFCs and HCFCs, cyclopentane is one of the best blowing agents

- Reasonably high MW (Q? how much?)
- appropriate boiling point (Q? what is the appropriate range) to vaporize during the exo-therm in PU foaming and remain vapor in the final product (to prevent condensation and foam collapse)
- Significantly enhances bubble nucleation (demonstrated by my PhD work) and yields higher cell number density (Dow, Minogue)
- Also cheap, chemically stable (Q?), no ODP (reason for phasing out CFCs and HCFCs used before the Montreal Protocol), anything else (Q?)

HCFCs and CFCs had all these properties except for the ODP of 0 (and secondarily, a low GWP) which is why they were banned in the Montreal Protocol.

### Cyclopentane has practical problems

- Highly flammable
- Not a low GWP
- Q? any others?

### Industry has moved to alternatives to meet regulation at great cost

HFOs and HCFOs are
- far less flammable
- have lower GWP
- no ODP
- higher MW

But they are
- less soluble in polyol (requiring siloxane Q? surfactants to emulsify)
- less stable (amines attack something Q? requiring something Q?)
- more expensive (Q? how much?)

The industry invested a lot to comply with regulations. Did they need to?

### SPOILER: HFOs and HCFOs

We'll find in the end that reducing flammability is only possible by adding fluorines, which dramatically changes the thermodynamic properties from cyclopentane and requires significant reformulation to compensate.


## How can we explore other alternatives to cyclopentane?

To answer this question, we frame it as a scientific question with the following

### Assumptions

- Blowing agent performance ~ ability to generate high cell number density in PU foams
- cell number density ~ nucleation rate
- nucleation rate ~ favorable thermodynamic properties (based on my PhD work)
- thermodynamic properties ~ PC-SAFT parameters (specifically m, sigma, and epsilon) E! what is PC-SAFT, why use it, and what do these parameters mean, and why are we ignoring the others (k + association parameters) Q? can we ignore association parameters in fluorinated and chlorinated compounds?

yielding our

### Scientific Questions (Goals)

*Which molecules have similar m, sigma, and epsilon to cyclopentane?* (proxy for blowing agent ability)

Secondarily, *which molecules have a suitable boiling point?* (vaporize during exo-therm of PU and remain vapor in final foam)

Thirdly, *which molecules have similar solubility in polyol?* Q? can we answer this with Henry's constant in n-hexane given that polyol has -OH groups? Q? or is Henry's constant useful as a validation of the PC-SAFT parameters' utility for predicting thermodynamic properties in mixtures?

# Proposed Solution: ML prediction of PC-SAFT parameters to filter against cyclopentane's

## Let's first consider all possible approaches to answering our scientific questions:

In order of increasing throughput and levels of abstraction:

1. Experimental measurement of thermodynamic properties and fitting of parameters (e.g., VLE, Q? what else) (~days/molecule)
2. Quantum simulations (COSMO-RS Q? what else Q?) (~hours/molecule)
3. Phenomenological/empirical models (coarse-grained models, ML) (~seconds/molecule)

To screen thousands of chemicals in one weekend, I need to use #3.

## Baseline: What models are available?

### SPT-PCSAFT has issues

SPT-PCSAFT is a published ML model for predicting PC-SAFT parameters of > 13k molecules (Q? what is the citation)

- Model weights are not available (Q? is this true or are model weights available?)
- Validation failed in our study (Q? is this true or did we find that the systematic error we initially saw was an artifact of our method?)

## Train Our Own Model

### On what dataset?

- Unfortunately, most datasets of PC-SAFT parameters are proprietary (e.g., Q? what are some examples?)
- SPT-PCSAFT is a large dataset, but only of predicted parameters, so training on it will compound on their errors (E! we tested training on this model and found some issues--explain them)
- Open-source datasets: Esper (~1800 molecules Q? what is exact number) and ML-SAFT (Q? how many?). Combined and deduplicated -> ~ 2k molecules (Q? exact number?)

### With what features?

Literature suggests features from
- RDKit (E! what is it why and what literature suggested it and why)
- Morgan Fingerprint (E! what is it why and what literature suggested it and why)
- 3D Conformer (E! what is it why and what literature suggested it and why)

(Q? is that true? What else? Why?). We excluded 3D Conformer features because they are more complex to calculate (Q? is that why?). They are more of an optimization.

SMILES was also considered, but poor because (Q? E! what was the issue with SMILES-based representations like ChemBERTa? Is this connected to the issues with SPT-PCSAFT)

### With what architecture?

In order of increasing need for data and learnability of features:

1. Group-contribution method: estimates PC-SAFT parameters based on human-derived rules based on functional groups in the molecule (Q? citation?)
2. Random Forest: (E! why did we choose this and what are the biases it introduces to help prevent over-fitting in the case of smaller datasets)
3. GNN: (chemprop E! what is it and GINEConv E! what is it) recommended by literature (Q? what literature and why useful for this based on message-passing and ability to learn features E!)

SPOILER: The Bitter Lesson suggests that GNNs should work best with a large enough dataset (E! where GNN shines in large dataset), but we ended up finding that RF was the best performer.

### Metrics of Success

A good model will have
- Low MAE (Q? what is MAE formula and why do we use it?)
- R^2 above the noise threshold (Q? why is R^2 appropriate?)
Q? anything else?

### Training

- Objective/loss: Q?
- Optimizer: I think AdamW but with what params and why? Q?
- Learning rate: Q?
- Stopping condition: Q?

### Evaluation

Q? what were the results that led to our decision to choose RF?
E! make a document of the primary results for ALL models showing why we choose RF and where other models might be valid

## Filter candidate molecules

Filters:
- PC-SAFT parameters predicted by selected ML model (m, sigma, epsilon) Q? what were the specific ranges we permitted (E! and why?)
- Boiling point (E! how is this calculated) (Q? what were the specific ranges we permitted?) (E! and why?)
- Henry's Constant (Q? are we using this for filtering?) (Q? what were the specific ranges we permitted?) (E! and why?)

Q? what were the results?
Q? what were the candidates with uncertainties? (Q? was it chlorinated olefins or were there others?)

### Scientific Rationale

Q? What was the scientific explanation for these candidates passing the filters?

### Limitations: What molecules is this not appropriate for?

- Fluorinated compounds were too far OOD (Q? what was our measurement of this?) so they failed on the 15-molecule validation against the literature values (E! explain this)
- Polar molecules (Q? Because we didn't account for association parameters, right? Or do we still have evidence of being able to make predictions for polar molecules reasonably well? What's the cutoff and what's the measurement of polarity?)

# Conclusion: No feasible candidates, but a useful approach

## Feasible Candidates

Those found are not feasible due to practical limitations:
- Q? what were they? list them here

## Valid Approach with Reasonable Accuracy

Q? what was the evidence of the validity of our approach (parameters -> PC-SAFT instead of directly predicting thermodynamic properties like boiling point; comparison to SPT-PCSAFT; other metrics (E! include a full report supporting the value of this approach and its unique contributions))

## Future Work

- Apply to finding thermodynamically similar alternatives to the leading blowing agents of today (HFOs): need larger dataset of measurements for fluorinated compounds (it was OOD for Esper and failed validation) and a dataset for estimating association parameters since fluorinated compounds are highly polar and can have H-bonds (Q? is that right?)
- Apply to finding thermodynamically similar alternatives of almost any organic molecule! This is demonstrated in the streamlit app (E!)
- Eventually the app should, given a molecule:
- check for OOD and uncertainty in prediction (reject if too high)
- predict parameters (could also put them through an EOS solver to generate thermodynamic property estimates and plots)
- filter molecules in the dataset by proximity of parameters (just rank by proximity)

Also with more data for association parameters and k we could predict those too (Q? are these even helpful for this question? Are they needed for polar molecules?)

## Big Picture: ML Accelerates Search for Candidates in New Spaces with Sufficient Experimental Data

If the datasets are too small, use theoretical coarse-grained models for high-throughput estimates (e.g., group-contribution method for PC-SAFT).
If the datasets are large, learn features (e.g., GNN, transformer).
Many datasets sit in the middle: somewhat representative, but not terribly rich or massive. In this case, ML needs features from science (RDKit and Morgan Fingerprints) and biases (Q? what's the correct term?) like those in RF.
Lastly, physical theories can provide a bridge between parameters that ML can predict and physical observables that can be measured.

### Bitter Lesson: How much data are needed to learn the physics itself?

With enough data, we should be able to make predictions as accurately as physical models (i.e., within experimental uncertainty).
E! do some research on where this has been successful in chemistry (which properties have massive datasets that have been trained and predicted from SMILES, for example?)