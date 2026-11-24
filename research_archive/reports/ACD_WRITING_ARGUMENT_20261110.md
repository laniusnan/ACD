# ACD paper argument: background, analysis, and method

**Status — 2026-11-11.** This is the primary writing guide for the ACD paper.
It records an argument structure inspired by the problem-analysis-method
progression used in recent KD work, not a new experimental conclusion.  It
supersedes prior outline-only guidance where the proposed modules appeared
before the representation failure they address.  Claims about broad accuracy,
universality, or state of the art remain contingent on the registered training
queue and must be updated only from completed final-reloaded evaluations.

## Paper question and answer

**Question.** Why can direct classifier-output matching be unreliable when a
teacher and a student have different classifier parameterizations?

**Answer to establish.** A conventional linear classifier represents each raw
logit as

\[
z_k = \mathbf{w}_k^\top\mathbf{h}+b_k
    = \|\mathbf{w}_k\|\|\mathbf{h}\|\cos\theta_k+b_k.
\]

Thus raw logits combine directional category relations with feature magnitude,
classifier-weight norm, and bias.  The latter quantities need not be
comparable across architectures or training trajectories.  ACD first makes the
student-side category coordinates angular (ACI), then transfers a
scale-invariant conditional non-target relation on that interface (SICD).

This is a hypothesis-and-analysis chain.  The formula proves the entanglement
in the representation; it does **not** by itself prove that every raw-logit KD
run has corrupted relations or that ACD improves every pair.  Use modal wording
(`can`, `may`) until the registered interaction and affine-control experiments
are complete.

## Narrative order

### Publication-safe diagnostic chain — 2026-11-11

ACD should follow a problem--diagnosis--response chain, but it must not borrow
LIFT/PLACE's capacity-gap claim or overstate what the present diagnostics show.
The paper's empirical symptom is a **calibration-invariant relation residual**:
after removing the best per-image positive scalar from a student's logits,
teacher--student disagreement can remain in relative class scores.  This is a
measurable endpoint phenomenon, not proof that a student has ignored a
semantic ground-truth angular relation.

The publication-safe chain is:

1. Classifier outputs are a universal, low-overhead KD interface, yet raw
   matching treats their class coordinates as directly comparable.
2. The classifier decomposition identifies why this assumption can be
   restrictive: a score mixes directional terms with feature magnitude,
   class-specific norm, and bias.
3. Existing output losses improve decomposition, normalization, or relation
   modeling, but leave the coordinate construction and supervision selection
   as separate questions.
4. The endpoint residual diagnostic shows that scalar confidence alone does
   not explain all teacher--student output disagreement.  Its angular-head
   comparison motivates, but does not prove, a coordinate-level response.
5. Fixed-direction and local-force probes show that parameterization can
   change conditional distributions and local KD forces.  They identify a
   plausible mechanism, without establishing an optimization trajectory.
6. ACI addresses *where* the student represents class alternatives; SICD
   addresses *what* it receives after standardization.  Their effectiveness is
   an experimental question answered by the controlled interaction and loss
   ablations, not by the diagnostic alone.

This yields an analysis module suitable for a mechanism paper only under the
following claim boundaries:

| Claim | Evidence that can support it now | What it cannot establish |
|---|---|---|
| Raw KD leaves disagreement beyond scalar confidence | Per-image oracle-calibrated relation-residual distributions on two completed seeds | The causal source of every residual or a universal failure of KD |
| Parameterization can alter output supervision and a local KD force | Fixed-direction checkpoint reconstruction and virtual-update probe | A persistent training-time conflict or final-accuracy harm |
| ACI has a KD-specific effect | Only a paired CE $\\times$ KD head-interaction result, reported with both signs | A claim inferred from cosine accuracy or the residual diagnostic alone |
| SICD selects a useful signal | Full/confidence/conditional loss interventions with stated gradient controls | Universal superiority, unless independent multi-pair and multi-seed tests agree |

The present evidence therefore supports the section title
**Diagnosing Calibration-Invariant Relation Residuals in Output KD**.  It does
not yet support an unconditional theory claim such as “classifier geometry
mismatch universally causes KD failure.”  Use ``can'', ``may'', and
``in our controlled setting'' until the missing trajectory and interaction
evidence is completed.

### Required separation between the endpoint diagnostic and the trajectory claim

The endpoint relation-residual figure may establish that a positive scalar
cannot explain all output disagreement.  It must not be described as direct
evidence that standard KD ``ignores angular relations.''  That stronger claim
requires matched raw-KD and ACD checkpoints from the same pair/seed, with a
pre-indexed probe reported at epochs $1$, $30$, $120$, and $240.  At each
checkpoint, report raw discrepancy, normalized conditional-relation
discrepancy, the raw/reference gradient cosine and norm ratio, feature norm,
classifier-norm dispersion, and bias-to-score ratio.  A trajectory in which
raw discrepancy falls while normalized conditional discrepancy remains high
would support the narrower statement that raw matching under-aligns this
defined angular-relation proxy.  Until then, the endpoint diagnostic remains
descriptive.

### 1. Introduction: why classifier outputs matter

Open from efficient deployment and KD, then state that KD has expanded from
output matching to intermediate features, relational signals, and structural
knowledge.  Classifier outputs are nevertheless a uniquely valuable universal
interface: they are available from virtually every classifier; output-based
supervision needs no intermediate-feature access, layer correspondence, or
architecture-specific adapter; and it adds little memory or compute overhead.
They are both the primary objective in pure logit KD and a common auxiliary
signal in feature-/relation-based pipelines.

Do **not** introduce ACI/SICD before naming the unresolved issue.  End the
Introduction problem setup with the question of whether raw teacher and student
logits actually occupy compatible classifier coordinates.

### 2. Background and problem formulation: what a logit contains

Define a linear classifier and derive the logit decomposition above.  Separate:

- directional category relations, the candidate transferable semantic signal;
- feature magnitude, classifier-weight norm, and bias, which are
  parameterization-dependent quantities.

This section should not claim that cosine classifiers, standardization, or
target/non-target splitting are new.  Its role is to state precisely the
object that a raw-KL loss compares and the invariances it lacks.

### 3. Analysis of classifier geometry mismatch: why raw matching can fail

Define **classifier geometry mismatch** as incompatibility between the teacher
and student classifier-induced category coordinates.  Across architectures or
training trajectories, direct raw-logit matching can force the student to
imitate parameterization-dependent quantities.  This can create conflicting
distillation gradients and distort the inter-class alternatives that output KD
is intended to convey.

The analysis must pair each assertion with evidence:

| Analysis question | Evidence and paper role | Boundary |
|---|---|---|
| Can raw radial/bias terms alter non-target ordering? | Fixed-checkpoint data-driven teaser | Descriptive probe, not a generalization result |
| Does the angular interface change the KD-specific head interaction? | CE-only versus KD head-interaction matrix | A cosine CE gain alone is not ACI evidence |
| Does conditional standardized relation help beyond ordinary cosine KD? | standardized-full and relation matched arms | Pair/seed coverage must be reported exactly |
| Are raw-logit objectives sensitive to teacher transformations? | shift/scale/temperature/relation affine controls | Interpret only if completed under the sealed protocol |

The goal is an analysis section that motivates a response to two distinct
problems, rather than an ablation retrospective.

### 4. Method: one module per diagnosed problem

| Diagnosed issue | Method response | Intended role |
|---|---|---|
| Raw student coordinates contain radial scale and bias effects | **Angular Coordinate Interface (ACI)** | Construct angular student category coordinates that emphasize directional relations |
| Full raw or full standardized supervision still contains non-transferable absolute-confidence content | **Scale-Invariant Conditional Distillation (SICD)** | Normalize outputs and distill conditional non-target relations, with the sealed gradient-strength matching rule |

ACI and SICD are consequently sequential rather than two unrelated tricks:
ACI defines a more compatible interface; SICD selects the information to
transfer through that interface.

### 5. Experiments: answer the analysis questions

1. Main KD comparisons across homogeneous and heterogeneous pairs.
2. Head × supervision interaction: compare the cosine effect under CE-only and
   KD, rather than claiming an angular-KD benefit from raw cosine accuracy.
3. SICD ablation: ordinary cosine KD, standardized-full, and matched
   conditional relation.
4. Affine controls and DKD controls: distinguish invariance/selectivity from
   a generic alternative loss explanation.
5. Efficiency and feature-/relation-KD composition, only if supported by
   completed comparable experiments.

## Consequences for title, abstract, and Introduction

### Title

The title should lead with the problem and name the method once.  A current
working choice is:

> **ACD: Bridging Classifier Geometry Mismatch in Knowledge Distillation**

It is short, has the desired problem-resolution impact, and avoids a second
`Distillation` after the colon.  `Geometry mismatch` is the paper's named
analysis object; use `geometry gap` only if the final prose avoids defining a
strict mismatch.

### Abstract

The abstract should remain a compact three-part argument, not reproduce the
Background/Analysis section headings:

1. classifier outputs are important, universal, and efficient, but conventional
   objectives underuse their information;
2. conventional classifier parameterization entangles semantic relations with
   architecture-specific effects, producing classifier geometry mismatch;
3. ACI establishes angular coordinates and SICD transfers scale-invariant
   conditional non-target relations; conclude only with results substantiated
   by completed experiments.

Avoid implying that feature-based KD is more accurate *because* of the
identified mismatch, or that ordinary cosine accuracy alone establishes an ACI
benefit.  `State-of-the-art`, `universal`, and `consistently significant` need
completed multi-pair evidence and should be removed or qualified otherwise.

### Introduction section plan

| Introduction block | Job | Approximate length |
|---|---|---:|
| P1: motivation and output-interface value | Why KD, then why classifier outputs are worth fixing | 150–180 words |
| P2: linear-classifier formulation | Equation and parameterization entanglement | 120–150 words |
| P3: analysis consequence | Geometry mismatch, visual intuition, evidence preview | 100–130 words |
| P4: ACD overview | ACI then SICD as direct responses | 120–150 words |
| P5: contributions | Method, analysis/evidence, validated experimental scope | 80–110 words |

The Background and Analysis can be separate main sections after Introduction
when page budget permits.  If merged into Intro, preserve the same logical
order and use the equation/teaser as the transition from P2 to P3.

## Figure implications

The revised plan separates **phenomenon → diagnostic analysis → supervision
selection → method**. The teaser should not explain both modules in advance.
Use the primary [figure reading and design guide](../paper_assets/acd_figures/README.md#evidence-first-figure-plan)
for layouts, axes, controls, observations, contrary patterns and evidence gaps.
This supersedes requiring the teaser to combine the problem, ACI and SICD.

| Paper question | Figure role | Claim boundary |
|---|---|---|
| Can fixed angular scores yield different supervision? | Empirical teaser: angles, class-dependent norms/biases, resulting distributions | Angular ordering is not semantic ground truth |
| Do these factors change the learning signal? | Analysis: factor sensitivity and gradient measurements | Gradient change is not automatically conflict or accuracy loss |
| Which supervision should be retained? | Analysis: confidence/conditional decomposition and matched interventions | Loss share is not usefulness; retain negative outcomes |
| How does ACD implement the response? | Method: unchanged teacher, student ACI, standardization, SICD and CE | No shared feature-space projection or removal of arbitrary teacher bias |

### Inspiration and source boundary

The inspiration from LIFT and PLACE is the problem–analysis–method argument.
Its full captions and Algorithm 1 could not be rechecked during the 2026-11-10
discussion because source retrieval failed. Descriptions of its Figures 1–3
remain provisional: do not cite this guide as verification that Figure 1 proves
capacity-gap causality or Algorithm 1 is merely the code version of Figure 2.
An algorithm may define a diagnostic or the training method; check the source.
Before Method, figures should establish the phenomenon and test its structure,
not repeatedly show failure. A performance gap alone does not identify a cause.

### Supervision and evidence boundaries

Ordinary KL KD matches softened distributions, not absolute raw logits, and
already ignores a common additive shift. A common positive scale changes
confidence but not ordering; class-dependent norms and biases can change
ordering. Standardization removes common positive scale/shift in the ideal
nondegenerate case, subject to numerical clamps, not arbitrary class-dependent
teacher effects. ACI changes the student head only.

Assess a KD-specific interface effect using paired head gains:
`(cosine KD − linear KD) − (cosine CE − linear CE)`, with an unchanged recipe.
Retain negative differences. Cosine-plus-SICD need not beat linear-plus-SICD to
define ACI's operation, but report existing comparisons. Neither factorization
nor synthetic figures establish nontransferability or gradient conflict.

**Status update — 2026-11-10:** four evidence-backed figure drafts are rendered
as editable SVG, vector PDF and PNG in the
[primary guide](../paper_assets/acd_figures/README.md#rendered-evidence-first-figures).
They show aggregate ranking sensitivity, structural normalization alongside
negative head interactions, component endpoints with the stronger full-loss
norm control retained, and the implemented method flow. No new inference,
gradient diagnostic, registration or training was performed. Per-example
probability panels and controlled gradient-sensitivity curves remain missing
evidence/presentation, not completed figures. These drafts do not establish
the original strong geometry-mismatch causal claim.

## Citation positioning

Use foundational KD for output matching; representative feature, relation, and
structural KD papers only to establish the field's breadth; and closest logit
methods (e.g., decoupled, standardized, or streamlined objectives) in the
problem/method contrast.  Related work should distinguish operations, not
claim novelty merely because both methods use relations or normalization.
