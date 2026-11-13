# SAST: Paper-Level Method Abstraction

Status: final method-positioning note, frozen on 2026-11-13.

This document records the paper-level abstraction of the proposed method. It
is intentionally not a laboratory notebook. The paper should describe the
stable mechanism shared across backbones, while profile search, scheduler
choices, checkpoint conversion, and other engineering details remain in the
implementation or supplementary reproducibility material.

## 1. Recommended Method Identity

**SAST: Source-Anchored Subspace Transfer** is a plug-in module for safely
learning from an auxiliary, non-dialogue visual emotion corpus while retaining
the behavior of a dialogue emotion recognizer.

The implementation also uses the internal name **CESA** for the
context-decoupled emotion-subspace adapter. In the paper, SAST should be the
method name and CESA may be introduced as its feature-adaptation block. This
keeps the terminology aligned with the code without presenting two competing
methods.

The central idea is:

> Transfer auxiliary visual emotion evidence through a source-defined emotion
> subspace, while preserving the source model as an explicit behavioral
> anchor and keeping non-dialogue auxiliary samples outside fabricated
> dialogue context.

## 2. Problem Formulation

Let the source dialogue corpus be

\[
\mathcal{D}_s=\{(X_i^s,Y_i^s)\},
\]

where each dialogue contains a sequence of visual utterances and emotion
labels. Let

\[
\mathcal{D}_a=\{(x_j^a,y_j^a)\}
\]

be an auxiliary corpus such as BOBSL, whose samples provide visual emotion
supervision but no authentic dialogue structure. Both corpora use the aligned
four-class emotion space \(\{A,N,J,S\}\).

A naive union of the two datasets asks one contextual model to absorb both
domain shift and context mismatch. The diagnostic experiments show that this
can produce class-dependent boundary changes and that dialogue context can
amplify either useful transfer or negative transfer. The goal is therefore not
generic domain invariance. It is to extract transferable emotion variation
from \(\mathcal{D}_a\) without allowing that variation to overwrite the source
decision geometry.

## 3. Architecture

### 3.1 Frozen source-behavior anchor

SAST begins with a source-supervised dialogue emotion backbone \(F_0\). The
anchor can be obtained from source training or from an existing joint model;
in both cases, it contains the original temporal, graph, or hypergraph context
model and classification head and is frozen during residual transfer. The term
*source-anchored* refers to the source-defined geometry and behavioral
constraints, rather than requiring one particular checkpoint initialization.

This anchor serves two purposes. It preserves a strong task-specific solution,
and it supplies a reference distribution against which every adapted source
prediction can be checked. SAST therefore learns a correction to an emotion
model rather than relearning the full model from a heterogeneous data union.

### 3.2 Source emotion subspace

For source visual features \(h\in\mathbb{R}^d\), SAST computes one centroid
\(\mu_c\) for each emotion class and the global centroid \(\bar\mu\). Singular
value decomposition of the centered class offsets gives an orthonormal basis

\[
U=\operatorname{SVD}\left([\mu_c-\bar\mu]_{c=1}^{C}\right),
\qquad U\in\mathbb{R}^{d\times r},
\]

where \(r\le C-1\). This basis captures source-discriminative emotion
directions while excluding most unrestricted domain-specific variation.

The subspace is deliberately source-defined. BOBSL supplies supervision for
learning a useful residual, but it does not redefine the geometry that the
source dialogue model already uses successfully.

### 3.3 Context-decoupled residual adaptation

For each visual utterance feature, the CESA block predicts a bounded
low-dimensional correction:

\[
z=\tanh\left(W_{up}\,\phi(W_{down}\,\operatorname{LN}(h))\right),
\qquad
\Delta h=\alpha Uz,
\qquad
\tilde h=h+\Delta h.
\]

The up-projection is initialized to zero. Transfer therefore starts exactly
from the anchor model and introduces only a gradual residual. The low-rank
projection and residual scale \(\alpha\) bound the directions and magnitude of
adaptation.

Crucially, this operation is applied at the utterance-feature level before the
backbone performs dialogue context propagation. Source utterances retain their
real dialogue sequence. Each BOBSL sample contributes local emotion evidence
without inventing speaker turns or temporal relations that are absent from the
data. This is the context-decoupled part of SAST.

For backbones whose decision interface benefits from additional flexibility,
the implementation supports a small zero-initialized residual in the shared
emotion-logit space. It remains bounded and additive, so the one-checkpoint
inference contract is unchanged.

### 3.4 Source-preserving transfer objective

The objective can be presented as four conceptually stable terms:

\[
\mathcal{L}=
\mathcal{L}_{disc}
+\lambda_{geo}\mathcal{L}_{geo}
+\lambda_{anchor}\mathcal{L}_{anchor}
+\lambda_{cap}\mathcal{L}_{cap}.
\]

- **Discriminative transfer** \(\mathcal{L}_{disc}\) combines supervised
  emotion classification on source and BOBSL samples.
- **Emotion geometry** \(\mathcal{L}_{geo}\) aligns adapted labeled samples
  with their source class prototypes in standardized subspace coordinates.
- **Anchor preservation** \(\mathcal{L}_{anchor}\) combines source-prediction
  consistency with a per-sample safety constraint that penalizes corrections
  which increase source classification loss relative to the frozen anchor.
- **Capacity control** \(\mathcal{L}_{cap}\) regularizes residual magnitude.

In implementation, anchor preservation is realized with prediction KL
stability and the safety loss

\[
\mathcal{L}_{safe}
=\frac{1}{n_s}\sum_i
\max\left(0,\ell(\tilde p_i^s,y_i^s)-
\ell(p_{0,i}^s,y_i^s)+m\right).
\]

Where a jointly trained peer is available, its useful transfer behavior can be
compressed into the adapter through a training-only consistency term. This is
an optional source of soft supervision within the same objective, not an
additional inference branch. The deployed model always consists of one
backbone and one adapter.

## 4. Training and Inference Contract

1. Obtain a source-supervised visual emotion backbone, optionally with prior
   joint training on the auxiliary corpus.
2. Freeze the source backbone and compute source class centroids and the
   source emotion basis.
3. Attach a zero-initialized CESA residual adapter before contextual
   propagation.
4. Train the adapter with source dialogues and context-free BOBSL samples
   under the source-preserving transfer objective.
5. Select the checkpoint using the source-side selection protocol.
6. Deploy one backbone-plus-adapter checkpoint on both the source test set and
   eJSL.

There is no inference-time ensemble, checkpoint routing, oracle domain label,
or eJSL-based checkpoint selection in the final method.

## 5. Motivation-to-Design Correspondence

| Observed issue | SAST response |
|---|---|
| BOBSL and dialogue corpora remain visually domain-separable | Transfer is restricted to source emotion directions instead of forcing global feature invariance. |
| Raw cross-domain emotion prototypes are not interchangeable | Source prototypes define the coordinate system; auxiliary labels guide a residual within it. |
| Plain joint training can move class priors and boundaries asymmetrically | The source backbone is frozen and every change is an additive, bounded correction. |
| Context can amplify transfer errors | Adaptation occurs before context propagation, while auxiliary examples are trained as context-free utterances. |
| Auxiliary supervision is useful but can erode source behavior | Source classification, KL stability, per-sample safety, and residual control retain the anchor. |
| Different dialogue backbones expose different internal interfaces | The same adapter operates on their common visual utterance representation and preserves each native context model. |

This correspondence is the main methodological story: every component answers
an experimentally observed transfer failure rather than adding generic model
capacity.

## 6. Innovation Positioning

The novelty claim should be made at the level of the complete transfer design,
not by claiming that SVD, residual adapters, or KL regularization are
individually new.

The defensible contributions are:

1. **Context-aware formulation of auxiliary transfer.** SAST explicitly
   separates local emotion evidence from dialogue context, allowing a
   non-dialogue corpus to train a dialogue recognizer without fabricated
   conversational structure.
2. **Source-defined emotion transfer geometry.** The module transfers along a
   compact class-structured subspace derived from the target recognizer,
   instead of aligning complete source and auxiliary feature distributions.
3. **Behaviorally anchored residual learning.** A frozen source model,
   zero-initialized bounded residual, and sample-level safety objective turn
   auxiliary learning into controlled correction rather than unrestricted
   joint optimization.
4. **Backbone-independent deployment.** SAST attaches to heterogeneous
   dialogue emotion architectures at the visual utterance interface and
   retains a single-checkpoint inference path.

Together these form a coherent method for safe auxiliary visual emotion
transfer under simultaneous domain and context mismatch.

## 7. Result Narrative

The main empirical message should emphasize the aligned BOBSL-5K,
five-seed study:

- Across five backbones and two source corpora, SAST improves the plain joint
  model on the source metric in 10/10 settings and on eJSL in 9/10 settings;
  the remaining difference is 0.0001 weighted F1.
- SAST beats both source-only and plain joint training on both source and eJSL
  in 7/10 backbone-corpus settings.
- On the two focal negative-transfer backbones, MMGCN and MAGTKD, all four
  MELD/IEMOCAP settings improve over both comparison systems on both metrics.
- The largest improvements over plain joint training occur on external eJSL,
  which is consistent with the module's purpose of preserving transferable
  emotion evidence while suppressing auxiliary-domain distortion.

These statements are supported by `docs/proposal3_final_results.md`. The paper
does not need to narrate every exploratory profile or rejected alternative.

## 8. Ablation Logic

The paper-facing ablation should follow the method's causal structure:

1. **Plain joint training:** establishes the negative-transfer problem.
2. **Without source preservation:** tests whether residual capacity alone is
   sufficient when source safety and prediction stability are removed.
3. **Without subspace restriction:** replaces the source emotion projection
   with a full-space residual around the same frozen anchor.
4. **Full SAST:** adds class-structured low-rank transfer and the complete
   source-preserving objective.
5. **Without context decoupling:** passes context-length-one auxiliary samples
   through the contextual path, testing the value of separating local and
   dialogue learning.

Existing plain-joint, source-preservation, full-space, context-ablation, and
final SAST runs provide the implementation basis for this presentation. Exact
tables should be generated from frozen artifacts rather than reconstructed
from prose.

## 9. Paper Abstraction Rules

The following principles are part of the handoff to the paper-writing agent:

- The paper is an abstraction of the implemented research contribution, not a
  chronological explanation of every experiment.
- Describe mechanisms that remain invariant across backbones: source anchor,
  emotion subspace, bounded residual, context decoupling, and source-preserving
  learning.
- Treat profile names, parameter sweeps, seed scheduling, GPU orchestration,
  checkpoint conversion, and dataset-specific tensor handling as engineering
  details rather than method components.
- It is valid to select the experiments most directly supporting the stated
  contribution, provided every reported number and protocol is accurate.
- State strong positive findings prominently. Keep limitations concise and
  scoped to the claims actually made; do not turn the method section into a
  complete failure analysis.
- Do not promote historical routing, oracle selection, multi-checkpoint
  mixtures, or rejected architectures as part of SAST.
- Do not claim universal per-seed dominance or universal domain invariance.
  The supported claim is robust mean improvement across the stated
  five-seed, two-corpus, multi-backbone protocol.
- Hyperparameters may differ by backbone and corpus. Present them in the
  implementation details or appendix, not as conceptual changes to SAST.

## 10. Paper-Ready Summary

> We propose Source-Anchored Subspace Transfer (SAST), a plug-in module for
> exploiting a context-free auxiliary visual emotion corpus in dialogue
> emotion recognition. SAST freezes a source-trained contextual backbone and
> derives a compact emotion subspace from source class centroids. A
> zero-initialized residual adapter then modifies each visual utterance only
> along these source-discriminative directions before dialogue context is
> modeled. Auxiliary samples supervise this local adapter without introducing
> artificial conversational links, while source classification, prediction
> consistency, and sample-level safety constraints preserve the anchor's
> behavior. SAST therefore transfers complementary visual emotion evidence as
> a controlled correction to the source model and retains a single-checkpoint,
> backbone-independent inference path.

## 11. Implementation and Evidence Map

Core implementation:

```text
new/cesa.py
MMGCN/train_eval_mmgcn_sast_video_anjs.py
new/MAGTKD/MELD/train_eval_magtkd_sast_video_anjs.py
new/ECERC/MELD/train_eval_ecerc_sast_video_anjs.py
new2/CMERC/train_eval_cmerc_cesa_visual_anjs.py
new2/ConxGNN/train_eval_conxgnn_sast_visual_anjs.py
```

Canonical local records:

```text
docs/MOTIVATION_EVIDENCE.md
docs/proposal1_preliminary_results.md
docs/proposal2_source_anchored_results.md
docs/proposal3_final_results.md
docs/EXPERIMENT_HANDOFF.md
```

Canonical server result table:

```text
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_report/proposal3_main_b5k.csv
```
