# Motivation and Mechanism Evidence for BOBSL Joint Transfer

Status date: 2026-11-13

This document records the empirical evidence behind the motivation for safe
BOBSL joint transfer. It is separate from the engineering and reproduction
handoff in `docs/EXPERIMENT_HANDOFF.md`.

The final method result remains Proposal 3 in
`docs/proposal3_final_results.md`. Historical routing, oracle, and held-out
mixture studies below are mechanism diagnostics, not the final inference
procedure.

## 1. Central Observation

Adding BOBSL to source-corpus training does not act as a uniformly beneficial
regularizer. It can improve the source corpus while damaging eJSL, or improve
eJSL while damaging the source corpus. The direction depends jointly on the
backbone, source corpus, and BOBSL amount.

The canonical B5K comparison from the final Proposal 3 report is:

| Source | Backbone | Source-only | Plain joint | Delta source | Delta eJSL | Regime |
|---|---|---:|---:|---:|---:|---|
| IEMOCAP | MMGCN | .2924 / .2072 | .3221 / .2090 | +.0297 | +.0018 | both up |
| IEMOCAP | MAGTKD | .3249 / .1096 | .3337 / .2708 | +.0089 | +.1612 | both up |
| IEMOCAP | CMERC | .3219 / .1763 | .2944 / .2522 | -.0275 | +.0759 | source erosion |
| IEMOCAP | ConxGNN | .3768 / .1180 | .3221 / .1745 | -.0547 | +.0565 | source erosion |
| IEMOCAP | ECERC | .2820 / .1720 | .3194 / .2309 | +.0374 | +.0589 | both up |
| MELD | MMGCN | .4151 / .3505 | .4401 / .2358 | +.0250 | -.1147 | external negative transfer |
| MELD | MAGTKD | .4468 / .3330 | .4646 / .2576 | +.0178 | -.0754 | external negative transfer |
| MELD | CMERC | .4664 / .2346 | .4671 / .2462 | +.0007 | +.0116 | both up |
| MELD | ConxGNN | .4301 / .2425 | .4491 / .2666 | +.0190 | +.0241 | both up |
| MELD | ECERC | .3560 / .1891 | .4268 / .2112 | +.0708 | +.0221 | both up |

Each score is source weighted F1 / eJSL weighted F1, averaged over seeds 35,
36, 37, 41, and 42. Exact values should be read from:

```text
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_report/proposal3_main_b5k.csv
```

Some older mechanism reports use earlier reproduced CMERC/ECERC/ConxGNN
anchors and therefore differ by a few thousandths from the frozen final table.
Those reports remain valid paired analyses of their saved checkpoints, but
their rounded headline values must not replace the Proposal 3 CSV.

## 2. The Useful Classification Is By Failure Mode

The experiments do not support a permanent taxonomy such as "MMGCN is always
negative-transfer" and "CMERC is always positive-transfer." They support two
recurring failure modes:

1. **External negative transfer.** Joint training improves the source corpus
   but overwrites behavior that transferred to eJSL. MELD MMGCN and MAGTKD are
   the clearest B5K examples.
2. **Source-anchor erosion.** Joint training learns BOBSL signals that improve
   eJSL, but moves away from the source optimum. IEMOCAP CMERC and ConxGNN are
   the clearest B5K examples.

ECERC is comparatively robust at B5K, while other backbones can move between
regimes as the source corpus or BOBSL budget changes. The paper-level
classification should therefore be phrased as **corpus-backbone transfer
regimes**, not immutable model families.

## 3. Auxiliary Scale Reveals Overload

Changing only the BOBSL budget can change the transfer regime. Selected
five-seed deltas from the cross-dataset trend report are:

| Source / Backbone | BOBSL | Delta source | Delta eJSL | Interpretation |
|---|---:|---:|---:|---|
| MELD / MMGCN | 2K | +.0169 | -.0421 | external harm begins |
| MELD / MMGCN | 5K | +.0250 | -.1147 | stronger source/eJSL trade-off |
| MELD / MMGCN | 10K | +.0332 | -.1294 | overload strengthens |
| MELD / MAGTKD | 1K | +.0045 | -.0629 | external harm already present |
| MELD / MAGTKD | 5K | +.0177 | -.0754 | source gain, eJSL loss |
| MELD / MAGTKD | 10K | +.0182 | -.0773 | trade-off persists |
| IEMOCAP / CMERC | 1K | +.0095 | +.1154 | both improve |
| IEMOCAP / CMERC | 2K | -.0219 | +.1182 | source erosion appears |
| IEMOCAP / CMERC | 5K | -.0275 | +.0758 | source erosion persists |
| IEMOCAP / ConxGNN | 1K | +.0033 | +.1486 | both improve |
| IEMOCAP / ConxGNN | 5K | -.0547 | +.0565 | source erosion |
| IEMOCAP / ConxGNN | 10K | -.0644 | +.0332 | stronger source erosion |

This supports a capacity/control problem rather than a simple lack-of-data
explanation: more auxiliary samples can increase one side of transfer while
decreasing the other.

Authoritative trend report:

```text
/raid_zoe/home/lr/maokeyu/sign/meld_iemocap_joint_trend_comparison/joint_cross_dataset_report.md
```

## 4. Joint Training Changes Class Decisions, Not Just Accuracy

The paired prediction audit compares source-only and B5K-joint predictions for
the same examples and seeds. Four representative failures are:

| Source / Backbone / Test | Delta wF1 | Disagreement | Fixed | Broken | Diagnostic |
|---|---:|---:|---:|---:|---|
| MELD / MMGCN / eJSL | -.1147 | .8073 | .2079 | .2915 | large external decision rewrite |
| MELD / MAGTKD / eJSL | -.0754 | .1864 | .0541 | .0923 | broken exceeds fixed |
| IEMOCAP / CMERC / IEMOCAP | -.0275 | .6181 | .1707 | .1994 | source decisions destabilize |
| IEMOCAP / ConxGNN / IEMOCAP | -.0547 | .5485 | .1600 | .1882 | source decisions destabilize |

The shifts are class-specific:

- MELD MMGCN on eJSL changes class F1 by `J=-.3357`, `N=-.2824`, and
  `S=+.3524`. Its output is not merely noisier; its decision prior moves
  sharply toward sadness.
- MELD MAGTKD on eJSL changes `J=-.2363`, while the small sadness gain cannot
  compensate for the lost angry/joy performance.
- IEMOCAP CMERC loses source `A=-.0717` and `S=-.0747`; source NLL increases
  by `+6.0926` in this saved run.
- IEMOCAP ConxGNN loses source `J=-.2329` and `N=-.0938`, while gaining A and
  S. This is a class-boundary rotation rather than uniform degradation.
- The opposite case is also visible: IEMOCAP MAGTKD improves eJSL by `+.1612`
  despite `.6911` prediction disagreement because its fixed rate `.2004`
  exceeds its broken rate `.1204`.

These observations motivate class-structured, bounded adaptation rather than
unconstrained full-model joint fine-tuning.

Artifacts:

```text
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/prediction_shift/report.md
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/prediction_shift/paired_seed_overall.csv
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/prediction_shift/paired_seed_per_class.csv
```

Reproduction script: `analysis/analyze_bobsl_b5k_prediction_shift.py`.

## 5. Shared Features Are Strongly Domain-Separable

The same TimeSformer extractor is used within each baseline pipeline, but
feature consistency does not imply domain-invariant emotion geometry.

Class-balanced linear domain probes obtain:

| Domain pair | Probe accuracy | Centroid cosine distance | RBF MMD |
|---|---:|---:|---:|
| BOBSL vs MELD | .9997 | .5112 | .1691 |
| BOBSL vs IEMOCAP | 1.0000 | .7310 | .3916 |
| BOBSL vs eJSL | 1.0000 | .4909 | .3804 |

A nearest-prototype emotion classifier trained on BOBSL features transfers at
only `.1351` wF1 to MELD, `.1815` to IEMOCAP, and `.0991` to eJSL. Thus BOBSL
contains useful supervision, but its raw visual emotion prototypes are not
directly interchangeable with source/eJSL prototypes.

This is supporting evidence for domain shift, not a causal proof by itself.
The paired prediction and context interventions provide the behavioral and
causal evidence.

Artifacts:

```text
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/visual_domain_geometry/pairwise_geometry.csv
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/visual_domain_geometry/prototype_transfer.csv
```

## 6. Dialogue Context Amplifies Both Benefit And Harm

BOBSL examples are length-one pseudo-dialogues, whereas MELD, IEMOCAP, and
eJSL contain dialogue context. The context ablation evaluates the same saved
models normally and with every utterance isolated as a context-length-one
dialogue.

The reported interaction is:

```text
(joint - source-only with normal context)
- (joint - source-only with context=1)
```

Positive interaction means context amplifies joint-training benefit; negative
interaction means it amplifies harm.

| Backbone / Source / Test | Normal joint gain | Context=1 gain | Interaction |
|---|---:|---:|---:|
| MAGTKD / IEMOCAP / eJSL | +.1607 | +.0171 | +.1435 |
| MAGTKD / MELD / eJSL | -.1125 | -.0057 | -.1068 |
| MAGTKD / MELD / MELD | +.0146 | -.0511 | +.0657 |
| CMERC / IEMOCAP / IEMOCAP | -.0230 | +.0407 | -.0638 |
| CMERC / IEMOCAP / eJSL | +.0527 | +.1128 | -.0601 |
| CMERC / MELD / eJSL | +.0147 | +.0778 | -.0631 |

The same MAGTKD context machinery strongly amplifies benefit with IEMOCAP but
harm with MELD. CMERC can improve as an isolated visual classifier while
losing source performance once dialogue context is restored. This is the
strongest evidence against assigning a fixed positive/negative label to a
backbone.

The causal conclusion is limited to these frozen checkpoints and the tested
normal-versus-isolated intervention. It does not prove that every context
layer is responsible.

Artifacts:

```text
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/context_ablation/report.md
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/context_ablation/mean_interactions.csv
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/context_ablation/seed_interactions.csv
```

Reproduction script: `analysis/evaluate_context_ablation.py`.

## 7. Historical Routing And Oracle Studies

Earlier experiments asked whether source-only and joint checkpoints contained
complementary decisions. They answer a useful diagnostic question, but they do
not define the final method.

### Classwise Pareto diagnostic

A classwise three-expert mixture selected weights on 20% of eJSL dialogues and
reported on the disjoint 80%. It produced positive source and held-out eJSL
deltas in 10/10 backbone/corpus cases. The expert ablation was:

| Experts | Passes | Mean source delta | Mean held-out eJSL delta |
|---|---:|---:|---:|
| source anchor only | 0/10 | +.0002 | -.0005 |
| joint anchor mixture | 9/10 | +.0021 | +.0336 |
| source + joint + auxiliary tri-expert | 10/10 | +.0010 | +.0365 |

This demonstrates expert complementarity and a reachable Pareto region. It
uses labeled eJSL calibration and multiple checkpoints, so it is an upper-bound
mechanism study, not a label-free or deployable Proposal 3 result.

### Epoch/profile oracle

The oracle analysis explicitly uses eJSL labels for profile and/or epoch
selection. Gains over its corresponding main runs reach `+.1034` for
MMGCN/IEMOCAP and `+.0290` for MAGTKD/MELD. This shows that checkpoint
trajectories contain substantially different transfer behavior, but those
numbers must never be reported as normal transfer performance.

Reports:

```text
/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_pareto/final_study/study_report.md
/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_oracle/comparison/oracle_report.md
```

## 8. Existing Visualizations

The following files still exist and remain useful as diagnostic figures:

```text
/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_pareto/final_study/pareto_scatter.png
/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_pareto/final_study/heldout_gain_bar.png
/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_pareto/final_study/expert_ablation.png
/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_pareto/final_study/expert_weight_heatmap.png
```

PDF versions are stored beside each PNG. Their roles are:

- `pareto_scatter`: source/eJSL trade-off and the existence of a Pareto region.
- `heldout_gain_bar`: held-out eJSL gains after the diagnostic calibration.
- `expert_ablation`: source-only, joint-mixture, and tri-expert comparison.
- `expert_weight_heatmap`: classwise expert preference.

These figures visualize the labeled held-out mixture study. They should be
labeled as diagnostic/upper-bound figures. They are not direct visualizations
of Proposal 3. A main-paper Proposal 3 figure should be regenerated from
`proposal3_main_b5k.csv` and clearly use the one-checkpoint results.

## 9. How The Evidence Connects To The Final Module

The experiment sequence is internally consistent:

1. **Plain joint baselines** expose two asymmetric failure modes and strong
   class/context-dependent decision shifts.
2. **Proposal 1** refines the exact plain-joint checkpoint. It beats plain
   joint on both metrics in 9/10 cases, but only beats source-only on both in
   6/10. This isolates inherited anchor damage as the remaining problem.
3. **Proposal 2** starts from a source anchor and constrains a residual to the
   source emotion subspace. Its mean deltas versus source-only are positive in
   10/10 cases, showing that anchor preservation addresses the failure found
   in Proposal 1.
4. **Proposal 3** locks backbone/corpus-specific profiles within the same
   plug-in module family. It beats both source-only and plain joint on both
   metrics in 7/10 cases, including all 4/4 MMGCN/MAGTKD hard cases.

The final module choices are therefore linked to observed problems:

- frozen anchor and source losses address source-anchor erosion;
- zero-initialized bounded residuals limit catastrophic decision changes;
- a low-rank source-emotion subspace encodes class structure without assuming
  raw cross-domain prototype equality;
- source-logit stability and safety terms protect transferable source behavior;
- optional peer distillation can recover useful joint behavior during
  training, while deployment still uses one checkpoint.

Proposal 3 does not claim universal dominance. Its three misses are
IEMOCAP/CMERC and IEMOCAP/ConxGNN versus the source-only source score, plus a
`.0001` eJSL shortfall versus plain joint for MELD/ConxGNN. Those limitations
are compatible with the earlier diagnosis of corpus-backbone-specific regimes.

## 10. Claims Supported By The Evidence

The evidence supports these statements:

- BOBSL joint training creates asymmetric, architecture- and corpus-dependent
  transfer rather than uniform regularization.
- Increasing BOBSL can strengthen either external negative transfer or source
  erosion.
- Joint training causes class-specific prior and boundary shifts.
- Dialogue context can causally amplify either auxiliary benefit or harm.
- The aligned visual features remain strongly domain-separable; direct
  prototype alignment is therefore unsafe as a universal solution.
- Source anchoring plus bounded class-structured adaptation is better aligned
  with the observed failure modes than unconstrained full-model joint tuning.

The evidence does not support these stronger statements:

- a fixed positive/negative category for each backbone across all corpora;
- universal per-seed improvement in all ten settings;
- a causal explanation based only on domain-probe accuracy;
- using routing/oracle results as label-free final performance;
- treating the historical mixture visualizations as Proposal 3 figures;
- claiming all checkpoint selection is on an untouched test set. The inherited
  scripts use the split named `source` for selection, which must be described
  exactly in any paper protocol.

## 11. Evidence Index

Primary final result:

```text
docs/proposal3_final_results.md
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_report/proposal3_main_b5k.csv
```

Baseline and mechanism evidence:

```text
/raid_zoe/home/lr/maokeyu/sign/meld_iemocap_joint_trend_comparison/
/raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary/
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/prediction_shift/
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/context_ablation/
/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/visual_domain_geometry/
```

Method-lineage evidence:

```text
docs/proposal1_preliminary_results.md
docs/proposal2_source_anchored_results.md
docs/proposal3_final_results.md
/raid_zoe/home/lr/maokeyu/sign/sast_plain_joint_refine/three_way/
/raid_zoe/home/lr/maokeyu/sign/proposal2_source_anchor/
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_report/
```

Diagnostic upper bounds and rejected alternatives:

```text
/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_oracle/
/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_pareto/
/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_v2/
/raid_zoe/home/lr/maokeyu/sign/decoupled_context_model/
/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/
```

The engineering paths, feature PKLs, checkpoints, and reproduction commands
are deliberately kept in `docs/EXPERIMENT_HANDOFF.md`.

The separate non-oracle text pilot is documented in
`docs/NONORACLE_TEXT_PILOT.md`.
