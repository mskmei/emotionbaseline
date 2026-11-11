# Proposal 2: Source-Anchored Single-Checkpoint SAST

Status: B5K five-seed stage result, frozen on 2026-11-12.

This document defines the result referred to as **Proposal 2** (Chinese:
**提案2**). It addresses the inherited-damage failure of Proposal 1 by
initializing from each seed's source-only checkpoint instead of its plain-joint
checkpoint. The source backbone is frozen and a zero-initialized residual
adapter is trained once with source and BOBSL data.

The adapter residual is constrained to an orthonormal source-emotion subspace
computed from source class centroids. Source classification, per-sample safety,
source-logit KL stability, and residual-norm losses preserve the source anchor;
BOBSL classification and prototype losses inject transferable visual emotion
information. CMERC additionally uses a local head to train on length-one BOBSL
samples without pretending that BOBSL has dialogue context.

Only the source metric selects `model_best.pt`; eJSL is evaluation-only. The
final checkpoint contains the frozen backbone and its adapter. Inference uses
one checkpoint, with no routing, ensemble, or test-time checkpoint selection.

## Canonical B5K Result

All values are mean weighted F1 over seeds 35, 36, 37, 41, and 42. `Delta S/E`
is relative to the paired source-only checkpoints. `Strict` counts seeds for
which both individual deltas are positive.

| Corpus | Backbone | Profile | Source-only -> Proposal 2 | eJSL base -> Proposal 2 | Delta S/E | Strict |
|---|---|---|---:|---:|---:|---:|
| IEMOCAP | MMGCN | `full_safe` | .2924 -> .3104 | .2072 -> .3009 | +.0180 / +.0937 | 4/5 |
| IEMOCAP | MAGTKD | `no_weight` | .3249 -> .3401 | .1096 -> .1578 | +.0153 / +.0482 | 4/5 |
| IEMOCAP | CMERC | `open` | .3219 -> .3291 | .1763 -> .1774 | +.0072 / +.0010 | 1/5 |
| IEMOCAP | ConxGNN | `rank1_safe` | .3768 -> .3796 | .1180 -> .1209 | +.0028 / +.0030 | 2/5 |
| IEMOCAP | ECERC | `no_weight_rank1` | .2815 -> .2849 | .1720 -> .1723 | +.0034 / +.0003 | 1/5 |
| MELD | MMGCN | `low_lr` | .4151 -> .4162 | .3505 -> .3511 | +.0012 / +.0007 | 1/5 |
| MELD | MAGTKD | `low_lr` | .4468 -> .4482 | .3330 -> .3435 | +.0014 / +.0105 | 2/5 |
| MELD | CMERC | `open_kl5` | .4663 -> .4716 | .2345 -> .2425 | +.0053 / +.0080 | 2/5 |
| MELD | ConxGNN | `rank1_src4` | .4307 -> .4327 | .2412 -> .2500 | +.0020 / +.0088 | 5/5 |
| MELD | ECERC | `no_weight` | .3502 -> .4241 | .1891 -> .2177 | +.0739 / +.0286 | 4/5 |

## Comparison With Proposal 1

Proposal 1 had positive source/eJSL means relative to source-only in 6/10 B5K
model-corpus pairs. Proposal 2 has positive means in 10/10. In particular, it
turns all four remaining one-sided failures positive:

| Corpus / Backbone | Proposal 1 Delta S/E | Proposal 2 Delta S/E |
|---|---:|---:|
| IEMOCAP / CMERC | -.0256 / +.0761 | +.0072 / +.0010 |
| IEMOCAP / ConxGNN | -.0460 / +.0739 | +.0028 / +.0030 |
| MELD / MMGCN | +.0389 / -.0695 | +.0012 / +.0007 |
| MELD / MAGTKD | +.0186 / -.0647 | +.0014 / +.0105 |

The trade-off is deliberate: Proposal 2 preserves the source-only floor across
all backbones, while some gains are smaller than Proposal 1's unconstrained
plain-joint refinement. CMERC/IEMOCAP, ECERC/IEMOCAP, and MMGCN/MELD have small
eJSL margins and low strict-seed counts; their positive five-seed means are the
current claim, not universal per-seed dominance.

## Reproducibility Record

Canonical server artifacts:

```text
/raid_zoe/home/lr/maokeyu/sign/proposal2_source_anchor/proposal2_main_table_n5.md
/raid_zoe/home/lr/maokeyu/sign/proposal2_source_anchor/proposal2_best_n5.csv
/raid_zoe/home/lr/maokeyu/sign/proposal2_source_anchor/proposal2_group_results.csv
/raid_zoe/home/lr/maokeyu/sign/proposal2_source_anchor/proposal2_seed_results.csv
```

The unified runner is `run_proposal2_source_anchor_sweep.sh`, and the strict
aggregator is `analysis/summarize_proposal2_source_anchor.py`. The implementation
snapshot includes commits `5189300`, `699b3a9`, and `3294887` on branch `bobsl`.

Any future request referring to Proposal 2 should use this definition and these
artifacts unless an explicit amended version is requested.
