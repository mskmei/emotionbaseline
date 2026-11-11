# Proposal 1: Strict Plain-Joint Refinement

Status: preliminary result snapshot, frozen on 2026-11-11.

This document defines the result referred to as **Proposal 1** (Chinese:
**提案1**). Proposal 1 initializes a plug-in emotion-subspace adapter from the
exact plain BOBSL+source joint checkpoint, selects one source-validation
checkpoint, and uses that single checkpoint for both source and eJSL tests.
There is no test-time routing or checkpoint ensemble.

## Canonical B5K Result

All values are five-seed mean weighted F1. `Ours-Joint` and `Ours-Source`
denote the changes relative to plain joint training and source-only training,
respectively. Each cell is `source / eJSL`.

| Corpus | Backbone | Source-only | Plain joint | Proposal 1 | Ours-Joint | Ours-Source |
|---|---|---:|---:|---:|---:|---:|
| IEMOCAP | MMGCN | .2924 / .2072 | .3221 / .2090 | .3227 / .2126 | +.0006 / +.0036 | +.0303 / +.0054 |
| IEMOCAP | MAGTKD | .3249 / .1096 | .3337 / .2708 | .3355 / .2778 | +.0017 / +.0071 | +.0106 / +.1683 |
| IEMOCAP | CMERC | .3219 / .1763 | .2944 / .2522 | .2963 / .2524 | +.0020 / +.0003 | -.0256 / +.0761 |
| IEMOCAP | ConxGNN | .3768 / .1180 | .3221 / .1745 | .3308 / .1919 | +.0087 / +.0174 | -.0460 / +.0739 |
| IEMOCAP | ECERC | .2820 / .1720 | .3194 / .2309 | .3286 / .2329 | +.0092 / +.0021 | +.0466 / +.0609 |
| MELD | MMGCN | .4151 / .3505 | .4401 / .2358 | .4539 / .2810 | +.0138 / +.0452 | +.0389 / -.0695 |
| MELD | MAGTKD | .4468 / .3330 | .4646 / .2576 | .4654 / .2683 | +.0009 / +.0107 | +.0186 / -.0647 |
| MELD | CMERC | .4664 / .2346 | .4671 / .2462 | .4717 / .2526 | +.0046 / +.0063 | +.0053 / +.0180 |
| MELD | ConxGNN | .4301 / .2425 | .4491 / .2666 | .4506 / .2666 | +.0015 / -.0001 | +.0205 / +.0240 |
| MELD | ECERC | .3560 / .1891 | .4268 / .2112 | .4442 / .2369 | +.0174 / +.0257 | +.0882 / +.0477 |

## Frozen Conclusions

- Proposal 1 beats plain joint training on both metrics in 9/10 B5K settings.
  The only miss is ConxGNN/MELD eJSL by 0.0001 weighted F1.
- It reverses the two original IEMOCAP negative-transfer cases, MMGCN and
  MAGTKD, relative to both plain joint and source-only training.
- It beats source-only training on both metrics in 6/10 B5K settings and on
  16/20 individual source/eJSL metrics.
- Its remaining failure mode is inherited damage in the plain-joint base:
  IEMOCAP CMERC/ConxGNN lose source performance, while MELD MMGCN/MAGTKD lose
  eJSL performance. Refining that checkpoint improves it but does not fully
  recover the source-only floor.
- B1K is generally the safest transfer regime. B10K more often exposes BOBSL
  overload. The full B1K/B5K/B10K results remain part of Proposal 1.

## Reproducibility Record

Canonical server artifacts:

```text
/raid_zoe/home/lr/maokeyu/sign/sast_plain_joint_refine/three_way/strict_three_way_best.md
/raid_zoe/home/lr/maokeyu/sign/sast_plain_joint_refine/three_way/strict_three_way_best.csv
/raid_zoe/home/lr/maokeyu/sign/sast_plain_joint_refine/three_way/strict_group_results.csv
/raid_zoe/home/lr/maokeyu/sign/sast_plain_joint_refine/three_way/strict_seed_results.csv
```

The code snapshot ends at Git commit `fccd759`. Important implementation and
reporting commits are `5306e51`, `9572109`, `676fb5c`, `960b821`, `e4308bf`,
`669831a`, `bac3400`, and `fccd759`.

Any future request referring to Proposal 1 should use this definition and
these artifacts unless an explicit amended version is requested.
