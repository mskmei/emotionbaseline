# Proposal 3: Final Plug-in Safe Joint Transfer

Status: final B5K five-seed result, frozen on 2026-11-13.

Proposal 3 is a plug-in residual emotion-subspace module. It is attached to an
existing visual emotion backbone, while the backbone is used as a frozen
source or joint anchor. Training may use a second frozen checkpoint as a peer
teacher, but the saved model and inference path contain one backbone and one
adapter only. There is no inference-time checkpoint routing, ensemble, or
eJSL-based epoch selection.

The search permits backbone- and corpus-specific hyperparameters. All main
results use BOBSL-5K and the five seeds 35, 36, 37, 41, and 42. Each checkpoint
is selected by source weighted F1.

## Final Main Table

Each cell is source weighted F1 / eJSL weighted F1.

| Corpus | Backbone | Source-only | Plain joint | Joint + module | Delta vs source | Delta vs joint | Pass both | Profile |
|---|---|---:|---:|---:|---:|---:|:---:|---|
| IEMOCAP | MMGCN | .2924 / .2072 | .3221 / .2090 | .3227 / .2126 | +.0303 / +.0054 | +.0006 / +.0036 | yes | `context_r1` |
| IEMOCAP | MAGTKD | .3249 / .1096 | .3337 / .2708 | .3376 / .3161 | +.0127 / +.2065 | +.0038 / +.0453 | yes | `decision_p0_w2_s025` |
| IEMOCAP | CMERC | .3219 / .1763 | .2944 / .2522 | .2963 / .2524 | -.0256 / +.0761 | +.0020 / +.0003 | no | `fullspace` |
| IEMOCAP | ConxGNN | .3768 / .1180 | .3221 / .1745 | .3308 / .1919 | -.0460 / +.0739 | +.0087 / +.0174 | no | `full` |
| IEMOCAP | ECERC | .2820 / .1720 | .3194 / .2309 | .3286 / .2329 | +.0466 / +.0609 | +.0092 / +.0021 | yes | `no_weight` |
| MELD | MMGCN | .4151 / .3505 | .4401 / .2358 | .4414 / .3715 | +.0264 / +.0210 | +.0013 / +.1357 | yes | `dual_better_w4` |
| MELD | MAGTKD | .4468 / .3330 | .4646 / .2576 | .4777 / .3406 | +.0309 / +.0077 | +.0131 / +.0831 | yes | `dual_all_w2` |
| MELD | CMERC | .4664 / .2346 | .4671 / .2462 | .4717 / .2526 | +.0053 / +.0180 | +.0046 / +.0063 | yes | `bobsl2` |
| MELD | ConxGNN | .4301 / .2425 | .4491 / .2666 | .4506 / .2666 | +.0205 / +.0240 | +.0015 / -.0001 | no | `rank1_b10_kl1_off31` |
| MELD | ECERC | .3560 / .1891 | .4268 / .2112 | .4442 / .2369 | +.0882 / +.0477 | +.0174 / +.0257 | yes | `no_weight` |

The MMGCN/MAGTKD hard requirement passes in all 4/4 model-corpus pairs. Across
all five backbones, 7/10 pairs beat both source-only and plain joint training
on both source and eJSL. MAGTKD/MELD also beats the stronger tuned drop80 joint
anchor, improving it from .4725/.3334 to .4777/.3406.

## Scale Exploration

B1K and B10K are supporting analyses rather than replacements for the aligned
B5K main table. Existing five-seed sweeps show that B1K is generally the safer
transfer regime. B10K more often exposes a source/eJSL trade-off. A new seed42
screen reached a strict double improvement for MMGCN/MELD B1K with
`dual_better_w4` (.4168/.3921), while no tested B10K setting gave a consistent
strict improvement across the hard pairs. These single-seed screens are kept
separate from the five-seed claims.

## Reproducibility Record

Canonical server artifacts:

```text
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_report/proposal3_main_b5k.md
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_report/proposal3_main_b5k.csv
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_report/proposal3_scale_fiveseed.csv
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_report/proposal3_scale_seed42_screen.csv
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_search/round10_drop80_anchor
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_search/drop80_five_seed_anchors
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_search/round11_magtkd_meld_five_seed
/raid_zoe/home/lr/maokeyu/sign/proposal3_final_search/round12_b1_b10_screen
```

The report can be regenerated with:

```bash
python analysis/build_proposal3_final_report.py
```

The implementation snapshot ends at Git commit `708b5d1`, with this document
added immediately afterward.
