# BOBSL + IEMOCAP video experiment

This migration keeps the existing BOBSL + MELD visual experiment intact and adds IEMOCAP as a second source dataset.

## Protocol

- IEMOCAP metadata and splits come from the repository files under `new/MAGTKD/datasets/IEMOCAP/`.
- The repository split contains 5,163 train, 647 dev, and 1,623 test utterances. Test is Session5.
- The mapped ANJS4 counts in those CSVs are train `A=2127 N=1221 J=1044 S=771`, dev `A=274 N=103 J=202 S=68`, and test `A=551 N=384 J=443 S=245`.
- To match the existing MELD experiment, IEMOCAP train and dev are folded into the training split. Session5 remains the source test split.
- Labels are mapped to the shared ANJS4 space:
  - `ang`, `fru` -> `A`
  - `neu` -> `N`
  - `hap`, `exc` -> `J`
  - `sad` -> `S`
- Each utterance segment uses the CSV start/end time with one second of context on both sides, matching the repository's previous IEMOCAP extractor.
- Eight uniformly spaced RGB frames are saved under `/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/frames_8_margin1`.
- TimeSformer and VideoMAE processor settings match the MELD visual feature pipeline.
- BOBSL samples remain context-one dialogues and use sample seed 123.
- Every baseline uses the same five configurations: IEMOCAP-only and joint training with 1k, 2k, 5k, or 10k sampled BOBSL utterances.
- Training seeds remain `35 36 37 41 42`.
- The best checkpoint is selected by source weighted F1, as in the prior MELD scripts. Metrics are then reported on IEMOCAP Session5 and eJSL from that same checkpoint.

The IEMOCAP labels are derived from the existing six-label metadata. This is different from selecting only four original IEMOCAP labels: no `fru` or `exc` examples are discarded.

## Existing features reused

The run scripts reuse the baseline-specific BOBSL and translated-eJSL PKLs from the completed MELD experiments. This avoids changing the feature origin for either comparison dataset. The scripts fail with the exact missing path if one is unavailable.

No new Python package is introduced. The new preparation stage uses `opencv-python`, and feature extraction uses the existing `torch`, `transformers`, and `numpy` dependencies.

## Server checks

Run a non-mutating layout, metadata, video, and existing-frame check:

```bash
python IEMOCAP/prepare_iemocap_anjs4_frames.py \
  --iemocap_root /raid_zoe/home/lr/maokeyu/IEMOCAP_full_release \
  --metadata_root "$PWD/new/MAGTKD/datasets/IEMOCAP" \
  --frame_root /raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/frames_8_margin1 \
  --manifest_root /raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/manifests \
  --check_only --strict
```

Inspect the generated report and split manifests:

```bash
python -m json.tool /raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/manifests/iemocap_anjs4_frame_report.json
head -n 3 /raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/manifests/iemocap_test_anjs4_frames.csv
```

Check all reused PKLs before starting the full run:

```bash
for p in \
  /raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/bobsl_anjs4_train_val.pkl \
  /raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl \
  /raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/bobsl_ecerc_video_anjs4_train_val.pkl \
  /raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/ejsl_ecerc_video_anjs4_translated.pkl \
  /raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/bobsl_magtkd_video_anjs4_train_val.pkl \
  /raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl \
  /raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/bobsl_cmerc_visual_anjs4_train_val.pkl \
  /raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl \
  /raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/bobsl_conxgnn_visual_anjs4_train_val.pkl \
  /raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/ejsl_conxgnn_visual_anjs4_translated.pkl
do
  test -s "$p" || echo "MISSING: $p"
done
```

## Running

Run all five baselines on two GPUs:

```bash
GPU0=0 GPU1=1 bash run_five_baselines_video_joint_bobsl_iemocap_ejsl.sh
```

The master script prepares frames, extracts IEMOCAP features once, writes all five native PKL formats, runs the baselines, and writes the combined report to:

```text
/raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary/iemocap_joint_report.md
```

Individual wrappers are also available:

```text
run_mmgcn_video_joint_bobsl_iemocap_ejsl.sh
run_ecerc_video_joint_bobsl_iemocap_ejsl.sh
run_magtkd_video_joint_bobsl_iemocap_ejsl.sh
run_cmerc_visual_joint_bobsl_iemocap_ejsl.sh
run_conxgnn_visual_joint_bobsl_iemocap_ejsl.sh
```

All stages are resumable by default. Set `REBUILD_IEMOCAP_FEATURES=1` or `REBUILD_JOINT=1` only when those artifacts must be regenerated.
