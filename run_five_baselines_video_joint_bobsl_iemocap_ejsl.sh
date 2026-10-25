#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
IEMOCAP_ROOT=${IEMOCAP_ROOT:-/raid_zoe/home/lr/maokeyu/IEMOCAP_full_release}
IEMOCAP_METADATA_ROOT=${IEMOCAP_METADATA_ROOT:-"$ROOT_DIR/new/MAGTKD/datasets/IEMOCAP"}
IEMOCAP_FRAME_ROOT=${IEMOCAP_FRAME_ROOT:-/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/frames_8_margin1}
IEMOCAP_MANIFEST_ROOT=${IEMOCAP_MANIFEST_ROOT:-/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/manifests}
IEMOCAP_FEATURE_CACHE=${IEMOCAP_FEATURE_CACHE:-/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/timesformer_feature_cache}
MASTER_LOG_ROOT=${MASTER_LOG_ROOT:-/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/logs}
PREPARE_FRAMES=${PREPARE_FRAMES:-1}
BUILD_FEATURES=${BUILD_FEATURES:-1}
REBUILD_IEMOCAP_FEATURES=${REBUILD_IEMOCAP_FEATURES:-0}
mkdir -p "$MASTER_LOG_ROOT"

if [[ "$PREPARE_FRAMES" == "1" ]]; then
  echo "[IEMOCAP-ALL] validate/extract utterance frames"
  python IEMOCAP/prepare_iemocap_anjs4_frames.py \
    --iemocap_root "$IEMOCAP_ROOT" \
    --metadata_root "$IEMOCAP_METADATA_ROOT" \
    --frame_root "$IEMOCAP_FRAME_ROOT" \
    --manifest_root "$IEMOCAP_MANIFEST_ROOT" \
    --num_frames 8 \
    --margin_seconds 1.0 \
    --strict 2>&1 | tee "$MASTER_LOG_ROOT/prepare_frames.log"
fi

feature_outputs=(
  /raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/features/iemocap_mmgcn_video_anjs4.pkl
  /raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/features/iemocap_ecerc_video_anjs4.pkl
  /raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/features/iemocap_magtkd_video_anjs4.pkl
  /raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/features/iemocap_cmerc_visual_anjs4.pkl
  /raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/features/iemocap_conxgnn_visual_anjs4.pkl
)
need_features="$REBUILD_IEMOCAP_FEATURES"
for path in "${feature_outputs[@]}"; do
  [[ -s "$path" ]] || need_features=1
done

if [[ "$BUILD_FEATURES" == "1" && "$need_features" == "1" ]]; then
  echo "[IEMOCAP-ALL] extract TimeSformer features once and write all five pkl schemas on GPU $GPU0"
  CUDA_VISIBLE_DEVICES="$GPU0" python IEMOCAP/build_iemocap_anjs4_baseline_features.py \
    --baselines all \
    --iemocap_root "$IEMOCAP_ROOT" \
    --metadata_root "$IEMOCAP_METADATA_ROOT" \
    --frame_root "$IEMOCAP_FRAME_ROOT" \
    --cache_dir "$IEMOCAP_FEATURE_CACHE" \
    --num_frames 8 \
    --fp16 2>&1 | tee "$MASTER_LOG_ROOT/build_all_features.log"
fi

for path in "${feature_outputs[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "[IEMOCAP-ALL] missing feature pkl: $path" >&2
    exit 1
  fi
done

worker0() {
  BASELINE=MMGCN GPU="$GPU0" PREPARE_FRAMES=0 BUILD_FEATURES=0 bash run_video_joint_bobsl_iemocap_ejsl.sh
  BASELINE=ECERC GPU="$GPU0" PREPARE_FRAMES=0 BUILD_FEATURES=0 bash run_video_joint_bobsl_iemocap_ejsl.sh
}

worker1() {
  BASELINE=MAGTKD GPU="$GPU1" PREPARE_FRAMES=0 BUILD_FEATURES=0 bash run_video_joint_bobsl_iemocap_ejsl.sh
  BASELINE=CMERC GPU="$GPU1" PREPARE_FRAMES=0 BUILD_FEATURES=0 bash run_video_joint_bobsl_iemocap_ejsl.sh
  BASELINE=ConxGNN GPU="$GPU1" PREPARE_FRAMES=0 BUILD_FEATURES=0 bash run_video_joint_bobsl_iemocap_ejsl.sh
}

echo "[IEMOCAP-ALL] GPU $GPU0: MMGCN -> ECERC"
echo "[IEMOCAP-ALL] GPU $GPU1: MAGTKD -> CMERC -> ConxGNN"
worker0 >"$MASTER_LOG_ROOT/gpu${GPU0}_worker.log" 2>&1 &
pid0=$!
worker1 >"$MASTER_LOG_ROOT/gpu${GPU1}_worker.log" 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "[IEMOCAP-ALL] at least one GPU worker failed; inspect $MASTER_LOG_ROOT" >&2
  exit "$status"
fi

python IEMOCAP/summarize_iemocap_joint_baselines.py \
  --out_dir /raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary \
  2>&1 | tee "$MASTER_LOG_ROOT/five_baseline_summary.log"

echo "[IEMOCAP-ALL] done"
echo "[IEMOCAP-ALL] summary=/raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary/iemocap_joint_report.md"
