#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE to MMGCN, ECERC, MAGTKD, CMERC, or ConxGNN}
SWEEP_TAG=${SWEEP_TAG:?Set a descriptive SWEEP_TAG such as lr100 or drop20}
GPU=${GPU:-0}
CONFIGS=${CONFIGS:-"iemocap_only joint_b5k joint_b10k"}
SEEDS=${SEEDS:-"35 36 37 41 42"}
if [[ -n "${CONFIGS_CSV:-}" ]]; then
  CONFIGS=${CONFIGS_CSV//,/ }
fi
if [[ -n "${SEEDS_CSV:-}" ]]; then
  SEEDS=${SEEDS_CSV//,/ }
fi

baseline_lower=$(printf '%s' "$BASELINE" | tr '[:upper:]' '[:lower:]')
work_root="/raid_zoe/home/lr/maokeyu/sign/${baseline_lower}_bobsl_iemocap_ejsl"
out_root=${OUT_ROOT:-"$work_root/video_joint_trend_sweep/$SWEEP_TAG"}
joint_pkl_root=${JOINT_PKL_ROOT:-"$work_root/video_joint_bobsl_iemocap/joint_pkls"}

echo "[IEMOCAP-TREND-SWEEP] baseline=$BASELINE tag=$SWEEP_TAG gpu=$GPU"
echo "[IEMOCAP-TREND-SWEEP] configs=$CONFIGS seeds=$SEEDS out=$out_root"
echo "[IEMOCAP-TREND-SWEEP] lr=${TRAIN_LR:-default} dropout=${TRAIN_DROPOUT:-default} batch=${TRAIN_BATCH_SIZE:-default}"

BASELINE="$BASELINE" \
GPU="$GPU" \
CONFIGS="$CONFIGS" \
SEEDS="$SEEDS" \
OUT_ROOT="$out_root" \
JOINT_PKL_ROOT="$joint_pkl_root" \
SWEEP_TAG="$SWEEP_TAG" \
PREPARE_FRAMES=0 \
BUILD_FEATURES=0 \
RESUME_DONE=${RESUME_DONE:-1} \
bash run_video_joint_bobsl_iemocap_ejsl.sh

echo "[IEMOCAP-TREND-SWEEP] done: $out_root"
