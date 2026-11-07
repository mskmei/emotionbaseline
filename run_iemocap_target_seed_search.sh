#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE to MAGTKD, ECERC, CMERC, or ConxGNN}
GPU=${GPU:-0}
SEARCH_SEED=${SEARCH_SEED:-42}
PREFIX=${PREFIX:-search_seed${SEARCH_SEED}}

run_case() {
  local tag="$1"
  shift
  echo "[IEMOCAP-TARGET-SEARCH] baseline=$BASELINE case=$tag seed=$SEARCH_SEED gpu=$GPU params=$*"
  env TRAIN_CLASS_WEIGHT=1 "$@" \
    BASELINE="$BASELINE" \
    SWEEP_TAG="${PREFIX}_${tag}" \
    CONFIGS_CSV="iemocap_only,${JOINT_CONFIG}" \
    SEEDS_CSV="$SEARCH_SEED" \
    GPU="$GPU" \
    bash run_iemocap_joint_trend_sweep.sh
}

case "$BASELINE" in
  MAGTKD)
    JOINT_CONFIG=joint_b2k
    run_case lr50 TRAIN_LR=0.00005
    run_case lr150 TRAIN_LR=0.00015
    run_case lr300 TRAIN_LR=0.0003
    run_case drop00 TRAIN_DROPOUT=0.0
    run_case drop20 TRAIN_DROPOUT=0.2
    run_case drop80 TRAIN_DROPOUT=0.8
    run_case temp3 TRAIN_TEMP=3.0
    run_case wd10 TRAIN_WEIGHT_DECAY=0.00001
    ;;
  ECERC)
    JOINT_CONFIG=joint_b2k
    run_case gamma05 TRAIN_LOSS_GAMMA=0.5
    run_case gamma15 TRAIN_LOSS_GAMMA=1.5
    run_case gamma3 TRAIN_LOSS_GAMMA=3.0
    run_case lr75 TRAIN_LR=0.0000075
    run_case lr15 TRAIN_LR=0.000015
    run_case lr30 TRAIN_LR=0.00003
    run_case wd10 TRAIN_WEIGHT_DECAY=0.0001
    run_case wd40 TRAIN_WEIGHT_DECAY=0.0004
    ;;
  CMERC)
    JOINT_CONFIG=joint_b1k
    run_case lr50 TRAIN_LR=0.00005
    run_case lr200 TRAIN_LR=0.0002
    run_case drop20 TRAIN_DROPOUT=0.2
    run_case drop60 TRAIN_DROPOUT=0.6
    run_case gamma1 TRAIN_FOCAL_GAMMA=1.0
    run_case gamma4 TRAIN_FOCAL_GAMMA=4.0
    run_case ce_cw CMERC_LOSS=ce
    run_case ce_nocw CMERC_LOSS=ce TRAIN_CLASS_WEIGHT=0
    ;;
  ConxGNN)
    JOINT_CONFIG=joint_b1k
    run_case lr100 TRAIN_LR=0.0001
    run_case lr200 TRAIN_LR=0.0002
    run_case lr500 TRAIN_LR=0.0005
    run_case drop20 TRAIN_DROPOUT=0.2
    run_case drop40 TRAIN_DROPOUT=0.4
    run_case drop60 TRAIN_DROPOUT=0.6
    run_case wd0 TRAIN_WEIGHT_DECAY=0.0
    run_case wd10 TRAIN_WEIGHT_DECAY=0.00001
    ;;
  *)
    echo "Unsupported BASELINE=$BASELINE" >&2
    exit 2
    ;;
esac

echo "[IEMOCAP-TARGET-SEARCH] done baseline=$BASELINE prefix=$PREFIX seed=$SEARCH_SEED"
