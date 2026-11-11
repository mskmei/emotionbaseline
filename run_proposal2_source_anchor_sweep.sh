#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE to MMGCN, MAGTKD, CMERC, ConxGNN, or ECERC}
GPU=${GPU:-0}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
BOBSL_K=${BOBSL_K:-5}
SEEDS=${SEEDS:-42}
EPOCHS=${EPOCHS:-}
RESUME_DONE=${RESUME_DONE:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/proposal2_source_anchor}
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}

echo "[PROPOSAL2] baseline=$BASELINE corpora=$CORPORA bobsl=${BOBSL_K}k seeds=$SEEDS gpu=$GPU epochs=${EPOCHS:-baseline-default}"
echo "[PROPOSAL2] one final checkpoint = frozen source-only backbone + source-anchored adapter"

case "$BASELINE" in
  MMGCN|MAGTKD)
    PROFILES=${PROFILES:-"full_safe low_lr very_low_lr b25_low safe_r3 no_weight open_r3 source4_r3"}
    BASELINE="$BASELINE" \
    GPU="$GPU" CORPORA="$CORPORA" BOBSL_K="$BOBSL_K" SEEDS="$SEEDS" EPOCHS="$EPOCHS" \
    PROFILES="$PROFILES" BASE_MODE=source_only RESUME_DONE="$RESUME_DONE" \
    WORK_ROOT="$WORK_ROOT" PYTHON_BIN="$PYTHON_BIN" \
      bash run_sast_negative_transfer_screen.sh
    ;;
  CMERC)
    PROFILES=${PROFILES:-"open open_kl1 open_kl5 fullspace fullspace_kl1 lowbobsl rank1_safe source4"}
    GPU="$GPU" CORPORA="$CORPORA" BOBSL_K="$BOBSL_K" SEEDS="$SEEDS" EPOCHS="$EPOCHS" \
    PROFILES="$PROFILES" BASE_MODE=source_only RESUME_DONE="$RESUME_DONE" \
    WORK_ROOT="$WORK_ROOT/CMERC" PYTHON_BIN="$PYTHON_BIN" \
      bash run_cesa_cmerc_screen.sh
    ;;
  ConxGNN)
    PROFILES=${PROFILES:-"rank1 rank1_safe rank1_b10_kl1 rank1_b25 rank1_src4 rank1_unweighted very_low_lr high_lr"}
    GPU="$GPU" CORPORA="$CORPORA" BOBSL_K="$BOBSL_K" SEEDS="$SEEDS" EPOCHS="$EPOCHS" \
    PROFILES="$PROFILES" BASE_MODE=source_only \
    WORK_ROOT="$WORK_ROOT/ConxGNN" PYTHON_BIN="$PYTHON_BIN" \
      bash run_conxgnn_sast_screen.sh
    ;;
  ECERC)
    PROFILES=${PROFILES:-"no_weight context_r3 safe_r3 low_lr rank1_safe b25_low source4_r3 scale1_r3"}
    GPU="$GPU" CORPORA="$CORPORA" BOBSL_K="$BOBSL_K" SEEDS="$SEEDS" EPOCHS="$EPOCHS" \
    PROFILES="$PROFILES" BASE_MODE=source_only RESUME_DONE="$RESUME_DONE" \
    WORK_ROOT="$WORK_ROOT/ECERC" PYTHON_BIN="$PYTHON_BIN" \
      bash run_ecerc_sast_screen.sh
    ;;
  *)
    echo "Unsupported BASELINE=$BASELINE" >&2
    exit 2
    ;;
esac
