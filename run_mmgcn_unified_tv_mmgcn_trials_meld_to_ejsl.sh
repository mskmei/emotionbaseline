#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}
MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$WORK_ROOT/meld_anjs4_unified.pkl"}
EJSL_UNIFIED_PKL=${EJSL_UNIFIED_PKL:-"$WORK_ROOT/ejsl_anjs4_unified.pkl"}
OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/runs_mmgcn_tv_trials"}

GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-10}
LONGER_EPOCHS=${LONGER_EPOCHS:-$EPOCHS}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-5.0}
TRIALS=${TRIALS:-"stable weighted_nll"}

mkdir -p "$OUT_ROOT"

if [ ! -f "$MELD_UNIFIED_PKL" ]; then
  echo "[MMGCN-TV-TRIALS] missing MELD_UNIFIED_PKL: $MELD_UNIFIED_PKL" >&2
  echo "[MMGCN-TV-TRIALS] run run_mmgcn_unified_meld_to_ejsl.sh once first, or set MELD_UNIFIED_PKL." >&2
  exit 1
fi

if [ ! -f "$EJSL_UNIFIED_PKL" ]; then
  echo "[MMGCN-TV-TRIALS] missing EJSL_UNIFIED_PKL: $EJSL_UNIFIED_PKL" >&2
  echo "[MMGCN-TV-TRIALS] run run_mmgcn_unified_meld_to_ejsl.sh once first, or set EJSL_UNIFIED_PKL." >&2
  exit 1
fi

run_trial() {
  local name="$1"
  local seed="$2"
  local epochs="$3"
  local lr="$4"
  local dropout="$5"
  local loss="$6"
  local focal_gamma="$7"
  local trial_out="$OUT_ROOT/$name"

  mkdir -p "$trial_out"
  echo "[MMGCN-TV-TRIALS][$name] seed=$seed epochs=$epochs lr=$lr dropout=$dropout loss=$loss focal_gamma=$focal_gamma"

  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$MELD_UNIFIED_PKL" \
    --external_test_pkl "$EJSL_UNIFIED_PKL" \
    --out_dir "$trial_out" \
    --modalities tv \
    --graph_type MMGCN \
    --epochs "$epochs" \
    --batch_size "$BATCH_SIZE" \
    --lr "$lr" \
    --dropout "$dropout" \
    --loss "$loss" \
    --focal_gamma "$focal_gamma" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --seed "$seed" | tee "$trial_out/train.log"
}

for trial in $TRIALS; do
  case "$trial" in
    stable)
      run_trial stable 43 "$EPOCHS" 0.0002 0.30 focal 1.5
      ;;
    weighted_nll)
      run_trial weighted_nll 44 "$EPOCHS" 0.0002 0.30 nll 2.0
      ;;
    seed_only)
      run_trial seed_only 45 "$EPOCHS" 0.0003 0.40 focal 2.0
      ;;
    longer)
      run_trial longer 46 "$LONGER_EPOCHS" 0.00015 0.35 focal 1.5
      ;;
    *)
      echo "[MMGCN-TV-TRIALS] unknown trial: $trial" >&2
      echo "[MMGCN-TV-TRIALS] valid: stable weighted_nll seed_only longer" >&2
      exit 1
      ;;
  esac
done

echo "[MMGCN-TV-TRIALS] done. reports under: $OUT_ROOT"
echo "[MMGCN-TV-TRIALS] look for: */tv/external_test_best_classification_report.txt"
