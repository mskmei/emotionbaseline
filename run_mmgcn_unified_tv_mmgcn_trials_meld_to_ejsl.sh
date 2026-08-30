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
TRIALS=${TRIALS:-"stable_seed41 stable_seed42 stable_seed47 stable_lr15 stable_dropout25 stable_gamma10"}

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
  local l2="$8"
  shift 8
  local trial_out="$OUT_ROOT/$name"

  mkdir -p "$trial_out"
  echo "[MMGCN-TV-TRIALS][$name] seed=$seed epochs=$epochs lr=$lr l2=$l2 dropout=$dropout loss=$loss focal_gamma=$focal_gamma extra=$*"

  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$MELD_UNIFIED_PKL" \
    --external_test_pkl "$EJSL_UNIFIED_PKL" \
    --out_dir "$trial_out" \
    --modalities tv \
    --graph_type MMGCN \
    --epochs "$epochs" \
    --batch_size "$BATCH_SIZE" \
    --lr "$lr" \
    --l2 "$l2" \
    --dropout "$dropout" \
    --loss "$loss" \
    --focal_gamma "$focal_gamma" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --seed "$seed" \
    "$@" | tee "$trial_out/train.log"
}

for trial in $TRIALS; do
  case "$trial" in
    stable)
      run_trial stable 43 "$EPOCHS" 0.0002 0.30 focal 1.5 0.00003
      ;;
    weighted_nll)
      run_trial weighted_nll 44 "$EPOCHS" 0.0002 0.30 nll 2.0 0.00003
      ;;
    seed_only)
      run_trial seed_only 45 "$EPOCHS" 0.0003 0.40 focal 2.0 0.00003
      ;;
    longer)
      run_trial longer 46 "$LONGER_EPOCHS" 0.00015 0.35 focal 1.5 0.00003
      ;;
    stable_seed41)
      run_trial stable_seed41 41 "$EPOCHS" 0.0002 0.30 focal 1.5 0.00003
      ;;
    stable_seed42)
      run_trial stable_seed42 42 "$EPOCHS" 0.0002 0.30 focal 1.5 0.00003
      ;;
    stable_seed47)
      run_trial stable_seed47 47 "$EPOCHS" 0.0002 0.30 focal 1.5 0.00003
      ;;
    stable_lr15)
      run_trial stable_lr15 43 "$EPOCHS" 0.00015 0.30 focal 1.5 0.00003
      ;;
    stable_lr25)
      run_trial stable_lr25 43 "$EPOCHS" 0.00025 0.30 focal 1.5 0.00003
      ;;
    stable_dropout25)
      run_trial stable_dropout25 43 "$EPOCHS" 0.0002 0.25 focal 1.5 0.00003
      ;;
    stable_dropout35)
      run_trial stable_dropout35 43 "$EPOCHS" 0.0002 0.35 focal 1.5 0.00003
      ;;
    stable_gamma10)
      run_trial stable_gamma10 43 "$EPOCHS" 0.0002 0.30 focal 1.0 0.00003
      ;;
    stable_gamma20)
      run_trial stable_gamma20 43 "$EPOCHS" 0.0002 0.30 focal 2.0 0.00003
      ;;
    stable_l2low)
      run_trial stable_l2low 43 "$EPOCHS" 0.0002 0.30 focal 1.5 0.00001
      ;;
    stable_l2high)
      run_trial stable_l2high 43 "$EPOCHS" 0.0002 0.30 focal 1.5 0.0001
      ;;
    stable_nocw)
      run_trial stable_nocw 43 "$EPOCHS" 0.0002 0.30 focal 1.5 0.00003 --no_class_weight
      ;;
    *)
      echo "[MMGCN-TV-TRIALS] unknown trial: $trial" >&2
      echo "[MMGCN-TV-TRIALS] valid: stable weighted_nll seed_only longer stable_seed41 stable_seed42 stable_seed47 stable_lr15 stable_lr25 stable_dropout25 stable_dropout35 stable_gamma10 stable_gamma20 stable_l2low stable_l2high stable_nocw" >&2
      exit 1
      ;;
  esac
done

echo "[MMGCN-TV-TRIALS] done. reports under: $OUT_ROOT"
echo "[MMGCN-TV-TRIALS] look for: */tv/external_test_best_classification_report.txt"
