#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}
MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$WORK_ROOT/meld_anjs4_unified.pkl"}
EJSL_UNIFIED_PKL=${EJSL_UNIFIED_PKL:-"$WORK_ROOT/ejsl_anjs4_unified.pkl"}
OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/runs_mmgcn_tv_search30"}

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-15}
RESUME=${RESUME:-1}
TOP_K=${TOP_K:-10}
TRIALS=${TRIALS:-"s01_seed35 s02_seed36 s03_seed37 s04_seed38 s05_seed39 s06_seed40 s07_seed48 s08_seed49 s09_lr10 s10_lr125 s11_lr15 s12_lr175 s13_lr225 s14_lr25 s15_dropout20 s16_dropout25 s17_dropout28 s18_dropout35 s19_dropout40 s20_gamma075 s21_gamma10 s22_gamma125 s23_gamma175 s24_gamma20 s25_batch4 s26_batch16 s27_l2low s28_l2high s29_usemodal s30_avlstm"}

mkdir -p "$OUT_ROOT"

if [ ! -f "$MELD_UNIFIED_PKL" ]; then
  echo "[MMGCN-TV-SEARCH30] missing MELD_UNIFIED_PKL: $MELD_UNIFIED_PKL" >&2
  exit 1
fi

if [ ! -f "$EJSL_UNIFIED_PKL" ]; then
  echo "[MMGCN-TV-SEARCH30] missing EJSL_UNIFIED_PKL: $EJSL_UNIFIED_PKL" >&2
  exit 1
fi

run_trial() {
  local name="$1"
  local seed="$2"
  local epochs="$3"
  local batch_size="$4"
  local lr="$5"
  local l2="$6"
  local dropout="$7"
  local loss="$8"
  local focal_gamma="$9"
  local max_grad_norm="${10}"
  shift 10
  local trial_out="$OUT_ROOT/$name"
  local best_summary="$trial_out/tv/external_test_best_summary.json"

  if [ "$RESUME" = "1" ] && [ -f "$best_summary" ]; then
    echo "[MMGCN-TV-SEARCH30][$name] skip existing $best_summary"
    return
  fi

  mkdir -p "$trial_out"
  echo "[MMGCN-TV-SEARCH30][$name] seed=$seed epochs=$epochs batch=$batch_size lr=$lr l2=$l2 dropout=$dropout loss=$loss gamma=$focal_gamma grad=$max_grad_norm extra=$*"

  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$MELD_UNIFIED_PKL" \
    --external_test_pkl "$EJSL_UNIFIED_PKL" \
    --out_dir "$trial_out" \
    --modalities tv \
    --graph_type MMGCN \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --l2 "$l2" \
    --dropout "$dropout" \
    --loss "$loss" \
    --focal_gamma "$focal_gamma" \
    --max_grad_norm "$max_grad_norm" \
    --seed "$seed" \
    "$@" | tee "$trial_out/train.log"
}

for trial in $TRIALS; do
  case "$trial" in
    s01_seed35)
      run_trial "$trial" 35 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s02_seed36)
      run_trial "$trial" 36 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s03_seed37)
      run_trial "$trial" 37 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s04_seed38)
      run_trial "$trial" 38 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s05_seed39)
      run_trial "$trial" 39 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s06_seed40)
      run_trial "$trial" 40 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s07_seed48)
      run_trial "$trial" 48 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s08_seed49)
      run_trial "$trial" 49 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s09_lr10)
      run_trial "$trial" 43 "$EPOCHS" 8 0.00010 0.00003 0.30 focal 1.5 5.0
      ;;
    s10_lr125)
      run_trial "$trial" 43 "$EPOCHS" 8 0.000125 0.00003 0.30 focal 1.5 5.0
      ;;
    s11_lr15)
      run_trial "$trial" 43 "$EPOCHS" 8 0.00015 0.00003 0.30 focal 1.5 5.0
      ;;
    s12_lr175)
      run_trial "$trial" 43 "$EPOCHS" 8 0.000175 0.00003 0.30 focal 1.5 5.0
      ;;
    s13_lr225)
      run_trial "$trial" 43 "$EPOCHS" 8 0.000225 0.00003 0.30 focal 1.5 5.0
      ;;
    s14_lr25)
      run_trial "$trial" 43 "$EPOCHS" 8 0.00025 0.00003 0.30 focal 1.5 5.0
      ;;
    s15_dropout20)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.20 focal 1.5 5.0
      ;;
    s16_dropout25)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.25 focal 1.5 5.0
      ;;
    s17_dropout28)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.28 focal 1.5 5.0
      ;;
    s18_dropout35)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.35 focal 1.5 5.0
      ;;
    s19_dropout40)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.40 focal 1.5 5.0
      ;;
    s20_gamma075)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 0.75 5.0
      ;;
    s21_gamma10)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.0 5.0
      ;;
    s22_gamma125)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.25 5.0
      ;;
    s23_gamma175)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.75 5.0
      ;;
    s24_gamma20)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 2.0 5.0
      ;;
    s25_batch4)
      run_trial "$trial" 43 "$EPOCHS" 4 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s26_batch16)
      run_trial "$trial" 43 "$EPOCHS" 16 0.0002 0.00003 0.30 focal 1.5 5.0
      ;;
    s27_l2low)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00001 0.30 focal 1.5 5.0
      ;;
    s28_l2high)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.0001 0.30 focal 1.5 5.0
      ;;
    s29_usemodal)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0 --use_modal
      ;;
    s30_avlstm)
      run_trial "$trial" 43 "$EPOCHS" 8 0.0002 0.00003 0.30 focal 1.5 5.0 --av_using_lstm
      ;;
    *)
      echo "[MMGCN-TV-SEARCH30] unknown trial: $trial" >&2
      exit 1
      ;;
  esac
done

python MMGCN/summarize_unified_trial_search.py \
  --root "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/search30_summary.csv" \
  --top_k "$TOP_K"

echo "[MMGCN-TV-SEARCH30] reports: $OUT_ROOT"
echo "[MMGCN-TV-SEARCH30] summary: $OUT_ROOT/search30_summary.csv"
