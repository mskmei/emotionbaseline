#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
PHASE=${PHASE:-screen}
SCREEN_SEED=${SCREEN_SEED:-42}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-"35 36 37 41 42"}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
PYTHON=${PYTHON:-python}
RESUME_DONE=${RESUME_DONE:-1}

TRAIN_PKL=${TRAIN_PKL:-/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/features/iemocap_magtkd_video_anjs4.pkl}
EJSL_PKL=${EJSL_PKL:-/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl}
OUT_ROOT=${OUT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/source_only_tuning}
LOG_ROOT="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

# profile|lr|dropout|head|class_weight|weight_power|label_smoothing|batch|epochs|heads|warmup|l2
SCREEN_PROFILES=(
  "canonical|0.0001|0.5|video|1|1.0|0.0|16|30|8|-1|0.000001"
  "lr30|0.00003|0.5|video|1|1.0|0.0|16|40|8|-1|0.000001"
  "lr50|0.00005|0.5|video|1|1.0|0.0|16|40|8|-1|0.000001"
  "lr75|0.000075|0.5|video|1|1.0|0.0|16|40|8|-1|0.000001"
  "lr150|0.00015|0.5|video|1|1.0|0.0|16|30|8|-1|0.000001"
  "lr300|0.0003|0.5|video|1|1.0|0.0|16|30|8|-1|0.000001"
  "drop10|0.0001|0.1|video|1|1.0|0.0|16|30|8|-1|0.000001"
  "drop30|0.0001|0.3|video|1|1.0|0.0|16|30|8|-1|0.000001"
  "drop70|0.0001|0.7|video|1|1.0|0.0|16|30|8|-1|0.000001"
  "drop90|0.0001|0.9|video|1|1.0|0.0|16|40|8|-1|0.000001"
  "smooth05|0.0001|0.5|video|1|1.0|0.05|16|30|8|-1|0.000001"
  "smooth10|0.0001|0.5|video|1|1.0|0.1|16|30|8|-1|0.000001"
  "smooth20|0.0001|0.5|video|1|1.0|0.2|16|30|8|-1|0.000001"
  "weight25|0.0001|0.5|video|1|0.25|0.0|16|30|8|-1|0.000001"
  "weight50|0.0001|0.5|video|1|0.5|0.0|16|30|8|-1|0.000001"
  "weight75|0.0001|0.5|video|1|0.75|0.0|16|30|8|-1|0.000001"
  "weight125|0.0001|0.5|video|1|1.25|0.0|16|30|8|-1|0.000001"
  "weight150|0.0001|0.5|video|1|1.5|0.0|16|30|8|-1|0.000001"
  "nocw_smooth10|0.0001|0.5|video|0|1.0|0.1|16|30|8|-1|0.000001"
  "batch8|0.0001|0.5|video|1|1.0|0.0|8|30|8|-1|0.000001"
  "batch32|0.0001|0.5|video|1|1.0|0.0|32|30|8|-1|0.000001"
  "heads4|0.0001|0.5|video|1|1.0|0.0|16|30|4|-1|0.000001"
  "heads16|0.0001|0.5|video|1|1.0|0.0|16|30|16|-1|0.000001"
  "warmup0|0.0001|0.5|video|1|1.0|0.0|16|30|8|0|0.000001"
  "wd100|0.0001|0.5|video|1|1.0|0.0|16|30|8|-1|0.0001"
  "text_cw|0.0001|0.5|text|1|1.0|0.0|16|30|8|-1|0.000001"
  "text_nocw|0.0001|0.5|text|0|1.0|0.0|16|30|8|-1|0.000001"
  "audio_cw|0.0001|0.5|audio|1|1.0|0.0|16|30|8|-1|0.000001"
  "audio_nocw|0.0001|0.5|audio|0|1.0|0.0|16|30|8|-1|0.000001"
  "reg_lr50_d30_w50_s10|0.00005|0.3|video|1|0.5|0.1|16|50|8|-1|0.000001"
  "reg_lr75_d30_w50_s10|0.000075|0.3|video|1|0.5|0.1|16|40|8|-1|0.000001"
  "reg_lr75_d70_w75_s05|0.000075|0.7|video|1|0.75|0.05|16|40|8|-1|0.000001"
)

# Set after the screen phase. The defaults retain the most plausible regularized
# candidates; TOP_PROFILES may override this list without editing the script.
DEFAULT_CONFIRM_PROFILES="lr300 smooth10 weight50 text_cw audio_cw reg_lr50_d30_w50_s10 reg_lr75_d30_w50_s10"

find_profile() {
  local wanted="$1" entry
  for entry in "${SCREEN_PROFILES[@]}"; do
    if [[ "${entry%%|*}" == "$wanted" ]]; then
      printf '%s\n' "$entry"
      return 0
    fi
  done
  echo "Unknown profile: $wanted" >&2
  return 2
}

run_one() {
  local entry="$1" seed="$2"
  local profile lr dropout head class_weight weight_power smoothing batch epochs heads warmup l2
  IFS='|' read -r profile lr dropout head class_weight weight_power smoothing batch epochs heads warmup l2 <<< "$entry"
  local out_dir="$OUT_ROOT/${profile}_seed${seed}"
  local summary="$out_dir/external_test_best_summary.json"
  local log="$LOG_ROOT/${profile}_seed${seed}.log"
  if [[ "$RESUME_DONE" == "1" && -s "$summary" ]]; then
    echo "[MAGTKD-SOURCE] skip $profile seed=$seed"
    return
  fi
  mkdir -p "$out_dir"
  local cmd=(
    "$PYTHON" new/MAGTKD/MELD/train_eval_magtkd_video_anjs.py
    --train_pkl "$TRAIN_PKL"
    --external_test_pkl "$EJSL_PKL"
    --out_dir "$out_dir"
    --modality video
    --output_head "$head"
    --seed "$seed"
    --epochs "$epochs"
    --batch_size "$batch"
    --num_workers 0
    --lr "$lr"
    --l2 "$l2"
    --dropout "$dropout"
    --hidden_dim 768
    --n_head "$heads"
    --temp 2.0
    --warmup_steps "$warmup"
    --class_weight_power "$weight_power"
    --label_smoothing "$smoothing"
    --max_grad_norm 10.0
    --selection_split source
    --selection_metric weighted_f1
  )
  if [[ "$class_weight" == "1" ]]; then cmd+=(--class_weight); fi
  echo "[MAGTKD-SOURCE] profile=$profile seed=$seed gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee "$log"
}

run_screen() {
  local index=0 entry
  for entry in "${SCREEN_PROFILES[@]}"; do
    if (( index % NUM_SHARDS == SHARD_INDEX )); then
      run_one "$entry" "$SCREEN_SEED"
    fi
    index=$((index + 1))
  done
}

run_confirm() {
  local profiles=${TOP_PROFILES:-$DEFAULT_CONFIRM_PROFILES}
  local profile seed entry
  for profile in $profiles; do
    entry=$(find_profile "$profile")
    for seed in $CONFIRM_SEEDS; do
      run_one "$entry" "$seed"
    done
  done
}

echo "[MAGTKD-SOURCE] phase=$PHASE gpu=$GPU shard=$SHARD_INDEX/$NUM_SHARDS out=$OUT_ROOT"
case "$PHASE" in
  screen) run_screen ;;
  confirm) run_confirm ;;
  all) run_screen; run_confirm ;;
  *) echo "PHASE must be screen, confirm, or all" >&2; exit 2 ;;
esac

"$PYTHON" analysis/summarize_magtkd_iemocap_source_tuning.py \
  --root "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/summary.csv"
