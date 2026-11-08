#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-python}
WORKER_ID=${WORKER_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-1}
PROFILES=${PROFILES:-"td00 td25 td50 td75"}
EPOCHS=${EPOCHS:-15}
SOURCE_REPEAT=${SOURCE_REPEAT:-2}
MIN_DOMAIN_SAMPLES=${MIN_DOMAIN_SAMPLES:-500}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/tv_nonoracle}
SOURCE_PKL=${SOURCE_PKL:-/raid_zoe/home/lr/maokeyu/sign/non_mmgcn_text_dropout_nonoracle/features/cmerc_meld_text_visual_anjs4.pkl}
EXTERNAL_PKL=${EXTERNAL_PKL:-/raid_zoe/home/lr/maokeyu/sign/non_mmgcn_text_dropout_nonoracle/features/cmerc_ejsl_text_visual_anjs4_nonoracle.pkl}
BOBSL_PKL=${BOBSL_PKL:-/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/bobsl_cmerc_visual_anjs4_train_val.pkl}
CONTEXT_ROOT=${CONTEXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/non_mmgcn_text_dropout_nonoracle/runs/CMERC}
JOINT_PKL=${JOINT_PKL:-$WORK_ROOT/features/meld_tv_bobsl5k.pkl}
OUT_ROOT=${OUT_ROOT:-$WORK_ROOT/runs}
LOG_ROOT=${LOG_ROOT:-$WORK_ROOT/logs}
SEED=${SEED:-42}

mkdir -p "$WORK_ROOT/features" "$OUT_ROOT" "$LOG_ROOT"
for required in "$SOURCE_PKL" "$EXTERNAL_PKL" "$BOBSL_PKL"; do
  if [[ ! -s "$required" ]]; then echo "[DE-CMERC-TV] missing required file: $required" >&2; exit 1; fi
done

if [[ ! -s "$JOINT_PKL" ]]; then
  if [[ "$WORKER_ID" == 0 ]]; then
    "$PYTHON_BIN" new2/CMERC/build_cmerc_joint_visual_pkl.py \
      --source_pkl "$SOURCE_PKL" --source_name MELD \
      --bobsl_train_val_pkl "$BOBSL_PKL" --out_pkl "$JOINT_PKL" \
      --max_bobsl_train_dialogues 5000 --bobsl_sample_seed 123
  else
    for _ in $(seq 1 120); do
      [[ -s "$JOINT_PKL" ]] && break
      sleep 5
    done
    if [[ ! -s "$JOINT_PKL" ]]; then echo "[DE-CMERC-TV] timed out waiting for $JOINT_PKL" >&2; exit 1; fi
  fi
fi

run_one() {
  local profile="$1"
  local condition="$2"
  local context_checkpoint="$CONTEXT_ROOT/tv_${profile}/model_best.pt"
  local out_dir="$OUT_ROOT/$profile/$condition"
  local log_path="$LOG_ROOT/${profile}_${condition}.log"
  local train_pkl="$SOURCE_PKL"
  local -a expert_args=(
    --gate_biases 1.5 0 -0.5 --distill_weight 0.2 --gate_penalty 0.01
    --bobsl_weight 0.75 --bobsl_fused_weight 0.5 --alignment_weight 0.1
  )
  if [[ "$condition" == joint_b5k ]]; then
    train_pkl="$JOINT_PKL"
    local source_init="$OUT_ROOT/$profile/source_only/model_best.pt"
    if [[ ! -s "$source_init" ]]; then echo "[DE-CMERC-TV] missing source init: $source_init" >&2; return 1; fi
    expert_args+=(
      --residual_bobsl_gate --residual_gate_bias -3.0
      --init_source_expert_checkpoint "$source_init" --freeze_source_expert --freeze_base_gate
    )
  fi
  if [[ -s "$out_dir/external_test_best_summary.json" ]]; then
    echo "[DE-CMERC-TV] skip existing $profile/$condition"
    return
  fi
  if [[ ! -s "$context_checkpoint" ]]; then echo "[DE-CMERC-TV] missing context: $context_checkpoint" >&2; return 1; fi
  mkdir -p "$out_dir"
  echo "[DE-CMERC-TV] worker=$WORKER_ID profile=$profile condition=$condition" | tee "$log_path"
  "$PYTHON_BIN" new2/CMERC/train_eval_domain_expert_cmerc.py \
    --train_pkl "$train_pkl" --source_pkl "$SOURCE_PKL" --external_test_pkl "$EXTERNAL_PKL" \
    --context_checkpoint "$context_checkpoint" --out_dir "$out_dir" --modality tv \
    --seed "$SEED" --epochs "$EPOCHS" --batch_size 16 --source_repeat "$SOURCE_REPEAT" \
    --min_domain_samples_per_epoch "$MIN_DOMAIN_SAMPLES" "${expert_args[@]}" 2>&1 | tee -a "$log_path"
}

task_index=0
failures=0
for profile in $PROFILES; do
  current=$task_index
  task_index=$((task_index + 1))
  if (( current % NUM_WORKERS != WORKER_ID )); then continue; fi
  for condition in source_only joint_b5k; do
    set +e
    run_one "$profile" "$condition"
    status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      failures=$((failures + 1))
      if [[ "$CONTINUE_ON_ERROR" != 1 ]]; then exit "$status"; fi
    fi
  done
done
echo "[DE-CMERC-TV] worker=$WORKER_ID done failures=$failures"
if [[ "$NUM_WORKERS" == 1 ]]; then
  "$PYTHON_BIN" analysis/summarize_domain_expert_cmerc_tv.py --root "$OUT_ROOT"
fi
