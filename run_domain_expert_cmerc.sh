#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-python}
WORKER_ID=${WORKER_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-1}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
CONDITIONS=${CONDITIONS:-"source_only joint_b5k"}
PROFILES=${PROFILES:-"cautious balanced transfer"}
SEEDS=${SEEDS:-"35"}
EPOCHS=${EPOCHS:-15}
SOURCE_REPEAT=${SOURCE_REPEAT:-2}
MIN_DOMAIN_SAMPLES=${MIN_DOMAIN_SAMPLES:-500}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
OUT_ROOT=${OUT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/screen}
SOURCE_INIT_ROOT=${SOURCE_INIT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/domain_expert_cmerc/screen}
WARMSTART_SOURCE_ONLY=${WARMSTART_SOURCE_ONLY:-0}
LOG_ROOT=${LOG_ROOT:-$OUT_ROOT/logs}
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

resolve_corpus() {
  local corpus="$1"
  local seed="$2"
  case "$corpus" in
    MELD)
      BATCH_SIZE=16
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/meld_cmerc_visual_anjs4.pkl
      JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld/joint_pkls/joint_b5k.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      CONTEXT_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld/meld_only_seed${seed}/model_best.pt
      ;;
    IEMOCAP)
      BATCH_SIZE=8
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/features/iemocap_cmerc_visual_anjs4.pkl
      JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      CONTEXT_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/iemocap_only_seed${seed}/model_best.pt
      ;;
    *) echo "[DE-CMERC] unknown corpus=$corpus" >&2; return 2 ;;
  esac
}

resolve_profile() {
  case "$1" in
    cautious)
      PROFILE_ARGS=(--gate_biases 2.5 0 -1.5 --distill_weight 0.5 --gate_penalty 0.05 --bobsl_weight 0.5 --bobsl_fused_weight 0.1 --alignment_weight 0.05)
      ;;
    balanced)
      PROFILE_ARGS=(--gate_biases 2.0 0 -1.0 --distill_weight 0.3 --gate_penalty 0.02 --bobsl_weight 0.5 --bobsl_fused_weight 0.25 --alignment_weight 0.05)
      ;;
    transfer)
      PROFILE_ARGS=(--gate_biases 1.5 0 -0.5 --distill_weight 0.2 --gate_penalty 0.01 --bobsl_weight 0.75 --bobsl_fused_weight 0.5 --alignment_weight 0.1)
      ;;
    *) echo "[DE-CMERC] unknown profile=$1" >&2; return 2 ;;
  esac
}

task_index=0
failures=0
for corpus in $CORPORA; do
  for condition in $CONDITIONS; do
    for profile in $PROFILES; do
      for seed in $SEEDS; do
        current=$task_index
        task_index=$((task_index + 1))
        if (( current % NUM_WORKERS != WORKER_ID )); then continue; fi
        resolve_corpus "$corpus" "$seed"
        resolve_profile "$profile"
        if [[ "$condition" == source_only ]]; then TRAIN_PKL="$SOURCE_PKL"; else TRAIN_PKL="$JOINT_PKL"; fi
        INIT_ARGS=()
        if [[ "$condition" == joint_b5k && "$WARMSTART_SOURCE_ONLY" == 1 ]]; then
          source_init="$SOURCE_INIT_ROOT/$corpus/source_only/${profile}_seed${seed}/model_best.pt"
          if [[ ! -s "$source_init" ]]; then echo "[DE-CMERC] missing source init: $source_init" >&2; exit 1; fi
          INIT_ARGS=(--init_source_expert_checkpoint "$source_init" --freeze_source_expert)
        fi
        out_dir="$OUT_ROOT/$corpus/$condition/${profile}_seed${seed}"
        done_path="$out_dir/external_test_best_summary.json"
        log_path="$LOG_ROOT/${corpus}_${condition}_${profile}_seed${seed}.log"
        if [[ -s "$done_path" ]]; then echo "[DE-CMERC] skip existing $done_path"; continue; fi
        for required in "$TRAIN_PKL" "$SOURCE_PKL" "$EXTERNAL_PKL" "$CONTEXT_CHECKPOINT"; do
          if [[ ! -s "$required" ]]; then echo "[DE-CMERC] missing required file: $required" >&2; exit 1; fi
        done
        mkdir -p "$out_dir"
        echo "[DE-CMERC] worker=$WORKER_ID task=$current $corpus/$condition/$profile/seed$seed" | tee "$log_path"
        set +e
        "$PYTHON_BIN" new2/CMERC/train_eval_domain_expert_cmerc.py \
          --train_pkl "$TRAIN_PKL" --source_pkl "$SOURCE_PKL" --external_test_pkl "$EXTERNAL_PKL" \
          --context_checkpoint "$CONTEXT_CHECKPOINT" --out_dir "$out_dir" --seed "$seed" --epochs "$EPOCHS" \
          --batch_size "$BATCH_SIZE" --source_repeat "$SOURCE_REPEAT" \
          --min_domain_samples_per_epoch "$MIN_DOMAIN_SAMPLES" "${PROFILE_ARGS[@]}" "${INIT_ARGS[@]}" 2>&1 | tee -a "$log_path"
        status=${PIPESTATUS[0]}
        set -e
        if [[ "$status" -ne 0 ]]; then
          failures=$((failures + 1)); echo "[DE-CMERC] failed status=$status log=$log_path" | tee -a "$log_path"
          if [[ "$CONTINUE_ON_ERROR" != 1 ]]; then exit "$status"; fi
        fi
      done
    done
  done
done
echo "[DE-CMERC] worker=$WORKER_ID done tasks=$task_index failures=$failures"
if [[ "$NUM_WORKERS" == 1 ]]; then "$PYTHON_BIN" analysis/summarize_domain_expert_cmerc.py --root "$OUT_ROOT"; fi
