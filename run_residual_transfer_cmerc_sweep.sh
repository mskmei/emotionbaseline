#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
PHASE=${PHASE:-source}
PROFILES=${PROFILES:-"s010 s025 s050 s025wide s025free"}
SEEDS=${SEEDS:-35}
EPOCHS=${EPOCHS:-15}
OUT_ROOT=${OUT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/residual_transfer_cmerc/screen}
SOURCE_ROOT=${SOURCE_ROOT:-$OUT_ROOT}
SOURCE_PROFILE_MELD=${SOURCE_PROFILE_MELD:-s025}
SOURCE_PROFILE_IEMOCAP=${SOURCE_PROFILE_IEMOCAP:-s025}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
mkdir -p "$OUT_ROOT/logs"

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
      SOURCE_PROFILE="$SOURCE_PROFILE_MELD"
      ;;
    IEMOCAP)
      BATCH_SIZE=8
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/features/iemocap_cmerc_visual_anjs4.pkl
      JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      CONTEXT_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/iemocap_only_seed${seed}/model_best.pt
      SOURCE_PROFILE="$SOURCE_PROFILE_IEMOCAP"
      ;;
    *) echo "[RT-CMERC] unknown corpus=$corpus" >&2; return 2 ;;
  esac
}

resolve_profile() {
  case "$1" in
    s010) PROFILE_ARGS=(--source_scale 0.10 --adapter_hidden_dim 128 --adapter_dropout 0.2 --base_distill_weight 2.0 --source_residual_penalty 0.01 --lr 0.00001) ;;
    s025) PROFILE_ARGS=(--source_scale 0.25 --adapter_hidden_dim 128 --adapter_dropout 0.2 --base_distill_weight 2.0 --source_residual_penalty 0.01 --lr 0.00003) ;;
    s050) PROFILE_ARGS=(--source_scale 0.50 --adapter_hidden_dim 128 --adapter_dropout 0.2 --base_distill_weight 5.0 --source_residual_penalty 0.02 --lr 0.00003) ;;
    s025wide) PROFILE_ARGS=(--source_scale 0.25 --adapter_hidden_dim 256 --adapter_dropout 0.3 --base_distill_weight 2.0 --source_residual_penalty 0.01 --lr 0.00002) ;;
    s025free) PROFILE_ARGS=(--source_scale 0.25 --adapter_hidden_dim 128 --adapter_dropout 0.2 --base_distill_weight 0.5 --source_residual_penalty 0.005 --lr 0.00003) ;;
    r25cw) PROFILE_ARGS=(--transfer_scale 0.25 --gate_bias -3.0 --classwise_gate --stability_weight 0.2 --gate_penalty 0.01) ;;
    r50cw) PROFILE_ARGS=(--transfer_scale 0.5 --gate_bias -3.0 --classwise_gate --stability_weight 0.2 --gate_penalty 0.01) ;;
    r100cw) PROFILE_ARGS=(--transfer_scale 1.0 --gate_bias -3.0 --classwise_gate --stability_weight 0.2 --gate_penalty 0.01) ;;
    r50scalar) PROFILE_ARGS=(--transfer_scale 0.5 --gate_bias -3.0 --no-classwise_gate --stability_weight 0.2 --gate_penalty 0.01) ;;
    r50open) PROFILE_ARGS=(--transfer_scale 0.5 --gate_bias -2.0 --classwise_gate --stability_weight 0.1 --gate_penalty 0.005) ;;
    r50cross) PROFILE_ARGS=(--transfer_scale 0.5 --gate_bias -3.0 --classwise_gate --cross_source_weight 0.1 --stability_weight 0.2 --gate_penalty 0.01) ;;
    r50stable) PROFILE_ARGS=(--transfer_scale 0.5 --gate_bias -3.5 --classwise_gate --stability_weight 1.0 --gate_penalty 0.02) ;;
    r100cross) PROFILE_ARGS=(--transfer_scale 1.0 --gate_bias -2.5 --classwise_gate --cross_source_weight 0.1 --stability_weight 0.2 --gate_penalty 0.01) ;;
    *) echo "[RT-CMERC] unknown profile=$1" >&2; return 2 ;;
  esac
}

run_one() {
  local corpus="$1"
  local condition="$2"
  local profile="$3"
  local seed="$4"
  resolve_corpus "$corpus" "$seed"
  resolve_profile "$profile"
  local train_pkl="$SOURCE_PKL"
  local init_args=()
  if [[ "$condition" == joint_b5k ]]; then
    train_pkl="$JOINT_PKL"
    local source_init="$SOURCE_ROOT/$corpus/source_only/${SOURCE_PROFILE}_seed${seed}/model_best.pt"
    if [[ ! -s "$source_init" ]]; then echo "[RT-CMERC] missing source init: $source_init" >&2; return 1; fi
    init_args=(--init_source_checkpoint "$source_init")
  fi
  local out_dir="$OUT_ROOT/$corpus/$condition/${profile}_seed${seed}"
  local done_path="$out_dir/external_test_best_summary.json"
  local log_path="$OUT_ROOT/logs/${corpus}_${condition}_${profile}_seed${seed}.log"
  if [[ -s "$done_path" ]]; then echo "[RT-CMERC] skip existing $done_path"; return 0; fi
  mkdir -p "$out_dir"
  echo "[RT-CMERC] $corpus/$condition/$profile/seed$seed source_profile=$SOURCE_PROFILE" | tee "$log_path"
  set +e
  "$PYTHON_BIN" new2/CMERC/train_eval_residual_transfer_cmerc.py \
    --train_pkl "$train_pkl" --source_pkl "$SOURCE_PKL" --external_test_pkl "$EXTERNAL_PKL" \
    --context_checkpoint "$CONTEXT_CHECKPOINT" --out_dir "$out_dir" --seed "$seed" --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" --source_repeat 2 --min_domain_samples_per_epoch 500 \
    "${PROFILE_ARGS[@]}" "${init_args[@]}" 2>&1 | tee -a "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 && "$CONTINUE_ON_ERROR" != 1 ]]; then return "$status"; fi
  return 0
}

for corpus in $CORPORA; do
  if [[ "$PHASE" == source || "$PHASE" == both ]]; then
    for profile in $PROFILES; do
      for seed in $SEEDS; do run_one "$corpus" source_only "$profile" "$seed"; done
    done
  elif [[ "$PHASE" == joint ]]; then
    for profile in $PROFILES; do
      for seed in $SEEDS; do run_one "$corpus" joint_b5k "$profile" "$seed"; done
    done
  else
    echo "[RT-CMERC] unsupported PHASE=$PHASE" >&2
    exit 2
  fi
done

"$PYTHON_BIN" analysis/summarize_residual_transfer.py --root "$OUT_ROOT"
