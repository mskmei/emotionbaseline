#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
PHASE=${PHASE:-source}
PROFILES=${PROFILES:-"s010 s025 s050"}
SEEDS=${SEEDS:-35}
EPOCHS=${EPOCHS:-15}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/residual_transfer_module/MMGCN}
OUT_ROOT=${OUT_ROOT:-$WORK_ROOT/runs}
CACHE_ROOT=${CACHE_ROOT:-$WORK_ROOT/cache}
SOURCE_ROOT=${SOURCE_ROOT:-$OUT_ROOT}
SOURCE_PROFILE_MELD=${SOURCE_PROFILE_MELD:-s025}
SOURCE_PROFILE_IEMOCAP=${SOURCE_PROFILE_IEMOCAP:-s025}
mkdir -p "$OUT_ROOT/logs" "$CACHE_ROOT"

resolve_corpus() {
  local corpus="$1" seed="$2"
  case "$corpus" in
    MELD)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/meld_anjs4_unified.pkl
      JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_pkls/meld_bobsl5000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl
      BASE_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld/meld_only_seed${seed}/video/model_best.pt
      SOURCE_PROFILE="$SOURCE_PROFILE_MELD"
      ;;
    IEMOCAP)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/features/iemocap_mmgcn_video_anjs4.pkl
      JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl
      BASE_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/iemocap_only_seed${seed}/video/model_best.pt
      SOURCE_PROFILE="$SOURCE_PROFILE_IEMOCAP"
      ;;
    *) echo "[MMGCN-RT] unknown corpus=$corpus" >&2; return 2 ;;
  esac
}

resolve_profile() {
  case "$1" in
    s010) PROFILE_ARGS=(--source_scale 0.10 --adapter_hidden_dim 128 --base_distill_weight 2.0 --source_residual_penalty 0.01 --lr 0.00001) ;;
    s025) PROFILE_ARGS=(--source_scale 0.25 --adapter_hidden_dim 128 --base_distill_weight 2.0 --source_residual_penalty 0.01 --lr 0.00003) ;;
    s050) PROFILE_ARGS=(--source_scale 0.50 --adapter_hidden_dim 128 --base_distill_weight 5.0 --source_residual_penalty 0.02 --lr 0.00003) ;;
    r25cw) PROFILE_ARGS=(--transfer_scale 0.25 --gate_bias -3.0 --classwise_gate --stability_weight 0.2 --gate_penalty 0.01) ;;
    r50cw) PROFILE_ARGS=(--transfer_scale 0.50 --gate_bias -3.0 --classwise_gate --stability_weight 0.2 --gate_penalty 0.01) ;;
    r100cw) PROFILE_ARGS=(--transfer_scale 1.0 --gate_bias -3.0 --classwise_gate --stability_weight 0.2 --gate_penalty 0.01) ;;
    r50open) PROFILE_ARGS=(--transfer_scale 0.50 --gate_bias -2.0 --classwise_gate --stability_weight 0.1 --gate_penalty 0.005) ;;
    r50cross) PROFILE_ARGS=(--transfer_scale 0.50 --gate_bias -3.0 --classwise_gate --cross_source_weight 0.1 --stability_weight 0.2 --gate_penalty 0.01) ;;
    r200open) PROFILE_ARGS=(--transfer_scale 2.0 --gate_bias -1.5 --classwise_gate --stability_weight 0.05 --gate_penalty 0.001) ;;
    r400open) PROFILE_ARGS=(--transfer_scale 4.0 --gate_bias -1.5 --classwise_gate --stability_weight 0.05 --gate_penalty 0.001) ;;
    r200cross) PROFILE_ARGS=(--transfer_scale 2.0 --gate_bias -1.5 --classwise_gate --cross_source_weight 0.1 --stability_weight 0.05 --gate_penalty 0.001) ;;
    r200scalar) PROFILE_ARGS=(--transfer_scale 2.0 --gate_bias -1.5 --no-classwise_gate --stability_weight 0.05 --gate_penalty 0.001) ;;
    *) echo "[MMGCN-RT] unknown profile=$1" >&2; return 2 ;;
  esac
}

ensure_cache() {
  local corpus="$1" seed="$2"
  local cache_dir="$CACHE_ROOT/$corpus/seed${seed}"
  if [[ -s "$cache_dir/joint_train.npz" && -s "$cache_dir/source_train.npz" && -s "$cache_dir/source_test.npz" && -s "$cache_dir/external_test.npz" ]]; then
    CACHE_DIR="$cache_dir"
    return
  fi
  mkdir -p "$cache_dir"
  "$PYTHON_BIN" MMGCN/build_residual_transfer_cache.py \
    --source_pkl "$SOURCE_PKL" --joint_pkl "$JOINT_PKL" --external_pkl "$EXTERNAL_PKL" \
    --checkpoint "$BASE_CHECKPOINT" --out_dir "$cache_dir" --batch_size 8
  CACHE_DIR="$cache_dir"
}

run_one() {
  local corpus="$1" condition="$2" profile="$3" seed="$4"
  resolve_corpus "$corpus" "$seed"
  ensure_cache "$corpus" "$seed"
  resolve_profile "$profile"
  local train_cache="$CACHE_DIR/source_train.npz"
  local init_args=()
  if [[ "$condition" == joint_b5k ]]; then
    train_cache="$CACHE_DIR/joint_train.npz"
    local source_init="$SOURCE_ROOT/$corpus/source_only/${SOURCE_PROFILE}_seed${seed}/model_best.pt"
    if [[ ! -s "$source_init" ]]; then echo "[MMGCN-RT] missing source init=$source_init" >&2; return 1; fi
    init_args=(--init_source_checkpoint "$source_init")
  fi
  local out_dir="$OUT_ROOT/$corpus/$condition/${profile}_seed${seed}"
  if [[ -s "$out_dir/external_test_best_summary.json" ]]; then echo "[MMGCN-RT] skip $out_dir"; return; fi
  mkdir -p "$out_dir"
  "$PYTHON_BIN" new/train_cached_residual_transfer.py \
    --train_cache "$train_cache" --source_test_cache "$CACHE_DIR/source_test.npz" \
    --external_test_cache "$CACHE_DIR/external_test.npz" --out_dir "$out_dir" \
    --seed "$seed" --epochs "$EPOCHS" "${PROFILE_ARGS[@]}" "${init_args[@]}" \
    2>&1 | tee "$OUT_ROOT/logs/${corpus}_${condition}_${profile}_seed${seed}.log"
}

for corpus in $CORPORA; do
  for profile in $PROFILES; do
    for seed in $SEEDS; do
      if [[ "$PHASE" == source ]]; then
        run_one "$corpus" source_only "$profile" "$seed"
      elif [[ "$PHASE" == joint ]]; then
        run_one "$corpus" joint_b5k "$profile" "$seed"
      else
        echo "[MMGCN-RT] unsupported PHASE=$PHASE" >&2
        exit 2
      fi
    done
  done
done

"$PYTHON_BIN" analysis/summarize_residual_transfer.py --root "$OUT_ROOT"
