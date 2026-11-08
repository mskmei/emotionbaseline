#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"
BASELINE=${BASELINE:?Set BASELINE=MMGCN or CMERC}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
PROFILES=${PROFILES:-"global reliability reliability15 visual visual15"}
SEEDS=${SEEDS:-35}
EPOCHS=${EPOCHS:-30}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate}
CACHE_ROOT=${CACHE_ROOT:-$WORK_ROOT/cache/$BASELINE}
OUT_ROOT=${OUT_ROOT:-$WORK_ROOT/runs/$BASELINE}
mkdir -p "$CACHE_ROOT" "$OUT_ROOT/logs"

resolve_paths() {
  local corpus="$1" seed="$2"
  case "$BASELINE/$corpus" in
    MMGCN/MELD)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/meld_anjs4_unified.pkl
      JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_pkls/meld_bobsl5000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl
      SOURCE_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld/meld_only_seed${seed}/video/model_best.pt
      JOINT_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_b5k_seed${seed}/video/model_best.pt
      ;;
    MMGCN/IEMOCAP)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/features/iemocap_mmgcn_video_anjs4.pkl
      JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl
      SOURCE_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/iemocap_only_seed${seed}/video/model_best.pt
      JOINT_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_b5k_seed${seed}/video/model_best.pt
      ;;
    CMERC/MELD)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/meld_cmerc_visual_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      SOURCE_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld/meld_only_seed${seed}/model_best.pt
      JOINT_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld/joint_b5k_seed${seed}/model_best.pt
      ;;
    CMERC/IEMOCAP)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/features/iemocap_cmerc_visual_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      SOURCE_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/iemocap_only_seed${seed}/model_best.pt
      JOINT_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_b5k_seed${seed}/model_best.pt
      ;;
    *) echo "[safe-gate] unsupported $BASELINE/$corpus" >&2; return 2 ;;
  esac
}

resolve_profile() {
  case "$1" in
    global) PROFILE_ARGS=(--mode global --max_scale 1.0 --gate_bias -1.0 --lr 0.01 --distill_weight 0.1 --gate_penalty 0.001) ;;
    reliability) PROFILE_ARGS=(--mode reliability --max_scale 1.0 --gate_bias -1.5 --lr 0.001 --distill_weight 0.2 --gate_penalty 0.005) ;;
    reliability15) PROFILE_ARGS=(--mode reliability --max_scale 1.5 --gate_bias -1.5 --lr 0.001 --distill_weight 0.2 --gate_penalty 0.005) ;;
    visual) PROFILE_ARGS=(--mode visual --max_scale 1.0 --gate_bias -1.5 --lr 0.0003 --distill_weight 0.2 --gate_penalty 0.005) ;;
    visual15) PROFILE_ARGS=(--mode visual --max_scale 1.5 --gate_bias -1.5 --lr 0.0003 --distill_weight 0.2 --gate_penalty 0.005) ;;
    reliability_free) PROFILE_ARGS=(--mode reliability --max_scale 1.0 --gate_bias -1.0 --lr 0.001 --distill_weight 0.0 --gate_penalty 0.0) ;;
    *) echo "[safe-gate] unknown profile=$1" >&2; return 2 ;;
  esac
}

ensure_cache() {
  local corpus="$1" seed="$2" cache_dir="$CACHE_ROOT/$corpus/seed${seed}"
  if [[ -s "$cache_dir/source_train.npz" && -s "$cache_dir/source_test.npz" && -s "$cache_dir/external_test.npz" ]]; then
    CACHE_DIR="$cache_dir"
    return
  fi
  mkdir -p "$cache_dir"
  if [[ "$BASELINE" == MMGCN ]]; then
    "$PYTHON_BIN" MMGCN/build_residual_transfer_cache.py \
      --source_pkl "$SOURCE_PKL" --joint_pkl "$JOINT_PKL" --external_pkl "$EXTERNAL_PKL" \
      --checkpoint "$SOURCE_CHECKPOINT" --joint_checkpoint "$JOINT_CHECKPOINT" --out_dir "$cache_dir"
  else
    "$PYTHON_BIN" new2/CMERC/build_safe_transfer_pair_cache.py \
      --source_pkl "$SOURCE_PKL" --external_pkl "$EXTERNAL_PKL" \
      --source_checkpoint "$SOURCE_CHECKPOINT" --joint_checkpoint "$JOINT_CHECKPOINT" --out_dir "$cache_dir"
  fi
  CACHE_DIR="$cache_dir"
}

for corpus in $CORPORA; do
  for seed in $SEEDS; do
    resolve_paths "$corpus" "$seed"
    ensure_cache "$corpus" "$seed"
    for profile in $PROFILES; do
      resolve_profile "$profile"
      out_dir="$OUT_ROOT/$corpus/${profile}_seed${seed}"
      if [[ -s "$out_dir/external_test_best_summary.json" ]]; then echo "[safe-gate] skip $out_dir"; continue; fi
      mkdir -p "$out_dir"
      "$PYTHON_BIN" new/train_safe_transfer_gate.py \
        --train_cache "$CACHE_DIR/source_train.npz" --source_test_cache "$CACHE_DIR/source_test.npz" \
        --external_test_cache "$CACHE_DIR/external_test.npz" --out_dir "$out_dir" \
        --seed "$seed" --epochs "$EPOCHS" "${PROFILE_ARGS[@]}" \
        2>&1 | tee "$OUT_ROOT/logs/${corpus}_${profile}_seed${seed}.log"
    done
  done
done

"$PYTHON_BIN" analysis/summarize_safe_transfer_gate.py --root "$OUT_ROOT" --baseline "$BASELINE"
