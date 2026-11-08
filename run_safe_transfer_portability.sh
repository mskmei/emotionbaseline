#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE=MAGTKD, ECERC, or ConxGNN}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-python}
SEEDS=${SEEDS:-"35 36 37 41 42"}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
PROFILES=${PROFILES:-"global reliability routing routing_visual"}
EPOCHS=${EPOCHS:-30}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_v2}
CACHE_ROOT=${CACHE_ROOT:-$WORK_ROOT/portable_cache/$BASELINE}
OUT_ROOT=${OUT_ROOT:-$WORK_ROOT/portable_runs/$BASELINE}
mkdir -p "$CACHE_ROOT" "$OUT_ROOT"

resolve_paths() {
  local corpus="$1" seed="$2"
  case "$BASELINE/$corpus" in
    MAGTKD/MELD)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/meld_magtkd_video_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl
      MODEL_ROOT=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld
      SOURCE_CHECKPOINT=$MODEL_ROOT/meld_only_seed${seed}/model_best.pt
      JOINT_CHECKPOINT=$MODEL_ROOT/joint_b5k_seed${seed}/model_best.pt
      ;;
    MAGTKD/IEMOCAP)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/features/iemocap_magtkd_video_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl
      MODEL_ROOT=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      SOURCE_CHECKPOINT=$MODEL_ROOT/iemocap_only_seed${seed}/model_best.pt
      JOINT_CHECKPOINT=$MODEL_ROOT/joint_b5k_seed${seed}/model_best.pt
      ;;
    ECERC/MELD)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/meld_ecerc_video_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/ejsl_ecerc_video_anjs4_translated.pkl
      MODEL_ROOT=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld
      SOURCE_CHECKPOINT=$MODEL_ROOT/meld_only_seed${seed}/model_best.pt
      JOINT_CHECKPOINT=$MODEL_ROOT/joint_b5k_seed${seed}/model_best.pt
      ;;
    ECERC/IEMOCAP)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/features/iemocap_ecerc_video_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/ejsl_ecerc_video_anjs4_translated.pkl
      MODEL_ROOT=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      SOURCE_CHECKPOINT=$MODEL_ROOT/iemocap_only_seed${seed}/model_best.pt
      JOINT_CHECKPOINT=$MODEL_ROOT/joint_b5k_seed${seed}/model_best.pt
      ;;
    ConxGNN/MELD)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/meld_conxgnn_visual_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/ejsl_conxgnn_visual_anjs4_translated.pkl
      MODEL_ROOT=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld
      SOURCE_CHECKPOINT=$MODEL_ROOT/meld_only_seed${seed}/model_best.pt
      JOINT_CHECKPOINT=$MODEL_ROOT/joint_b5k_seed${seed}/model_best.pt
      ;;
    ConxGNN/IEMOCAP)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/features/iemocap_conxgnn_visual_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/ejsl_conxgnn_visual_anjs4_translated.pkl
      MODEL_ROOT=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      SOURCE_CHECKPOINT=$MODEL_ROOT/iemocap_only_seed${seed}/model_best.pt
      JOINT_CHECKPOINT=$MODEL_ROOT/joint_b5k_seed${seed}/model_best.pt
      ;;
    *) echo "[portable-transfer] unsupported $BASELINE/$corpus" >&2; exit 2 ;;
  esac
}

for corpus in $CORPORA; do
  for seed in $SEEDS; do
    resolve_paths "$corpus" "$seed"
    cache_dir="$CACHE_ROOT/$corpus/seed${seed}"
    if [[ -s "$cache_dir/source_train.npz" && -s "$cache_dir/source_test.npz" && -s "$cache_dir/external_test.npz" ]]; then
      echo "[portable-transfer] reuse $cache_dir"
      continue
    fi
    "$PYTHON_BIN" new/build_safe_transfer_pair_cache.py \
      --baseline "$BASELINE" --source_pkl "$SOURCE_PKL" --external_pkl "$EXTERNAL_PKL" \
      --source_checkpoint "$SOURCE_CHECKPOINT" --joint_checkpoint "$JOINT_CHECKPOINT" \
      --out_dir "$cache_dir" --batch_size 8
  done
done

BASELINE="$BASELINE" GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" \
CORPORA="$CORPORA" PROFILES="$PROFILES" SEEDS="$SEEDS" EPOCHS="$EPOCHS" VALID_RATIO=0 \
CACHE_ROOT="$CACHE_ROOT" OUT_ROOT="$OUT_ROOT" \
bash run_safe_transfer_strict.sh
