#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

CORPUS=${CORPUS:?Set CORPUS=MELD or IEMOCAP}
PROFILE=${PROFILE:-rank1}
SEEDS=${SEEDS:-"35 36 37 41 42"}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/conxgnn_sast/fiveseed}

case "$CORPUS" in
  MELD)
    EPOCHS=${EPOCHS:-10}
    TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld/joint_pkls/joint_b5k.pkl
    EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/ejsl_conxgnn_visual_anjs4_translated.pkl
    BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld
    BASE_PREFIX=meld_only
    ;;
  IEMOCAP)
    EPOCHS=${EPOCHS:-20}
    TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl
    EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/ejsl_conxgnn_visual_anjs4_translated.pkl
    BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
    BASE_PREFIX=iemocap_only
    ;;
  *) echo "Unknown CORPUS=$CORPUS" >&2; exit 2 ;;
esac

case "$PROFILE" in
  rank1) PROFILE_ARGS=(--subspace_rank 1) ;;
  rank3) PROFILE_ARGS=(--subspace_rank 3) ;;
  full) PROFILE_ARGS=(--subspace_rank 3 --projection full) ;;
  rank1_safe) PROFILE_ARGS=(--subspace_rank 1 --safety_weight 2.0 --residual_weight 0.005) ;;
  rank1_b10) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0001 --residual_scale 1.0 --bobsl_weight 0.10 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005) ;;
  rank1_b25) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.25 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005) ;;
  rank1_src4) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.25 --source_weight 4.0 --safety_weight 4.0 --residual_weight 0.01) ;;
  *) echo "Unknown PROFILE=$PROFILE" >&2; exit 2 ;;
esac

mkdir -p "$WORK_ROOT/logs"
for seed in $SEEDS; do
  out_dir="$WORK_ROOT/${CORPUS,,}/${PROFILE}_seed${seed}"
  log_path="$WORK_ROOT/logs/${CORPUS,,}_${PROFILE}_seed${seed}.log"
  if [[ -s "$out_dir/summary.json" ]]; then
    echo "[SAST-5] skip $out_dir"
    continue
  fi
  echo "[SAST-5] corpus=$CORPUS profile=$PROFILE seed=$seed gpu=$GPU" | tee "$log_path"
  "$PYTHON_BIN" new2/ConxGNN/train_eval_conxgnn_sast_visual_anjs.py \
    --train_pkl "$TRAIN_PKL" \
    --external_test_pkl "$EXTERNAL_PKL" \
    --base_checkpoint "$BASE_ROOT/${BASE_PREFIX}_seed${seed}/model_best.pt" \
    --out_dir "$out_dir" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --source_batch_size 8 \
    --bobsl_batch_size 64 \
    --eval_batch_size 8 \
    --adapter_dim 128 \
    --adapter_dropout 0.2 \
    --residual_scale 2.0 \
    --lr 0.0005 \
    --source_weight 1.0 \
    --bobsl_weight 1.0 \
    --safety_weight 0.5 \
    --residual_weight 0.001 \
    --class_weight \
    "${PROFILE_ARGS[@]}" 2>&1 | tee -a "$log_path"
done
