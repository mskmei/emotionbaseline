#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
BOBSL_K=${BOBSL_K:-5}
PROFILES=${PROFILES:-"rank1 rank3 full rank1_safe rank1_b10 rank1_b25 rank1_src4 rank1_unweighted rank3_b25 rank3_safe rank3_open very_low_lr high_lr scale05 scale4"}
SEED=${SEED:-42}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/conxgnn_sast/screen}
EPOCHS_OVERRIDE=${EPOCHS:-}

set_paths() {
  case "$1" in
    MELD)
      EPOCHS=${EPOCHS_OVERRIDE:-10}
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld/joint_pkls/joint_b${BOBSL_K}k.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/ejsl_conxgnn_visual_anjs4_translated.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld
      BASE_PREFIX=meld_only
      ;;
    IEMOCAP)
      EPOCHS=${EPOCHS_OVERRIDE:-20}
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl${BOBSL_K}000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/ejsl_conxgnn_visual_anjs4_translated.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      BASE_PREFIX=iemocap_only
      ;;
    *) echo "Unknown CORPUS=$1" >&2; return 2 ;;
  esac
}

profile_args() {
  case "$1" in
    rank1) PROFILE_ARGS=(--subspace_rank 1) ;;
    rank3) PROFILE_ARGS=(--subspace_rank 3) ;;
    full) PROFILE_ARGS=(--subspace_rank 3 --projection full) ;;
    rank1_safe) PROFILE_ARGS=(--subspace_rank 1 --safety_weight 2.0 --residual_weight 0.005) ;;
    rank1_b10) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0001 --residual_scale 1.0 --bobsl_weight 0.10 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005) ;;
    rank1_b25) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.25 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005) ;;
    rank1_src4) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.25 --source_weight 4.0 --safety_weight 4.0 --residual_weight 0.01) ;;
    rank1_unweighted) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.25 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005 --unweighted_adapter) ;;
    rank3_b25) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.25 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005) ;;
    rank3_safe) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.5 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.01) ;;
    rank3_open) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0005 --residual_scale 4.0 --bobsl_weight 1.0 --source_weight 1.0 --safety_weight 0.25 --residual_weight 0.0001) ;;
    very_low_lr) PROFILE_ARGS=(--subspace_rank 1 --lr 0.00005 --residual_scale 1.0 --bobsl_weight 0.25 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005) ;;
    high_lr) PROFILE_ARGS=(--subspace_rank 1 --lr 0.001 --residual_scale 1.0 --bobsl_weight 0.25 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005) ;;
    scale05) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 0.5 --bobsl_weight 0.25 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005) ;;
    scale4) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 4.0 --bobsl_weight 0.25 --source_weight 2.0 --safety_weight 2.0 --residual_weight 0.005) ;;
    *) echo "Unknown profile=$1" >&2; return 2 ;;
  esac
}

mkdir -p "$WORK_ROOT/logs"
for corpus in $CORPORA; do
  set_paths "$corpus"
  for profile in $PROFILES; do
    profile_args "$profile"
    out_dir="$WORK_ROOT/${corpus,,}/b${BOBSL_K}k/${profile}_seed${SEED}"
    log_path="$WORK_ROOT/logs/${corpus,,}_b${BOBSL_K}k_${profile}_seed${SEED}.log"
    if [[ -s "$out_dir/summary.json" ]]; then
      echo "[SAST-SCREEN] skip $out_dir"
      continue
    fi
    echo "[SAST-SCREEN] corpus=$corpus profile=$profile seed=$SEED gpu=$GPU" | tee "$log_path"
    "$PYTHON_BIN" new2/ConxGNN/train_eval_conxgnn_sast_visual_anjs.py \
      --train_pkl "$TRAIN_PKL" \
      --external_test_pkl "$EXTERNAL_PKL" \
      --base_checkpoint "$BASE_ROOT/${BASE_PREFIX}_seed${SEED}/model_best.pt" \
      --out_dir "$out_dir" \
      --seed "$SEED" \
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
done
