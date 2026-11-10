#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

CORPORA=${CORPORA:-"MELD IEMOCAP"}
BOBSL_K=${BOBSL_K:-5}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
BASE_MODE=${BASE_MODE:-source_only}
SEEDS=${SEEDS:-42}
PROFILES=${PROFILES:-"context_r3 context_r1 proto10_r3 safe_r3 source2_r3 low_lr very_low_lr rank1_low rank1_safe b25_low source4_r3 scale05_r3 scale1_r3 no_weight"}
EPOCHS=${EPOCHS:-12}
RESUME_DONE=${RESUME_DONE:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ecerc_sast/screen}

set_paths() {
  case "$1" in
    MELD)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_pkls/joint_b${BOBSL_K}k.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/ejsl_ecerc_video_anjs4_translated.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld
      BASE_PREFIX=meld_only
      ;;
    IEMOCAP)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl${BOBSL_K}000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/ejsl_ecerc_video_anjs4_translated.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      BASE_PREFIX=iemocap_only
      ;;
    *) echo "Unsupported corpus=$1" >&2; return 2 ;;
  esac
}

checkpoint_for_seed() {
  if [[ "$BASE_MODE" == plain_joint ]]; then
    printf '%s/joint_b%sk_seed%s/model_best.pt\n' "$BASE_ROOT" "$BOBSL_K" "$1"
  elif [[ "$BASE_MODE" == source_only ]]; then
    printf '%s/%s_seed%s/model_best.pt\n' "$BASE_ROOT" "$BASE_PREFIX" "$1"
  else
    echo "Unsupported BASE_MODE=$BASE_MODE" >&2
    return 2
  fi
}

profile_args() {
  case "$1" in
    context_r3) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.0) ;;
    context_r1) PROFILE_ARGS=(--subspace_rank 1 --prototype_weight 0.0) ;;
    proto10_r3) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.1) ;;
    safe_r3) PROFILE_ARGS=(--subspace_rank 3 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --prototype_weight 0.1) ;;
    source2_r3) PROFILE_ARGS=(--subspace_rank 3 --source_weight 2.0 --prototype_weight 0.1) ;;
    low_lr) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --prototype_weight 0.1) ;;
    very_low_lr) PROFILE_ARGS=(--subspace_rank 3 --lr 0.00005 --prototype_weight 0.1) ;;
    rank1_low) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0001 --prototype_weight 0.0) ;;
    rank1_safe) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --prototype_weight 0.0) ;;
    b25_low) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --bobsl_weight 0.25 --safety_weight 1.0 --stability_weight 0.5 --prototype_weight 0.1) ;;
    source4_r3) PROFILE_ARGS=(--subspace_rank 3 --source_weight 4.0 --bobsl_weight 0.5 --safety_weight 2.0 --stability_weight 0.5 --prototype_weight 0.1) ;;
    scale05_r3) PROFILE_ARGS=(--subspace_rank 3 --residual_scale 0.5 --lr 0.0002 --prototype_weight 0.1) ;;
    scale1_r3) PROFILE_ARGS=(--subspace_rank 3 --residual_scale 1.0 --lr 0.0002 --prototype_weight 0.1) ;;
    no_weight) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.1 --no-class_weight) ;;
    *) echo "Unknown profile=$1" >&2; return 2 ;;
  esac
}

mkdir -p "$WORK_ROOT/logs"
for corpus in $CORPORA; do
  set_paths "$corpus"
  for profile in $PROFILES; do
    profile_args "$profile"
    for seed in $SEEDS; do
      out_dir="$WORK_ROOT/${corpus,,}/b${BOBSL_K}k/${profile}_seed${seed}"
      log_path="$WORK_ROOT/logs/${corpus,,}_b${BOBSL_K}k_${profile}_seed${seed}.log"
      if [[ "$RESUME_DONE" == 1 && -s "$out_dir/summary.json" ]]; then
        echo "[SAST-ECERC] skip $out_dir"
        continue
      fi
      checkpoint=$(checkpoint_for_seed "$seed")
      echo "[SAST-ECERC] corpus=$corpus bobsl=${BOBSL_K}k base_mode=$BASE_MODE profile=$profile seed=$seed gpu=$GPU" | tee "$log_path"
      "$PYTHON_BIN" new/ECERC/MELD/train_eval_ecerc_sast_video_anjs.py \
        --train_pkl "$TRAIN_PKL" \
        --external_test_pkl "$EXTERNAL_PKL" \
        --base_checkpoint "$checkpoint" \
        --out_dir "$out_dir" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --source_batch_size 8 \
        --bobsl_batch_size 64 \
        --eval_batch_size 8 \
        --adapter_dim 128 \
        --adapter_dropout 0.2 \
        --lr 0.0005 \
        --residual_scale 2.0 \
        --source_weight 1.0 \
        --bobsl_weight 1.0 \
        --safety_weight 0.5 \
        --stability_weight 0.1 \
        --prototype_weight 0.1 \
        --residual_weight 0.001 \
        --max_grad_norm 5.0 \
        "${PROFILE_ARGS[@]}" 2>&1 | tee -a "$log_path"
    done
  done
done

"$PYTHON_BIN" analysis/summarize_sast_negative_transfer.py \
  --root "$WORK_ROOT" \
  --out_csv "$WORK_ROOT/screen_summary.csv"
