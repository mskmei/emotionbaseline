#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE=MMGCN or MAGTKD}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
SEEDS=${SEEDS:-42}
PROFILES=${PROFILES:-"context_r3 context_r1 proto10_r3 proto50_r3 safe_r3 open_r3 source2_r3 full_safe low_lr no_weight"}
EPOCHS=${EPOCHS:-12}
RESUME_DONE=${RESUME_DONE:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/sast_negative_transfer}

case "$BASELINE" in
  MMGCN)
    TRAINER=MMGCN/train_eval_mmgcn_sast_video_anjs.py
    TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_pkls/meld_bobsl5000_trainonly_sample123.pkl
    EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl
    BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld
    checkpoint_for_seed() { printf '%s/meld_only_seed%s/video/model_best.pt\n' "$BASE_ROOT" "$1"; }
    ;;
  MAGTKD)
    TRAINER=new/MAGTKD/MELD/train_eval_magtkd_sast_video_anjs.py
    TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_pkls/joint_b5k.pkl
    EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl
    BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld
    checkpoint_for_seed() { printf '%s/meld_only_seed%s/model_best.pt\n' "$BASE_ROOT" "$1"; }
    ;;
  *) echo "Unsupported BASELINE=$BASELINE" >&2; exit 2 ;;
esac

profile_args() {
  case "$1" in
    context_r3) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.0) ;;
    context_r1) PROFILE_ARGS=(--subspace_rank 1 --prototype_weight 0.0) ;;
    proto10_r3) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.1) ;;
    proto50_r3) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.5) ;;
    safe_r3) PROFILE_ARGS=(--subspace_rank 3 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --prototype_weight 0.1) ;;
    open_r3) PROFILE_ARGS=(--subspace_rank 3 --residual_scale 4.0 --safety_weight 0.25 --stability_weight 0.05 --prototype_weight 0.1 --residual_weight 0.0001) ;;
    source2_r3) PROFILE_ARGS=(--subspace_rank 3 --source_weight 2.0 --prototype_weight 0.1) ;;
    full_safe) PROFILE_ARGS=(--projection full --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --prototype_weight 0.1) ;;
    low_lr) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --prototype_weight 0.1) ;;
    no_weight) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.1 --no-class_weight) ;;
    *) echo "Unknown profile=$1" >&2; return 2 ;;
  esac
}

mkdir -p "$WORK_ROOT/$BASELINE/logs"
for profile in $PROFILES; do
  profile_args "$profile"
  for seed in $SEEDS; do
    out_dir="$WORK_ROOT/$BASELINE/${profile}_seed${seed}"
    log_path="$WORK_ROOT/$BASELINE/logs/${profile}_seed${seed}.log"
    if [[ "$RESUME_DONE" == 1 && -s "$out_dir/summary.json" ]]; then
      echo "[SAST-SCREEN] skip $out_dir"
      continue
    fi
    checkpoint=$(checkpoint_for_seed "$seed")
    echo "[SAST-SCREEN] baseline=$BASELINE profile=$profile seed=$seed gpu=$GPU" | tee "$log_path"
    "$PYTHON_BIN" "$TRAINER" \
      --train_pkl "$TRAIN_PKL" \
      --external_test_pkl "$EXTERNAL_PKL" \
      --base_checkpoint "$checkpoint" \
      --out_dir "$out_dir" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
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

"$PYTHON_BIN" analysis/summarize_sast_negative_transfer.py \
  --root "$WORK_ROOT" \
  --out_csv "$WORK_ROOT/screen_summary.csv"
