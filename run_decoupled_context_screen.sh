#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
PROFILES=${PROFILES:-"source_only joint_tiny joint_010 joint_015 joint_low joint_mid joint_high joint_wide"}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-50}
RESUME_DONE=${RESUME_DONE:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/decoupled_context_model/screen}

corpus_paths() {
  case "$1" in
    MELD)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/meld_cmerc_visual_anjs4.pkl
      JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld/joint_pkls/joint_b5k.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      ;;
    IEMOCAP)
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/features/iemocap_cmerc_visual_anjs4.pkl
      JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      ;;
    *) echo "Unknown corpus: $1" >&2; return 2 ;;
  esac
}

profile_args() {
  case "$1" in
    source_only)
      TRAIN_PKL=$SOURCE_PKL
      PROFILE_ARGS=(--bobsl_weight 0.0)
      ;;
    joint_low)
      TRAIN_PKL=$JOINT_PKL
      PROFILE_ARGS=(--bobsl_weight 0.25)
      ;;
    joint_tiny)
      TRAIN_PKL=$JOINT_PKL
      PROFILE_ARGS=(--bobsl_weight 0.05)
      ;;
    joint_010)
      TRAIN_PKL=$JOINT_PKL
      PROFILE_ARGS=(--bobsl_weight 0.1)
      ;;
    joint_015)
      TRAIN_PKL=$JOINT_PKL
      PROFILE_ARGS=(--bobsl_weight 0.15)
      ;;
    joint_mid)
      TRAIN_PKL=$JOINT_PKL
      PROFILE_ARGS=(--bobsl_weight 0.5)
      ;;
    joint_high)
      TRAIN_PKL=$JOINT_PKL
      PROFILE_ARGS=(--bobsl_weight 1.0)
      ;;
    joint_wide)
      TRAIN_PKL=$JOINT_PKL
      PROFILE_ARGS=(--bobsl_weight 0.5 --hidden_dim 384 --adapter_dim 192 --n_heads 6 --dropout 0.35)
      ;;
    *) echo "Unknown profile: $1" >&2; return 2 ;;
  esac
}

mkdir -p "$WORK_ROOT/logs"
for corpus in $CORPORA; do
  corpus_paths "$corpus"
  for profile in $PROFILES; do
    profile_args "$profile"
    out_dir="$WORK_ROOT/${corpus,,}/${profile}_seed${SEED}"
    log_path="$WORK_ROOT/logs/${corpus,,}_${profile}_seed${SEED}.log"
    if [[ "$RESUME_DONE" == 1 && -s "$out_dir/external_test_best_summary.json" ]]; then
      echo "[DLCEN-SCREEN] skip $out_dir"
      continue
    fi
    mkdir -p "$out_dir"
    echo "[DLCEN-SCREEN] corpus=$corpus profile=$profile seed=$SEED gpu=$GPU" | tee "$log_path"
    "$PYTHON_BIN" new/train_eval_decoupled_context_anjs.py \
      --train_pkl "$TRAIN_PKL" \
      --external_test_pkl "$EXTERNAL_PKL" \
      --out_dir "$out_dir" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --batch_size 16 \
      --bobsl_batch_size 64 \
      --eval_batch_size 16 \
      --lr 0.0003 \
      --weight_decay 0.0001 \
      --hidden_dim 256 \
      --adapter_dim 128 \
      --dropout 0.3 \
      --adapter_dropout 0.2 \
      --local_loss_weight 0.25 \
      --context_loss_weight 0.25 \
      --shift_loss_weight 0.1 \
      --calibration_weight 0.05 \
      --residual_weight 0.01 \
      --max_grad_norm 5.0 \
      "${PROFILE_ARGS[@]}" 2>&1 | tee -a "$log_path"
  done
done

"$PYTHON_BIN" analysis/summarize_decoupled_context.py --root "$WORK_ROOT"
