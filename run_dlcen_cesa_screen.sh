#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
PROFILES=${PROFILES:-"small balanced open strong_safety"}
SEED=${SEED:-42}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/dlcen_cesa/screen_gated}

corpus_paths() {
  case "$1" in
    MELD)
      EPOCHS=${EPOCHS:-15}
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld/joint_pkls/joint_b5k.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      BASE_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/decoupled_context_model/screen/meld/source_only_seed42/model_best.pt
      ;;
    IEMOCAP)
      EPOCHS=${EPOCHS:-30}
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      BASE_CHECKPOINT=/raid_zoe/home/lr/maokeyu/sign/decoupled_context_model/screen/iemocap/source_only_seed42/model_best.pt
      ;;
    *) echo "Unknown corpus=$1" >&2; return 2 ;;
  esac
}

profile_args() {
  case "$1" in
    small) PROFILE_ARGS=(--lr 0.0001 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.01 --gate_bias -4.0) ;;
    balanced) PROFILE_ARGS=(--lr 0.0003 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.001 --gate_bias -3.0) ;;
    open) PROFILE_ARGS=(--lr 0.0005 --transfer_scale 2.0 --safety_weight 0.5 --residual_weight 0.001 --gate_bias -3.0) ;;
    strong_safety) PROFILE_ARGS=(--lr 0.0005 --transfer_scale 2.0 --safety_weight 2.0 --residual_weight 0.001 --gate_bias -4.0) ;;
    ungated_b10) PROFILE_ARGS=(--lr 0.0003 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.001 --bobsl_weight 0.10 --disable_transfer_gate) ;;
    ungated_b25) PROFILE_ARGS=(--lr 0.0003 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.001 --bobsl_weight 0.25 --disable_transfer_gate) ;;
    gated_b25) PROFILE_ARGS=(--lr 0.0003 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.001 --bobsl_weight 0.25 --gate_bias -4.0) ;;
    proto10) PROFILE_ARGS=(--lr 0.0003 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.001 --bobsl_weight 0.25 --prototype_weight 0.10 --disable_transfer_gate) ;;
    proto50) PROFILE_ARGS=(--lr 0.0003 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.001 --bobsl_weight 0.25 --prototype_weight 0.50 --disable_transfer_gate) ;;
    proto100) PROFILE_ARGS=(--lr 0.0003 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.001 --bobsl_weight 0.25 --prototype_weight 1.0 --disable_transfer_gate) ;;
    proto50_gated) PROFILE_ARGS=(--lr 0.0003 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.001 --bobsl_weight 0.25 --prototype_weight 0.50 --gate_bias -4.0) ;;
    *) echo "Unknown profile=$1" >&2; return 2 ;;
  esac
}

mkdir -p "$WORK_ROOT/logs"
for corpus in $CORPORA; do
  unset EPOCHS
  corpus_paths "$corpus"
  for profile in $PROFILES; do
    profile_args "$profile"
    out_dir="$WORK_ROOT/${corpus,,}/${profile}_seed${SEED}"
    log_path="$WORK_ROOT/logs/${corpus,,}_${profile}_seed${SEED}.log"
    if [[ -s "$out_dir/external_test_best_summary.json" ]]; then
      echo "[DLCEN-CESA-SCREEN] skip $out_dir"
      continue
    fi
    echo "[DLCEN-CESA-SCREEN] corpus=$corpus profile=$profile gpu=$GPU" | tee "$log_path"
    "$PYTHON_BIN" new/train_eval_dlcen_cesa_anjs.py \
      --train_pkl "$TRAIN_PKL" \
      --external_test_pkl "$EXTERNAL_PKL" \
      --base_checkpoint "$BASE_CHECKPOINT" \
      --out_dir "$out_dir" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --source_batch_size 16 \
      --bobsl_batch_size 64 \
      --eval_batch_size 16 \
      --bobsl_weight 1.0 \
      --local_source_weight 0.1 \
      --max_grad_norm 5.0 \
      "${PROFILE_ARGS[@]}" 2>&1 | tee -a "$log_path"
  done
done
