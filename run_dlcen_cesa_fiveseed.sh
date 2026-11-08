#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

CORPUS=${CORPUS:?Set CORPUS=MELD or IEMOCAP}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
SEEDS=${SEEDS:-"35 36 37 41 42"}
PROFILE=${PROFILE:-open}
RESUME_DONE=${RESUME_DONE:-1}
RUN_TRANSFER=${RUN_TRANSFER:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/dlcen_cesa/fiveseed}

case "$CORPUS" in
  MELD)
    SOURCE_EPOCHS=${SOURCE_EPOCHS:-60}
    TRANSFER_EPOCHS=${TRANSFER_EPOCHS:-15}
    SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/meld_cmerc_visual_anjs4.pkl
    JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld/joint_pkls/joint_b5k.pkl
    EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
    ;;
  IEMOCAP)
    SOURCE_EPOCHS=${SOURCE_EPOCHS:-60}
    TRANSFER_EPOCHS=${TRANSFER_EPOCHS:-30}
    SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/features/iemocap_cmerc_visual_anjs4.pkl
    JOINT_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl
    EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
    ;;
  *) echo "Unknown CORPUS=$CORPUS" >&2; exit 2 ;;
esac

profile_args() {
  case "$PROFILE" in
    small) PROFILE_ARGS=(--lr 0.0001 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.01) ;;
    balanced) PROFILE_ARGS=(--lr 0.0003 --transfer_scale 1.0 --safety_weight 1.0 --residual_weight 0.001) ;;
    open) PROFILE_ARGS=(--lr 0.0005 --transfer_scale 2.0 --safety_weight 0.5 --residual_weight 0.001) ;;
    strong_safety) PROFILE_ARGS=(--lr 0.0005 --transfer_scale 2.0 --safety_weight 2.0 --residual_weight 0.001) ;;
    *) echo "Unknown PROFILE=$PROFILE" >&2; exit 2 ;;
  esac
}
profile_args

mkdir -p "$WORK_ROOT/logs"
for seed in $SEEDS; do
  source_dir="$WORK_ROOT/source_backbones/${CORPUS,,}/seed${seed}"
  source_log="$WORK_ROOT/logs/${CORPUS,,}_source_seed${seed}.log"
  if [[ "$RESUME_DONE" != 1 || ! -s "$source_dir/model_best.pt" ]]; then
    echo "[DLCEN-CESA-5] source corpus=$CORPUS seed=$seed gpu=$GPU" | tee "$source_log"
    "$PYTHON_BIN" new/train_eval_decoupled_context_anjs.py \
      --train_pkl "$SOURCE_PKL" \
      --external_test_pkl "$EXTERNAL_PKL" \
      --out_dir "$source_dir" \
      --seed "$seed" \
      --epochs "$SOURCE_EPOCHS" \
      --batch_size 16 \
      --eval_batch_size 16 \
      --lr 0.0003 \
      --weight_decay 0.0001 \
      --hidden_dim 256 \
      --adapter_dim 128 \
      --dropout 0.3 \
      --adapter_dropout 0.2 \
      --bobsl_weight 0.0 \
      --local_loss_weight 0.25 \
      --context_loss_weight 0.25 \
      --shift_loss_weight 0.1 \
      --calibration_weight 0.05 \
      --residual_weight 0.01 \
      --max_grad_norm 5.0 2>&1 | tee -a "$source_log"
  fi

  if [[ "$RUN_TRANSFER" != 1 ]]; then
    continue
  fi

  transfer_dir="$WORK_ROOT/${CORPUS,,}/${PROFILE}_seed${seed}"
  transfer_log="$WORK_ROOT/logs/${CORPUS,,}_${PROFILE}_seed${seed}.log"
  if [[ "$RESUME_DONE" == 1 && -s "$transfer_dir/external_test_best_summary.json" ]]; then
    echo "[DLCEN-CESA-5] skip $transfer_dir"
    continue
  fi
  echo "[DLCEN-CESA-5] transfer corpus=$CORPUS profile=$PROFILE seed=$seed gpu=$GPU" | tee "$transfer_log"
  "$PYTHON_BIN" new/train_eval_dlcen_cesa_anjs.py \
    --train_pkl "$JOINT_PKL" \
    --external_test_pkl "$EXTERNAL_PKL" \
    --base_checkpoint "$source_dir/model_best.pt" \
    --out_dir "$transfer_dir" \
    --seed "$seed" \
    --epochs "$TRANSFER_EPOCHS" \
    --source_batch_size 16 \
    --bobsl_batch_size 64 \
    --eval_batch_size 16 \
    --bobsl_weight 1.0 \
    --local_source_weight 0.1 \
    --max_grad_norm 5.0 \
    "${PROFILE_ARGS[@]}" 2>&1 | tee -a "$transfer_log"
done

if [[ "$RUN_TRANSFER" == 1 ]]; then
  "$PYTHON_BIN" analysis/summarize_dlcen_cesa.py --root "$WORK_ROOT"
fi
