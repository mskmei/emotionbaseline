#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
TRAIN_PKL=${TRAIN_PKL:-/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_pkls/joint_b5k.pkl}
EXTERNAL_PKL=${EXTERNAL_PKL:-/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/proposal2_dual_anchor_search/magtkd_joint_anchor}
SEEDS=${SEEDS:-42}
PROFILES=${PROFILES:-"base lr50 lr75 lr150 lr200 lr300 drop20 drop30 drop40 drop60 drop70 drop80 nocw temp10 temp15 temp30 temp40 batch8 batch32 l2low l2high"}
EPOCHS=${EPOCHS:-40}
RESUME_DONE=${RESUME_DONE:-1}

profile_args() {
  LR=0.0001
  L2=0.000001
  DROPOUT=0.5
  TEMP=2.0
  BATCH_SIZE=16
  CLASS_WEIGHT=1
  case "$1" in
    base) ;;
    lr50) LR=0.00005 ;;
    lr75) LR=0.000075 ;;
    lr150) LR=0.00015 ;;
    lr200) LR=0.0002 ;;
    lr300) LR=0.0003 ;;
    drop20) DROPOUT=0.2 ;;
    drop30) DROPOUT=0.3 ;;
    drop40) DROPOUT=0.4 ;;
    drop60) DROPOUT=0.6 ;;
    drop70) DROPOUT=0.7 ;;
    drop80) DROPOUT=0.8 ;;
    nocw) CLASS_WEIGHT=0 ;;
    temp10) TEMP=1.0 ;;
    temp15) TEMP=1.5 ;;
    temp30) TEMP=3.0 ;;
    temp40) TEMP=4.0 ;;
    batch8) BATCH_SIZE=8 ;;
    batch32) BATCH_SIZE=32 ;;
    l2low) L2=0.0000001 ;;
    l2high) L2=0.00001 ;;
    *) echo "Unknown profile=$1" >&2; return 2 ;;
  esac
}

mkdir -p "$WORK_ROOT/logs"
for profile in $PROFILES; do
  profile_args "$profile"
  for seed in $SEEDS; do
    out_dir="$WORK_ROOT/${profile}_seed${seed}"
    log_path="$WORK_ROOT/logs/${profile}_seed${seed}.log"
    if [[ "$RESUME_DONE" == 1 && -s "$out_dir/external_test_best_summary.json" ]]; then
      echo "[MAGTKD-ANCHOR] skip $out_dir"
      continue
    fi
    class_weight_args=()
    if [[ "$CLASS_WEIGHT" == 1 ]]; then class_weight_args+=(--class_weight); fi
    echo "[MAGTKD-ANCHOR] profile=$profile seed=$seed gpu=$GPU lr=$LR drop=$DROPOUT temp=$TEMP batch=$BATCH_SIZE l2=$L2 cw=$CLASS_WEIGHT" | tee "$log_path"
    "$PYTHON_BIN" new/MAGTKD/MELD/train_eval_magtkd_video_anjs.py \
      --train_pkl "$TRAIN_PKL" \
      --external_test_pkl "$EXTERNAL_PKL" \
      --out_dir "$out_dir" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_workers 0 \
      --lr "$LR" \
      --l2 "$L2" \
      --dropout "$DROPOUT" \
      --hidden_dim 768 \
      --n_head 8 \
      --temp "$TEMP" \
      --output_head video \
      --max_grad_norm 10.0 \
      --selection_split source \
      --selection_metric weighted_f1 \
      --save_epoch_every 0 \
      "${class_weight_args[@]}" 2>&1 | tee -a "$log_path"
  done
done

"$PYTHON_BIN" analysis/summarize_magtkd_anchor_sweep.py \
  --root "$WORK_ROOT" \
  --out_csv "$WORK_ROOT/anchor_summary.csv"
