#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl}
UNIFIED_ROOT=${UNIFIED_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}
INV_ROOT=${INV_ROOT:-"$WORK_ROOT/gap_investigation"}

BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-"$WORK_ROOT/bobsl_anjs4_train_val.pkl"}
BOBSL_TEST_PKL=${BOBSL_TEST_PKL:-"$WORK_ROOT/bobsl_anjs4_test.pkl"}
PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-"$WORK_ROOT/bobsl_video_pretrain/video/model_best.pt"}
MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$UNIFIED_ROOT/meld_anjs4_unified.pkl"}
EJSL_UNIFIED_PKL=${EJSL_UNIFIED_PKL:-"$UNIFIED_ROOT/ejsl_anjs4_unified.pkl"}
ORIGINAL_RUN_ROOT=${ORIGINAL_RUN_ROOT:-"$WORK_ROOT/meld_v_tv_finetune_from_bobsl_video"}

MODALITIES=${MODALITIES:-"video tv"}
GRAPH_TYPE=${GRAPH_TYPE:-MMGCN}
EPOCHS=${EPOCHS:-15}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-0.0003}
LOW_LR=${LOW_LR:-0.0001}
L2=${L2:-0.00003}
DROPOUT=${DROPOUT:-0.4}
LOSS=${LOSS:-focal}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-5.0}
SEED=${SEED:-43}
SAVE_EPOCH_EVERY=${SAVE_EPOCH_EVERY:-5}
RUN_FINETUNE=${RUN_FINETUNE:-1}
RUN_DIRECT_EVAL=${RUN_DIRECT_EVAL:-1}

mkdir -p "$INV_ROOT/gpu1"

for required in "$BOBSL_TRAIN_VAL_PKL" "$BOBSL_TEST_PKL" "$PRETRAIN_CHECKPOINT" "$MELD_UNIFIED_PKL" "$EJSL_UNIFIED_PKL"; do
  if [ ! -f "$required" ]; then
    echo "[MMGCN-GAP-GPU1] missing required file: $required" >&2
    exit 1
  fi
done

if [ "$RUN_DIRECT_EVAL" = "1" ]; then
  echo "[MMGCN-GAP-GPU1] direct eval of BOBSL video checkpoint before any MELD fine-tune"
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/eval_unified_checkpoint.py \
    --checkpoint "$PRETRAIN_CHECKPOINT" \
    --test_pkl "$BOBSL_TEST_PKL" "$MELD_UNIFIED_PKL" "$EJSL_UNIFIED_PKL" \
    --out_dir "$INV_ROOT/gpu1/direct_bobsl_checkpoint_eval" \
    --modalities video \
    --batch_size "$BATCH_SIZE" \
    --num_workers 0 | tee "$INV_ROOT/gpu1/direct_bobsl_checkpoint_eval.log"
fi

if [ "$RUN_FINETUNE" = "1" ]; then
  echo "[MMGCN-GAP-GPU1] variant C: BOBSL init, eJSL oracle selection"
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$MELD_UNIFIED_PKL" \
    --external_test_pkl "$EJSL_UNIFIED_PKL" \
    --out_dir "$INV_ROOT/gpu1/from_bobsl_external_select" \
    --modalities $MODALITIES \
    --graph_type "$GRAPH_TYPE" \
    --init_checkpoint "$PRETRAIN_CHECKPOINT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --l2 "$L2" \
    --dropout "$DROPOUT" \
    --loss "$LOSS" \
    --focal_gamma "$FOCAL_GAMMA" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --selection_split external \
    --selection_metric weighted_f1 \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    --seed "$SEED" | tee "$INV_ROOT/gpu1/from_bobsl_external_select.log"

  echo "[MMGCN-GAP-GPU1] variant D: BOBSL init, lower LR, select best by MELD/source"
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$MELD_UNIFIED_PKL" \
    --external_test_pkl "$EJSL_UNIFIED_PKL" \
    --out_dir "$INV_ROOT/gpu1/from_bobsl_low_lr_source_select" \
    --modalities $MODALITIES \
    --graph_type "$GRAPH_TYPE" \
    --init_checkpoint "$PRETRAIN_CHECKPOINT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LOW_LR" \
    --l2 "$L2" \
    --dropout "$DROPOUT" \
    --loss "$LOSS" \
    --focal_gamma "$FOCAL_GAMMA" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --selection_split source \
    --selection_metric weighted_f1 \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    --seed "$SEED" | tee "$INV_ROOT/gpu1/from_bobsl_low_lr_source_select.log"
fi

echo "[MMGCN-GAP-GPU1] offline diagnosis"
python MMGCN/analyze_unified_transfer_gap.py \
  --bobsl_train_val_pkl "$BOBSL_TRAIN_VAL_PKL" \
  --bobsl_test_pkl "$BOBSL_TEST_PKL" \
  --meld_pkl "$MELD_UNIFIED_PKL" \
  --ejsl_pkl "$EJSL_UNIFIED_PKL" \
  --run_roots "$ORIGINAL_RUN_ROOT" "$INV_ROOT/gpu1/from_bobsl_external_select" "$INV_ROOT/gpu1/from_bobsl_low_lr_source_select" \
  --out_dir "$INV_ROOT/gpu1/diagnosis" \
  --max_items 5000 \
  --seed "$SEED" | tee "$INV_ROOT/gpu1/diagnosis.log"

echo "[MMGCN-GAP-GPU1] done: $INV_ROOT/gpu1"
