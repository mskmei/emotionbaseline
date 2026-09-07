#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

export TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ejsl_txt_en_openai}

GPU=${GPU:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/telme_bobsl_meld_ejsl}
INV_ROOT=${INV_ROOT:-"$WORK_ROOT/gap_investigation_gpu1"}
BOBSL_VIDEO_CKPT=${BOBSL_VIDEO_CKPT:-"$WORK_ROOT/bobsl_video_pretrain/student_video/total_student.bin"}

MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
SOURCE_TXT_ROOT=${SOURCE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}

MODALITIES=${MODALITIES:-"video tv"}
SEED=${SEED:-43}
TEACHER_EPOCHS=${TEACHER_EPOCHS:-10}
STUDENT_EPOCHS=${STUDENT_EPOCHS:-10}
FUSION_EPOCHS=${FUSION_EPOCHS:-10}
TEACHER_LR=${TEACHER_LR:-1e-6}
STUDENT_LR=${STUDENT_LR:-1e-5}
LOW_STUDENT_LR=${LOW_STUDENT_LR:-3e-6}
FUSION_LR=${FUSION_LR:-1e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-16}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-4}
TEST_NUM_WORKERS=${TEST_NUM_WORKERS:-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-10.0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
MAX_EJSL_SAMPLES=${MAX_EJSL_SAMPLES:-0}
SAVE_EPOCH_EVERY=${SAVE_EPOCH_EVERY:-5}

RUN_SCRATCH_CONTROL=${RUN_SCRATCH_CONTROL:-1}
RUN_BOBSL_FREEZE=${RUN_BOBSL_FREEZE:-1}
RUN_BOBSL_LOW_LR=${RUN_BOBSL_LOW_LR:-1}
WAIT_FOR_BOBSL_CKPT=${WAIT_FOR_BOBSL_CKPT:-1}
WAIT_MINUTES=${WAIT_MINUTES:-360}

mkdir -p "$INV_ROOT"

if [ "$TRANSLATED_TXT_ROOT" = "$SOURCE_TXT_ROOT" ]; then
  echo "[TELME-GAP-GPU1] refusing to overwrite SOURCE_TXT_ROOT: $SOURCE_TXT_ROOT" >&2
  exit 1
fi
if [ ! -d "$TRANSLATED_TXT_ROOT" ]; then
  echo "[TELME-GAP-GPU1] missing translated eJSL txt root: $TRANSLATED_TXT_ROOT" >&2
  exit 1
fi

eval_variant() {
  local name="$1"
  local init_ckpt="$2"
  local student_lr="$3"
  local freeze_backbone="$4"
  local save_root="$INV_ROOT/$name/save_model"
  local out_root="$INV_ROOT/$name/ejsl_outputs"
  mkdir -p "$save_root" "$out_root"

  local train_flags=()
  if [ -n "$init_ckpt" ]; then
    train_flags+=(--video_init_checkpoint "$init_ckpt")
  fi
  if [ "$freeze_backbone" = "1" ]; then
    train_flags+=(--freeze_video_backbone)
  fi

  echo "[TELME-GAP-GPU1][$name] fine-tune MELD4 modalities=$MODALITIES student_lr=$student_lr freeze=$freeze_backbone"
  CUDA_VISIBLE_DEVICES="$GPU" python MELD/train_telme_meld4_modalities.py \
    --data_root "$MELD_RAW_ROOT" \
    --save_root "$save_root" \
    --modalities $MODALITIES \
    --teacher_epochs "$TEACHER_EPOCHS" \
    --student_epochs "$STUDENT_EPOCHS" \
    --fusion_epochs "$FUSION_EPOCHS" \
    --teacher_lr "$TEACHER_LR" \
    --student_lr "$student_lr" \
    --fusion_lr "$FUSION_LR" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --num_workers "$TRAIN_NUM_WORKERS" \
    --seed "$SEED" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    "${train_flags[@]}" | tee "$INV_ROOT/$name/train.log"

  for MOD in $MODALITIES; do
    case "$MOD" in
      t) EVAL_MOD=text ;;
      v|visual) EVAL_MOD=video ;;
      tv|vt|vl|text_video|video_text) EVAL_MOD=tv ;;
      text|video) EVAL_MOD="$MOD" ;;
      *) continue ;;
    esac

    echo "[TELME-GAP-GPU1][$name][$EVAL_MOD] evaluate translated eJSL"
    CUDA_VISIBLE_DEVICES="$GPU" python IEMOCAP/inference_ejsl_frame.py \
      --frame_root "$FRAME_ROOT" \
      --mp4_root "$MP4_ROOT" \
      --txt_root "$TRANSLATED_TXT_ROOT" \
      --save_model_root "$save_root/$EVAL_MOD" \
      --batch_size "$TEST_BATCH_SIZE" \
      --num_workers "$TEST_NUM_WORKERS" \
      --seed "$SEED" \
      --save_dir "$out_root" \
      --report_prefix "telme_${name}_${EVAL_MOD}_ejsl" \
      --eval_modality "$EVAL_MOD" \
      --checkpoint_dataset MELD4 \
      --fusion_input_order audio_video \
      --missing_audio_strategy zero_hidden \
      --max_samples "$MAX_EJSL_SAMPLES" \
      --save_predictions | tee "$INV_ROOT/$name/${EVAL_MOD}_ejsl.log"
  done
}

if [ "$RUN_SCRATCH_CONTROL" = "1" ]; then
  eval_variant "scratch_meld4" "" "$STUDENT_LR" "0"
fi

if [ ! -f "$BOBSL_VIDEO_CKPT" ] && [ "$WAIT_FOR_BOBSL_CKPT" = "1" ]; then
  echo "[TELME-GAP-GPU1] waiting for BOBSL checkpoint: $BOBSL_VIDEO_CKPT"
  waited=0
  while [ ! -f "$BOBSL_VIDEO_CKPT" ] && [ "$waited" -lt "$WAIT_MINUTES" ]; do
    sleep 60
    waited=$((waited + 1))
    echo "[TELME-GAP-GPU1] waited ${waited}m/${WAIT_MINUTES}m"
  done
fi

if [ ! -f "$BOBSL_VIDEO_CKPT" ]; then
  echo "[TELME-GAP-GPU1] BOBSL checkpoint is still missing; skip BOBSL-init variants: $BOBSL_VIDEO_CKPT" >&2
  exit 0
fi

if [ "$RUN_BOBSL_FREEZE" = "1" ]; then
  eval_variant "bobsl_init_freeze_backbone" "$BOBSL_VIDEO_CKPT" "$STUDENT_LR" "1"
fi

if [ "$RUN_BOBSL_LOW_LR" = "1" ]; then
  eval_variant "bobsl_init_low_student_lr" "$BOBSL_VIDEO_CKPT" "$LOW_STUDENT_LR" "0"
fi

echo "[TELME-GAP-GPU1] done: $INV_ROOT"
