#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

export TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ejsl_txt_en_openai}

BOBSL_ROOT=${BOBSL_ROOT:-/raid_zoe/home/lr/wangyi/sign/bobsl}
MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
SOURCE_TXT_ROOT=${SOURCE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/telme_bobsl_meld_ejsl}
BOBSL_SAVE_ROOT=${BOBSL_SAVE_ROOT:-"$WORK_ROOT/bobsl_video_pretrain"}
MELD_SAVE_ROOT=${MELD_SAVE_ROOT:-"$WORK_ROOT/meld_v_tv_finetune_from_bobsl_video"}
OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/ejsl_outputs"}
BOBSL_VIDEO_CKPT=${BOBSL_VIDEO_CKPT:-"$BOBSL_SAVE_ROOT/student_video/total_student.bin"}

GPU=${GPU:-0}
SEED=${SEED:-42}
MODALITIES=${MODALITIES:-"video tv"}
TRANSLATE=${TRANSLATE:-0}
TRANSLATION_CACHE=${TRANSLATION_CACHE:-"$TRANSLATED_TXT_ROOT/translation_cache.jsonl"}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-openai}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}

RUN_BOBSL_PRETRAIN=${RUN_BOBSL_PRETRAIN:-1}
REUSE_BOBSL_PRETRAIN=${REUSE_BOBSL_PRETRAIN:-0}
RUN_MELD_FINETUNE=${RUN_MELD_FINETUNE:-1}
REUSE_SHARED=${REUSE_SHARED:-0}
REUSE_FUSION=${REUSE_FUSION:-0}

BOBSL_EPOCHS=${BOBSL_EPOCHS:-30}
BOBSL_BATCH_SIZE=${BOBSL_BATCH_SIZE:-8}
BOBSL_NUM_WORKERS=${BOBSL_NUM_WORKERS:-4}
BOBSL_LR=${BOBSL_LR:-1e-5}
BOBSL_WEIGHT_DECAY=${BOBSL_WEIGHT_DECAY:-0.0}
BOBSL_MIN_SCORE=${BOBSL_MIN_SCORE:-0.0}
BOBSL_LIMIT_TRAIN=${BOBSL_LIMIT_TRAIN:-0}
BOBSL_LIMIT_VAL=${BOBSL_LIMIT_VAL:-0}
BOBSL_LIMIT_TEST=${BOBSL_LIMIT_TEST:-0}
BOBSL_FREEZE_VIDEO_BACKBONE=${BOBSL_FREEZE_VIDEO_BACKBONE:-0}

TEACHER_EPOCHS=${TEACHER_EPOCHS:-10}
STUDENT_EPOCHS=${STUDENT_EPOCHS:-10}
FUSION_EPOCHS=${FUSION_EPOCHS:-10}
TEACHER_LR=${TEACHER_LR:-1e-6}
STUDENT_LR=${STUDENT_LR:-1e-5}
FUSION_LR=${FUSION_LR:-1e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-16}

TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-4}
TEST_NUM_WORKERS=${TEST_NUM_WORKERS:-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-10.0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
MAX_EJSL_SAMPLES=${MAX_EJSL_SAMPLES:-0}
FREEZE_VIDEO_BACKBONE=${FREEZE_VIDEO_BACKBONE:-0}
SAVE_EPOCH_EVERY=${SAVE_EPOCH_EVERY:-5}

mkdir -p "$WORK_ROOT" "$BOBSL_SAVE_ROOT" "$MELD_SAVE_ROOT" "$OUT_ROOT"

if [ "$TRANSLATED_TXT_ROOT" = "$SOURCE_TXT_ROOT" ]; then
  echo "[TELME-BOBSL] refusing to overwrite SOURCE_TXT_ROOT: $SOURCE_TXT_ROOT" >&2
  exit 1
fi

if [ "$TRANSLATE" = "1" ]; then
  echo "[TELME-BOBSL] translate eJSL txt to English"
  python JSL/translate_ejsl_txt_root.py \
    --input_txt_root "$SOURCE_TXT_ROOT" \
    --output_txt_root "$TRANSLATED_TXT_ROOT" \
    --dial_list "$DIAL_LIST" \
    --cache_jsonl "$TRANSLATION_CACHE" \
    --backend "$TRANSLATION_BACKEND" \
    --openai_model "$OPENAI_MODEL"
elif [ ! -d "$TRANSLATED_TXT_ROOT" ]; then
  echo "[TELME-BOBSL] missing translated eJSL txt root: $TRANSLATED_TXT_ROOT" >&2
  echo "[TELME-BOBSL] set TRANSLATE=1 or point TRANSLATED_TXT_ROOT to the existing translated txt directory." >&2
  exit 1
fi

BOBSL_FLAGS=()
if [ "$BOBSL_FREEZE_VIDEO_BACKBONE" = "1" ]; then
  BOBSL_FLAGS+=(--freeze_video_backbone)
fi

if [ "$RUN_BOBSL_PRETRAIN" = "1" ]; then
  if [ "$REUSE_BOBSL_PRETRAIN" = "1" ] && [ -f "$BOBSL_VIDEO_CKPT" ]; then
    echo "[TELME-BOBSL] reuse BOBSL video checkpoint: $BOBSL_VIDEO_CKPT"
  else
    echo "[TELME-BOBSL] pretrain TELME video student on BOBSL"
    CUDA_VISIBLE_DEVICES="$GPU" python MELD/train_telme_bobsl_video.py \
      --bobsl_root "$BOBSL_ROOT" \
      --save_root "$BOBSL_SAVE_ROOT" \
      --epochs "$BOBSL_EPOCHS" \
      --batch_size "$BOBSL_BATCH_SIZE" \
      --num_workers "$BOBSL_NUM_WORKERS" \
      --lr "$BOBSL_LR" \
      --weight_decay "$BOBSL_WEIGHT_DECAY" \
      --max_grad_norm "$MAX_GRAD_NORM" \
      --seed "$SEED" \
      --min_score "$BOBSL_MIN_SCORE" \
      --limit_train "$BOBSL_LIMIT_TRAIN" \
      --limit_val "$BOBSL_LIMIT_VAL" \
      --limit_test "$BOBSL_LIMIT_TEST" \
      --save_epoch_every "$SAVE_EPOCH_EVERY" \
      "${BOBSL_FLAGS[@]}" | tee "$BOBSL_SAVE_ROOT/telme_bobsl_video_pretrain.log"
  fi
fi

if [ ! -f "$BOBSL_VIDEO_CKPT" ]; then
  echo "[TELME-BOBSL] missing BOBSL video checkpoint: $BOBSL_VIDEO_CKPT" >&2
  exit 1
fi

TRAIN_FLAGS=()
if [ "$REUSE_SHARED" = "1" ]; then
  TRAIN_FLAGS+=(--reuse_shared)
fi
if [ "$REUSE_FUSION" = "1" ]; then
  TRAIN_FLAGS+=(--reuse_fusion)
fi
if [ "$FREEZE_VIDEO_BACKBONE" = "1" ]; then
  TRAIN_FLAGS+=(--freeze_video_backbone)
fi

if [ "$RUN_MELD_FINETUNE" = "1" ]; then
  echo "[TELME-BOBSL] fine-tune on MELD4 modalities: $MODALITIES"
  CUDA_VISIBLE_DEVICES="$GPU" python MELD/train_telme_meld4_modalities.py \
    --data_root "$MELD_RAW_ROOT" \
    --save_root "$MELD_SAVE_ROOT" \
    --modalities $MODALITIES \
    --teacher_epochs "$TEACHER_EPOCHS" \
    --student_epochs "$STUDENT_EPOCHS" \
    --fusion_epochs "$FUSION_EPOCHS" \
    --teacher_lr "$TEACHER_LR" \
    --student_lr "$STUDENT_LR" \
    --fusion_lr "$FUSION_LR" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --num_workers "$TRAIN_NUM_WORKERS" \
    --seed "$SEED" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --video_init_checkpoint "$BOBSL_VIDEO_CKPT" \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    "${TRAIN_FLAGS[@]}" | tee "$MELD_SAVE_ROOT/telme_meld4_finetune.log"
fi

for MOD in $MODALITIES; do
  case "$MOD" in
    t) EVAL_MOD=text ;;
    v|visual) EVAL_MOD=video ;;
    tv|vt|vl|text_video|video_text) EVAL_MOD=tv ;;
    text|video) EVAL_MOD="$MOD" ;;
    *)
      echo "[TELME-BOBSL] unsupported modality in MODALITIES: $MOD" >&2
      exit 1
      ;;
  esac

  MODEL_ROOT="$MELD_SAVE_ROOT/$EVAL_MOD"
  PREFIX="telme_bobsl_meld4_${EVAL_MOD}_ejsl"
  LOG_PATH="$OUT_ROOT/${PREFIX}.log"

  echo "[TELME-BOBSL][$EVAL_MOD] evaluate translated eJSL"
  CUDA_VISIBLE_DEVICES="$GPU" python IEMOCAP/inference_ejsl_frame.py \
    --frame_root "$FRAME_ROOT" \
    --mp4_root "$MP4_ROOT" \
    --txt_root "$TRANSLATED_TXT_ROOT" \
    --save_model_root "$MODEL_ROOT" \
    --batch_size "$TEST_BATCH_SIZE" \
    --num_workers "$TEST_NUM_WORKERS" \
    --seed "$SEED" \
    --save_dir "$OUT_ROOT" \
    --report_prefix "$PREFIX" \
    --eval_modality "$EVAL_MOD" \
    --checkpoint_dataset MELD4 \
    --fusion_input_order audio_video \
    --missing_audio_strategy zero_hidden \
    --max_samples "$MAX_EJSL_SAMPLES" \
    --save_predictions | tee "$LOG_PATH"
done

echo "[TELME-BOBSL] BOBSL report: $BOBSL_SAVE_ROOT/bobsl_test_best_classification_report.txt"
echo "[TELME-BOBSL] eJSL reports: $OUT_ROOT"
for MOD in $MODALITIES; do
  case "$MOD" in
    t) EVAL_MOD=text ;;
    v|visual) EVAL_MOD=video ;;
    tv|vt|vl|text_video|video_text) EVAL_MOD=tv ;;
    text|video) EVAL_MOD="$MOD" ;;
    *) continue ;;
  esac
  REPORT_PATH="$OUT_ROOT/telme_bobsl_meld4_${EVAL_MOD}_ejsl_classification_report.txt"
  echo "===== ${EVAL_MOD} ====="
  if [ -f "$REPORT_PATH" ]; then
    sed -n '1,20p' "$REPORT_PATH"
  else
    echo "missing report: $REPORT_PATH"
  fi
done
