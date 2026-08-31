#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}
TELME_SAVE_ROOT=${TELME_SAVE_ROOT:-./MELD/save_model_meld4_modalities}
OUT_ROOT=${OUT_ROOT:-./IEMOCAP/outputs_ejsl_telme_meld4_modalities_translated}

FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
SOURCE_TXT_ROOT=${SOURCE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-./IEMOCAP/ejsl_txt_en_openai}
TRANSLATION_CACHE=${TRANSLATION_CACHE:-"$TRANSLATED_TXT_ROOT/translation_cache.jsonl"}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-openai}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}

MODALITIES=${MODALITIES:-"text video tv"}
GPU=${GPU:-0}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-4}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-16}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-4}
TEST_NUM_WORKERS=${TEST_NUM_WORKERS:-4}

TEACHER_EPOCHS=${TEACHER_EPOCHS:-10}
STUDENT_EPOCHS=${STUDENT_EPOCHS:-10}
FUSION_EPOCHS=${FUSION_EPOCHS:-10}
TEACHER_LR=${TEACHER_LR:-1e-6}
STUDENT_LR=${STUDENT_LR:-1e-5}
FUSION_LR=${FUSION_LR:-1e-5}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-10.0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
MAX_EJSL_SAMPLES=${MAX_EJSL_SAMPLES:-0}
TRANSLATE=${TRANSLATE:-1}
REUSE_SHARED=${REUSE_SHARED:-0}
REUSE_FUSION=${REUSE_FUSION:-0}

mkdir -p "$OUT_ROOT"

if [ "$TRANSLATED_TXT_ROOT" = "$SOURCE_TXT_ROOT" ]; then
  echo "[TELME-MELD4] refusing to overwrite SOURCE_TXT_ROOT: $SOURCE_TXT_ROOT" >&2
  exit 1
fi

for csv_path in \
  "$MELD_RAW_ROOT/train_meld_emo.csv" \
  "$MELD_RAW_ROOT/dev_meld_emo.csv" \
  "$MELD_RAW_ROOT/test_meld_emo.csv"; do
  if [ ! -f "$csv_path" ]; then
    echo "[TELME-MELD4] missing MELD csv: $csv_path" >&2
    exit 1
  fi
done

if [ "$TRANSLATE" = "1" ]; then
  echo "[TELME-MELD4] translate eJSL txt to English"
  python JSL/translate_ejsl_txt_root.py \
    --input_txt_root "$SOURCE_TXT_ROOT" \
    --output_txt_root "$TRANSLATED_TXT_ROOT" \
    --dial_list "$DIAL_LIST" \
    --cache_jsonl "$TRANSLATION_CACHE" \
    --backend "$TRANSLATION_BACKEND" \
    --openai_model "$OPENAI_MODEL"
fi

TRAIN_FLAGS=()
if [ "$REUSE_SHARED" = "1" ]; then
  TRAIN_FLAGS+=(--reuse_shared)
fi
if [ "$REUSE_FUSION" = "1" ]; then
  TRAIN_FLAGS+=(--reuse_fusion)
fi

echo "[TELME-MELD4] train MELD ANJS4 modalities: $MODALITIES"
CUDA_VISIBLE_DEVICES="$GPU" python MELD/train_telme_meld4_modalities.py \
  --data_root "$MELD_RAW_ROOT" \
  --save_root "$TELME_SAVE_ROOT" \
  --modalities $MODALITIES \
  --teacher_epochs "$TEACHER_EPOCHS" \
  --student_epochs "$STUDENT_EPOCHS" \
  --fusion_epochs "$FUSION_EPOCHS" \
  --teacher_lr "$TEACHER_LR" \
  --student_lr "$STUDENT_LR" \
  --fusion_lr "$FUSION_LR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$TRAIN_NUM_WORKERS" \
  --seed "$SEED" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  "${TRAIN_FLAGS[@]}" | tee "$OUT_ROOT/telme_meld4_train.log"

for MOD in $MODALITIES; do
  case "$MOD" in
    t) EVAL_MOD=text ;;
    v|visual) EVAL_MOD=video ;;
    tv|vt|vl|text_video|video_text) EVAL_MOD=tv ;;
    text|video) EVAL_MOD="$MOD" ;;
    *)
      echo "[TELME-MELD4] unsupported modality in MODALITIES: $MOD" >&2
      exit 1
      ;;
  esac

  MODEL_ROOT="$TELME_SAVE_ROOT/$EVAL_MOD"
  PREFIX="telme_meld4_${EVAL_MOD}_ejsl"
  LOG_PATH="$OUT_ROOT/${PREFIX}.log"

  echo "[TELME-MELD4][$EVAL_MOD] test eJSL"
  CUDA_VISIBLE_DEVICES="$GPU" python IEMOCAP/inference_ejsl_frame.py \
    --frame_root "$FRAME_ROOT" \
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

echo "[TELME-MELD4] eJSL summary"
for MOD in $MODALITIES; do
  case "$MOD" in
    t) EVAL_MOD=text ;;
    v|visual) EVAL_MOD=video ;;
    tv|vt|vl|text_video|video_text) EVAL_MOD=tv ;;
    text|video) EVAL_MOD="$MOD" ;;
    *) continue ;;
  esac
  REPORT_PATH="$OUT_ROOT/telme_meld4_${EVAL_MOD}_ejsl_classification_report.txt"
  echo "===== ${EVAL_MOD} ====="
  if [ -f "$REPORT_PATH" ]; then
    sed -n '1,20p' "$REPORT_PATH"
  else
    echo "missing report: $REPORT_PATH"
  fi
done
