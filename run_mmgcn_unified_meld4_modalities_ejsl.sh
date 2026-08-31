#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
SOURCE_TXT_ROOT=${SOURCE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}
TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-"$WORK_ROOT/ejsl_txt_en_openai"}
TRANSLATION_CACHE=${TRANSLATION_CACHE:-"$TRANSLATED_TXT_ROOT/translation_cache.jsonl"}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-openai}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}

MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$WORK_ROOT/meld_anjs4_unified.pkl"}
EJSL_UNIFIED_PKL=${EJSL_UNIFIED_PKL:-"$WORK_ROOT/ejsl_anjs4_unified.pkl"}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-"$WORK_ROOT/feature_cache"}
OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/runs_meld4_modalities"}

GPU=${GPU:-0}
TRANSLATE=${TRANSLATE:-1}
REBUILD_FEATURES=${REBUILD_FEATURES:-0}
MODALITIES=${MODALITIES:-"text video tv"}
GRAPH_TYPE=${GRAPH_TYPE:-MMGCN}
EPOCHS=${EPOCHS:-15}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-0.0003}
L2=${L2:-0.00003}
DROPOUT=${DROPOUT:-0.4}
LOSS=${LOSS:-focal}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-5.0}
SEED=${SEED:-42}

mkdir -p "$WORK_ROOT" "$OUT_ROOT"

if [ "$TRANSLATED_TXT_ROOT" = "$SOURCE_TXT_ROOT" ]; then
  echo "[MMGCN-MELD4] refusing to overwrite SOURCE_TXT_ROOT: $SOURCE_TXT_ROOT" >&2
  exit 1
fi

if [ "$TRANSLATE" = "1" ]; then
  echo "[MMGCN-MELD4] translate eJSL txt to English"
  python JSL/translate_ejsl_txt_root.py \
    --input_txt_root "$SOURCE_TXT_ROOT" \
    --output_txt_root "$TRANSLATED_TXT_ROOT" \
    --dial_list "$DIAL_LIST" \
    --cache_jsonl "$TRANSLATION_CACHE" \
    --backend "$TRANSLATION_BACKEND" \
    --openai_model "$OPENAI_MODEL"
fi

if [ "$REBUILD_FEATURES" = "1" ] || [ ! -f "$MELD_UNIFIED_PKL" ] || [ ! -f "$EJSL_UNIFIED_PKL" ]; then
  echo "[MMGCN-MELD4] build same-origin ANJS4 MELD/eJSL feature pkl"
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/build_unified_meld_ejsl_pkl.py \
    --meld_root "$MELD_RAW_ROOT" \
    --ejsl_txt_root "$TRANSLATED_TXT_ROOT" \
    --ejsl_dial_list "$DIAL_LIST" \
    --ejsl_frame_root "$FRAME_ROOT" \
    --ejsl_mp4_root "$MP4_ROOT" \
    --out_meld_pkl "$MELD_UNIFIED_PKL" \
    --out_ejsl_pkl "$EJSL_UNIFIED_PKL" \
    --cache_dir "$FEATURE_CACHE_DIR" \
    --fp16
fi

echo "[MMGCN-MELD4] train on MELD ANJS4 and evaluate eJSL modalities: $MODALITIES"
CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
  --train_pkl "$MELD_UNIFIED_PKL" \
  --external_test_pkl "$EJSL_UNIFIED_PKL" \
  --out_dir "$OUT_ROOT" \
  --modalities $MODALITIES \
  --graph_type "$GRAPH_TYPE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --l2 "$L2" \
  --dropout "$DROPOUT" \
  --loss "$LOSS" \
  --focal_gamma "$FOCAL_GAMMA" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --seed "$SEED"

echo "[MMGCN-MELD4] reports: $OUT_ROOT"
for MOD in $MODALITIES; do
  case "$MOD" in
    t|text|l) EVAL_MOD=text ;;
    v|video|visual) EVAL_MOD=video ;;
    tv|vt|vl|text_video|video_text) EVAL_MOD=tv ;;
    *) continue ;;
  esac
  REPORT_PATH="$OUT_ROOT/$EVAL_MOD/external_test_best_classification_report.txt"
  echo "===== ${EVAL_MOD} ====="
  if [ -f "$REPORT_PATH" ]; then
    sed -n '1,20p' "$REPORT_PATH"
  else
    echo "missing report: $REPORT_PATH"
  fi
done
