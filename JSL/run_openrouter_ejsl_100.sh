#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before calling OpenRouter.}"

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
VIDEO_ROOT=${VIDEO_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
OUT_ROOT=${OUT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/openrouter_ejsl_20}
STRUCTURE_TXT_ROOT=${STRUCTURE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
OUTPUT_TXT_ROOT=${OUTPUT_TXT_ROOT:-"$OUT_ROOT/txt_roots"}
WRITE_TXT_TREE=${WRITE_TXT_TREE:-0}

MODELS=${MODELS:-"google/gemini-3.5-flash google/gemini-3-pro-preview"}
LIMIT=${LIMIT:-20}
SELECTION=${SELECTION:-balanced}
INPUT_MODE=${INPUT_MODE:-frames}
NUM_FRAMES=${NUM_FRAMES:-32}
IMAGE_MAX_SIDE=${IMAGE_MAX_SIDE:-640}
JPEG_QUALITY=${JPEG_QUALITY:-78}
MAX_TOKENS=${MAX_TOKENS:-96}
TEMPERATURE=${TEMPERATURE:-0.1}
TOP_P=${TOP_P:-1.0}
REASONING_EFFORT=${REASONING_EFFORT:-minimal}
SLEEP_SEC=${SLEEP_SEC:-0.2}

mkdir -p "$OUT_ROOT"

python JSL/inspect_ejsl_frame_fps.py \
  --dial_list "$DIAL_LIST" \
  --frame_root "$FRAME_ROOT" \
  --video_root "$VIDEO_ROOT" \
  --limit "$LIMIT" \
  --out_csv "$OUT_ROOT/ejsl_frame_fps_rows.csv" \
  --summary_json "$OUT_ROOT/ejsl_frame_fps_summary.json"

MODEL_FLAGS=()
for model in $MODELS; do
  MODEL_FLAGS+=(--model "$model")
done

TXT_TREE_FLAGS=()
if [ "$WRITE_TXT_TREE" = "1" ]; then
  TXT_TREE_FLAGS+=(--write_txt_tree --output_txt_root "$OUTPUT_TXT_ROOT")
fi

python JSL/openrouter_ejsl_video_text.py \
  --dial_list "$DIAL_LIST" \
  --frame_root "$FRAME_ROOT" \
  --video_root "$VIDEO_ROOT" \
  --out_root "$OUT_ROOT" \
  "${MODEL_FLAGS[@]}" \
  --limit "$LIMIT" \
  --selection "$SELECTION" \
  --input_mode "$INPUT_MODE" \
  --num_frames "$NUM_FRAMES" \
  --image_max_side "$IMAGE_MAX_SIDE" \
  --jpeg_quality "$JPEG_QUALITY" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --reasoning_effort "$REASONING_EFFORT" \
  --structure_txt_root "$STRUCTURE_TXT_ROOT" \
  "${TXT_TREE_FLAGS[@]}" \
  --sleep_sec "$SLEEP_SEC"

echo "[OpenRouter-eJSL] out root: $OUT_ROOT"
echo "[OpenRouter-eJSL] cost summary: $OUT_ROOT/cost_summary.json"
echo "[OpenRouter-eJSL] reviews: $OUT_ROOT/reviews"
if [ "$WRITE_TXT_TREE" = "1" ]; then
  echo "[OpenRouter-eJSL] replacement txt root(s): $OUTPUT_TXT_ROOT"
fi
