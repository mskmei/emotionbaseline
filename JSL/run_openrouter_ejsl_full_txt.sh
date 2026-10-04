#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before calling OpenRouter.}"

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
VIDEO_ROOT=${VIDEO_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
STRUCTURE_TXT_ROOT=${STRUCTURE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}

OUT_ROOT=${OUT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/openrouter_ejsl_full}
OUTPUT_TXT_ROOT=${OUTPUT_TXT_ROOT:-"$OUT_ROOT/txt_roots"}

MODELS=${MODELS:-"google/gemini-3.5-flash"}
LIMIT=${LIMIT:-0}
INPUT_MODE=${INPUT_MODE:-frames}
NUM_FRAMES=${NUM_FRAMES:-32}
IMAGE_MAX_SIDE=${IMAGE_MAX_SIDE:-640}
JPEG_QUALITY=${JPEG_QUALITY:-78}
MAX_TOKENS=${MAX_TOKENS:-96}
TEMPERATURE=${TEMPERATURE:-0.1}
TOP_P=${TOP_P:-1.0}
REASONING_EFFORT=${REASONING_EFFORT:-minimal}
SLEEP_SEC=${SLEEP_SEC:-0.2}

mkdir -p "$OUT_ROOT" "$OUTPUT_TXT_ROOT"

MODEL_FLAGS=()
for model in $MODELS; do
  MODEL_FLAGS+=(--model "$model")
done

python JSL/openrouter_ejsl_video_text.py \
  --dial_list "$DIAL_LIST" \
  --frame_root "$FRAME_ROOT" \
  --video_root "$VIDEO_ROOT" \
  --structure_txt_root "$STRUCTURE_TXT_ROOT" \
  --out_root "$OUT_ROOT" \
  --output_txt_root "$OUTPUT_TXT_ROOT" \
  --write_txt_tree \
  "${MODEL_FLAGS[@]}" \
  --limit "$LIMIT" \
  --selection first \
  --input_mode "$INPUT_MODE" \
  --num_frames "$NUM_FRAMES" \
  --image_max_side "$IMAGE_MAX_SIDE" \
  --jpeg_quality "$JPEG_QUALITY" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --reasoning_effort "$REASONING_EFFORT" \
  --sleep_sec "$SLEEP_SEC"

echo "[OpenRouter-eJSL-full] out root: $OUT_ROOT"
echo "[OpenRouter-eJSL-full] cost summary: $OUT_ROOT/cost_summary.json"
echo "[OpenRouter-eJSL-full] replacement txt root(s): $OUTPUT_TXT_ROOT"
