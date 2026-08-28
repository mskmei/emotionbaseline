#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

: "${JSL_MODEL_DIR:?Set JSL_MODEL_DIR to the fine-tuned JSL model directory.}"
: "${OUTPUT_TXT_ROOT:?Set OUTPUT_TXT_ROOT to your writable non-oracle eJSL txt root.}"

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
VIDEO_ROOT=${VIDEO_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
STRUCTURE_TXT_ROOT=${STRUCTURE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
KEYPOINT_CACHE_DIR=${KEYPOINT_CACHE_DIR:-"$OUTPUT_TXT_ROOT/.keypoints"}
PREDICTIONS_JSONL=${PREDICTIONS_JSONL:-"$OUTPUT_TXT_ROOT/non_oracle_predictions.jsonl"}

GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_VISUAL_TOKENS=${NUM_VISUAL_TOKENS:-64}
SAMPLE_FPS=${SAMPLE_FPS:-10}
MAX_FRAMES=${MAX_FRAMES:-0}
MODEL_COMPLEXITY=${MODEL_COMPLEXITY:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}

CUDA_VISIBLE_DEVICES="$GPU" python JSL/generate_ejsl_non_oracle_txt.py \
  --model_dir "$JSL_MODEL_DIR" \
  --dial_list "$DIAL_LIST" \
  --video_root "$VIDEO_ROOT" \
  --frame_root "$FRAME_ROOT" \
  --structure_txt_root "$STRUCTURE_TXT_ROOT" \
  --output_txt_root "$OUTPUT_TXT_ROOT" \
  --keypoint_cache_dir "$KEYPOINT_CACHE_DIR" \
  --predictions_jsonl "$PREDICTIONS_JSONL" \
  --batch_size "$BATCH_SIZE" \
  --num_visual_tokens "$NUM_VISUAL_TOKENS" \
  --sample_fps "$SAMPLE_FPS" \
  --max_frames "$MAX_FRAMES" \
  --model_complexity "$MODEL_COMPLEXITY" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --resume

echo "[JSL] non-oracle txt root: $OUTPUT_TXT_ROOT"
