#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

: "${JSHUWA_VIDEO_DIR:?Set JSHUWA_VIDEO_DIR to local YouTube videos named by yid.}"
: "${JSHUWA_SUBTITLE_TEXT:?Set JSHUWA_SUBTITLE_TEXT to JSONL/CSV containing Japanese text for J-Shuwa rows.}"
: "${JSL_WORK_DIR:?Set JSL_WORK_DIR to a writable work directory for manifests/keypoints/model.}"

JSHUWA_MANIFEST=${JSHUWA_MANIFEST:-"$JSL_WORK_DIR/jshuwa_train_manifest.csv"}
JSHUWA_KEYPOINT_DIR=${JSHUWA_KEYPOINT_DIR:-"$JSL_WORK_DIR/keypoints"}
JSHUWA_KEYPOINT_MANIFEST=${JSHUWA_KEYPOINT_MANIFEST:-"$JSL_WORK_DIR/jshuwa_train_keypoints.csv"}
JSL_MODEL_DIR=${JSL_MODEL_DIR:-"$JSL_WORK_DIR/qwen3_jsl_lora"}

SOURCE=${SOURCE:-all}
SAMPLE_FPS=${SAMPLE_FPS:-10}
MAX_FRAMES=${MAX_FRAMES:-0}
MODEL_COMPLEXITY=${MODEL_COMPLEXITY:-1}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-1.7B}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LR=${LR:-2e-4}
NUM_VISUAL_TOKENS=${NUM_VISUAL_TOKENS:-64}
MAX_TARGET_TOKENS=${MAX_TARGET_TOKENS:-128}
GPU=${GPU:-0}

mkdir -p "$JSL_WORK_DIR"

python JSL/prepare_jshuwa_manifest.py \
  --video_dir "$JSHUWA_VIDEO_DIR" \
  --subtitle_text "$JSHUWA_SUBTITLE_TEXT" \
  --source "$SOURCE" \
  --out_csv "$JSHUWA_MANIFEST"

python JSL/extract_mediapipe_keypoints.py \
  --manifest_csv "$JSHUWA_MANIFEST" \
  --output_dir "$JSHUWA_KEYPOINT_DIR" \
  --out_manifest_csv "$JSHUWA_KEYPOINT_MANIFEST" \
  --sample_fps "$SAMPLE_FPS" \
  --max_frames "$MAX_FRAMES" \
  --model_complexity "$MODEL_COMPLEXITY" \
  --resume

CUDA_VISIBLE_DEVICES="$GPU" python JSL/train_jsl_translation.py \
  --manifest_csv "$JSHUWA_KEYPOINT_MANIFEST" \
  --output_dir "$JSL_MODEL_DIR" \
  --base_model "$BASE_MODEL" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --lr "$LR" \
  --num_visual_tokens "$NUM_VISUAL_TOKENS" \
  --max_target_tokens "$MAX_TARGET_TOKENS" \
  --bf16

echo "[JSL] saved model: $JSL_MODEL_DIR"
