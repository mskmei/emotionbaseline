#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

JSL_WORK_DIR=${JSL_WORK_DIR:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle}
HF_HOME=${HF_HOME:-"$JSL_WORK_DIR/hf_home"}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$JSL_WORK_DIR/hf_datasets_cache"}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$JSL_WORK_DIR/hf_models"}
export HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE

JSHUWA_METADATA_CSV=${JSHUWA_METADATA_CSV:-"$JSL_WORK_DIR/manifests/jshuwa_metadata_train.csv"}
JSHUWA_VIDEO_DIR=${JSHUWA_VIDEO_DIR:-"$JSL_WORK_DIR/jshuwa_youtube_videos"}
JSHUWA_SUBTITLE_DIR=${JSHUWA_SUBTITLE_DIR:-"$JSL_WORK_DIR/jshuwa_youtube_subtitles"}
JSHUWA_MANIFEST=${JSHUWA_MANIFEST:-"$JSL_WORK_DIR/manifests/jshuwa_cc_train_manifest.csv"}
JSHUWA_KEYPOINT_DIR=${JSHUWA_KEYPOINT_DIR:-"$JSL_WORK_DIR/keypoints/jshuwa_cc"}
JSHUWA_KEYPOINT_MANIFEST=${JSHUWA_KEYPOINT_MANIFEST:-"$JSL_WORK_DIR/manifests/jshuwa_cc_train_keypoints.csv"}
JSL_MODEL_DIR=${JSL_MODEL_DIR:-"$JSL_WORK_DIR/models/qwen3_jsl_lora_cc"}
OUTPUT_TXT_ROOT=${OUTPUT_TXT_ROOT:-"$JSL_WORK_DIR/ejsl_nonoracle_txt"}
EJSL_KEYPOINT_CACHE_DIR=${EJSL_KEYPOINT_CACHE_DIR:-"$JSL_WORK_DIR/keypoints/ejsl_1920"}
PREDICTIONS_JSONL=${PREDICTIONS_JSONL:-"$JSL_WORK_DIR/ejsl_nonoracle_predictions.jsonl"}

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
EJSL_VIDEO_ROOT=${EJSL_VIDEO_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
EJSL_FRAME_ROOT=${EJSL_FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
STRUCTURE_TXT_ROOT=${STRUCTURE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-1.7B}
GPU=${GPU:-0}
SAMPLE_FPS=${SAMPLE_FPS:-10}
MAX_FRAMES=${MAX_FRAMES:-0}
MODEL_COMPLEXITY=${MODEL_COMPLEXITY:-1}
NUM_VISUAL_TOKENS=${NUM_VISUAL_TOKENS:-64}
MAX_TARGET_TOKENS=${MAX_TARGET_TOKENS:-128}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LR=${LR:-2e-4}
MAX_YIDS=${MAX_YIDS:-0}
MAX_ROWS=${MAX_ROWS:-0}
EJSL_BATCH_SIZE=${EJSL_BATCH_SIZE:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}

mkdir -p "$JSL_WORK_DIR" "$JSL_WORK_DIR/manifests"

python JSL/download_jshuwa_metadata.py \
  --out_csv "$JSHUWA_METADATA_CSV"

python JSL/build_jshuwa_cc_manifest.py \
  --metadata_csv "$JSHUWA_METADATA_CSV" \
  --video_dir "$JSHUWA_VIDEO_DIR" \
  --subtitle_dir "$JSHUWA_SUBTITLE_DIR" \
  --out_csv "$JSHUWA_MANIFEST" \
  --source cc \
  --download_videos \
  --download_subtitles \
  --skip_missing \
  --max_yids "$MAX_YIDS" \
  --max_rows "$MAX_ROWS"

python JSL/extract_mediapipe_keypoints.py \
  --manifest_csv "$JSHUWA_MANIFEST" \
  --output_dir "$JSHUWA_KEYPOINT_DIR" \
  --out_manifest_csv "$JSHUWA_KEYPOINT_MANIFEST" \
  --sample_fps "$SAMPLE_FPS" \
  --max_frames "$MAX_FRAMES" \
  --model_complexity "$MODEL_COMPLEXITY" \
  --resume \
  --skip_errors

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

CUDA_VISIBLE_DEVICES="$GPU" python JSL/generate_ejsl_non_oracle_txt.py \
  --model_dir "$JSL_MODEL_DIR" \
  --dial_list "$DIAL_LIST" \
  --video_root "$EJSL_VIDEO_ROOT" \
  --frame_root "$EJSL_FRAME_ROOT" \
  --structure_txt_root "$STRUCTURE_TXT_ROOT" \
  --output_txt_root "$OUTPUT_TXT_ROOT" \
  --keypoint_cache_dir "$EJSL_KEYPOINT_CACHE_DIR" \
  --predictions_jsonl "$PREDICTIONS_JSONL" \
  --batch_size "$EJSL_BATCH_SIZE" \
  --num_visual_tokens "$NUM_VISUAL_TOKENS" \
  --sample_fps "$SAMPLE_FPS" \
  --max_frames "$MAX_FRAMES" \
  --model_complexity "$MODEL_COMPLEXITY" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --resume

echo "[JSL] done"
echo "[JSL] model: $JSL_MODEL_DIR"
echo "[JSL] non-oracle txt root: $OUTPUT_TXT_ROOT"
