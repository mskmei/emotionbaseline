#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

DATA_ENV=${DATA_ENV:-telme39}
JSL_WORK_DIR=${JSL_WORK_DIR:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle_qwen3_4b_full_clean}

HF_HOME=${HF_HOME:-"$JSL_WORK_DIR/hf_home"}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$JSL_WORK_DIR/hf_datasets_cache"}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$JSL_WORK_DIR/hf_models"}
export HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE

JSHUWA_METADATA_CSV=${JSHUWA_METADATA_CSV:-"$JSL_WORK_DIR/manifests/jshuwa_metadata_train.csv"}
JSHUWA_VIDEO_DIR=${JSHUWA_VIDEO_DIR:-"$JSL_WORK_DIR/jshuwa_youtube_videos"}

CC_CLEAN_MANIFEST=${CC_CLEAN_MANIFEST:-"$JSL_WORK_DIR/manifests/jshuwa_cc_all_clean_manifest.csv"}
HARDSUB_JSONL=${HARDSUB_JSONL:-"$JSL_WORK_DIR/manifests/jshuwa_hardsub_qwen3vl32b_subtitles.jsonl"}
HARDSUB_RAW_MANIFEST=${HARDSUB_RAW_MANIFEST:-"$JSL_WORK_DIR/manifests/jshuwa_hardsub_qwen3vl32b_raw_manifest.csv"}
HARDSUB_CLEAN_MANIFEST=${HARDSUB_CLEAN_MANIFEST:-"$JSL_WORK_DIR/manifests/jshuwa_hardsub_qwen3vl32b_clean_manifest.csv"}
HARDSUB_CLEAN_REPORT=${HARDSUB_CLEAN_REPORT:-"$JSL_WORK_DIR/manifests/jshuwa_hardsub_qwen3vl32b_clean_report.json"}
COMBINED_CLEAN_MANIFEST=${COMBINED_CLEAN_MANIFEST:-"$JSL_WORK_DIR/manifests/jshuwa_cc_hardsub_qwen3vl32b_clean_manifest.csv"}

VLLM_API_BASE=${VLLM_API_BASE:-http://localhost:8000/v1}
QWEN3_VL_MODEL=${QWEN3_VL_MODEL:-Qwen/Qwen3-VL-32B-Thinking}
MAX_TOKENS=${MAX_TOKENS:-256}
CONTEXT_SEGMENTS=${CONTEXT_SEGMENTS:-1}
MAX_IMAGE_SIDE=${MAX_IMAGE_SIDE:-1280}
JPEG_QUALITY=${JPEG_QUALITY:-90}
REQUEST_RETRIES=${REQUEST_RETRIES:-3}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-180}
SKIP_ERRORS=${SKIP_ERRORS:-0}

YTDLP_EXTRACTOR_ARGS=${YTDLP_EXTRACTOR_ARGS:-youtube:player_client=android_vr}
YTDLP_EXTRACTOR_ARGS_CANDIDATES=${YTDLP_EXTRACTOR_ARGS_CANDIDATES:-"youtube:player_client=android_vr;youtube:player_client=android;youtube:player_client=ios;youtube:player_client=web;youtube:player_client=mweb"}
YTDLP_VIDEO_FORMAT=${YTDLP_VIDEO_FORMAT:-18/best[height<=360][ext=mp4]/best[height<=480][ext=mp4]/best}
YTDLP_COOKIES=${YTDLP_COOKIES:-}
YTDLP_COOKIES_FROM_BROWSER=${YTDLP_COOKIES_FROM_BROWSER:-}
MAX_YIDS=${MAX_YIDS:-0}
MAX_ROWS=${MAX_ROWS:-0}

CLEAN_MIN_DURATION=${CLEAN_MIN_DURATION:-0.6}
CLEAN_MAX_DURATION=${CLEAN_MAX_DURATION:-20.0}
CLEAN_MIN_TEXT_CHARS=${CLEAN_MIN_TEXT_CHARS:-4}
CLEAN_MAX_TEXT_CHARS=${CLEAN_MAX_TEXT_CHARS:-80}
CLEAN_MAX_SAME_VIDEO_DUP=${CLEAN_MAX_SAME_VIDEO_DUP:-1}
CLEAN_MAX_DUP=${CLEAN_MAX_DUP:-10}
CLEAN_MAX_SHORT_DUP=${CLEAN_MAX_SHORT_DUP:-3}
CLEAN_DROP_META_TERMS=${CLEAN_DROP_META_TERMS:-1}

mkdir -p "$JSL_WORK_DIR/manifests"

run_data_py() {
  conda run -n "$DATA_ENV" python "$@"
}

YTDLP_FLAGS=()
if [ -n "$YTDLP_COOKIES" ]; then
  YTDLP_FLAGS+=(--cookies "$YTDLP_COOKIES")
fi
if [ -n "$YTDLP_COOKIES_FROM_BROWSER" ]; then
  YTDLP_FLAGS+=(--cookies_from_browser "$YTDLP_COOKIES_FROM_BROWSER")
fi

ERROR_FLAGS=()
if [ "$SKIP_ERRORS" = "1" ]; then
  ERROR_FLAGS+=(--skip_errors)
fi

echo "[J-Shuwa-hardsub] work dir=$JSL_WORK_DIR"
echo "[J-Shuwa-hardsub] VLM endpoint=$VLLM_API_BASE model=$QWEN3_VL_MODEL"
echo "[J-Shuwa-hardsub] reuse CC clean manifest=$CC_CLEAN_MANIFEST"

if [ ! -s "$JSHUWA_METADATA_CSV" ]; then
  run_data_py JSL/download_jshuwa_metadata.py \
    --out_csv "$JSHUWA_METADATA_CSV"
fi

run_data_py JSL/extract_jshuwa_hardsub_subtitles_vlm.py \
  --metadata_csv "$JSHUWA_METADATA_CSV" \
  --video_dir "$JSHUWA_VIDEO_DIR" \
  --out_jsonl "$HARDSUB_JSONL" \
  --out_csv "$HARDSUB_RAW_MANIFEST" \
  --api_base "$VLLM_API_BASE" \
  --model "$QWEN3_VL_MODEL" \
  --max_tokens "$MAX_TOKENS" \
  --timeout "$REQUEST_TIMEOUT" \
  --request_retries "$REQUEST_RETRIES" \
  --context_segments "$CONTEXT_SEGMENTS" \
  --max_image_side "$MAX_IMAGE_SIDE" \
  --jpeg_quality "$JPEG_QUALITY" \
  --video_format "$YTDLP_VIDEO_FORMAT" \
  --extractor_args "$YTDLP_EXTRACTOR_ARGS" \
  --extractor_args_candidates "$YTDLP_EXTRACTOR_ARGS_CANDIDATES" \
  --download_videos \
  --skip_missing \
  --resume \
  --max_yids "$MAX_YIDS" \
  --max_rows "$MAX_ROWS" \
  "${ERROR_FLAGS[@]}" \
  "${YTDLP_FLAGS[@]}"

CLEAN_FLAGS=()
if [ "$CLEAN_DROP_META_TERMS" = "1" ]; then
  CLEAN_FLAGS+=(--drop_meta_terms)
fi

run_data_py JSL/clean_jshuwa_manifest.py \
  --in_csv "$HARDSUB_RAW_MANIFEST" \
  --out_csv "$HARDSUB_CLEAN_MANIFEST" \
  --report_json "$HARDSUB_CLEAN_REPORT" \
  --min_duration "$CLEAN_MIN_DURATION" \
  --max_duration "$CLEAN_MAX_DURATION" \
  --min_text_chars "$CLEAN_MIN_TEXT_CHARS" \
  --max_text_chars "$CLEAN_MAX_TEXT_CHARS" \
  --max_same_video_duplicate "$CLEAN_MAX_SAME_VIDEO_DUP" \
  --max_duplicate_text "$CLEAN_MAX_DUP" \
  --max_short_duplicate_text "$CLEAN_MAX_SHORT_DUP" \
  "${CLEAN_FLAGS[@]}"

if [ ! -s "$CC_CLEAN_MANIFEST" ]; then
  echo "[J-Shuwa-hardsub] missing CC clean manifest: $CC_CLEAN_MANIFEST" >&2
  echo "[J-Shuwa-hardsub] hardsub clean manifest is still ready: $HARDSUB_CLEAN_MANIFEST" >&2
  exit 0
fi

run_data_py JSL/combine_jshuwa_manifests.py \
  --in_csv "$CC_CLEAN_MANIFEST" \
  --in_csv "$HARDSUB_CLEAN_MANIFEST" \
  --out_csv "$COMBINED_CLEAN_MANIFEST" \
  --dedupe sample_id

echo "[J-Shuwa-hardsub] done"
echo "[J-Shuwa-hardsub] hardsub jsonl: $HARDSUB_JSONL"
echo "[J-Shuwa-hardsub] hardsub raw manifest: $HARDSUB_RAW_MANIFEST"
echo "[J-Shuwa-hardsub] hardsub clean manifest: $HARDSUB_CLEAN_MANIFEST"
echo "[J-Shuwa-hardsub] combined CC+hardsub clean manifest: $COMBINED_CLEAN_MANIFEST"
