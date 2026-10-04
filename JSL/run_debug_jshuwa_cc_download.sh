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
OUT_CSV=${OUT_CSV:-"$JSL_WORK_DIR/manifests/jshuwa_cc_download_debug_manifest.csv"}

MAX_YIDS=${MAX_YIDS:-20}
MAX_ROWS=${MAX_ROWS:-100}
YTDLP_EXTRACTOR_ARGS=${YTDLP_EXTRACTOR_ARGS:-youtube:player_client=android_vr}
YTDLP_EXTRACTOR_ARGS_CANDIDATES=${YTDLP_EXTRACTOR_ARGS_CANDIDATES:-"youtube:player_client=android_vr;youtube:player_client=android;youtube:player_client=ios;youtube:player_client=web;youtube:player_client=mweb"}
YTDLP_VIDEO_FORMAT=${YTDLP_VIDEO_FORMAT:-18/best[height<=360][ext=mp4]/best[height<=480][ext=mp4]/best}
YTDLP_COOKIES=${YTDLP_COOKIES:-}
YTDLP_COOKIES_FROM_BROWSER=${YTDLP_COOKIES_FROM_BROWSER:-}
YTDLP_VERBOSE=${YTDLP_VERBOSE:-1}

mkdir -p "$JSL_WORK_DIR/manifests" "$JSHUWA_VIDEO_DIR" "$JSHUWA_SUBTITLE_DIR"

if [ ! -s "$JSHUWA_METADATA_CSV" ]; then
  python JSL/download_jshuwa_metadata.py \
    --out_csv "$JSHUWA_METADATA_CSV"
fi

YTDLP_FLAGS=()
if [ -n "$YTDLP_COOKIES" ]; then
  YTDLP_FLAGS+=(--cookies "$YTDLP_COOKIES")
fi
if [ -n "$YTDLP_COOKIES_FROM_BROWSER" ]; then
  YTDLP_FLAGS+=(--cookies_from_browser "$YTDLP_COOKIES_FROM_BROWSER")
fi
if [ "$YTDLP_VERBOSE" = "1" ]; then
  YTDLP_FLAGS+=(--verbose_ytdlp)
fi

python JSL/build_jshuwa_cc_manifest.py \
  --metadata_csv "$JSHUWA_METADATA_CSV" \
  --video_dir "$JSHUWA_VIDEO_DIR" \
  --subtitle_dir "$JSHUWA_SUBTITLE_DIR" \
  --out_csv "$OUT_CSV" \
  --source cc \
  --video_format "$YTDLP_VIDEO_FORMAT" \
  --extractor_args "$YTDLP_EXTRACTOR_ARGS" \
  --extractor_args_candidates "$YTDLP_EXTRACTOR_ARGS_CANDIDATES" \
  --download_videos \
  --download_subtitles \
  --skip_missing \
  --max_yids "$MAX_YIDS" \
  --max_rows "$MAX_ROWS" \
  "${YTDLP_FLAGS[@]}"

echo "[J-Shuwa-debug] manifest: $OUT_CSV"
echo "[J-Shuwa-debug] video dir: $JSHUWA_VIDEO_DIR"
echo "[J-Shuwa-debug] subtitle dir: $JSHUWA_SUBTITLE_DIR"
