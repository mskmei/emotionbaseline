#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
VIDEO_ROOT=${VIDEO_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
OUT_ROOT=${OUT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/ejsl_frame_fps}
LIMIT=${LIMIT:-0}

mkdir -p "$OUT_ROOT"

python JSL/inspect_ejsl_frame_fps.py \
  --dial_list "$DIAL_LIST" \
  --frame_root "$FRAME_ROOT" \
  --video_root "$VIDEO_ROOT" \
  --limit "$LIMIT" \
  --out_csv "$OUT_ROOT/ejsl_frame_fps_rows.csv" \
  --summary_json "$OUT_ROOT/ejsl_frame_fps_summary.json"

echo "[eJSL-FPS] rows: $OUT_ROOT/ejsl_frame_fps_rows.csv"
echo "[eJSL-FPS] summary: $OUT_ROOT/ejsl_frame_fps_summary.json"
