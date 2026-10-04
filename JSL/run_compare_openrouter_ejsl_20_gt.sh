#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

REVIEW_CSV=${REVIEW_CSV:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/openrouter_ejsl_20/reviews/review_google__gemini-3.5-flash.csv}
STRUCTURE_TXT_ROOT=${STRUCTURE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
LIMIT=${LIMIT:-0}
SHOW_META=${SHOW_META:-1}

ARGS=()
if [ "$LIMIT" != "0" ]; then
  ARGS+=(--limit "$LIMIT")
fi
if [ "$SHOW_META" = "1" ]; then
  ARGS+=(--show_meta)
fi

python JSL/compare_openrouter_review_with_gt.py \
  --review_csv "$REVIEW_CSV" \
  --structure_txt_root "$STRUCTURE_TXT_ROOT" \
  "${ARGS[@]}"
