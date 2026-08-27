#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
TXT_ROOT=${TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
SAVE_MODEL_ROOT=${SAVE_MODEL_ROOT:-/raid_zoe/home/lr/maokeyu/sign/open/IEMOCAP/save_model}
SAVE_DIR=${SAVE_DIR:-./IEMOCAP/outputs_ejsl_ablation_tv}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}

mkdir -p "$SAVE_DIR"

for MOD in text video; do
  PREFIX="telme_ejsl_${MOD}_only"
  LOG_PATH="$SAVE_DIR/${PREFIX}.log"
  echo "[TELME][$MOD] start"
  python IEMOCAP/inference_ejsl_frame.py \
    --frame_root "$FRAME_ROOT" \
    --txt_root "$TXT_ROOT" \
    --save_model_root "$SAVE_MODEL_ROOT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --save_dir "$SAVE_DIR" \
    --report_prefix "$PREFIX" \
    --eval_modality "$MOD" \
    --save_predictions | tee "$LOG_PATH"
done

echo "[TELME] summary"
for MOD in text video; do
  PREFIX="telme_ejsl_${MOD}_only"
  REPORT_PATH="$SAVE_DIR/${PREFIX}_classification_report.txt"
  echo "===== ${MOD} ====="
  if [ -f "$REPORT_PATH" ]; then
    sed -n '1,20p' "$REPORT_PATH"
  else
    echo "missing report: $REPORT_PATH"
  fi
done
