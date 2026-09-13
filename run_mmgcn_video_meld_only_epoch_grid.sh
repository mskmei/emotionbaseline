#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl}
UNIFIED_ROOT=${UNIFIED_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}
OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/video_meld_only_epoch_grid"}

MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
export TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ejsl_txt_en_openai}

MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$UNIFIED_ROOT/meld_anjs4_unified.pkl"}
EJSL_UNIFIED_PKL=${EJSL_UNIFIED_PKL:-"$UNIFIED_ROOT/ejsl_anjs4_unified.pkl"}
MELD_EJSL_FEATURE_CACHE_DIR=${MELD_EJSL_FEATURE_CACHE_DIR:-"$UNIFIED_ROOT/feature_cache"}

SEEDS=${SEEDS:-"35 36 37 41 42"}
MAX_EPOCHS=${MAX_EPOCHS:-50}
EPOCH_GRID=${EPOCH_GRID:-"1 5 10 15 20 25 30 35 40 45 50"}
GRAPH_TYPE=${GRAPH_TYPE:-MMGCN}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-0.0003}
L2=${L2:-0.00003}
DROPOUT=${DROPOUT:-0.40}
LOSS=${LOSS:-focal}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-5.0}
SELECTION_SPLIT=${SELECTION_SPLIT:-source}
SELECTION_METRIC=${SELECTION_METRIC:-weighted_f1}
SAVE_EPOCH_EVERY=${SAVE_EPOCH_EVERY:-0}
REBUILD_MELD_EJSL_FEATURES=${REBUILD_MELD_EJSL_FEATURES:-0}
RESUME=${RESUME:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}

mkdir -p "$OUT_ROOT"

if [ ! -d "$TRANSLATED_TXT_ROOT" ]; then
  echo "[MMGCN-MELD-ONLY-GRID] missing translated eJSL txt root: $TRANSLATED_TXT_ROOT" >&2
  exit 1
fi

run_logged() {
  local log_path="$1"
  shift
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$@" | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e
  if [ "$status" -ne 0 ]; then
    echo "[MMGCN-MELD-ONLY-GRID] command failed with status=$status" | tee "$log_path.failed"
    if [ "$CONTINUE_ON_ERROR" = "1" ]; then
      echo "[MMGCN-MELD-ONLY-GRID] CONTINUE_ON_ERROR=1, continuing"
      return 0
    fi
    exit "$status"
  fi
}

write_config() {
  local path="$1"
  shift
  python3 - "$path" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = {}
for item in sys.argv[2:]:
    key, sep, value = item.partition("=")
    if sep:
        data[key] = value
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

max_finished_epoch() {
  local metrics_csv="$1"
  if [ ! -f "$metrics_csv" ]; then
    echo 0
    return
  fi
  python3 - "$metrics_csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_epoch = 0
with path.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        try:
            max_epoch = max(max_epoch, int(float(row.get("epoch") or 0)))
        except ValueError:
            pass
print(max_epoch)
PY
}

if [ "$REBUILD_MELD_EJSL_FEATURES" = "1" ] || [ ! -f "$MELD_UNIFIED_PKL" ] || [ ! -f "$EJSL_UNIFIED_PKL" ]; then
  echo "[MMGCN-MELD-ONLY-GRID] build same-origin MELD/eJSL features from translated eJSL text"
  run_logged "$OUT_ROOT/build_meld_ejsl_features.log" python MMGCN/build_unified_meld_ejsl_pkl.py \
    --meld_root "$MELD_RAW_ROOT" \
    --ejsl_txt_root "$TRANSLATED_TXT_ROOT" \
    --ejsl_dial_list "$DIAL_LIST" \
    --ejsl_frame_root "$FRAME_ROOT" \
    --ejsl_mp4_root "$MP4_ROOT" \
    --out_meld_pkl "$MELD_UNIFIED_PKL" \
    --out_ejsl_pkl "$EJSL_UNIFIED_PKL" \
    --cache_dir "$MELD_EJSL_FEATURE_CACHE_DIR" \
    --fp16
fi

for required in "$MELD_UNIFIED_PKL" "$EJSL_UNIFIED_PKL"; do
  if [ ! -f "$required" ]; then
    echo "[MMGCN-MELD-ONLY-GRID] missing required file: $required" >&2
    exit 1
  fi
done

for seed in $SEEDS; do
  trial="meld_only_seed${seed}"
  trial_out="$OUT_ROOT/$trial"
  metrics_csv="$trial_out/video/mmgcn_unified_video_epoch_metrics.csv"
  finished_epoch="$(max_finished_epoch "$metrics_csv")"

  if [ "$RESUME" = "1" ] && [ "$finished_epoch" -ge "$MAX_EPOCHS" ]; then
    echo "[MMGCN-MELD-ONLY-GRID][$trial] skip existing metrics through epoch $finished_epoch"
    continue
  fi

  mkdir -p "$trial_out"
  write_config "$trial_out/config.json" \
    "trial=$trial" \
    "seed=$seed" \
    "max_epochs=$MAX_EPOCHS" \
    "epoch_grid=$EPOCH_GRID" \
    "train_pkl=$MELD_UNIFIED_PKL" \
    "external_test_pkl=$EJSL_UNIFIED_PKL" \
    "lr=$LR" \
    "l2=$L2" \
    "dropout=$DROPOUT" \
    "loss=$LOSS" \
    "focal_gamma=$FOCAL_GAMMA" \
    "batch_size=$BATCH_SIZE" \
    "selection_split=$SELECTION_SPLIT" \
    "selection_metric=$SELECTION_METRIC"

  echo "[MMGCN-MELD-ONLY-GRID][$trial] train to MAX_EPOCHS=$MAX_EPOCHS; grid=$EPOCH_GRID"
  run_logged "$trial_out/train.log" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$MELD_UNIFIED_PKL" \
    --external_test_pkl "$EJSL_UNIFIED_PKL" \
    --out_dir "$trial_out" \
    --modalities video \
    --graph_type "$GRAPH_TYPE" \
    --epochs "$MAX_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --l2 "$L2" \
    --dropout "$DROPOUT" \
    --loss "$LOSS" \
    --focal_gamma "$FOCAL_GAMMA" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --selection_split "$SELECTION_SPLIT" \
    --selection_metric "$SELECTION_METRIC" \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    --seed "$seed"
done

python3 MMGCN/summarize_meld_only_epoch_grid.py \
  --root "$OUT_ROOT" \
  --epochs "$EPOCH_GRID" \
  --out_seed_csv "$OUT_ROOT/meld_only_epoch_grid_seed_rows.csv" \
  --out_group_csv "$OUT_ROOT/meld_only_epoch_grid_summary.csv" \
  --out_txt "$OUT_ROOT/meld_only_epoch_grid_summary.txt" \
  | tee "$OUT_ROOT/meld_only_epoch_grid_summary.log"

echo "[MMGCN-MELD-ONLY-GRID] done: $OUT_ROOT"
echo "[MMGCN-MELD-ONLY-GRID] summary: $OUT_ROOT/meld_only_epoch_grid_summary.txt"
echo "[MMGCN-MELD-ONLY-GRID] seed rows: $OUT_ROOT/meld_only_epoch_grid_seed_rows.csv"
