#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl}
UNIFIED_ROOT=${UNIFIED_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}
OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/video_joint_bobsl_meld"}
JOINT_PKL_ROOT=${JOINT_PKL_ROOT:-"$OUT_ROOT/joint_pkls"}

BOBSL_ROOT=${BOBSL_ROOT:-/raid_zoe/home/lr/wangyi/sign/bobsl}
MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
export TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ejsl_txt_en_openai}

BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-"$WORK_ROOT/bobsl_anjs4_train_val.pkl"}
BOBSL_TEST_PKL=${BOBSL_TEST_PKL:-"$WORK_ROOT/bobsl_anjs4_test.pkl"}
BOBSL_FEATURE_CACHE_DIR=${BOBSL_FEATURE_CACHE_DIR:-"$WORK_ROOT/bobsl_feature_cache"}
MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$UNIFIED_ROOT/meld_anjs4_unified.pkl"}
EJSL_UNIFIED_PKL=${EJSL_UNIFIED_PKL:-"$UNIFIED_ROOT/ejsl_anjs4_unified.pkl"}
MELD_EJSL_FEATURE_CACHE_DIR=${MELD_EJSL_FEATURE_CACHE_DIR:-"$UNIFIED_ROOT/feature_cache"}

SEEDS=${SEEDS:-"35 36 37 41 42"}
CONFIGS=${CONFIGS:-"meld_only joint_b2k joint_b5k joint_b10k joint_b5k_lr100 joint_b5k_drop20"}
FT_EPOCHS=${FT_EPOCHS:-15}
FT_GRAPH_TYPE=${FT_GRAPH_TYPE:-MMGCN}
BOBSL_SAMPLE_SEED=${BOBSL_SAMPLE_SEED:-123}
INCLUDE_BOBSL_VAL=${INCLUDE_BOBSL_VAL:-0}
RESUME=${RESUME:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
REBUILD_BOBSL_FEATURES=${REBUILD_BOBSL_FEATURES:-0}
REBUILD_MELD_EJSL_FEATURES=${REBUILD_MELD_EJSL_FEATURES:-0}
REBUILD_JOINT_PKL=${REBUILD_JOINT_PKL:-0}
SAVE_EPOCH_EVERY=${SAVE_EPOCH_EVERY:-0}
TOP_K=${TOP_K:-20}

mkdir -p "$OUT_ROOT" "$JOINT_PKL_ROOT"

if [ ! -d "$TRANSLATED_TXT_ROOT" ]; then
  echo "[MMGCN-JOINT] missing translated eJSL txt root: $TRANSLATED_TXT_ROOT" >&2
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
    echo "[MMGCN-JOINT] command failed with status=$status" | tee "$log_path.failed"
    if [ "$CONTINUE_ON_ERROR" = "1" ]; then
      echo "[MMGCN-JOINT] CONTINUE_ON_ERROR=1, continuing"
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

if [ "$REBUILD_BOBSL_FEATURES" = "1" ] || [ ! -f "$BOBSL_TRAIN_VAL_PKL" ] || [ ! -f "$BOBSL_TEST_PKL" ]; then
  echo "[MMGCN-JOINT] build BOBSL unified video features"
  run_logged "$OUT_ROOT/build_bobsl_features.log" python MMGCN/build_unified_bobsl_pkl.py \
    --bobsl_root "$BOBSL_ROOT" \
    --out_train_val_pkl "$BOBSL_TRAIN_VAL_PKL" \
    --out_test_pkl "$BOBSL_TEST_PKL" \
    --cache_dir "$BOBSL_FEATURE_CACHE_DIR" \
    --fp16
fi

if [ "$REBUILD_MELD_EJSL_FEATURES" = "1" ] || [ ! -f "$MELD_UNIFIED_PKL" ] || [ ! -f "$EJSL_UNIFIED_PKL" ]; then
  echo "[MMGCN-JOINT] build same-origin MELD/eJSL features from translated eJSL text"
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

for required in "$BOBSL_TRAIN_VAL_PKL" "$MELD_UNIFIED_PKL" "$EJSL_UNIFIED_PKL"; do
  if [ ! -f "$required" ]; then
    echo "[MMGCN-JOINT] missing required file: $required" >&2
    exit 1
  fi
done

joint_pkl_for() {
  local max_bobsl="$1"
  local suffix="trainonly"
  if [ "$INCLUDE_BOBSL_VAL" = "1" ]; then
    suffix="trainval"
  fi
  echo "$JOINT_PKL_ROOT/meld_bobsl${max_bobsl}_${suffix}_sample${BOBSL_SAMPLE_SEED}.pkl"
}

build_joint_if_needed() {
  local max_bobsl="$1"
  local out_pkl="$2"
  if [ "$REBUILD_JOINT_PKL" = "1" ] || [ ! -f "$out_pkl" ]; then
    echo "[MMGCN-JOINT] build joint pkl max_bobsl=$max_bobsl out=$out_pkl"
    local cmd=(
      python MMGCN/build_joint_meld_bobsl_video_pkl.py
      --meld_pkl "$MELD_UNIFIED_PKL"
      --bobsl_train_val_pkl "$BOBSL_TRAIN_VAL_PKL"
      --out_pkl "$out_pkl"
      --max_bobsl_train_dialogues "$max_bobsl"
      --bobsl_sample_seed "$BOBSL_SAMPLE_SEED"
      --force_bobsl_context1
    )
    if [ "$INCLUDE_BOBSL_VAL" = "1" ]; then
      cmd+=(--include_bobsl_val)
    fi
    run_logged "$out_pkl.build.log" "${cmd[@]}"
  else
    echo "[MMGCN-JOINT] skip existing joint pkl $out_pkl"
  fi
}

run_one() {
  local config="$1"
  local seed="$2"
  local train_pkl="$3"
  local bobsl_max="$4"
  local lr="$5"
  local dropout="$6"
  local loss="$7"
  local gamma="$8"
  shift 8
  local extra_args=("$@")
  local trial="${config}_seed${seed}"
  local out_dir="$OUT_ROOT/$trial"
  local best_summary="$out_dir/video/external_test_best_summary.json"
  if [ "$RESUME" = "1" ] && [ -f "$best_summary" ]; then
    echo "[MMGCN-JOINT] skip existing $best_summary"
    return
  fi
  mkdir -p "$out_dir"
  write_config "$out_dir/config.json" \
    "trial=$trial" \
    "config=$config" \
    "seed=$seed" \
    "joint_pkl=$train_pkl" \
    "bobsl_max=$bobsl_max" \
    "bobsl_include_val=$INCLUDE_BOBSL_VAL" \
    "lr=$lr" \
    "dropout=$dropout" \
    "loss=$loss" \
    "focal_gamma=$gamma" \
    "extra_args=${extra_args[*]}"
  echo "[MMGCN-JOINT][$trial] train_pkl=$train_pkl"
  run_logged "$out_dir/train.log" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$train_pkl" \
    --external_test_pkl "$EJSL_UNIFIED_PKL" \
    --out_dir "$out_dir" \
    --modalities video \
    --graph_type "$FT_GRAPH_TYPE" \
    --epochs "$FT_EPOCHS" \
    --batch_size 8 \
    --lr "$lr" \
    --l2 0.00003 \
    --dropout "$dropout" \
    --loss "$loss" \
    --focal_gamma "$gamma" \
    --max_grad_norm 5.0 \
    --selection_split source \
    --selection_metric weighted_f1 \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    --seed "$seed" \
    "${extra_args[@]}"
}

for config in $CONFIGS; do
  case "$config" in
    meld_only)
      train_pkl="$MELD_UNIFIED_PKL"
      bobsl_max=0
      lr=0.0003
      dropout=0.40
      loss=focal
      gamma=2.0
      extra_args=()
      ;;
    joint_b2k)
      train_pkl="$(joint_pkl_for 2000)"
      build_joint_if_needed 2000 "$train_pkl"
      bobsl_max=2000
      lr=0.0003
      dropout=0.40
      loss=focal
      gamma=2.0
      extra_args=()
      ;;
    joint_b5k)
      train_pkl="$(joint_pkl_for 5000)"
      build_joint_if_needed 5000 "$train_pkl"
      bobsl_max=5000
      lr=0.0003
      dropout=0.40
      loss=focal
      gamma=2.0
      extra_args=()
      ;;
    joint_b10k)
      train_pkl="$(joint_pkl_for 10000)"
      build_joint_if_needed 10000 "$train_pkl"
      bobsl_max=10000
      lr=0.0003
      dropout=0.40
      loss=focal
      gamma=2.0
      extra_args=()
      ;;
    joint_b5k_lr100)
      train_pkl="$(joint_pkl_for 5000)"
      build_joint_if_needed 5000 "$train_pkl"
      bobsl_max=5000
      lr=0.0001
      dropout=0.40
      loss=focal
      gamma=2.0
      extra_args=()
      ;;
    joint_b5k_drop20)
      train_pkl="$(joint_pkl_for 5000)"
      build_joint_if_needed 5000 "$train_pkl"
      bobsl_max=5000
      lr=0.0003
      dropout=0.20
      loss=focal
      gamma=2.0
      extra_args=()
      ;;
    *)
      echo "[MMGCN-JOINT] unknown config: $config" >&2
      exit 1
      ;;
  esac

  for seed in $SEEDS; do
    run_one "$config" "$seed" "$train_pkl" "$bobsl_max" "$lr" "$dropout" "$loss" "$gamma" "${extra_args[@]}"
  done
done

python3 MMGCN/summarize_video_joint_runs.py \
  --root "$OUT_ROOT" \
  --top_k "$TOP_K" | tee "$OUT_ROOT/video_joint_group_averages.log"

echo "[MMGCN-JOINT] done: $OUT_ROOT"
echo "[MMGCN-JOINT] runs: $OUT_ROOT/video_joint_runs.csv"
echo "[MMGCN-JOINT] group averages: $OUT_ROOT/video_joint_group_averages.csv"
