#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TRANSLATED_TXT_ROOT="${TRANSLATED_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ejsl_txt_en_openai}"

WORK_ROOT="${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl}"
FEATURE_ROOT="${FEATURE_ROOT:-$WORK_ROOT/features}"
OUT_ROOT="${OUT_ROOT:-$WORK_ROOT/visual_joint_bobsl_meld}"
JOINT_PKL_ROOT="${JOINT_PKL_ROOT:-$OUT_ROOT/joint_pkls}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$FEATURE_ROOT/cache}"

BOBSL_ROOT="${BOBSL_ROOT:-/raid_zoe/home/lr/wangyi/sign/bobsl}"
MELD_RAW_ROOT="${MELD_RAW_ROOT:-/raid_zoe/home/lr/maokeyu/sign/emotionbaseline/dataset/MELD.Raw}"
EJSL_DIAL_LIST="${EJSL_DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}"
EJSL_FRAME_ROOT="${EJSL_FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}"
EJSL_MP4_ROOT="${EJSL_MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}"

MELD_PKL="${MELD_PKL:-$FEATURE_ROOT/meld_conxgnn_visual_anjs4.pkl}"
EJSL_PKL="${EJSL_PKL:-$FEATURE_ROOT/ejsl_conxgnn_visual_anjs4_translated.pkl}"
BOBSL_TRAIN_VAL_PKL="${BOBSL_TRAIN_VAL_PKL:-$FEATURE_ROOT/bobsl_conxgnn_visual_anjs4_train_val.pkl}"
BOBSL_TEST_PKL="${BOBSL_TEST_PKL:-$FEATURE_ROOT/bobsl_conxgnn_visual_anjs4_test.pkl}"

VIDEO_MODEL="${VIDEO_MODEL:-facebook/timesformer-base-finetuned-k400}"
VIDEO_PROCESSOR="${VIDEO_PROCESSOR:-MCG-NJU/videomae-base}"
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-}"
NUM_FRAMES="${NUM_FRAMES:-8}"
VIDEO_MAX_SECONDS="${VIDEO_MAX_SECONDS:-30}"

SEEDS="${SEEDS:-35 36 37 41 42}"
CONFIGS="${CONFIGS:-meld_only joint_b1k joint_b5k joint_b10k}"
BOBSL_SAMPLE_SEED="${BOBSL_SAMPLE_SEED:-123}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00000001}"
DROP_RATE="${DROP_RATE:-0.8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
INTER_SIZE="${INTER_SIZE:-256}"
LOSS="${LOSS:-ce}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.5}"
CLASS_WEIGHT="${CLASS_WEIGHT:-1}"
CE_LOSS_PARAM="${CE_LOSS_PARAM:-1.0}"
CB_LOSS_PARAM="${CB_LOSS_PARAM:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-5.0}"
SAVE_EPOCH_EVERY="${SAVE_EPOCH_EVERY:-0}"
REBUILD_FEATURES="${REBUILD_FEATURES:-0}"
REBUILD_JOINT="${REBUILD_JOINT:-0}"
RESUME_DONE="${RESUME_DONE:-1}"

export EJSL_PKL EPOCHS BATCH_SIZE LR WEIGHT_DECAY DROP_RATE HIDDEN_SIZE INTER_SIZE LOSS FOCAL_GAMMA CLASS_WEIGHT CE_LOSS_PARAM CB_LOSS_PARAM VIDEO_MODEL VIDEO_PROCESSOR TRANSLATED_TXT_ROOT

mkdir -p "$FEATURE_ROOT" "$OUT_ROOT" "$JOINT_PKL_ROOT" "$LOG_ROOT" "$FEATURE_CACHE_DIR"

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"
  echo "[CONXGNN-RUN] $*" | tee "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

build_features_if_needed() {
  local skip_args=()
  if [[ "$REBUILD_FEATURES" != "1" && -s "$MELD_PKL" ]]; then
    skip_args+=(--skip_meld)
  fi
  if [[ "$REBUILD_FEATURES" != "1" && -s "$EJSL_PKL" ]]; then
    skip_args+=(--skip_ejsl)
  fi
  if [[ "$REBUILD_FEATURES" != "1" && -s "$BOBSL_TRAIN_VAL_PKL" && -s "$BOBSL_TEST_PKL" ]]; then
    skip_args+=(--skip_bobsl)
  fi
  if [[ "${#skip_args[@]}" -eq 3 ]]; then
    echo "[CONXGNN-RUN] feature pkls already exist under $FEATURE_ROOT"
    return
  fi

  run_logged "$LOG_ROOT/build_features.log" \
    python new2/ConxGNN/build_conxgnn_visual_anjs_features.py \
      --bobsl_root "$BOBSL_ROOT" \
      --meld_root "$MELD_RAW_ROOT" \
      --ejsl_txt_root "$TRANSLATED_TXT_ROOT" \
      --ejsl_dial_list "$EJSL_DIAL_LIST" \
      --ejsl_frame_root "$EJSL_FRAME_ROOT" \
      --ejsl_mp4_root "$EJSL_MP4_ROOT" \
      --out_meld_pkl "$MELD_PKL" \
      --out_ejsl_pkl "$EJSL_PKL" \
      --out_bobsl_train_val_pkl "$BOBSL_TRAIN_VAL_PKL" \
      --out_bobsl_test_pkl "$BOBSL_TEST_PKL" \
      --cache_dir "$FEATURE_CACHE_DIR" \
      --video_model "$VIDEO_MODEL" \
      --video_processor "$VIDEO_PROCESSOR" \
      --local_model_root "$LOCAL_MODEL_ROOT" \
      --num_frames "$NUM_FRAMES" \
      --video_max_seconds "$VIDEO_MAX_SECONDS" \
      --fp16 \
      "${skip_args[@]}"
}

joint_pkl_for_config() {
  local config="$1"
  local max_bobsl="$2"
  local out_pkl="$JOINT_PKL_ROOT/${config}.pkl"
  if [[ "$REBUILD_JOINT" == "1" || ! -s "$out_pkl" ]]; then
    run_logged "$LOG_ROOT/build_${config}.log" \
      python new2/ConxGNN/build_conxgnn_joint_visual_pkl.py \
        --meld_pkl "$MELD_PKL" \
        --bobsl_train_val_pkl "$BOBSL_TRAIN_VAL_PKL" \
        --out_pkl "$out_pkl" \
        --max_bobsl_train_dialogues "$max_bobsl" \
        --bobsl_sample_seed "$BOBSL_SAMPLE_SEED" >&2
  fi
  printf '%s\n' "$out_pkl"
}

write_config_json() {
  local out_dir="$1"
  local config="$2"
  local seed="$3"
  local bobsl_max="$4"
  local train_pkl="$5"
  mkdir -p "$out_dir"
  python - "$out_dir/config.json" "$config" "$seed" "$bobsl_max" "$train_pkl" <<'PY'
import json
import os
import sys

path, config, seed, bobsl_max, train_pkl = sys.argv[1:6]
data = {
    "baseline": "ConxGNN",
    "config": config,
    "seed": int(seed),
    "bobsl_max": int(bobsl_max),
    "train_pkl": train_pkl,
    "external_test_pkl": os.environ["EJSL_PKL"],
    "source_test": "MELD test",
    "external_test": "translated eJSL test",
    "modality": "visual",
    "epochs": int(os.environ["EPOCHS"]),
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "lr": float(os.environ["LR"]),
    "weight_decay": float(os.environ["WEIGHT_DECAY"]),
    "dropout": float(os.environ["DROP_RATE"]),
    "hidden_size": int(os.environ["HIDDEN_SIZE"]),
    "inter_size": int(os.environ["INTER_SIZE"]),
    "loss": os.environ["LOSS"],
    "focal_gamma": float(os.environ["FOCAL_GAMMA"]),
    "class_weight": os.environ["CLASS_WEIGHT"] == "1",
    "ce_loss_param": float(os.environ["CE_LOSS_PARAM"]),
    "cb_loss_param": float(os.environ["CB_LOSS_PARAM"]),
    "video_model": os.environ["VIDEO_MODEL"],
    "video_processor": os.environ["VIDEO_PROCESSOR"],
    "translated_txt_root": os.environ["TRANSLATED_TXT_ROOT"],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

run_trial() {
  local config="$1"
  local seed="$2"
  local bobsl_max="$3"
  local train_pkl="$4"
  local out_dir="$OUT_ROOT/${config}_seed${seed}"
  local done_path="$out_dir/external_test_best_summary.json"
  if [[ "$RESUME_DONE" == "1" && -s "$done_path" ]]; then
    echo "[CONXGNN-RUN] skip existing $out_dir"
    return
  fi

  write_config_json "$out_dir" "$config" "$seed" "$bobsl_max" "$train_pkl"
  local class_weight_args=()
  if [[ "$CLASS_WEIGHT" == "1" ]]; then
    class_weight_args+=(--class_weight)
  fi

  run_logged "$LOG_ROOT/${config}_seed${seed}.log" \
    python new2/ConxGNN/train_eval_conxgnn_visual_anjs.py \
      --train_pkl "$train_pkl" \
      --external_test_pkl "$EJSL_PKL" \
      --out_dir "$out_dir" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --drop_rate "$DROP_RATE" \
      --hidden_size "$HIDDEN_SIZE" \
      --inter_size "$INTER_SIZE" \
      --loss "$LOSS" \
      --focal_gamma "$FOCAL_GAMMA" \
      --ce_loss_param "$CE_LOSS_PARAM" \
      --cb_loss_param "$CB_LOSS_PARAM" \
      --max_grad_norm "$MAX_GRAD_NORM" \
      --selection_split source \
      --selection_metric weighted_f1 \
      --save_epoch_every "$SAVE_EPOCH_EVERY" \
      "${class_weight_args[@]}"
}

build_features_if_needed

for config in $CONFIGS; do
  case "$config" in
    meld_only)
      bobsl_max=0
      train_pkl="$MELD_PKL"
      ;;
    joint_b1k)
      bobsl_max=1000
      train_pkl="$(joint_pkl_for_config "$config" "$bobsl_max")"
      ;;
    joint_b5k)
      bobsl_max=5000
      train_pkl="$(joint_pkl_for_config "$config" "$bobsl_max")"
      ;;
    joint_b10k)
      bobsl_max=10000
      train_pkl="$(joint_pkl_for_config "$config" "$bobsl_max")"
      ;;
    *)
      echo "[CONXGNN-RUN] unknown config: $config" >&2
      exit 2
      ;;
  esac
  for seed in $SEEDS; do
    run_trial "$config" "$seed" "$bobsl_max" "$train_pkl"
  done
done

python new/summarize_video_joint_runs.py \
  --root "$OUT_ROOT" \
  --baseline ConxGNN \
  --out_runs_csv "$OUT_ROOT/visual_joint_runs.csv" \
  --out_group_csv "$OUT_ROOT/visual_joint_group_averages.csv" \
  --out_txt "$OUT_ROOT/visual_joint_group_averages.txt"

echo "[CONXGNN-RUN] done. Summary: $OUT_ROOT/visual_joint_group_averages.txt"
