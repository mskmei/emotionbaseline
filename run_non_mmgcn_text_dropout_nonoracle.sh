#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/non_mmgcn_text_dropout_nonoracle}
FEATURE_ROOT=${FEATURE_ROOT:-$WORK_ROOT/features}
RUN_ROOT=${RUN_ROOT:-$WORK_ROOT/runs}
LOG_ROOT=${LOG_ROOT:-$WORK_ROOT/logs}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-$WORK_ROOT/feature_cache}

NONORACLE_TXT_ROOT=${NONORACLE_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_old_model_ejsl_split_finetune/ejsl_nonoracle_txt_best}
TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_four_nonoracle_mmgcn/translated_txt/old_pretrained_split}
TRANSLATE_IF_MISSING=${TRANSLATE_IF_MISSING:-0}
TRANSLATION_CACHE=${TRANSLATION_CACHE:-/raid_zoe/home/lr/maokeyu/sign/jsl_four_nonoracle_mmgcn/translation_cache.jsonl}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-openai}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}

MELD_RAW_ROOT=${MELD_RAW_ROOT:-/raid_zoe/home/lr/maokeyu/sign/emotionbaseline/dataset/MELD.Raw}
BOBSL_ROOT=${BOBSL_ROOT:-/raid_zoe/home/lr/wangyi/sign/bobsl}
EJSL_DIAL_LIST=${EJSL_DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
EJSL_FRAME_ROOT=${EJSL_FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
EJSL_MP4_ROOT=${EJSL_MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}

GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
SEED=${SEED:-42}
TEXT_DROPOUTS=${TEXT_DROPOUTS:-"0 0.25 0.5 0.75"}
TEXT_DROPOUT_SCOPE=${TEXT_DROPOUT_SCOPE:-utterance}
REBUILD_FEATURES=${REBUILD_FEATURES:-0}
RERUN_TRAIN=${RERUN_TRAIN:-0}

VIDEO_MODEL=${VIDEO_MODEL:-facebook/timesformer-base-finetuned-k400}
VIDEO_PROCESSOR=${VIDEO_PROCESSOR:-MCG-NJU/videomae-base}
NUM_FRAMES=${NUM_FRAMES:-8}
VIDEO_MAX_SECONDS=${VIDEO_MAX_SECONDS:-30}
TEXT_MODEL=${TEXT_MODEL:-roberta-large}
TEXT_POOLING=${TEXT_POOLING:-cls}
TEXT_MAX_LENGTH=${TEXT_MAX_LENGTH:-96}
TEXT_BATCH_SIZE=${TEXT_BATCH_SIZE:-32}

ECERC_EPOCHS=${ECERC_EPOCHS:-40}
ECERC_BATCH_SIZE=${ECERC_BATCH_SIZE:-32}
ECERC_LR=${ECERC_LR:-0.00001}
ECERC_L2=${ECERC_L2:-0.0002}
ECERC_LOSS_GAMMA=${ECERC_LOSS_GAMMA:-1.0}
ECERC_HIDDEN_SIZE=${ECERC_HIDDEN_SIZE:-128}
ECERC_BASE_LAYER=${ECERC_BASE_LAYER:-1}
ECERC_CLASS_WEIGHT=${ECERC_CLASS_WEIGHT:-1}

MAGTKD_EPOCHS=${MAGTKD_EPOCHS:-30}
MAGTKD_BATCH_SIZE=${MAGTKD_BATCH_SIZE:-16}
MAGTKD_LR=${MAGTKD_LR:-0.0001}
MAGTKD_L2=${MAGTKD_L2:-0.000001}
MAGTKD_DROPOUT=${MAGTKD_DROPOUT:-0.5}
MAGTKD_HIDDEN_DIM=${MAGTKD_HIDDEN_DIM:-768}
MAGTKD_N_HEAD=${MAGTKD_N_HEAD:-8}
MAGTKD_TEMP=${MAGTKD_TEMP:-2.0}
MAGTKD_CLASS_WEIGHT=${MAGTKD_CLASS_WEIGHT:-1}

CMERC_EPOCHS=${CMERC_EPOCHS:-40}
CMERC_BATCH_SIZE=${CMERC_BATCH_SIZE:-16}
CMERC_LR=${CMERC_LR:-0.0001}
CMERC_L2=${CMERC_L2:-0.00003}
CMERC_DROPOUT=${CMERC_DROPOUT:-0.4}
CMERC_LOSS=${CMERC_LOSS:-focal}
CMERC_FOCAL_GAMMA=${CMERC_FOCAL_GAMMA:-2.5}
CMERC_CLASS_WEIGHT=${CMERC_CLASS_WEIGHT:-0}

CONXGNN_EPOCHS=${CONXGNN_EPOCHS:-30}
CONXGNN_BATCH_SIZE=${CONXGNN_BATCH_SIZE:-64}
CONXGNN_LR=${CONXGNN_LR:-0.0003}
CONXGNN_WEIGHT_DECAY=${CONXGNN_WEIGHT_DECAY:-0.00000001}
CONXGNN_DROP_RATE=${CONXGNN_DROP_RATE:-0.8}
CONXGNN_HIDDEN_SIZE=${CONXGNN_HIDDEN_SIZE:-128}
CONXGNN_INTER_SIZE=${CONXGNN_INTER_SIZE:-256}
CONXGNN_LOSS=${CONXGNN_LOSS:-ce}
CONXGNN_FOCAL_GAMMA=${CONXGNN_FOCAL_GAMMA:-2.5}
CONXGNN_CLASS_WEIGHT=${CONXGNN_CLASS_WEIGHT:-1}

TELME_TEACHER_EPOCHS=${TELME_TEACHER_EPOCHS:-10}
TELME_STUDENT_EPOCHS=${TELME_STUDENT_EPOCHS:-10}
TELME_FUSION_EPOCHS=${TELME_FUSION_EPOCHS:-10}
TELME_BATCH_SIZE=${TELME_BATCH_SIZE:-4}
TELME_TRAIN_NUM_WORKERS=${TELME_TRAIN_NUM_WORKERS:-16}
TELME_TEST_BATCH_SIZE=${TELME_TEST_BATCH_SIZE:-4}
TELME_TEST_NUM_WORKERS=${TELME_TEST_NUM_WORKERS:-4}
TELME_TEACHER_LR=${TELME_TEACHER_LR:-0.000001}
TELME_STUDENT_LR=${TELME_STUDENT_LR:-0.00001}
TELME_FUSION_LR=${TELME_FUSION_LR:-0.00001}

mkdir -p "$FEATURE_ROOT" "$RUN_ROOT" "$LOG_ROOT" "$FEATURE_CACHE_DIR"

rate_label() {
  python - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(f"td{int(round(value * 100)):02d}")
PY
}

require_path() {
  local label="$1"
  local path="$2"
  if [ ! -e "$path" ]; then
    echo "[NON-MMGCN-TD] missing $label: $path" >&2
    exit 1
  fi
}

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"
  echo "[NON-MMGCN-TD] $*" | tee "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

write_config_json() {
  local out_dir="$1"
  local baseline="$2"
  local run_name="$3"
  local modality="$4"
  local dropout="$5"
  local train_pkl="$6"
  local ejsl_pkl="$7"
  mkdir -p "$out_dir"
  python - "$out_dir/config.json" "$baseline" "$run_name" "$modality" "$dropout" "$train_pkl" "$ejsl_pkl" <<'PY'
import json
import os
import sys

path, baseline, run_name, modality, dropout, train_pkl, ejsl_pkl = sys.argv[1:8]
data = {
    "baseline": baseline,
    "run_name": run_name,
    "modality": modality,
    "train_text_dropout": float(dropout),
    "train_text_dropout_scope": os.environ["TEXT_DROPOUT_SCOPE"],
    "seed": int(os.environ["SEED"]),
    "train_pkl": train_pkl,
    "external_test_pkl": ejsl_pkl,
    "source_test": "MELD test",
    "external_test": "translated eJSL non-oracle test",
    "nonoracle_txt_root": os.environ["NONORACLE_TXT_ROOT"],
    "translated_txt_root": os.environ["TRANSLATED_TXT_ROOT"],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

ensure_translated_txt() {
  require_path "non-oracle txt root" "$NONORACLE_TXT_ROOT"
  if [ -d "$TRANSLATED_TXT_ROOT" ] && find "$TRANSLATED_TXT_ROOT" -name '*.txt' -print -quit | grep -q .; then
    echo "[NON-MMGCN-TD] reuse translated txt root: $TRANSLATED_TXT_ROOT"
    return
  fi
  if [ "$TRANSLATE_IF_MISSING" != "1" ]; then
    echo "[NON-MMGCN-TD] translated txt root is missing or empty: $TRANSLATED_TXT_ROOT" >&2
    echo "[NON-MMGCN-TD] set TRANSLATED_TXT_ROOT to the existing translation or TRANSLATE_IF_MISSING=1." >&2
    exit 1
  fi
  run_logged "$LOG_ROOT/translate_nonoracle.log" \
    python JSL/translate_ejsl_txt_root.py \
      --input_txt_root "$NONORACLE_TXT_ROOT" \
      --output_txt_root "$TRANSLATED_TXT_ROOT" \
      --dial_list "$EJSL_DIAL_LIST" \
      --cache_jsonl "$TRANSLATION_CACHE" \
      --backend "$TRANSLATION_BACKEND" \
      --openai_model "$OPENAI_MODEL"
}

feature_skip_args() {
  local meld_pkl="$1"
  local ejsl_pkl="$2"
  local args=(--skip_bobsl)
  if [ "$REBUILD_FEATURES" != "1" ] && [ -s "$meld_pkl" ]; then
    args+=(--skip_meld)
  fi
  if [ "$REBUILD_FEATURES" != "1" ] && [ -s "$ejsl_pkl" ]; then
    args+=(--skip_ejsl)
  fi
  printf '%s\n' "${args[@]}"
}

build_ecerc_features() {
  local meld_pkl="$FEATURE_ROOT/ecerc_meld_text_video_anjs4.pkl"
  local ejsl_pkl="$FEATURE_ROOT/ecerc_ejsl_text_video_anjs4_nonoracle.pkl"
  if [ "$REBUILD_FEATURES" != "1" ] && [ -s "$meld_pkl" ] && [ -s "$ejsl_pkl" ]; then
    echo "[ECERC] reuse features: $meld_pkl / $ejsl_pkl"
    return
  fi
  mapfile -t skip_args < <(feature_skip_args "$meld_pkl" "$ejsl_pkl")
  run_logged "$LOG_ROOT/ecerc_build_features.log" \
    python new/ECERC/MELD/build_ecerc_video_anjs_features.py \
      --bobsl_root "$BOBSL_ROOT" \
      --meld_root "$MELD_RAW_ROOT" \
      --ejsl_txt_root "$TRANSLATED_TXT_ROOT" \
      --ejsl_dial_list "$EJSL_DIAL_LIST" \
      --ejsl_frame_root "$EJSL_FRAME_ROOT" \
      --ejsl_mp4_root "$EJSL_MP4_ROOT" \
      --out_meld_pkl "$meld_pkl" \
      --out_ejsl_pkl "$ejsl_pkl" \
      --out_bobsl_train_val_pkl "$FEATURE_ROOT/unused_ecerc_bobsl_train_val.pkl" \
      --out_bobsl_test_pkl "$FEATURE_ROOT/unused_ecerc_bobsl_test.pkl" \
      --cache_dir "$FEATURE_CACHE_DIR" \
      --video_model "$VIDEO_MODEL" \
      --video_processor "$VIDEO_PROCESSOR" \
      --num_frames "$NUM_FRAMES" \
      --video_max_seconds "$VIDEO_MAX_SECONDS" \
      --use_text_features \
      --text_model "$TEXT_MODEL" \
      --text_pooling "$TEXT_POOLING" \
      --text_max_length "$TEXT_MAX_LENGTH" \
      --text_batch_size "$TEXT_BATCH_SIZE" \
      --fp16 \
      "${skip_args[@]}"
}

build_magtkd_features() {
  local meld_pkl="$FEATURE_ROOT/magtkd_meld_text_video_anjs4.pkl"
  local ejsl_pkl="$FEATURE_ROOT/magtkd_ejsl_text_video_anjs4_nonoracle.pkl"
  if [ "$REBUILD_FEATURES" != "1" ] && [ -s "$meld_pkl" ] && [ -s "$ejsl_pkl" ]; then
    echo "[MAGTKD] reuse features: $meld_pkl / $ejsl_pkl"
    return
  fi
  mapfile -t skip_args < <(feature_skip_args "$meld_pkl" "$ejsl_pkl")
  run_logged "$LOG_ROOT/magtkd_build_features.log" \
    python new/MAGTKD/MELD/build_magtkd_video_anjs_features.py \
      --bobsl_root "$BOBSL_ROOT" \
      --meld_root "$MELD_RAW_ROOT" \
      --ejsl_txt_root "$TRANSLATED_TXT_ROOT" \
      --ejsl_dial_list "$EJSL_DIAL_LIST" \
      --ejsl_frame_root "$EJSL_FRAME_ROOT" \
      --ejsl_mp4_root "$EJSL_MP4_ROOT" \
      --out_meld_pkl "$meld_pkl" \
      --out_ejsl_pkl "$ejsl_pkl" \
      --out_bobsl_train_val_pkl "$FEATURE_ROOT/unused_magtkd_bobsl_train_val.pkl" \
      --out_bobsl_test_pkl "$FEATURE_ROOT/unused_magtkd_bobsl_test.pkl" \
      --cache_dir "$FEATURE_CACHE_DIR" \
      --video_model "$VIDEO_MODEL" \
      --video_processor "$VIDEO_PROCESSOR" \
      --num_frames "$NUM_FRAMES" \
      --video_max_seconds "$VIDEO_MAX_SECONDS" \
      --hidden_dim "$MAGTKD_HIDDEN_DIM" \
      --use_text_features \
      --text_model "$TEXT_MODEL" \
      --text_pooling "$TEXT_POOLING" \
      --text_max_length "$TEXT_MAX_LENGTH" \
      --text_batch_size "$TEXT_BATCH_SIZE" \
      --fp16 \
      "${skip_args[@]}"
}

build_cmerc_features() {
  local meld_pkl="$FEATURE_ROOT/cmerc_meld_text_visual_anjs4.pkl"
  local ejsl_pkl="$FEATURE_ROOT/cmerc_ejsl_text_visual_anjs4_nonoracle.pkl"
  if [ "$REBUILD_FEATURES" != "1" ] && [ -s "$meld_pkl" ] && [ -s "$ejsl_pkl" ]; then
    echo "[CMERC] reuse features: $meld_pkl / $ejsl_pkl"
    return
  fi
  mapfile -t skip_args < <(feature_skip_args "$meld_pkl" "$ejsl_pkl")
  run_logged "$LOG_ROOT/cmerc_build_features.log" \
    python new2/CMERC/build_cmerc_visual_anjs_features.py \
      --bobsl_root "$BOBSL_ROOT" \
      --meld_root "$MELD_RAW_ROOT" \
      --ejsl_txt_root "$TRANSLATED_TXT_ROOT" \
      --ejsl_dial_list "$EJSL_DIAL_LIST" \
      --ejsl_frame_root "$EJSL_FRAME_ROOT" \
      --ejsl_mp4_root "$EJSL_MP4_ROOT" \
      --out_meld_pkl "$meld_pkl" \
      --out_ejsl_pkl "$ejsl_pkl" \
      --out_bobsl_train_val_pkl "$FEATURE_ROOT/unused_cmerc_bobsl_train_val.pkl" \
      --out_bobsl_test_pkl "$FEATURE_ROOT/unused_cmerc_bobsl_test.pkl" \
      --cache_dir "$FEATURE_CACHE_DIR" \
      --video_model "$VIDEO_MODEL" \
      --video_processor "$VIDEO_PROCESSOR" \
      --num_frames "$NUM_FRAMES" \
      --video_max_seconds "$VIDEO_MAX_SECONDS" \
      --use_text_features \
      --text_model "$TEXT_MODEL" \
      --text_pooling "$TEXT_POOLING" \
      --text_max_length "$TEXT_MAX_LENGTH" \
      --text_batch_size "$TEXT_BATCH_SIZE" \
      --fp16 \
      "${skip_args[@]}"
}

build_conxgnn_features() {
  local meld_pkl="$FEATURE_ROOT/conxgnn_meld_text_visual_anjs4.pkl"
  local ejsl_pkl="$FEATURE_ROOT/conxgnn_ejsl_text_visual_anjs4_nonoracle.pkl"
  if [ "$REBUILD_FEATURES" != "1" ] && [ -s "$meld_pkl" ] && [ -s "$ejsl_pkl" ]; then
    echo "[CONXGNN] reuse features: $meld_pkl / $ejsl_pkl"
    return
  fi
  mapfile -t skip_args < <(feature_skip_args "$meld_pkl" "$ejsl_pkl")
  run_logged "$LOG_ROOT/conxgnn_build_features.log" \
    python new2/ConxGNN/build_conxgnn_visual_anjs_features.py \
      --bobsl_root "$BOBSL_ROOT" \
      --meld_root "$MELD_RAW_ROOT" \
      --ejsl_txt_root "$TRANSLATED_TXT_ROOT" \
      --ejsl_dial_list "$EJSL_DIAL_LIST" \
      --ejsl_frame_root "$EJSL_FRAME_ROOT" \
      --ejsl_mp4_root "$EJSL_MP4_ROOT" \
      --out_meld_pkl "$meld_pkl" \
      --out_ejsl_pkl "$ejsl_pkl" \
      --out_bobsl_train_val_pkl "$FEATURE_ROOT/unused_conxgnn_bobsl_train_val.pkl" \
      --out_bobsl_test_pkl "$FEATURE_ROOT/unused_conxgnn_bobsl_test.pkl" \
      --cache_dir "$FEATURE_CACHE_DIR" \
      --video_model "$VIDEO_MODEL" \
      --video_processor "$VIDEO_PROCESSOR" \
      --num_frames "$NUM_FRAMES" \
      --video_max_seconds "$VIDEO_MAX_SECONDS" \
      --use_text_features \
      --text_model "$TEXT_MODEL" \
      --text_pooling "$TEXT_POOLING" \
      --text_max_length "$TEXT_MAX_LENGTH" \
      --text_batch_size "$TEXT_BATCH_SIZE" \
      --fp16 \
      "${skip_args[@]}"
}

run_ecerc_trial() {
  local modality="$1"
  local dropout="$2"
  local run_name="$3"
  local meld_pkl="$FEATURE_ROOT/ecerc_meld_text_video_anjs4.pkl"
  local ejsl_pkl="$FEATURE_ROOT/ecerc_ejsl_text_video_anjs4_nonoracle.pkl"
  local out_dir="$RUN_ROOT/ECERC/$run_name"
  if [ "$RERUN_TRAIN" != "1" ] && [ -s "$out_dir/external_test_best_summary.json" ]; then
    echo "[ECERC] skip existing $out_dir"
    return
  fi
  write_config_json "$out_dir" ECERC "$run_name" "$modality" "$dropout" "$meld_pkl" "$ejsl_pkl"
  local class_weight_args=()
  if [ "$ECERC_CLASS_WEIGHT" = "1" ]; then class_weight_args+=(--class_weight); fi
  run_logged "$LOG_ROOT/ecerc_${run_name}.log" \
    python new/ECERC/MELD/train_eval_ecerc_video_anjs.py \
      --train_pkl "$meld_pkl" \
      --external_test_pkl "$ejsl_pkl" \
      --out_dir "$out_dir" \
      --modality "$modality" \
      --train_text_dropout "$dropout" \
      --train_text_dropout_scope "$TEXT_DROPOUT_SCOPE" \
      --seed "$SEED" \
      --epochs "$ECERC_EPOCHS" \
      --batch_size "$ECERC_BATCH_SIZE" \
      --lr "$ECERC_LR" \
      --l2 "$ECERC_L2" \
      --loss_gamma "$ECERC_LOSS_GAMMA" \
      --base_layer "$ECERC_BASE_LAYER" \
      --hidden_size "$ECERC_HIDDEN_SIZE" \
      --max_grad_norm 5.0 \
      --selection_split source \
      --selection_metric weighted_f1 \
      "${class_weight_args[@]}"
}

run_magtkd_trial() {
  local modality="$1"
  local dropout="$2"
  local run_name="$3"
  local meld_pkl="$FEATURE_ROOT/magtkd_meld_text_video_anjs4.pkl"
  local ejsl_pkl="$FEATURE_ROOT/magtkd_ejsl_text_video_anjs4_nonoracle.pkl"
  local out_dir="$RUN_ROOT/MAGTKD/$run_name"
  if [ "$RERUN_TRAIN" != "1" ] && [ -s "$out_dir/external_test_best_summary.json" ]; then
    echo "[MAGTKD] skip existing $out_dir"
    return
  fi
  write_config_json "$out_dir" MAGTKD "$run_name" "$modality" "$dropout" "$meld_pkl" "$ejsl_pkl"
  local class_weight_args=()
  if [ "$MAGTKD_CLASS_WEIGHT" = "1" ]; then class_weight_args+=(--class_weight); fi
  run_logged "$LOG_ROOT/magtkd_${run_name}.log" \
    python new/MAGTKD/MELD/train_eval_magtkd_video_anjs.py \
      --train_pkl "$meld_pkl" \
      --external_test_pkl "$ejsl_pkl" \
      --out_dir "$out_dir" \
      --modality "$modality" \
      --output_head auto \
      --train_text_dropout "$dropout" \
      --train_text_dropout_scope "$TEXT_DROPOUT_SCOPE" \
      --seed "$SEED" \
      --epochs "$MAGTKD_EPOCHS" \
      --batch_size "$MAGTKD_BATCH_SIZE" \
      --lr "$MAGTKD_LR" \
      --l2 "$MAGTKD_L2" \
      --dropout "$MAGTKD_DROPOUT" \
      --hidden_dim "$MAGTKD_HIDDEN_DIM" \
      --n_head "$MAGTKD_N_HEAD" \
      --temp "$MAGTKD_TEMP" \
      --max_grad_norm 10.0 \
      --selection_split source \
      --selection_metric weighted_f1 \
      "${class_weight_args[@]}"
}

run_cmerc_trial() {
  local modality="$1"
  local dropout="$2"
  local run_name="$3"
  local meld_pkl="$FEATURE_ROOT/cmerc_meld_text_visual_anjs4.pkl"
  local ejsl_pkl="$FEATURE_ROOT/cmerc_ejsl_text_visual_anjs4_nonoracle.pkl"
  local out_dir="$RUN_ROOT/CMERC/$run_name"
  if [ "$RERUN_TRAIN" != "1" ] && [ -s "$out_dir/external_test_best_summary.json" ]; then
    echo "[CMERC] skip existing $out_dir"
    return
  fi
  write_config_json "$out_dir" CMERC "$run_name" "$modality" "$dropout" "$meld_pkl" "$ejsl_pkl"
  local class_weight_args=()
  if [ "$CMERC_CLASS_WEIGHT" = "1" ]; then class_weight_args+=(--class_weight); fi
  run_logged "$LOG_ROOT/cmerc_${run_name}.log" \
    python new2/CMERC/train_eval_cmerc_visual_anjs.py \
      --train_pkl "$meld_pkl" \
      --external_test_pkl "$ejsl_pkl" \
      --out_dir "$out_dir" \
      --modality "$modality" \
      --train_text_dropout "$dropout" \
      --train_text_dropout_scope "$TEXT_DROPOUT_SCOPE" \
      --seed "$SEED" \
      --epochs "$CMERC_EPOCHS" \
      --batch_size "$CMERC_BATCH_SIZE" \
      --lr "$CMERC_LR" \
      --l2 "$CMERC_L2" \
      --dropout "$CMERC_DROPOUT" \
      --loss "$CMERC_LOSS" \
      --focal_gamma "$CMERC_FOCAL_GAMMA" \
      --max_grad_norm 5.0 \
      --selection_split source \
      --selection_metric weighted_f1 \
      "${class_weight_args[@]}"
}

run_conxgnn_trial() {
  local modality="$1"
  local dropout="$2"
  local run_name="$3"
  local meld_pkl="$FEATURE_ROOT/conxgnn_meld_text_visual_anjs4.pkl"
  local ejsl_pkl="$FEATURE_ROOT/conxgnn_ejsl_text_visual_anjs4_nonoracle.pkl"
  local out_dir="$RUN_ROOT/ConxGNN/$run_name"
  if [ "$RERUN_TRAIN" != "1" ] && [ -s "$out_dir/external_test_best_summary.json" ]; then
    echo "[CONXGNN] skip existing $out_dir"
    return
  fi
  write_config_json "$out_dir" ConxGNN "$run_name" "$modality" "$dropout" "$meld_pkl" "$ejsl_pkl"
  local class_weight_args=()
  if [ "$CONXGNN_CLASS_WEIGHT" = "1" ]; then class_weight_args+=(--class_weight); fi
  run_logged "$LOG_ROOT/conxgnn_${run_name}.log" \
    python new2/ConxGNN/train_eval_conxgnn_visual_anjs.py \
      --train_pkl "$meld_pkl" \
      --external_test_pkl "$ejsl_pkl" \
      --out_dir "$out_dir" \
      --modality "$modality" \
      --train_text_dropout "$dropout" \
      --train_text_dropout_scope "$TEXT_DROPOUT_SCOPE" \
      --seed "$SEED" \
      --epochs "$CONXGNN_EPOCHS" \
      --batch_size "$CONXGNN_BATCH_SIZE" \
      --lr "$CONXGNN_LR" \
      --weight_decay "$CONXGNN_WEIGHT_DECAY" \
      --drop_rate "$CONXGNN_DROP_RATE" \
      --hidden_size "$CONXGNN_HIDDEN_SIZE" \
      --inter_size "$CONXGNN_INTER_SIZE" \
      --loss "$CONXGNN_LOSS" \
      --focal_gamma "$CONXGNN_FOCAL_GAMMA" \
      --max_grad_norm 5.0 \
      --selection_split source \
      --selection_metric weighted_f1 \
      "${class_weight_args[@]}"
}

run_telme_trial() {
  local modality="$1"
  local dropout="$2"
  local run_name="$3"
  local run_dir="$RUN_ROOT/TelME/$run_name"
  local save_root="$run_dir/save_model"
  local eval_dir="$run_dir/eval"
  local done_path="$eval_dir/telme_meld4_${modality}_ejsl_summary.json"
  if [ "$RERUN_TRAIN" != "1" ] && [ -s "$done_path" ]; then
    echo "[TelME] skip existing $run_dir"
    return
  fi
  mkdir -p "$run_dir" "$eval_dir" "$save_root"
  write_config_json "$run_dir" TelME "$run_name" "$modality" "$dropout" "raw_meld_meld4" "raw_ejsl_frames"

  local shared_source="$RUN_ROOT/TelME/t_only/save_model/shared"
  local reuse_args=()
  if [ "$modality" = "tv" ] && [ -d "$shared_source" ] && [ ! -e "$save_root/shared" ]; then
    ln -s "$shared_source" "$save_root/shared"
  fi
  if [ -e "$save_root/shared/teacher.bin" ]; then
    reuse_args+=(--reuse_shared)
  fi

  run_logged "$LOG_ROOT/telme_${run_name}_train.log" \
    python MELD/train_telme_meld4_modalities.py \
      --data_root "$MELD_RAW_ROOT" \
      --save_root "$save_root" \
      --modalities "$modality" \
      --teacher_epochs "$TELME_TEACHER_EPOCHS" \
      --student_epochs "$TELME_STUDENT_EPOCHS" \
      --fusion_epochs "$TELME_FUSION_EPOCHS" \
      --teacher_lr "$TELME_TEACHER_LR" \
      --student_lr "$TELME_STUDENT_LR" \
      --fusion_lr "$TELME_FUSION_LR" \
      --batch_size "$TELME_BATCH_SIZE" \
      --num_workers "$TELME_TRAIN_NUM_WORKERS" \
      --seed "$SEED" \
      --max_grad_norm 10.0 \
      --train_text_dropout "$dropout" \
      --train_text_dropout_scope "$TEXT_DROPOUT_SCOPE" \
      "${reuse_args[@]}"

  run_logged "$LOG_ROOT/telme_${run_name}_ejsl.log" \
    python IEMOCAP/inference_ejsl_frame.py \
      --frame_root "$EJSL_FRAME_ROOT" \
      --txt_root "$TRANSLATED_TXT_ROOT" \
      --save_model_root "$save_root/$modality" \
      --batch_size "$TELME_TEST_BATCH_SIZE" \
      --num_workers "$TELME_TEST_NUM_WORKERS" \
      --seed "$SEED" \
      --save_dir "$eval_dir" \
      --report_prefix "telme_meld4_${modality}_ejsl" \
      --eval_modality "$modality" \
      --checkpoint_dataset MELD4 \
      --fusion_input_order audio_video \
      --missing_audio_strategy zero_hidden \
      --save_predictions
}

run_tv_grid() {
  local runner="$1"
  "$runner" text 0 t_only
  for rate in $TEXT_DROPOUTS; do
    "$runner" tv "$rate" "tv_$(rate_label "$rate")"
  done
}

worker_gpu0() {
  export CUDA_VISIBLE_DEVICES="$GPU0"
  build_ecerc_features
  run_tv_grid run_ecerc_trial
  build_magtkd_features
  run_tv_grid run_magtkd_trial
  run_tv_grid run_telme_trial
}

worker_gpu1() {
  export CUDA_VISIBLE_DEVICES="$GPU1"
  build_cmerc_features
  run_tv_grid run_cmerc_trial
  build_conxgnn_features
  run_tv_grid run_conxgnn_trial
}

export NONORACLE_TXT_ROOT TRANSLATED_TXT_ROOT TEXT_DROPOUT_SCOPE SEED
ensure_translated_txt
require_path "MELD raw root" "$MELD_RAW_ROOT"
require_path "eJSL frame root" "$EJSL_FRAME_ROOT"

echo "[NON-MMGCN-TD] work root: $WORK_ROOT"
echo "[NON-MMGCN-TD] GPUs: group0=$GPU0 group1=$GPU1"
echo "[NON-MMGCN-TD] text dropout rates: $TEXT_DROPOUTS scope=$TEXT_DROPOUT_SCOPE"

(worker_gpu0) &
pid0=$!
(worker_gpu1) &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [ "$status" -ne 0 ]; then
  echo "[NON-MMGCN-TD] at least one worker failed with status=$status" >&2
  exit "$status"
fi

python summarize_non_mmgcn_text_dropout_nonoracle.py \
  --root "$RUN_ROOT" \
  --out_csv "$WORK_ROOT/non_mmgcn_text_dropout_summary.csv" \
  --out_json "$WORK_ROOT/non_mmgcn_text_dropout_summary.json"

echo "[NON-MMGCN-TD] done"
echo "[NON-MMGCN-TD] summary csv: $WORK_ROOT/non_mmgcn_text_dropout_summary.csv"
echo "[NON-MMGCN-TD] summary json: $WORK_ROOT/non_mmgcn_text_dropout_summary.json"
