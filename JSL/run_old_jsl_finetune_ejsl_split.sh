#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

DATA_ENV=${DATA_ENV:-telme39}
TRAIN_ENV=${TRAIN_ENV:-base}
GPU=${GPU:-0}

WORK_DIR=${WORK_DIR:-/raid_zoe/home/lr/maokeyu/sign/jsl_old_model_ejsl_split_finetune}
OLD_MODEL_DIR=${OLD_MODEL_DIR:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/models/qwen3_jsl_lora_cc_e10_lr5e5}

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
EJSL_VIDEO_ROOT=${EJSL_VIDEO_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
EJSL_FRAME_ROOT=${EJSL_FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
STRUCTURE_TXT_ROOT=${STRUCTURE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}

MANIFEST=${MANIFEST:-"$WORK_DIR/manifests/ejsl_1920_oracle_keypoints_train80_test20.csv"}
SPLIT_CSV_DIR=${SPLIT_CSV_DIR:-"$WORK_DIR/manifests/splits"}
KEYPOINT_CACHE_DIR=${KEYPOINT_CACHE_DIR:-"$WORK_DIR/keypoints/ejsl_1920"}
FINETUNE_DIR=${FINETUNE_DIR:-"$WORK_DIR/models/old_jsl_ejsl_train80"}
EVAL_ROOT=${EVAL_ROOT:-"$WORK_DIR/eval_checkpoints"}
BEST_TXT_ROOT=${BEST_TXT_ROOT:-"$WORK_DIR/ejsl_nonoracle_txt_best"}
BEST_PREDICTIONS_JSONL=${BEST_PREDICTIONS_JSONL:-"$WORK_DIR/ejsl_nonoracle_predictions_best.jsonl"}
FINAL_METRICS_JSON=${FINAL_METRICS_JSON:-"$WORK_DIR/ejsl_1920_best_metrics.json"}
FINAL_METRICS_CSV=${FINAL_METRICS_CSV:-"$WORK_DIR/ejsl_1920_best_per_sample.csv"}

SEED=${SEED:-42}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
SAMPLE_FPS=${SAMPLE_FPS:-10}
MAX_FRAMES=${MAX_FRAMES:-0}
MODEL_COMPLEXITY=${MODEL_COMPLEXITY:-1}
NUM_VISUAL_TOKENS=${NUM_VISUAL_TOKENS:-64}
MAX_TARGET_TOKENS=${MAX_TARGET_TOKENS:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
SELECTION_METRIC=${SELECTION_METRIC:-mean_rougeL_f1_char}
REBUILD_MANIFEST=${REBUILD_MANIFEST:-0}
REGEN=${REGEN:-0}

mkdir -p "$WORK_DIR/manifests" "$EVAL_ROOT"

run_data_py() {
  conda run -n "$DATA_ENV" python "$@"
}

run_train_py() {
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$TRAIN_ENV" python "$@"
}

run_train_py_cpu() {
  conda run -n "$TRAIN_ENV" python "$@"
}

echo "[JSL-eJSL-split] old model: $OLD_MODEL_DIR"
echo "[JSL-eJSL-split] work dir: $WORK_DIR"

if [ "$REBUILD_MANIFEST" = "1" ] || [ ! -s "$MANIFEST" ]; then
  run_data_py JSL/build_ejsl_oracle_keypoint_manifest.py \
    --dial_list "$DIAL_LIST" \
    --video_root "$EJSL_VIDEO_ROOT" \
    --frame_root "$EJSL_FRAME_ROOT" \
    --structure_txt_root "$STRUCTURE_TXT_ROOT" \
    --keypoint_cache_dir "$KEYPOINT_CACHE_DIR" \
    --out_csv "$MANIFEST" \
    --split_csv_dir "$SPLIT_CSV_DIR" \
    --train_ratio "$TRAIN_RATIO" \
    --seed "$SEED" \
    --sample_fps "$SAMPLE_FPS" \
    --max_frames "$MAX_FRAMES" \
    --model_complexity "$MODEL_COMPLEXITY" \
    --resume
fi

run_train_py JSL/train_jsl_translation.py \
  --manifest_csv "$MANIFEST" \
  --output_dir "$FINETUNE_DIR" \
  --init_model_dir "$OLD_MODEL_DIR" \
  --train_split train \
  --valid_split test \
  --num_visual_tokens "$NUM_VISUAL_TOKENS" \
  --max_target_tokens "$MAX_TARGET_TOKENS" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --save_epochs 1 \
  --bf16

mapfile -t CHECKPOINTS < <(find "$FINETUNE_DIR" -maxdepth 1 -type d -name 'checkpoint-epoch*' | sort)
if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  echo "[JSL-eJSL-split] no checkpoint dirs under $FINETUNE_DIR" >&2
  exit 1
fi

for ckpt in "${CHECKPOINTS[@]}"; do
  ckpt_name="$(basename "$ckpt")"
  ckpt_eval_dir="$EVAL_ROOT/$ckpt_name"
  mkdir -p "$ckpt_eval_dir"
  echo "[JSL-eJSL-split] evaluate checkpoint=$ckpt_name"

  for split in train test; do
    pred_jsonl="$ckpt_eval_dir/${split}_predictions.jsonl"
    if [ "$REGEN" = "1" ] || [ ! -s "$pred_jsonl" ]; then
      run_train_py JSL/generate_jsl_manifest_predictions.py \
        --model_dir "$ckpt" \
        --manifest_csv "$MANIFEST" \
        --split "$split" \
        --out_jsonl "$pred_jsonl" \
        --batch_size "$BATCH_SIZE" \
        --num_visual_tokens "$NUM_VISUAL_TOKENS" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --bf16
    fi

    run_train_py_cpu JSL/evaluate_ejsl_predictions.py \
      --predictions_jsonl "$pred_jsonl" \
      --dial_list "$DIAL_LIST" \
      --structure_txt_root "$STRUCTURE_TXT_ROOT" \
      --sample_id_csv "$SPLIT_CSV_DIR/${split}.csv" \
      --out_json "$ckpt_eval_dir/${split}_metrics.json" \
      --out_csv "$ckpt_eval_dir/${split}_per_sample.csv"
  done
  cp "$ckpt_eval_dir/test_metrics.json" "$ckpt_eval_dir/metrics.json"
done

run_train_py_cpu JSL/select_best_ejsl_eval.py \
  --eval_root "$EVAL_ROOT" \
  --metric "$SELECTION_METRIC" \
  --out_csv "$EVAL_ROOT/checkpoint_ranking.csv" \
  --out_json "$EVAL_ROOT/best_checkpoint.json"

BEST_CKPT="$(run_train_py_cpu - <<PY
import json
from pathlib import Path
print(json.loads((Path("$EVAL_ROOT") / "best_checkpoint.json").read_text())["best_checkpoint"])
PY
)"
BEST_MODEL_DIR="$FINETUNE_DIR/$BEST_CKPT"

run_train_py JSL/generate_ejsl_non_oracle_txt.py \
  --model_dir "$BEST_MODEL_DIR" \
  --dial_list "$DIAL_LIST" \
  --video_root "$EJSL_VIDEO_ROOT" \
  --frame_root "$EJSL_FRAME_ROOT" \
  --structure_txt_root "$STRUCTURE_TXT_ROOT" \
  --output_txt_root "$BEST_TXT_ROOT" \
  --keypoint_cache_dir "$KEYPOINT_CACHE_DIR" \
  --predictions_jsonl "$BEST_PREDICTIONS_JSONL" \
  --batch_size "$BATCH_SIZE" \
  --num_visual_tokens "$NUM_VISUAL_TOKENS" \
  --sample_fps "$SAMPLE_FPS" \
  --max_frames "$MAX_FRAMES" \
  --model_complexity "$MODEL_COMPLEXITY" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --resume

run_train_py_cpu JSL/evaluate_ejsl_predictions.py \
  --predictions_jsonl "$BEST_PREDICTIONS_JSONL" \
  --dial_list "$DIAL_LIST" \
  --structure_txt_root "$STRUCTURE_TXT_ROOT" \
  --out_json "$FINAL_METRICS_JSON" \
  --out_csv "$FINAL_METRICS_CSV"

echo "[JSL-eJSL-split] done"
echo "[JSL-eJSL-split] best checkpoint: $BEST_CKPT"
echo "[JSL-eJSL-split] checkpoint ranking: $EVAL_ROOT/checkpoint_ranking.csv"
echo "[JSL-eJSL-split] final txt root: $BEST_TXT_ROOT"
echo "[JSL-eJSL-split] final metrics: $FINAL_METRICS_JSON"
