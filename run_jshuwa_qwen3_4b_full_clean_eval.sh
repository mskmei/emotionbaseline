#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

DATA_ENV=${DATA_ENV:-telme39}
TRAIN_ENV=${TRAIN_ENV:-base}
GPU=${GPU:-0}

JSL_WORK_DIR=${JSL_WORK_DIR:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle_qwen3_4b_full_clean}
HF_HOME=${HF_HOME:-"$JSL_WORK_DIR/hf_home"}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$JSL_WORK_DIR/hf_datasets_cache"}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$JSL_WORK_DIR/hf_models"}
export HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE

JSHUWA_METADATA_CSV=${JSHUWA_METADATA_CSV:-"$JSL_WORK_DIR/manifests/jshuwa_metadata_train.csv"}
JSHUWA_VIDEO_DIR=${JSHUWA_VIDEO_DIR:-"$JSL_WORK_DIR/jshuwa_youtube_videos"}
JSHUWA_SUBTITLE_DIR=${JSHUWA_SUBTITLE_DIR:-"$JSL_WORK_DIR/jshuwa_youtube_subtitles"}
JSHUWA_RAW_MANIFEST=${JSHUWA_RAW_MANIFEST:-"$JSL_WORK_DIR/manifests/jshuwa_cc_all_raw_manifest.csv"}
JSHUWA_CLEAN_MANIFEST=${JSHUWA_CLEAN_MANIFEST:-"$JSL_WORK_DIR/manifests/jshuwa_cc_all_clean_manifest.csv"}
JSHUWA_CLEAN_REPORT=${JSHUWA_CLEAN_REPORT:-"$JSL_WORK_DIR/manifests/jshuwa_cc_all_clean_report.json"}
JSHUWA_KEYPOINT_DIR=${JSHUWA_KEYPOINT_DIR:-"$JSL_WORK_DIR/keypoints/jshuwa_cc_all_clean"}
JSHUWA_KEYPOINT_MANIFEST=${JSHUWA_KEYPOINT_MANIFEST:-"$JSL_WORK_DIR/manifests/jshuwa_cc_all_clean_keypoints.csv"}

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
EJSL_VIDEO_ROOT=${EJSL_VIDEO_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
EJSL_FRAME_ROOT=${EJSL_FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
STRUCTURE_TXT_ROOT=${STRUCTURE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
EJSL_KEYPOINT_CACHE_DIR=${EJSL_KEYPOINT_CACHE_DIR:-"$JSL_WORK_DIR/keypoints/ejsl_1920"}

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
MODEL_TAG=${MODEL_TAG:-qwen3_4b_jshuwa_all_clean}
MODEL_DIR=${MODEL_DIR:-"$JSL_WORK_DIR/models/$MODEL_TAG"}
EVAL_ROOT=${EVAL_ROOT:-"$JSL_WORK_DIR/ejsl_eval/$MODEL_TAG"}
FINAL_TXT_ROOT=${FINAL_TXT_ROOT:-"$JSL_WORK_DIR/ejsl_nonoracle_txt_best"}

YTDLP_EXTRACTOR_ARGS=${YTDLP_EXTRACTOR_ARGS:-youtube:player_client=android_vr}
YTDLP_EXTRACTOR_ARGS_CANDIDATES=${YTDLP_EXTRACTOR_ARGS_CANDIDATES:-"youtube:player_client=android_vr;youtube:player_client=android;youtube:player_client=ios;youtube:player_client=web;youtube:player_client=mweb"}
YTDLP_VIDEO_FORMAT=${YTDLP_VIDEO_FORMAT:-18/best[height<=360][ext=mp4]/best[height<=480][ext=mp4]/best}
YTDLP_COOKIES=${YTDLP_COOKIES:-}
YTDLP_COOKIES_FROM_BROWSER=${YTDLP_COOKIES_FROM_BROWSER:-}

SAMPLE_FPS=${SAMPLE_FPS:-10}
MAX_FRAMES=${MAX_FRAMES:-0}
MODEL_COMPLEXITY=${MODEL_COMPLEXITY:-1}

CLEAN_MIN_DURATION=${CLEAN_MIN_DURATION:-0.6}
CLEAN_MAX_DURATION=${CLEAN_MAX_DURATION:-20.0}
CLEAN_MIN_TEXT_CHARS=${CLEAN_MIN_TEXT_CHARS:-4}
CLEAN_MAX_TEXT_CHARS=${CLEAN_MAX_TEXT_CHARS:-80}
CLEAN_MAX_SAME_VIDEO_DUP=${CLEAN_MAX_SAME_VIDEO_DUP:-1}
CLEAN_MAX_DUP=${CLEAN_MAX_DUP:-10}
CLEAN_MAX_SHORT_DUP=${CLEAN_MAX_SHORT_DUP:-3}
CLEAN_DROP_META_TERMS=${CLEAN_DROP_META_TERMS:-1}

EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LR=${LR:-5e-5}
NUM_VISUAL_TOKENS=${NUM_VISUAL_TOKENS:-64}
PROJECTOR_HIDDEN=${PROJECTOR_HIDDEN:-2048}
MAX_TARGET_TOKENS=${MAX_TARGET_TOKENS:-96}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-2}
EVAL_LIMIT=${EVAL_LIMIT:-0}
SELECTION_METRIC=${SELECTION_METRIC:-corpus_bleu4_char}
RESUME_GENERATION=${RESUME_GENERATION:-1}

mkdir -p "$JSL_WORK_DIR/manifests" "$JSL_WORK_DIR/models" "$EVAL_ROOT"

run_data_py() {
  conda run -n "$DATA_ENV" python "$@"
}

run_train_py() {
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$TRAIN_ENV" python "$@"
}

YTDLP_FLAGS=()
if [ -n "$YTDLP_COOKIES" ]; then
  YTDLP_FLAGS+=(--cookies "$YTDLP_COOKIES")
fi
if [ -n "$YTDLP_COOKIES_FROM_BROWSER" ]; then
  YTDLP_FLAGS+=(--cookies_from_browser "$YTDLP_COOKIES_FROM_BROWSER")
fi

echo "[JSL-FULL] data env=$DATA_ENV train env=$TRAIN_ENV base_model=$BASE_MODEL"
echo "[JSL-FULL] work dir=$JSL_WORK_DIR"

run_data_py JSL/download_jshuwa_metadata.py \
  --out_csv "$JSHUWA_METADATA_CSV"

run_data_py JSL/build_jshuwa_cc_manifest.py \
  --metadata_csv "$JSHUWA_METADATA_CSV" \
  --video_dir "$JSHUWA_VIDEO_DIR" \
  --subtitle_dir "$JSHUWA_SUBTITLE_DIR" \
  --out_csv "$JSHUWA_RAW_MANIFEST" \
  --source cc \
  --video_format "$YTDLP_VIDEO_FORMAT" \
  --extractor_args "$YTDLP_EXTRACTOR_ARGS" \
  --extractor_args_candidates "$YTDLP_EXTRACTOR_ARGS_CANDIDATES" \
  --download_videos \
  --download_subtitles \
  --skip_missing \
  --max_yids 0 \
  --max_rows 0 \
  "${YTDLP_FLAGS[@]}"

CLEAN_FLAGS=()
if [ "$CLEAN_DROP_META_TERMS" = "1" ]; then
  CLEAN_FLAGS+=(--drop_meta_terms)
fi

run_data_py JSL/clean_jshuwa_manifest.py \
  --in_csv "$JSHUWA_RAW_MANIFEST" \
  --out_csv "$JSHUWA_CLEAN_MANIFEST" \
  --report_json "$JSHUWA_CLEAN_REPORT" \
  --min_duration "$CLEAN_MIN_DURATION" \
  --max_duration "$CLEAN_MAX_DURATION" \
  --min_text_chars "$CLEAN_MIN_TEXT_CHARS" \
  --max_text_chars "$CLEAN_MAX_TEXT_CHARS" \
  --max_same_video_duplicate "$CLEAN_MAX_SAME_VIDEO_DUP" \
  --max_duplicate_text "$CLEAN_MAX_DUP" \
  --max_short_duplicate_text "$CLEAN_MAX_SHORT_DUP" \
  "${CLEAN_FLAGS[@]}"

run_data_py JSL/extract_mediapipe_keypoints.py \
  --manifest_csv "$JSHUWA_CLEAN_MANIFEST" \
  --output_dir "$JSHUWA_KEYPOINT_DIR" \
  --out_manifest_csv "$JSHUWA_KEYPOINT_MANIFEST" \
  --sample_fps "$SAMPLE_FPS" \
  --max_frames "$MAX_FRAMES" \
  --model_complexity "$MODEL_COMPLEXITY" \
  --resume \
  --skip_errors

run_train_py JSL/check_qwen_env.py \
  --base_model "$BASE_MODEL"

run_train_py JSL/train_jsl_translation.py \
  --manifest_csv "$JSHUWA_KEYPOINT_MANIFEST" \
  --output_dir "$MODEL_DIR" \
  --base_model "$BASE_MODEL" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --lr "$LR" \
  --num_visual_tokens "$NUM_VISUAL_TOKENS" \
  --projector_hidden "$PROJECTOR_HIDDEN" \
  --max_target_tokens "$MAX_TARGET_TOKENS" \
  --save_epochs 1 \
  --bf16

CHECKPOINTS=()
while IFS= read -r ckpt; do
  CHECKPOINTS+=("$ckpt")
done < <(find "$MODEL_DIR" -maxdepth 1 -type d -name 'checkpoint-epoch*' | sort)
CHECKPOINTS+=("$MODEL_DIR")

GEN_RESUME_FLAG=()
if [ "$RESUME_GENERATION" = "1" ]; then
  GEN_RESUME_FLAG+=(--resume)
fi

for ckpt in "${CHECKPOINTS[@]}"; do
  ckpt_name="$(basename "$ckpt")"
  ckpt_eval_dir="$EVAL_ROOT/$ckpt_name"
  pred_jsonl="$ckpt_eval_dir/predictions.jsonl"
  txt_root="$ckpt_eval_dir/txt"
  mkdir -p "$ckpt_eval_dir"
  echo "[JSL-FULL] generate/eval checkpoint=$ckpt_name"

  run_train_py JSL/generate_ejsl_non_oracle_txt.py \
    --model_dir "$ckpt" \
    --dial_list "$DIAL_LIST" \
    --video_root "$EJSL_VIDEO_ROOT" \
    --frame_root "$EJSL_FRAME_ROOT" \
    --structure_txt_root "$STRUCTURE_TXT_ROOT" \
    --output_txt_root "$txt_root" \
    --keypoint_cache_dir "$EJSL_KEYPOINT_CACHE_DIR" \
    --predictions_jsonl "$pred_jsonl" \
    --batch_size "$GEN_BATCH_SIZE" \
    --num_visual_tokens "$NUM_VISUAL_TOKENS" \
    --sample_fps "$SAMPLE_FPS" \
    --max_frames "$MAX_FRAMES" \
    --model_complexity "$MODEL_COMPLEXITY" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --limit "$EVAL_LIMIT" \
    "${GEN_RESUME_FLAG[@]}"

  run_train_py JSL/evaluate_ejsl_predictions.py \
    --predictions_jsonl "$pred_jsonl" \
    --dial_list "$DIAL_LIST" \
    --structure_txt_root "$STRUCTURE_TXT_ROOT" \
    --out_json "$ckpt_eval_dir/metrics.json" \
    --out_csv "$ckpt_eval_dir/per_sample_metrics.csv" \
    --limit "$EVAL_LIMIT"
done

run_train_py JSL/select_best_ejsl_eval.py \
  --eval_root "$EVAL_ROOT" \
  --metric "$SELECTION_METRIC" \
  --out_csv "$EVAL_ROOT/checkpoint_ranking.csv" \
  --out_json "$EVAL_ROOT/best_checkpoint.json"

BEST_CKPT="$(python - <<PY
import json
from pathlib import Path
p = Path("$EVAL_ROOT") / "best_checkpoint.json"
print(json.loads(p.read_text())["best_checkpoint"])
PY
)"
ln -sfn "$EVAL_ROOT/$BEST_CKPT/txt" "$FINAL_TXT_ROOT"

echo "[JSL-FULL] done"
echo "[JSL-FULL] raw manifest: $JSHUWA_RAW_MANIFEST"
echo "[JSL-FULL] clean manifest: $JSHUWA_CLEAN_MANIFEST"
echo "[JSL-FULL] clean report: $JSHUWA_CLEAN_REPORT"
echo "[JSL-FULL] model dir: $MODEL_DIR"
echo "[JSL-FULL] eval root: $EVAL_ROOT"
echo "[JSL-FULL] best checkpoint: $BEST_CKPT"
echo "[JSL-FULL] best txt root symlink: $FINAL_TXT_ROOT"
