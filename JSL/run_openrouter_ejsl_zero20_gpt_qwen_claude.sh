#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before calling OpenRouter.}"

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
VIDEO_ROOT=${VIDEO_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
STRUCTURE_TXT_ROOT=${STRUCTURE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
OUT_ROOT=${OUT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/openrouter_ejsl_zero20_gpt_qwen_claude}

MODELS=${MODELS:-"openai/gpt-5.6-sol qwen/qwen3-vl-235b-a22b-instruct anthropic/claude-opus-5.5"}
LIMIT=${LIMIT:-20}
SELECTION=${SELECTION:-balanced}
INPUT_MODE=${INPUT_MODE:-frames}
SAMPLE_FPS=${SAMPLE_FPS:-4}
FRAME_DIR_FPS=${FRAME_DIR_FPS:-4}
NUM_FRAMES=${NUM_FRAMES:-48}
IMAGE_MAX_SIDE=${IMAGE_MAX_SIDE:-512}
JPEG_QUALITY=${JPEG_QUALITY:-70}
MAX_TOKENS=${MAX_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
REASONING_EFFORT=${REASONING_EFFORT:-}
DISABLE_REASONING=${DISABLE_REASONING:-1}
SLEEP_SEC=${SLEEP_SEC:-0.3}
RESUME=${RESUME:-1}

mkdir -p "$OUT_ROOT"

python JSL/inspect_ejsl_frame_fps.py \
  --dial_list "$DIAL_LIST" \
  --frame_root "$FRAME_ROOT" \
  --video_root "$VIDEO_ROOT" \
  --limit "$LIMIT" \
  --out_csv "$OUT_ROOT/ejsl_frame_fps_rows.csv" \
  --summary_json "$OUT_ROOT/ejsl_frame_fps_summary.json"

MODEL_FLAGS=()
for model in $MODELS; do
  MODEL_FLAGS+=(--model "$model")
done

RESUME_FLAGS=()
if [ "$RESUME" = "0" ]; then
  RESUME_FLAGS+=(--no_resume)
fi

REASONING_FLAGS=()
if [ "$DISABLE_REASONING" = "1" ]; then
  REASONING_FLAGS+=(--disable_reasoning)
fi

python JSL/openrouter_ejsl_video_text.py \
  --dial_list "$DIAL_LIST" \
  --frame_root "$FRAME_ROOT" \
  --video_root "$VIDEO_ROOT" \
  --out_root "$OUT_ROOT" \
  "${MODEL_FLAGS[@]}" \
  --limit "$LIMIT" \
  --selection "$SELECTION" \
  --input_mode "$INPUT_MODE" \
  --num_frames "$NUM_FRAMES" \
  --sample_fps "$SAMPLE_FPS" \
  --frame_dir_fps "$FRAME_DIR_FPS" \
  --image_max_side "$IMAGE_MAX_SIDE" \
  --jpeg_quality "$JPEG_QUALITY" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --reasoning_effort "$REASONING_EFFORT" \
  "${REASONING_FLAGS[@]}" \
  --response_mode plain_translation \
  --structure_txt_root "$STRUCTURE_TXT_ROOT" \
  --sleep_sec "$SLEEP_SEC" \
  "${RESUME_FLAGS[@]}"

mkdir -p "$OUT_ROOT/metrics"
for model in $MODELS; do
  model_name="$(printf '%s' "$model" | sed -E 's/[^A-Za-z0-9_.-]+/__/g; s/^_+//; s/_+$//')"
  predictions="$OUT_ROOT/responses/predictions_${model_name}.jsonl"
  python JSL/evaluate_ejsl_predictions.py \
    --predictions_jsonl "$predictions" \
    --dial_list "$DIAL_LIST" \
    --structure_txt_root "$STRUCTURE_TXT_ROOT" \
    --sample_id_csv "$OUT_ROOT/selected_samples.csv" \
    --out_json "$OUT_ROOT/metrics/${model_name}.json" \
    --out_csv "$OUT_ROOT/metrics/${model_name}_per_sample.csv"
done

python JSL/summarize_openrouter_zero_metrics.py \
  --out_root "$OUT_ROOT" \
  --models $MODELS \
  --out_csv "$OUT_ROOT/zero20_bleu_summary.csv" \
  --out_json "$OUT_ROOT/zero20_bleu_summary.json"

echo "[OpenRouter-zero20] out root: $OUT_ROOT"
echo "[OpenRouter-zero20] selected samples: $OUT_ROOT/selected_samples.csv"
echo "[OpenRouter-zero20] reviews: $OUT_ROOT/reviews"
echo "[OpenRouter-zero20] metrics: $OUT_ROOT/metrics"
echo "[OpenRouter-zero20] BLEU summary: $OUT_ROOT/zero20_bleu_summary.csv"
