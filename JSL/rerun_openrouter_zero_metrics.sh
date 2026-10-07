#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT_DIR"

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
STRUCTURE_TXT_ROOT=${STRUCTURE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
OUT_ROOT=${OUT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/openrouter_ejsl_zero20_gpt_qwen_claude}
MODELS=${MODELS:-"openai/gpt-5.6-sol qwen/qwen3-vl-235b-a22b-instruct"}

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
  --out_csv "$OUT_ROOT/zero20_bleu_rouge_summary.csv" \
  --out_json "$OUT_ROOT/zero20_bleu_rouge_summary.json"

echo "[OpenRouter-zero-metrics] metrics: $OUT_ROOT/metrics"
echo "[OpenRouter-zero-metrics] summary csv: $OUT_ROOT/zero20_bleu_rouge_summary.csv"
