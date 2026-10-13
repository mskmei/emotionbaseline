#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_tv_text_dropout_nonoracle}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-"$WORK_ROOT/feature_cache"}
TRANSLATED_ROOT=${TRANSLATED_ROOT:-"$WORK_ROOT/translated_txt"}
SWEEP_ROOT=${SWEEP_ROOT:-"$WORK_ROOT/candidates"}
MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/meld_anjs4_unified.pkl}

OLD_PRETRAINED_SPLIT_TXT_ROOT=${OLD_PRETRAINED_SPLIT_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_old_model_ejsl_split_finetune/ejsl_nonoracle_txt_best}
OLD_JSHUWA_TXT_ROOT=${OLD_JSHUWA_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/ejsl_nonoracle_txt}

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}

TEXT_DROPOUTS=${TEXT_DROPOUTS:-"0 0.25 0.5 0.75"}
TEXT_DROPOUT_SCOPE=${TEXT_DROPOUT_SCOPE:-utterance}

TRANSLATE_NONORACLE=${TRANSLATE_NONORACLE:-1}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-openai}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}
TRANSLATION_CACHE=${TRANSLATION_CACHE:-"$WORK_ROOT/translation_cache.jsonl"}
SEED_TRANSLATION_CACHE=${SEED_TRANSLATION_CACHE:-1}
TRANSLATE_FORCE=${TRANSLATE_FORCE:-0}

REBUILD_EJSL_FEATURES=${REBUILD_EJSL_FEATURES:-0}
RERUN_TRAIN=${RERUN_TRAIN:-0}

GRAPH_TYPE=${GRAPH_TYPE:-MMGCN}
EPOCHS=${EPOCHS:-15}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-0.0003}
L2=${L2:-0.00003}
DROPOUT=${DROPOUT:-0.4}
LOSS=${LOSS:-focal}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-5.0}
SEED=${SEED:-42}
SELECTION_SPLIT=${SELECTION_SPLIT:-source}
SELECTION_METRIC=${SELECTION_METRIC:-weighted_f1}

mkdir -p "$WORK_ROOT/features" "$FEATURE_CACHE_DIR" "$TRANSLATED_ROOT" "$SWEEP_ROOT"

sanitize_name() {
  python - "$1" <<'PY'
import re, sys
name = sys.argv[1]
name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
print(name or "candidate")
PY
}

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
    echo "[MMGCN-TV-TEXT-DROP] missing $label: $path" >&2
    exit 1
  fi
}

require_path "MELD_UNIFIED_PKL" "$MELD_UNIFIED_PKL"
require_path "old_pretrained_split txt root" "$OLD_PRETRAINED_SPLIT_TXT_ROOT"
require_path "old_jshuwa/qwen1.4 txt root" "$OLD_JSHUWA_TXT_ROOT"

if [ "$SEED_TRANSLATION_CACHE" = "1" ] && [ ! -s "$TRANSLATION_CACHE" ]; then
  mkdir -p "$(dirname "$TRANSLATION_CACHE")"
  for cache in \
    /raid_zoe/home/lr/maokeyu/sign/jsl_four_nonoracle_mmgcn/translation_cache.jsonl \
    /raid_zoe/home/lr/maokeyu/sign/mmgcn_nonoracle_text_sweep/translation_cache.jsonl
  do
    if [ -f "$cache" ]; then
      echo "[MMGCN-TV-TEXT-DROP] seed translation cache from $cache"
      cat "$cache" >> "$TRANSLATION_CACHE"
    fi
  done
fi

CANDIDATE_NAMES=(
  old_pretrained_split
  old_jshuwa_qwen14_pretrain
)
CANDIDATE_ROOTS=(
  "$OLD_PRETRAINED_SPLIT_TXT_ROOT"
  "$OLD_JSHUWA_TXT_ROOT"
)

echo "[MMGCN-TV-TEXT-DROP] work root: $WORK_ROOT"
echo "[MMGCN-TV-TEXT-DROP] MELD pkl: $MELD_UNIFIED_PKL"
echo "[MMGCN-TV-TEXT-DROP] text dropout rates: $TEXT_DROPOUTS scope=$TEXT_DROPOUT_SCOPE"

for idx in "${!CANDIDATE_NAMES[@]}"; do
  candidate="$(sanitize_name "${CANDIDATE_NAMES[$idx]}")"
  source_txt_root="${CANDIDATE_ROOTS[$idx]}"
  translated_txt_root="$TRANSLATED_ROOT/$candidate"
  ejsl_pkl="$WORK_ROOT/features/${candidate}_ejsl_anjs4.pkl"

  echo "[MMGCN-TV-TEXT-DROP] ===== text candidate=$candidate ====="
  echo "[MMGCN-TV-TEXT-DROP] source txt root: $source_txt_root"

  eval_txt_root="$source_txt_root"
  if [ "$TRANSLATE_NONORACLE" = "1" ]; then
    eval_txt_root="$translated_txt_root"
    if [ "$TRANSLATE_FORCE" = "1" ] || [ ! -f "$translated_txt_root/.translation_complete" ]; then
      echo "[MMGCN-TV-TEXT-DROP] translate Japanese -> English: $translated_txt_root"
      python JSL/translate_ejsl_txt_root.py \
        --input_txt_root "$source_txt_root" \
        --output_txt_root "$translated_txt_root" \
        --dial_list "$DIAL_LIST" \
        --cache_jsonl "$TRANSLATION_CACHE" \
        --backend "$TRANSLATION_BACKEND" \
        --openai_model "$OPENAI_MODEL"
    else
      echo "[MMGCN-TV-TEXT-DROP] reuse translated txt root: $translated_txt_root"
    fi
  fi

  if [ "$REBUILD_EJSL_FEATURES" = "1" ] || [ ! -f "$ejsl_pkl" ]; then
    echo "[MMGCN-TV-TEXT-DROP] build eJSL pkl: $ejsl_pkl"
    CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/build_unified_meld_ejsl_pkl.py \
      --meld_root "$MELD_RAW_ROOT" \
      --ejsl_txt_root "$eval_txt_root" \
      --ejsl_dial_list "$DIAL_LIST" \
      --ejsl_frame_root "$FRAME_ROOT" \
      --ejsl_mp4_root "$MP4_ROOT" \
      --out_meld_pkl "$WORK_ROOT/features/unused_meld_for_${candidate}.pkl" \
      --out_ejsl_pkl "$ejsl_pkl" \
      --cache_dir "$FEATURE_CACHE_DIR" \
      --skip_meld \
      --fp16
  else
    echo "[MMGCN-TV-TEXT-DROP] reuse eJSL pkl: $ejsl_pkl"
  fi

  for text_dropout in $TEXT_DROPOUTS; do
    label="$(rate_label "$text_dropout")"
    run_candidate="${candidate}_${label}"
    candidate_dir="$SWEEP_ROOT/$run_candidate"
    run_dir="$candidate_dir/runs"
    mkdir -p "$candidate_dir"

    cat > "$candidate_dir/candidate_meta.json" <<JSON
{
  "candidate": "$run_candidate",
  "base_candidate": "$candidate",
  "txt_root": "$source_txt_root",
  "translated_txt_root": "$eval_txt_root",
  "ejsl_pkl": "$ejsl_pkl",
  "meld_pkl": "$MELD_UNIFIED_PKL",
  "translate_nonoracle": "$TRANSLATE_NONORACLE",
  "translation_backend": "$TRANSLATION_BACKEND",
  "train_text_dropout": "$text_dropout",
  "train_text_dropout_scope": "$TEXT_DROPOUT_SCOPE"
}
JSON

    if [ "$RERUN_TRAIN" = "1" ] || [ ! -f "$run_dir/tv/external_test_best_summary.json" ]; then
      echo "[MMGCN-TV-TEXT-DROP] train tv candidate=$run_candidate text_dropout=$text_dropout"
      CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
        --train_pkl "$MELD_UNIFIED_PKL" \
        --external_test_pkl "$ejsl_pkl" \
        --out_dir "$run_dir" \
        --modalities tv \
        --graph_type "$GRAPH_TYPE" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --l2 "$L2" \
        --dropout "$DROPOUT" \
        --train_text_dropout "$text_dropout" \
        --train_text_dropout_scope "$TEXT_DROPOUT_SCOPE" \
        --loss "$LOSS" \
        --focal_gamma "$FOCAL_GAMMA" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --seed "$SEED" \
        --selection_split "$SELECTION_SPLIT" \
        --selection_metric "$SELECTION_METRIC"
    else
      echo "[MMGCN-TV-TEXT-DROP] reuse MMGCN report: $run_dir/tv"
    fi
  done
done

python MMGCN/summarize_nonoracle_text_sweep.py \
  --sweep_root "$SWEEP_ROOT" \
  --out_csv "$WORK_ROOT/tv_text_dropout_rank.csv" \
  --out_json "$WORK_ROOT/tv_text_dropout_rank.json" \
  --rank_metric weighted_f1

echo "[MMGCN-TV-TEXT-DROP] done"
echo "[MMGCN-TV-TEXT-DROP] rank csv: $WORK_ROOT/tv_text_dropout_rank.csv"
echo "[MMGCN-TV-TEXT-DROP] rank json: $WORK_ROOT/tv_text_dropout_rank.json"
