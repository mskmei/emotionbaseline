#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}

JSL_FULL_EVAL_ROOT=${JSL_FULL_EVAL_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle_qwen3_4b_full_clean/ejsl_eval/qwen3_4b_jshuwa_all_clean}
OLD_JSL_TXT_ROOT=${OLD_JSL_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/ejsl_nonoracle_txt}
EXTRA_TXT_ROOTS=${EXTRA_TXT_ROOTS:-}

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_nonoracle_text_sweep}
MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$WORK_ROOT/features/meld_anjs4_unified.pkl"}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-"$WORK_ROOT/feature_cache"}
SWEEP_ROOT=${SWEEP_ROOT:-"$WORK_ROOT/candidates"}
TRANSLATED_ROOT=${TRANSLATED_ROOT:-"$WORK_ROOT/translated_txt"}
TRANSLATION_CACHE=${TRANSLATION_CACHE:-"$WORK_ROOT/translation_cache.jsonl"}

GPU=${GPU:-0}
MODALITIES=${MODALITIES:-"text tv"}
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

TRANSLATE_NONORACLE=${TRANSLATE_NONORACLE:-1}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-openai}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}
TRANSLATE_FORCE=${TRANSLATE_FORCE:-0}
REBUILD_MELD_FEATURES=${REBUILD_MELD_FEATURES:-0}
REBUILD_EJSL_FEATURES=${REBUILD_EJSL_FEATURES:-0}
RERUN_TRAIN=${RERUN_TRAIN:-0}

mkdir -p "$WORK_ROOT/features" "$FEATURE_CACHE_DIR" "$SWEEP_ROOT" "$TRANSLATED_ROOT"

sanitize_name() {
  python - "$1" <<'PY'
import re, sys
name = sys.argv[1]
name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
print(name or "candidate")
PY
}

add_candidate() {
  local name="$1"
  local root="$2"
  if [ -d "$root" ]; then
    CANDIDATE_NAMES+=("$name")
    CANDIDATE_ROOTS+=("$root")
  else
    echo "[MMGCN-NONORACLE] skip missing txt root: $name -> $root"
  fi
}

CANDIDATE_NAMES=()
CANDIDATE_ROOTS=()

if [ -d "$JSL_FULL_EVAL_ROOT" ]; then
  while IFS= read -r txt_root; do
    ckpt_name="$(basename "$(dirname "$txt_root")")"
    add_candidate "qwen3_4b_${ckpt_name}" "$txt_root"
  done < <(find "$JSL_FULL_EVAL_ROOT" -mindepth 2 -maxdepth 2 -type d -name txt | sort)
else
  echo "[MMGCN-NONORACLE] missing JSL_FULL_EVAL_ROOT: $JSL_FULL_EVAL_ROOT"
fi

add_candidate "old_jshuwa_nonoracle" "$OLD_JSL_TXT_ROOT"

if [ -n "$EXTRA_TXT_ROOTS" ]; then
  IFS=':' read -r -a EXTRA_ROOT_ARRAY <<< "$EXTRA_TXT_ROOTS"
  for extra_root in "${EXTRA_ROOT_ARRAY[@]}"; do
    [ -n "$extra_root" ] || continue
    add_candidate "extra_$(sanitize_name "$(basename "$extra_root")")" "$extra_root"
  done
fi

if [ "${#CANDIDATE_NAMES[@]}" -eq 0 ]; then
  echo "[MMGCN-NONORACLE] no candidate txt roots found" >&2
  exit 1
fi

echo "[MMGCN-NONORACLE] candidates=${#CANDIDATE_NAMES[@]}"
for i in "${!CANDIDATE_NAMES[@]}"; do
  echo "  ${CANDIDATE_NAMES[$i]} -> ${CANDIDATE_ROOTS[$i]}"
done

if [ "$REBUILD_MELD_FEATURES" = "1" ] || [ ! -f "$MELD_UNIFIED_PKL" ]; then
  echo "[MMGCN-NONORACLE] build/rebuild MELD pkl: $MELD_UNIFIED_PKL"
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/build_unified_meld_ejsl_pkl.py \
    --meld_root "$MELD_RAW_ROOT" \
    --out_meld_pkl "$MELD_UNIFIED_PKL" \
    --cache_dir "$FEATURE_CACHE_DIR" \
    --skip_ejsl \
    --fp16
fi

for i in "${!CANDIDATE_NAMES[@]}"; do
  candidate_raw="${CANDIDATE_NAMES[$i]}"
  candidate="$(sanitize_name "$candidate_raw")"
  source_txt_root="${CANDIDATE_ROOTS[$i]}"
  candidate_dir="$SWEEP_ROOT/$candidate"
  translated_txt_root="$TRANSLATED_ROOT/$candidate"
  ejsl_pkl="$WORK_ROOT/features/${candidate}_ejsl_anjs4.pkl"
  run_dir="$candidate_dir/runs"
  mkdir -p "$candidate_dir"

  echo "[MMGCN-NONORACLE] ===== candidate=$candidate ====="
  echo "[MMGCN-NONORACLE] source txt root: $source_txt_root"

  eval_txt_root="$source_txt_root"
  if [ "$TRANSLATE_NONORACLE" = "1" ]; then
    eval_txt_root="$translated_txt_root"
    if [ "$TRANSLATE_FORCE" = "1" ] || [ ! -f "$translated_txt_root/.translation_complete" ]; then
      echo "[MMGCN-NONORACLE] translate Japanese non-oracle txt -> English: $translated_txt_root"
      python JSL/translate_ejsl_txt_root.py \
        --input_txt_root "$source_txt_root" \
        --output_txt_root "$translated_txt_root" \
        --dial_list "$DIAL_LIST" \
        --cache_jsonl "$TRANSLATION_CACHE" \
        --backend "$TRANSLATION_BACKEND" \
        --openai_model "$OPENAI_MODEL"
    else
      echo "[MMGCN-NONORACLE] reuse translated txt root: $translated_txt_root"
    fi
  fi

  if [ "$REBUILD_EJSL_FEATURES" = "1" ] || [ ! -f "$ejsl_pkl" ]; then
    echo "[MMGCN-NONORACLE] build eJSL pkl: $ejsl_pkl"
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
  fi

  cat > "$candidate_dir/candidate_meta.json" <<JSON
{
  "candidate": "$candidate",
  "candidate_raw": "$candidate_raw",
  "txt_root": "$source_txt_root",
  "translated_txt_root": "$eval_txt_root",
  "ejsl_pkl": "$ejsl_pkl",
  "meld_pkl": "$MELD_UNIFIED_PKL",
  "translate_nonoracle": "$TRANSLATE_NONORACLE",
  "translation_backend": "$TRANSLATION_BACKEND"
}
JSON

  if [ "$RERUN_TRAIN" = "1" ] || [ ! -f "$run_dir/text/external_test_best_summary.json" ] || [ ! -f "$run_dir/tv/external_test_best_summary.json" ]; then
    echo "[MMGCN-NONORACLE] train MELD -> evaluate eJSL modalities=$MODALITIES"
    CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
      --train_pkl "$MELD_UNIFIED_PKL" \
      --external_test_pkl "$ejsl_pkl" \
      --out_dir "$run_dir" \
      --modalities $MODALITIES \
      --graph_type "$GRAPH_TYPE" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --l2 "$L2" \
      --dropout "$DROPOUT" \
      --loss "$LOSS" \
      --focal_gamma "$FOCAL_GAMMA" \
      --max_grad_norm "$MAX_GRAD_NORM" \
      --seed "$SEED" \
      --selection_split "$SELECTION_SPLIT" \
      --selection_metric "$SELECTION_METRIC"
  else
    echo "[MMGCN-NONORACLE] reuse existing MMGCN reports: $run_dir"
  fi
done

python MMGCN/summarize_nonoracle_text_sweep.py \
  --sweep_root "$SWEEP_ROOT" \
  --out_csv "$WORK_ROOT/nonoracle_text_rank.csv" \
  --out_json "$WORK_ROOT/nonoracle_text_rank.json" \
  --rank_metric weighted_f1

echo "[MMGCN-NONORACLE] done"
echo "[MMGCN-NONORACLE] rank csv: $WORK_ROOT/nonoracle_text_rank.csv"
echo "[MMGCN-NONORACLE] rank json: $WORK_ROOT/nonoracle_text_rank.json"
