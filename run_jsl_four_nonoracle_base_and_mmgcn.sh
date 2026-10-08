#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-1.7B}
OLD_MODEL_DIR=${OLD_MODEL_DIR:-/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/models/qwen3_jsl_lora_cc}

OLD_SPLIT_WORK=${OLD_SPLIT_WORK:-/raid_zoe/home/lr/maokeyu/sign/jsl_old_model_ejsl_split_finetune}
OLD_SOLO_WORK=${OLD_SOLO_WORK:-/raid_zoe/home/lr/maokeyu/sign/jsl_old_model_ejsl_solo_finetune}
BASE_SPLIT_WORK=${BASE_SPLIT_WORK:-/raid_zoe/home/lr/maokeyu/sign/jsl_base_model_ejsl_split_finetune}
BASE_SOLO_WORK=${BASE_SOLO_WORK:-/raid_zoe/home/lr/maokeyu/sign/jsl_base_model_ejsl_solo_finetune}

OLD_SPLIT_TXT_ROOT=${OLD_SPLIT_TXT_ROOT:-"$OLD_SPLIT_WORK/ejsl_nonoracle_txt_best"}
OLD_SOLO_TXT_ROOT=${OLD_SOLO_TXT_ROOT:-"$OLD_SOLO_WORK/ejsl_nonoracle_txt_best"}
BASE_SPLIT_TXT_ROOT=${BASE_SPLIT_TXT_ROOT:-"$BASE_SPLIT_WORK/ejsl_nonoracle_txt_best"}
BASE_SOLO_TXT_ROOT=${BASE_SOLO_TXT_ROOT:-"$BASE_SOLO_WORK/ejsl_nonoracle_txt_best"}

RUN_OLD_SPLIT=${RUN_OLD_SPLIT:-1}
RUN_OLD_SOLO=${RUN_OLD_SOLO:-1}
RUN_BASE_SPLIT=${RUN_BASE_SPLIT:-1}
RUN_BASE_SOLO=${RUN_BASE_SOLO:-1}
OLD_SPLIT_EPOCHS=${OLD_SPLIT_EPOCHS:-20}
OLD_SOLO_EPOCHS=${OLD_SOLO_EPOCHS:-40}
BASE_SPLIT_EPOCHS=${BASE_SPLIT_EPOCHS:-20}
BASE_SOLO_EPOCHS=${BASE_SOLO_EPOCHS:-40}
OLD_REGEN=${OLD_REGEN:-1}
BASE_REGEN=${BASE_REGEN:-1}

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/jsl_four_nonoracle_mmgcn}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-"$WORK_ROOT/feature_cache"}
MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$WORK_ROOT/features/meld_anjs4_unified.pkl"}
TRANSLATED_ROOT=${TRANSLATED_ROOT:-"$WORK_ROOT/translated_txt"}
TRANSLATION_CACHE=${TRANSLATION_CACHE:-"$WORK_ROOT/translation_cache.jsonl"}
SWEEP_ROOT=${SWEEP_ROOT:-"$WORK_ROOT/candidates"}

TRANSLATE_NONORACLE=${TRANSLATE_NONORACLE:-1}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-openai}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}
TRANSLATE_FORCE=${TRANSLATE_FORCE:-0}

REBUILD_MELD_FEATURES=${REBUILD_MELD_FEATURES:-0}
REBUILD_EJSL_FEATURES=${REBUILD_EJSL_FEATURES:-0}
RERUN_TRAIN=${RERUN_TRAIN:-0}

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

mkdir -p "$WORK_ROOT/features" "$FEATURE_CACHE_DIR" "$TRANSLATED_ROOT" "$SWEEP_ROOT"

sanitize_name() {
  python - "$1" <<'PY'
import re, sys
name = sys.argv[1]
name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
print(name or "candidate")
PY
}

require_txt_root() {
  local name="$1"
  local root="$2"
  if [ ! -d "$root" ]; then
    echo "[JSL-4NONORACLE] missing txt root for $name: $root" >&2
    exit 1
  fi
}

echo "[JSL-4NONORACLE] ===== Step 1: old-pretrained non-oracle reruns ====="
if [ "$RUN_OLD_SPLIT" = "1" ]; then
  echo "[JSL-4NONORACLE] run old-pretrained split -> $OLD_SPLIT_WORK"
  USE_PRETRAINED=1 \
  OLD_MODEL_DIR="$OLD_MODEL_DIR" \
  BASE_MODEL="$BASE_MODEL" \
  WORK_DIR="$OLD_SPLIT_WORK" \
  GPU="$GPU" \
  EPOCHS="$OLD_SPLIT_EPOCHS" \
  REGEN="$OLD_REGEN" \
  bash JSL/run_old_jsl_finetune_ejsl_split.sh
else
  echo "[JSL-4NONORACLE] skip old-pretrained split"
fi

if [ "$RUN_OLD_SOLO" = "1" ]; then
  echo "[JSL-4NONORACLE] run old-pretrained solo -> $OLD_SOLO_WORK"
  USE_PRETRAINED=1 \
  OLD_MODEL_DIR="$OLD_MODEL_DIR" \
  BASE_MODEL="$BASE_MODEL" \
  WORK_DIR="$OLD_SOLO_WORK" \
  GPU="$GPU" \
  EPOCHS="$OLD_SOLO_EPOCHS" \
  REGEN="$OLD_REGEN" \
  bash JSL/run_old_jsl_finetune_ejsl_solo.sh
else
  echo "[JSL-4NONORACLE] skip old-pretrained solo"
fi

echo "[JSL-4NONORACLE] ===== Step 2: base-Qwen non-oracle runs ====="
if [ "$RUN_BASE_SPLIT" = "1" ]; then
  echo "[JSL-4NONORACLE] run base split -> $BASE_SPLIT_WORK"
  USE_PRETRAINED=0 \
  BASE_MODEL="$BASE_MODEL" \
  WORK_DIR="$BASE_SPLIT_WORK" \
  GPU="$GPU" \
  EPOCHS="$BASE_SPLIT_EPOCHS" \
  REGEN="$BASE_REGEN" \
  bash JSL/run_old_jsl_finetune_ejsl_split.sh
else
  echo "[JSL-4NONORACLE] skip base split"
fi

if [ "$RUN_BASE_SOLO" = "1" ]; then
  echo "[JSL-4NONORACLE] run base solo -> $BASE_SOLO_WORK"
  USE_PRETRAINED=0 \
  BASE_MODEL="$BASE_MODEL" \
  WORK_DIR="$BASE_SOLO_WORK" \
  GPU="$GPU" \
  EPOCHS="$BASE_SOLO_EPOCHS" \
  REGEN="$BASE_REGEN" \
  bash JSL/run_old_jsl_finetune_ejsl_solo.sh
else
  echo "[JSL-4NONORACLE] skip base solo"
fi

NAMES=(
  old_pretrained_split
  old_pretrained_solo
  qwen_base_split
  qwen_base_solo
)
ROOTS=(
  "$OLD_SPLIT_TXT_ROOT"
  "$OLD_SOLO_TXT_ROOT"
  "$BASE_SPLIT_TXT_ROOT"
  "$BASE_SOLO_TXT_ROOT"
)

echo "[JSL-4NONORACLE] ===== Step 3: validate four txt roots ====="
for idx in "${!NAMES[@]}"; do
  require_txt_root "${NAMES[$idx]}" "${ROOTS[$idx]}"
  echo "  ${NAMES[$idx]} -> ${ROOTS[$idx]}"
done

echo "[JSL-4NONORACLE] ===== Step 4: build/reuse MELD features ====="
if [ "$REBUILD_MELD_FEATURES" = "1" ] || [ ! -f "$MELD_UNIFIED_PKL" ]; then
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/build_unified_meld_ejsl_pkl.py \
    --meld_root "$MELD_RAW_ROOT" \
    --out_meld_pkl "$MELD_UNIFIED_PKL" \
    --cache_dir "$FEATURE_CACHE_DIR" \
    --skip_ejsl \
    --fp16
else
  echo "[JSL-4NONORACLE] reuse MELD pkl: $MELD_UNIFIED_PKL"
fi

echo "[JSL-4NONORACLE] ===== Step 5: translate, build eJSL features, run MMGCN ====="
for idx in "${!NAMES[@]}"; do
  raw_name="${NAMES[$idx]}"
  candidate="$(sanitize_name "$raw_name")"
  source_txt_root="${ROOTS[$idx]}"
  candidate_dir="$SWEEP_ROOT/$candidate"
  translated_txt_root="$TRANSLATED_ROOT/$candidate"
  ejsl_pkl="$WORK_ROOT/features/${candidate}_ejsl_anjs4.pkl"
  run_dir="$candidate_dir/runs"
  mkdir -p "$candidate_dir"

  echo "[JSL-4NONORACLE] ----- candidate=$candidate -----"
  echo "[JSL-4NONORACLE] source txt root: $source_txt_root"

  eval_txt_root="$source_txt_root"
  if [ "$TRANSLATE_NONORACLE" = "1" ]; then
    eval_txt_root="$translated_txt_root"
    if [ "$TRANSLATE_FORCE" = "1" ] || [ ! -f "$translated_txt_root/.translation_complete" ]; then
      echo "[JSL-4NONORACLE] translate Japanese -> English: $translated_txt_root"
      python JSL/translate_ejsl_txt_root.py \
        --input_txt_root "$source_txt_root" \
        --output_txt_root "$translated_txt_root" \
        --dial_list "$DIAL_LIST" \
        --cache_jsonl "$TRANSLATION_CACHE" \
        --backend "$TRANSLATION_BACKEND" \
        --openai_model "$OPENAI_MODEL"
    else
      echo "[JSL-4NONORACLE] reuse translated txt root: $translated_txt_root"
    fi
  fi

  if [ "$REBUILD_EJSL_FEATURES" = "1" ] || [ ! -f "$ejsl_pkl" ]; then
    echo "[JSL-4NONORACLE] build eJSL pkl: $ejsl_pkl"
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
    echo "[JSL-4NONORACLE] reuse eJSL pkl: $ejsl_pkl"
  fi

  cat > "$candidate_dir/candidate_meta.json" <<JSON
{
  "candidate": "$candidate",
  "txt_root": "$source_txt_root",
  "translated_txt_root": "$eval_txt_root",
  "ejsl_pkl": "$ejsl_pkl",
  "meld_pkl": "$MELD_UNIFIED_PKL",
  "translate_nonoracle": "$TRANSLATE_NONORACLE",
  "translation_backend": "$TRANSLATION_BACKEND"
}
JSON

  if [ "$RERUN_TRAIN" = "1" ] || [ ! -f "$run_dir/text/external_test_best_summary.json" ] || [ ! -f "$run_dir/tv/external_test_best_summary.json" ]; then
    echo "[JSL-4NONORACLE] train MELD -> evaluate eJSL modalities=$MODALITIES"
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
    echo "[JSL-4NONORACLE] reuse MMGCN reports: $run_dir"
  fi
done

echo "[JSL-4NONORACLE] ===== Step 6: summarize ====="
python MMGCN/summarize_nonoracle_text_sweep.py \
  --sweep_root "$SWEEP_ROOT" \
  --out_csv "$WORK_ROOT/four_nonoracle_text_rank.csv" \
  --out_json "$WORK_ROOT/four_nonoracle_text_rank.json" \
  --rank_metric weighted_f1

echo "[JSL-4NONORACLE] done"
echo "[JSL-4NONORACLE] rank csv: $WORK_ROOT/four_nonoracle_text_rank.csv"
echo "[JSL-4NONORACLE] rank json: $WORK_ROOT/four_nonoracle_text_rank.json"
