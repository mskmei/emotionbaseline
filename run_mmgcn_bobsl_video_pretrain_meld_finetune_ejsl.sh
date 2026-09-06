#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BOBSL_ROOT=${BOBSL_ROOT:-/raid_zoe/home/lr/wangyi/sign/bobsl}
MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
SOURCE_TXT_ROOT=${SOURCE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl}
UNIFIED_ROOT=${UNIFIED_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}
TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-"$UNIFIED_ROOT/ejsl_txt_en_openai"}
TRANSLATION_CACHE=${TRANSLATION_CACHE:-"$TRANSLATED_TXT_ROOT/translation_cache.jsonl"}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-openai}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}

BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-"$WORK_ROOT/bobsl_anjs4_train_val.pkl"}
BOBSL_TEST_PKL=${BOBSL_TEST_PKL:-"$WORK_ROOT/bobsl_anjs4_test.pkl"}
BOBSL_FEATURE_CACHE_DIR=${BOBSL_FEATURE_CACHE_DIR:-"$WORK_ROOT/bobsl_feature_cache"}

MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$UNIFIED_ROOT/meld_anjs4_unified.pkl"}
EJSL_UNIFIED_PKL=${EJSL_UNIFIED_PKL:-"$UNIFIED_ROOT/ejsl_anjs4_unified.pkl"}
MELD_EJSL_FEATURE_CACHE_DIR=${MELD_EJSL_FEATURE_CACHE_DIR:-"$UNIFIED_ROOT/feature_cache"}

BOBSL_OUT_ROOT=${BOBSL_OUT_ROOT:-"$WORK_ROOT/bobsl_video_pretrain"}
FINETUNE_OUT_ROOT=${FINETUNE_OUT_ROOT:-"$WORK_ROOT/meld_v_tv_finetune_from_bobsl_video"}
PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-"$BOBSL_OUT_ROOT/video/model_best.pt"}

GPU=${GPU:-0}
TRANSLATE=${TRANSLATE:-0}
REBUILD_BOBSL_FEATURES=${REBUILD_BOBSL_FEATURES:-0}
REBUILD_MELD_EJSL_FEATURES=${REBUILD_MELD_EJSL_FEATURES:-0}
RUN_BOBSL_PRETRAIN=${RUN_BOBSL_PRETRAIN:-1}
RUN_MELD_FINETUNE=${RUN_MELD_FINETUNE:-1}

BOBSL_GRAPH_TYPE=${BOBSL_GRAPH_TYPE:-DeepGCN}
BOBSL_EPOCHS=${BOBSL_EPOCHS:-30}
BOBSL_BATCH_SIZE=${BOBSL_BATCH_SIZE:-32}
BOBSL_LR=${BOBSL_LR:-0.0003}
BOBSL_L2=${BOBSL_L2:-0.00003}
BOBSL_DROPOUT=${BOBSL_DROPOUT:-0.4}
BOBSL_MIN_SCORE=${BOBSL_MIN_SCORE:-0.0}
BOBSL_SELECTION_SPLIT=${BOBSL_SELECTION_SPLIT:-source}
BOBSL_LIMIT_TRAIN=${BOBSL_LIMIT_TRAIN:-0}
BOBSL_LIMIT_VAL=${BOBSL_LIMIT_VAL:-0}
BOBSL_LIMIT_TEST=${BOBSL_LIMIT_TEST:-0}
BOBSL_TEXT_DIM=${BOBSL_TEXT_DIM:-1024}
BOBSL_AUDIO_DIM=${BOBSL_AUDIO_DIM:-768}
BOBSL_N_SPEAKERS=${BOBSL_N_SPEAKERS:-9}

MELD_MODALITIES=${MELD_MODALITIES:-"video tv"}
MELD_GRAPH_TYPE=${MELD_GRAPH_TYPE:-MMGCN}
MELD_EPOCHS=${MELD_EPOCHS:-15}
MELD_BATCH_SIZE=${MELD_BATCH_SIZE:-8}
MELD_LR=${MELD_LR:-0.0003}
MELD_L2=${MELD_L2:-0.00003}
MELD_DROPOUT=${MELD_DROPOUT:-0.4}
MELD_SELECTION_SPLIT=${MELD_SELECTION_SPLIT:-source}

LOSS=${LOSS:-focal}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-5.0}
SEED=${SEED:-42}
SAVE_EPOCH_EVERY=${SAVE_EPOCH_EVERY:-5}

mkdir -p "$WORK_ROOT" "$BOBSL_OUT_ROOT" "$FINETUNE_OUT_ROOT"

if [ "$TRANSLATED_TXT_ROOT" = "$SOURCE_TXT_ROOT" ]; then
  echo "[BOBSL-MMGCN] refusing to overwrite SOURCE_TXT_ROOT: $SOURCE_TXT_ROOT" >&2
  exit 1
fi

if [ "$TRANSLATE" = "1" ]; then
  echo "[BOBSL-MMGCN] translate eJSL txt to English"
  python JSL/translate_ejsl_txt_root.py \
    --input_txt_root "$SOURCE_TXT_ROOT" \
    --output_txt_root "$TRANSLATED_TXT_ROOT" \
    --dial_list "$DIAL_LIST" \
    --cache_jsonl "$TRANSLATION_CACHE" \
    --backend "$TRANSLATION_BACKEND" \
    --openai_model "$OPENAI_MODEL"
elif [ ! -d "$TRANSLATED_TXT_ROOT" ]; then
  echo "[BOBSL-MMGCN] missing translated eJSL txt root: $TRANSLATED_TXT_ROOT" >&2
  echo "[BOBSL-MMGCN] set TRANSLATE=1 or point TRANSLATED_TXT_ROOT to the existing translated txt directory." >&2
  exit 1
fi

if [ "$REBUILD_BOBSL_FEATURES" = "1" ] || [ ! -f "$BOBSL_TRAIN_VAL_PKL" ] || [ ! -f "$BOBSL_TEST_PKL" ]; then
  echo "[BOBSL-MMGCN] build BOBSL ANJS4 visual features"
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/build_unified_bobsl_pkl.py \
    --bobsl_root "$BOBSL_ROOT" \
    --out_train_val_pkl "$BOBSL_TRAIN_VAL_PKL" \
    --out_test_pkl "$BOBSL_TEST_PKL" \
    --cache_dir "$BOBSL_FEATURE_CACHE_DIR" \
    --min_score "$BOBSL_MIN_SCORE" \
    --limit_train "$BOBSL_LIMIT_TRAIN" \
    --limit_val "$BOBSL_LIMIT_VAL" \
    --limit_test "$BOBSL_LIMIT_TEST" \
    --text_dim "$BOBSL_TEXT_DIM" \
    --audio_dim "$BOBSL_AUDIO_DIM" \
    --n_speakers "$BOBSL_N_SPEAKERS" \
    --fp16
fi

if [ "$REBUILD_MELD_EJSL_FEATURES" = "1" ] || [ ! -f "$MELD_UNIFIED_PKL" ] || [ ! -f "$EJSL_UNIFIED_PKL" ]; then
  echo "[BOBSL-MMGCN] build same-origin MELD/eJSL feature pkl"
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/build_unified_meld_ejsl_pkl.py \
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

if [ "$RUN_BOBSL_PRETRAIN" = "1" ]; then
  echo "[BOBSL-MMGCN] pretrain video branch on BOBSL; best is selected by BOBSL val"
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$BOBSL_TRAIN_VAL_PKL" \
    --external_test_pkl "$BOBSL_TEST_PKL" \
    --out_dir "$BOBSL_OUT_ROOT" \
    --modalities video \
    --graph_type "$BOBSL_GRAPH_TYPE" \
    --epochs "$BOBSL_EPOCHS" \
    --batch_size "$BOBSL_BATCH_SIZE" \
    --lr "$BOBSL_LR" \
    --l2 "$BOBSL_L2" \
    --dropout "$BOBSL_DROPOUT" \
    --loss "$LOSS" \
    --focal_gamma "$FOCAL_GAMMA" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --selection_split "$BOBSL_SELECTION_SPLIT" \
    --selection_metric weighted_f1 \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    --seed "$SEED"
fi

if [ ! -f "$PRETRAIN_CHECKPOINT" ]; then
  echo "[BOBSL-MMGCN] missing pretrain checkpoint: $PRETRAIN_CHECKPOINT" >&2
  exit 1
fi

if [ "$RUN_MELD_FINETUNE" = "1" ]; then
  echo "[BOBSL-MMGCN] finetune on MELD ANJS4 and evaluate translated eJSL: $MELD_MODALITIES"
  CUDA_VISIBLE_DEVICES="$GPU" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$MELD_UNIFIED_PKL" \
    --external_test_pkl "$EJSL_UNIFIED_PKL" \
    --out_dir "$FINETUNE_OUT_ROOT" \
    --modalities $MELD_MODALITIES \
    --graph_type "$MELD_GRAPH_TYPE" \
    --init_checkpoint "$PRETRAIN_CHECKPOINT" \
    --epochs "$MELD_EPOCHS" \
    --batch_size "$MELD_BATCH_SIZE" \
    --lr "$MELD_LR" \
    --l2 "$MELD_L2" \
    --dropout "$MELD_DROPOUT" \
    --loss "$LOSS" \
    --focal_gamma "$FOCAL_GAMMA" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --selection_split "$MELD_SELECTION_SPLIT" \
    --selection_metric weighted_f1 \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    --seed "$SEED"
fi

echo "[BOBSL-MMGCN] BOBSL pretrain reports: $BOBSL_OUT_ROOT/video"
echo "[BOBSL-MMGCN] MELD fine-tune reports: $FINETUNE_OUT_ROOT"
for REPORT_PATH in \
  "$BOBSL_OUT_ROOT/video/source_test_best_classification_report.txt" \
  "$BOBSL_OUT_ROOT/video/external_test_best_classification_report.txt" \
  "$FINETUNE_OUT_ROOT/video/external_test_best_classification_report.txt" \
  "$FINETUNE_OUT_ROOT/tv/external_test_best_classification_report.txt"; do
  echo "===== $REPORT_PATH ====="
  if [ -f "$REPORT_PATH" ]; then
    sed -n '1,20p' "$REPORT_PATH"
  else
    echo "missing report"
  fi
done
