#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR/MMGCN"

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
TXT_ROOT=${TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
SOURCE_PKL=${SOURCE_PKL:-./IEMOCAP_features/IEMOCAP_features.pkl}
DIAL_PKL=${DIAL_PKL:-/raid_zoe/home/lr/maokeyu/sign/dial_test_iemocap.pkl}
OUT_ROOT=${OUT_ROOT:-./saved/dial_eval_ablation_tv}
GPU=${GPU:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-20}
REBUILD_DIAL_PKL=${REBUILD_DIAL_PKL:-1}

mkdir -p "$OUT_ROOT"

if [ "$REBUILD_DIAL_PKL" = "1" ] || [ ! -f "$DIAL_PKL" ]; then
  python build_dial_test_pkl.py \
    --dial_list "$DIAL_LIST" \
    --txt_root "$TXT_ROOT" \
    --frame_root "$FRAME_ROOT" \
    --mp4_root "$MP4_ROOT" \
    --dataset IEMOCAP \
    --source_pkl "$SOURCE_PKL" \
    --out_pkl "$DIAL_PKL"
fi

for MOD in text video; do
  SAVE_DIR="$OUT_ROOT/${MOD}_only"
  LOG_PATH="$SAVE_DIR/mmgcn_${MOD}_only.log"
  mkdir -p "$SAVE_DIR"
  echo "[MMGCN][$MOD] start"
  CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --base-model LSTM \
    --graph-model \
    --nodal-attention \
    --dropout 0.4 \
    --lr 0.0003 \
    --batch-size "$BATCH_SIZE" \
    --l2 0.00003 \
    --graph_type MMGCN \
    --epochs "$EPOCHS" \
    --graph_construct direct \
    --multi_modal \
    --mm_fusion_mthd concat_subsequently \
    --modals avl \
    --Dataset IEMOCAP \
    --Deep_GCN_nlayers 4 \
    --class-weight \
    --use_speaker \
    --dial_test_path "$DIAL_PKL" \
    --dial_eval_every 1 \
    --dial_save_dir "$SAVE_DIR" \
    --input_modality "$MOD" | tee "$LOG_PATH"
done

echo "[MMGCN] summary"
for MOD in text video; do
  SAVE_DIR="$OUT_ROOT/${MOD}_only"
  LOG_PATH="$SAVE_DIR/mmgcn_${MOD}_only.log"
  echo "===== ${MOD} ====="
  if [ -f "$LOG_PATH" ]; then
    grep -E 'dial_epoch:|Best DIAL Weighted-F1' "$LOG_PATH" | tail -n 8 || echo "no summary lines found in $LOG_PATH"
  else
    echo "missing log: $LOG_PATH"
  fi
done
