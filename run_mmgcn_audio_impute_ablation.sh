#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR/MMGCN"

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
TXT_ROOT=${TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
SOURCE_PKL=${SOURCE_PKL:-./IEMOCAP_features/IEMOCAP_features.pkl}
OUT_ROOT=${OUT_ROOT:-./saved/dial_eval_audio_impute}
PKL_ROOT=${PKL_ROOT:-/raid_zoe/home/lr/maokeyu/sign}
GPU=${GPU:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-20}
AUDIO_STRATEGIES=${AUDIO_STRATEGIES:-"source_mean pure_noise"}
AUDIO_NOISE_SCALE=${AUDIO_NOISE_SCALE:-1.0}
REBUILD_DIAL_PKL=${REBUILD_DIAL_PKL:-1}

mkdir -p "$OUT_ROOT"

for STRATEGY in $AUDIO_STRATEGIES; do
  DIAL_PKL="$PKL_ROOT/dial_test_iemocap_audio_${STRATEGY}.pkl"
  SAVE_DIR="$OUT_ROOT/${STRATEGY}"
  LOG_PATH="$SAVE_DIR/mmgcn_audio_${STRATEGY}.log"
  mkdir -p "$SAVE_DIR"

  if [ "$REBUILD_DIAL_PKL" = "1" ] || [ ! -f "$DIAL_PKL" ]; then
    echo "[MMGCN][audio=$STRATEGY] build pkl"
    python build_dial_test_pkl.py \
      --dial_list "$DIAL_LIST" \
      --txt_root "$TXT_ROOT" \
      --frame_root "$FRAME_ROOT" \
      --mp4_root "$MP4_ROOT" \
      --dataset IEMOCAP \
      --source_pkl "$SOURCE_PKL" \
      --missing_audio_strategy "$STRATEGY" \
      --audio_noise_scale "$AUDIO_NOISE_SCALE" \
      --out_pkl "$DIAL_PKL"
  fi

  echo "[MMGCN][audio=$STRATEGY] train/eval"
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
    --dial_save_dir "$SAVE_DIR" | tee "$LOG_PATH"
done

echo "[MMGCN audio imputation] summary"
for STRATEGY in $AUDIO_STRATEGIES; do
  LOG_PATH="$OUT_ROOT/${STRATEGY}/mmgcn_audio_${STRATEGY}.log"
  echo "===== ${STRATEGY} ====="
  if [ -f "$LOG_PATH" ]; then
    grep -E 'dial_epoch:|Best DIAL Weighted-F1' "$LOG_PATH" | tail -n 8 || echo "no summary lines found in $LOG_PATH"
  else
    echo "missing log: $LOG_PATH"
  fi
done
