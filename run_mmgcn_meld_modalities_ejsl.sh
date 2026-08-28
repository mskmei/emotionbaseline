#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR/MMGCN"

DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
TXT_ROOT=${TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}

SOURCE_PKL=${SOURCE_PKL:-./MELD_features/MELD_features_raw1.pkl}
PKL_ROOT=${PKL_ROOT:-/raid_zoe/home/lr/maokeyu/sign}
DIAL_PKL=${DIAL_PKL:-"$PKL_ROOT/dial_test_meld_zero_audio.pkl"}
OUT_ROOT=${OUT_ROOT:-./saved/dial_eval_meld_modalities}

MODALITIES=${MODALITIES:-"text video tv"}
GPU=${GPU:-0}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-20}
DIAL_EVAL_EVERY=${DIAL_EVAL_EVERY:-$EPOCHS}
REBUILD_DIAL_PKL=${REBUILD_DIAL_PKL:-1}

LR=${LR:-0.0003}
L2=${L2:-0.00003}
DROPOUT=${DROPOUT:-0.4}
DEEP_GCN_NLAYERS=${DEEP_GCN_NLAYERS:-4}
AUDIO_NOISE_SCALE=${AUDIO_NOISE_SCALE:-1.0}

mkdir -p "$OUT_ROOT"

if [ ! -f "$SOURCE_PKL" ]; then
  echo "[MMGCN] missing MELD source feature pkl: $SOURCE_PKL" >&2
  echo "[MMGCN] set SOURCE_PKL to MELD_features_raw1.pkl." >&2
  exit 1
fi

if [ "$REBUILD_DIAL_PKL" = "1" ] || [ ! -f "$DIAL_PKL" ]; then
  echo "[MMGCN] build eJSL feature pkl in MELD feature space"
  python build_dial_test_pkl.py \
    --dial_list "$DIAL_LIST" \
    --txt_root "$TXT_ROOT" \
    --frame_root "$FRAME_ROOT" \
    --mp4_root "$MP4_ROOT" \
    --dataset MELD \
    --source_pkl "$SOURCE_PKL" \
    --missing_audio_strategy zero \
    --audio_noise_scale "$AUDIO_NOISE_SCALE" \
    --seed "$SEED" \
    --out_pkl "$DIAL_PKL"
fi

for MOD in $MODALITIES; do
  case "$MOD" in
    t) EVAL_MOD=text ;;
    v|visual) EVAL_MOD=video ;;
    tv|vt|vl|text_video|video_text) EVAL_MOD=tv ;;
    text|video) EVAL_MOD="$MOD" ;;
    *)
      echo "[MMGCN] unsupported modality in MODALITIES: $MOD" >&2
      exit 1
      ;;
  esac

  SAVE_DIR="$OUT_ROOT/${EVAL_MOD}"
  LOG_PATH="$SAVE_DIR/mmgcn_meld_${EVAL_MOD}_ejsl.log"
  mkdir -p "$SAVE_DIR"

  echo "[MMGCN][$EVAL_MOD] train MELD and test eJSL"
  CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --base-model LSTM \
    --graph-model \
    --nodal-attention \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --l2 "$L2" \
    --graph_type MMGCN \
    --epochs "$EPOCHS" \
    --graph_construct direct \
    --multi_modal \
    --mm_fusion_mthd concat_subsequently \
    --modals avl \
    --Dataset MELD \
    --Deep_GCN_nlayers "$DEEP_GCN_NLAYERS" \
    --use_speaker \
    --dial_test_path "$DIAL_PKL" \
    --dial_eval_every "$DIAL_EVAL_EVERY" \
    --dial_save_dir "$SAVE_DIR" \
    --input_modality "$EVAL_MOD" | tee "$LOG_PATH"
done

echo "[MMGCN] eJSL summary"
for MOD in $MODALITIES; do
  case "$MOD" in
    t) EVAL_MOD=text ;;
    v|visual) EVAL_MOD=video ;;
    tv|vt|vl|text_video|video_text) EVAL_MOD=tv ;;
    text|video) EVAL_MOD="$MOD" ;;
    *) continue ;;
  esac
  LOG_PATH="$OUT_ROOT/${EVAL_MOD}/mmgcn_meld_${EVAL_MOD}_ejsl.log"
  echo "===== ${EVAL_MOD} ====="
  if [ -f "$LOG_PATH" ]; then
    grep -E 'dial_epoch:|Best DIAL Weighted-F1' "$LOG_PATH" | tail -n 8 || echo "no summary lines found in $LOG_PATH"
  else
    echo "missing log: $LOG_PATH"
  fi
done
