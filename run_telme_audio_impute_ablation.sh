#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
TXT_ROOT=${TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
SAVE_MODEL_ROOT=${SAVE_MODEL_ROOT:-/raid_zoe/home/lr/maokeyu/sign/open/IEMOCAP/save_model}
SAVE_DIR=${SAVE_DIR:-./IEMOCAP/outputs_ejsl_audio_impute}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}

AUDIO_STRATEGIES=${AUDIO_STRATEGIES:-"source_mean pure_noise"}
AUDIO_STATS_PATH=${AUDIO_STATS_PATH:-"$SAVE_DIR/iemocap_audio_hidden_stats.npz"}
AUDIO_STATS_SOURCE_CSV=${AUDIO_STATS_SOURCE_CSV:-./dataset/IEMOCAP_full_release/IEMOCAP_train.csv}
AUDIO_STATS_MAX_SAMPLES=${AUDIO_STATS_MAX_SAMPLES:-512}
AUDIO_STATS_HF_DATASET=${AUDIO_STATS_HF_DATASET:-AbstractTTS/IEMOCAP}
AUDIO_STATS_HF_SPLIT=${AUDIO_STATS_HF_SPLIT:-train}
AUDIO_STATS_HF_CACHE_DIR=${AUDIO_STATS_HF_CACHE_DIR:-}
AUDIO_NOISE_SCALE=${AUDIO_NOISE_SCALE:-1.0}

mkdir -p "$SAVE_DIR"

for STRATEGY in $AUDIO_STRATEGIES; do
  PREFIX="telme_ejsl_audio_${STRATEGY}"
  LOG_PATH="$SAVE_DIR/${PREFIX}.log"
  echo "[TELME][audio=$STRATEGY] start"
  python IEMOCAP/inference_ejsl_frame.py \
    --frame_root "$FRAME_ROOT" \
    --txt_root "$TXT_ROOT" \
    --save_model_root "$SAVE_MODEL_ROOT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --save_dir "$SAVE_DIR" \
    --report_prefix "$PREFIX" \
    --eval_modality full \
    --missing_audio_strategy "$STRATEGY" \
    --audio_stats_path "$AUDIO_STATS_PATH" \
    --audio_stats_source_csv "$AUDIO_STATS_SOURCE_CSV" \
    --audio_stats_max_samples "$AUDIO_STATS_MAX_SAMPLES" \
    --audio_stats_hf_dataset "$AUDIO_STATS_HF_DATASET" \
    --audio_stats_hf_split "$AUDIO_STATS_HF_SPLIT" \
    --audio_stats_hf_cache_dir "$AUDIO_STATS_HF_CACHE_DIR" \
    --audio_noise_scale "$AUDIO_NOISE_SCALE" \
    --save_predictions | tee "$LOG_PATH"
done

echo "[TELME audio imputation] summary"
for STRATEGY in $AUDIO_STRATEGIES; do
  PREFIX="telme_ejsl_audio_${STRATEGY}"
  REPORT_PATH="$SAVE_DIR/${PREFIX}_classification_report.txt"
  echo "===== ${STRATEGY} ====="
  if [ -f "$REPORT_PATH" ]; then
    sed -n '1,20p' "$REPORT_PATH"
  else
    echo "missing report: $REPORT_PATH"
  fi
done
