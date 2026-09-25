#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export TRANSLATED_TXT_ROOT="${TRANSLATED_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ejsl_txt_en_openai}"

# Default is analysis-only, so this is safe to run after existing experiments.
# Set RUN_EXPERIMENTS=1 to resume/run the natural paired sweep before summarizing.
RUN_EXPERIMENTS="${RUN_EXPERIMENTS:-0}"
BASELINES="${BASELINES:-cmerc conxgnn ecerc}"
SEEDS="${SEEDS:-35 36 37 41 42}"
CONFIGS="${CONFIGS:-meld_only joint_b1k joint_b5k joint_b10k}"

CMERC_ROOT="${CMERC_ROOT:-/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld}"
CONXGNN_ROOT="${CONXGNN_ROOT:-/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld}"
ECERC_ROOT="${ECERC_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld}"
SUMMARY_ROOT="${SUMMARY_ROOT:-/raid_zoe/home/lr/maokeyu/sign/joint_tradeoff_natural_summary}"

CMERC_GPU="${CMERC_GPU:-${GPU:-0}}"
CONXGNN_GPU="${CONXGNN_GPU:-${GPU:-0}}"
ECERC_GPU="${ECERC_GPU:-${GPU:-0}}"

run_one_baseline() {
  local baseline="$1"
  case "$baseline" in
    cmerc)
      echo "[TRADEOFF] run CMERC natural sweep"
      GPU="$CMERC_GPU" SEEDS="$SEEDS" CONFIGS="$CONFIGS" RESUME_DONE="${RESUME_DONE:-1}" \
        OUT_ROOT="$CMERC_ROOT" bash run_cmerc_visual_joint_bobsl_meld_ejsl.sh
      ;;
    conxgnn)
      echo "[TRADEOFF] run ConxGNN natural sweep"
      GPU="$CONXGNN_GPU" SEEDS="$SEEDS" CONFIGS="$CONFIGS" RESUME_DONE="${RESUME_DONE:-1}" \
        OUT_ROOT="$CONXGNN_ROOT" bash run_conxgnn_visual_joint_bobsl_meld_ejsl.sh
      ;;
    ecerc)
      echo "[TRADEOFF] run ECERC natural sweep"
      GPU="$ECERC_GPU" SEEDS="$SEEDS" CONFIGS="$CONFIGS" RESUME_DONE="${RESUME_DONE:-1}" \
        OUT_ROOT="$ECERC_ROOT" bash run_ecerc_video_joint_bobsl_meld_ejsl.sh
      ;;
    *)
      echo "[TRADEOFF] unknown baseline: $baseline" >&2
      exit 2
      ;;
  esac
}

if [[ "$RUN_EXPERIMENTS" == "1" ]]; then
  for baseline in $BASELINES; do
    run_one_baseline "$baseline"
  done
else
  echo "[TRADEOFF] RUN_EXPERIMENTS=0, only summarizing existing outputs"
fi

python new/summarize_joint_tradeoff.py \
  --root "CMERC=$CMERC_ROOT" \
  --root "ConxGNN=$CONXGNN_ROOT" \
  --root "ECERC=$ECERC_ROOT" \
  --out_dir "$SUMMARY_ROOT"

echo "[TRADEOFF] summary root: $SUMMARY_ROOT"
