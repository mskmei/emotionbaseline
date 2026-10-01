#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
SWEEP_TAG="${SWEEP_TAG:-}"

if [[ -n "$SWEEP_TAG" ]]; then
  DEFAULT_MMGCN_ROOT="/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/tradeoff_sweeps/$SWEEP_TAG"
  DEFAULT_MAGTKD_ROOT="/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/tradeoff_sweeps/$SWEEP_TAG"
  DEFAULT_ECERC_ROOT="/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/tradeoff_sweeps/$SWEEP_TAG"
  DEFAULT_CMERC_ROOT="/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/tradeoff_sweeps/$SWEEP_TAG"
  DEFAULT_CONXGNN_ROOT="/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/tradeoff_sweeps/$SWEEP_TAG"
  DEFAULT_OUT_DIR="/raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary/$SWEEP_TAG"
else
  DEFAULT_MMGCN_ROOT="/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld"
  DEFAULT_MAGTKD_ROOT="/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld"
  DEFAULT_ECERC_ROOT="/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld"
  DEFAULT_CMERC_ROOT="/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld"
  DEFAULT_CONXGNN_ROOT="/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/visual_joint_bobsl_meld"
  DEFAULT_OUT_DIR="/raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary"
fi

MMGCN_ROOT="${MMGCN_ROOT:-$DEFAULT_MMGCN_ROOT}"
MAGTKD_ROOT="${MAGTKD_ROOT:-$DEFAULT_MAGTKD_ROOT}"
ECERC_ROOT="${ECERC_ROOT:-$DEFAULT_ECERC_ROOT}"
CMERC_ROOT="${CMERC_ROOT:-$DEFAULT_CMERC_ROOT}"
CONXGNN_ROOT="${CONXGNN_ROOT:-$DEFAULT_CONXGNN_ROOT}"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"

"$PYTHON" new/summarize_five_baseline_per_class.py \
  --root "MMGCN=$MMGCN_ROOT|video" \
  --root "MAGTKD=$MAGTKD_ROOT" \
  --root "ECERC=$ECERC_ROOT" \
  --root "CMERC=$CMERC_ROOT" \
  --root "ConxGNN=$CONXGNN_ROOT" \
  --out_dir "$OUT_DIR"

echo "[five-baseline-per-class] out_dir=$OUT_DIR"
