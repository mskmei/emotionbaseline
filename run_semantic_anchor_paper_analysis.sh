#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}

"$PYTHON_BIN" analysis/analyze_semantic_anchor_paper.py \
  --cmerc_root /raid_zoe/home/lr/maokeyu/sign/cesa_cmerc/ablations \
  --conx_root /raid_zoe/home/lr/maokeyu/sign/conxgnn_sast/fiveseed \
  --meld_csv /raid_zoe/home/lr/maokeyu/sign/five_baseline_joint_per_class_summary/five_baseline_per_class_config_means.csv \
  --iemocap_csv /raid_zoe/home/lr/maokeyu/sign/five_baseline_bobsl_iemocap_ejsl_summary/iemocap_joint_config_means.csv \
  --out_dir /raid_zoe/home/lr/maokeyu/sign/semantic_anchor_paper
