#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}

"$PYTHON_BIN" analysis/analyze_semantic_anchor_paper.py \
  --cmerc_root /raid_zoe/home/lr/maokeyu/sign/cesa_cmerc/ablations \
  --conx_root /raid_zoe/home/lr/maokeyu/sign/conxgnn_sast/fiveseed \
  --out_dir /raid_zoe/home/lr/maokeyu/sign/semantic_anchor_paper
