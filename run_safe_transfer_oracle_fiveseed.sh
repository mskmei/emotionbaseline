#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE to the baseline name}
GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate_oracle}
CACHE_ROOT=${CACHE_ROOT:?Set CACHE_ROOT to the baseline cache containing MELD and IEMOCAP}
OUT_ROOT=${OUT_ROOT:-$WORK_ROOT/runs/$BASELINE}

echo "[safe-gate-oracle] baseline=$BASELINE"
echo "[safe-gate-oracle] eJSL labels select checkpoints; these results are exploratory only"

BASELINE="$BASELINE" GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" \
CORPORA="${CORPORA:-MELD IEMOCAP}" \
PROFILES="${PROFILES:-global reliability routing routing_visual}" \
SEEDS="${SEEDS:-35 36 37 41 42}" EPOCHS="${EPOCHS:-30}" \
VALID_RATIO=0 SELECTION_SPLIT=external \
WORK_ROOT="$WORK_ROOT" CACHE_ROOT="$CACHE_ROOT" OUT_ROOT="$OUT_ROOT" \
bash run_safe_transfer_strict.sh
