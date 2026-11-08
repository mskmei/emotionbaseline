#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE=MMGCN or CMERC}
GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
EPOCHS=${EPOCHS:-30}

BASELINE="$BASELINE" \
GPU="$GPU" \
PYTHON_BIN="$PYTHON_BIN" \
CORPORA="MELD IEMOCAP" \
PROFILES="routing routing_visual" \
SEEDS="35 36 37 41 42" \
EPOCHS="$EPOCHS" \
VALID_RATIO=0 \
bash run_safe_transfer_strict.sh
