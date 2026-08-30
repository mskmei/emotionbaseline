#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}

export MODALITIES=${MODALITIES:-tv}
export GRAPH_TYPE=${GRAPH_TYPE:-DeepGCN}
export TRANSLATE=${TRANSLATE:-0}
export REBUILD_FEATURES=${REBUILD_FEATURES:-0}
export OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/runs_deepgcn_tv"}

echo "[MMGCN-DEEPGCN-TV] MODALITIES=$MODALITIES GRAPH_TYPE=$GRAPH_TYPE"
echo "[MMGCN-DEEPGCN-TV] OUT_ROOT=$OUT_ROOT"
bash run_mmgcn_unified_meld_to_ejsl.sh
