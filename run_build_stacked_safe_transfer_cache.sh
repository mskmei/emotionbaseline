#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE}
CACHE_ROOT=${CACHE_ROOT:?Set CACHE_ROOT}
MAIN_ROOT=${MAIN_ROOT:?Set MAIN_ROOT}
THRESHOLD_CSV=${THRESHOLD_CSV:?Set THRESHOLD_CSV}
OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT}
PYTHON_BIN=${PYTHON_BIN:-python}
SEEDS=${SEEDS:-"35 36 37 41 42"}
EXPERT_BRANCH=${EXPERT_BRANCH:-joint}

for corpus in MELD IEMOCAP; do
  for seed in $SEEDS; do
    profile=$(awk -F, -v baseline="$BASELINE" -v corpus="$corpus" -v seed="$seed" '
      NR > 1 && $1 == baseline && $2 == corpus && $3 == seed {print $4; exit}
    ' "$THRESHOLD_CSV")
    checkpoint="$MAIN_ROOT/$corpus/${profile}_seed${seed}/model_best.pt"
    out_dir="$OUT_ROOT/$corpus/seed${seed}"
    if [[ -s "$out_dir/external_test.npz" ]]; then
      echo "[stack-cache] skip $out_dir"
      continue
    fi
    if [[ ! -s "$checkpoint" ]]; then
      echo "[stack-cache] missing checkpoint: $checkpoint" >&2
      exit 2
    fi
    "$PYTHON_BIN" new/build_stacked_safe_transfer_cache.py \
      --cache_dir "$CACHE_ROOT/$corpus/seed${seed}" \
      --checkpoint "$checkpoint" --out_dir "$out_dir" --device cpu \
      --expert_branch "$EXPERT_BRANCH"
  done
done
