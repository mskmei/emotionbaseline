#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE=MMGCN or CMERC}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
PROFILES=${PROFILES:-"global reliability routing routing_visual"}
SEEDS=${SEEDS:-35}
EPOCHS=${EPOCHS:-30}
VALID_RATIO=${VALID_RATIO:-0.15}
SELECTION_SPLIT=${SELECTION_SPLIT:-auto}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/safe_transfer_gate}
CACHE_ROOT=${CACHE_ROOT:-$WORK_ROOT/cache/$BASELINE}
OUT_ROOT=${OUT_ROOT:-$WORK_ROOT/strict_runs/$BASELINE}
mkdir -p "$OUT_ROOT/logs"

resolve_profile() {
  case "$1" in
    global)
      PROFILE_ARGS=(--mode global --max_scale 1.0 --gate_bias -1.0 --lr 0.01 --distill_weight 0.1 --gate_penalty 0.001)
      ;;
    reliability)
      PROFILE_ARGS=(--mode reliability --max_scale 1.0 --gate_bias -1.5 --lr 0.001 --distill_weight 0.2 --gate_penalty 0.005)
      ;;
    routing)
      PROFILE_ARGS=(--mode reliability --max_scale 1.0 --gate_bias -2.0 --lr 0.0003 --distill_weight 0.1 --gate_penalty 0.001 --routing_weight 0.5 --routing_margin 0.05)
      ;;
    routing_visual)
      PROFILE_ARGS=(--mode visual --max_scale 1.0 --gate_bias -2.0 --lr 0.0003 --distill_weight 0.1 --gate_penalty 0.001 --routing_weight 0.5 --routing_margin 0.05)
      ;;
    *) echo "[safe-gate-strict] unknown profile=$1" >&2; return 2 ;;
  esac
}

for corpus in $CORPORA; do
  for seed in $SEEDS; do
    cache_dir="$CACHE_ROOT/$corpus/seed${seed}"
    for filename in source_train.npz source_test.npz external_test.npz; do
      if [[ ! -s "$cache_dir/$filename" ]]; then
        echo "[safe-gate-strict] missing cache: $cache_dir/$filename" >&2
        exit 2
      fi
    done
    for profile in $PROFILES; do
      resolve_profile "$profile"
      out_dir="$OUT_ROOT/$corpus/${profile}_seed${seed}"
      if [[ -s "$out_dir/external_test_best_summary.json" ]]; then
        echo "[safe-gate-strict] skip $out_dir"
        continue
      fi
      mkdir -p "$out_dir"
      "$PYTHON_BIN" new/train_safe_transfer_gate.py \
        --train_cache "$cache_dir/source_train.npz" \
        --source_test_cache "$cache_dir/source_test.npz" \
        --external_test_cache "$cache_dir/external_test.npz" \
        --out_dir "$out_dir" --seed "$seed" --epochs "$EPOCHS" \
        --gate_valid_ratio "$VALID_RATIO" --split_seed 2026 \
        --selection_split "$SELECTION_SPLIT" --allow_source_fallback \
        "${PROFILE_ARGS[@]}" \
        2>&1 | tee "$OUT_ROOT/logs/${corpus}_${profile}_seed${seed}.log"
    done
  done
done

"$PYTHON_BIN" analysis/summarize_safe_transfer_gate.py --root "$OUT_ROOT" --baseline "${BASELINE}-strict"
