#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE}
CACHE_ROOT=${CACHE_ROOT:?Set CACHE_ROOT}
THRESHOLD_CSV=${THRESHOLD_CSV:?Set THRESHOLD_CSV}
OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT}
GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
SEEDS=${SEEDS:-"35 36 37 41 42"}
EPOCHS=${EPOCHS:-40}
SOURCE_FLOOR_TOLERANCE=${SOURCE_FLOOR_TOLERANCE:-0}
CONFIGS=${CONFIGS:-"v015 v025 v035 v050 v065 v080 v100 v120 v150 r025 r050 r075 r100 s025 s050 s075 s100 vs025 vs050 vs075 vs100 g025 g050 g100 v035_i01 v035_i03 v035_i06 v050_i03 v065_i03 v100_i03 v035_p01 v035_p03 v035_p06 v050_p03 v065_p03 v100_p03 v035_ip v050_ip v065_ip v100_ip v035_dist0 v035_dist4 v035_route1 v035_bias3 v035_bias1 v035_lr1 v035_lr10 v035_h64 vs035_ip vs100_ip"}

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT_ROOT/logs"

resolve_config() {
  COMMON_ARGS=(--mode visual --max_scale 0.35 --gate_bias -2.0 --lr 0.0003 --weight_decay 0.0001 --hidden_dim 32 --visual_gate_dim 16 --dropout 0.1 --distill_weight 0.1 --gate_penalty 0.001 --routing_weight 0.5 --routing_margin 0.05)
  EXTRA_ARGS=()
  case "$1" in
    v015) EXTRA_ARGS=(--max_scale 0.15) ;;
    v025) EXTRA_ARGS=(--max_scale 0.25) ;;
    v035) ;;
    v050) EXTRA_ARGS=(--max_scale 0.50) ;;
    v065) EXTRA_ARGS=(--max_scale 0.65) ;;
    v080) EXTRA_ARGS=(--max_scale 0.80) ;;
    v100) EXTRA_ARGS=(--max_scale 1.00) ;;
    v120) EXTRA_ARGS=(--max_scale 1.20) ;;
    v150) EXTRA_ARGS=(--max_scale 1.50) ;;
    r025) EXTRA_ARGS=(--mode reliability --max_scale 0.25) ;;
    r050) EXTRA_ARGS=(--mode reliability --max_scale 0.50) ;;
    r075) EXTRA_ARGS=(--mode reliability --max_scale 0.75) ;;
    r100) EXTRA_ARGS=(--mode reliability --max_scale 1.00) ;;
    s025) EXTRA_ARGS=(--mode sample --max_scale 0.25) ;;
    s050) EXTRA_ARGS=(--mode sample --max_scale 0.50) ;;
    s075) EXTRA_ARGS=(--mode sample --max_scale 0.75) ;;
    s100) EXTRA_ARGS=(--mode sample --max_scale 1.00) ;;
    vs025) EXTRA_ARGS=(--mode visual_sample --max_scale 0.25) ;;
    vs050) EXTRA_ARGS=(--mode visual_sample --max_scale 0.50) ;;
    vs075) EXTRA_ARGS=(--mode visual_sample --max_scale 0.75) ;;
    vs100) EXTRA_ARGS=(--mode visual_sample --max_scale 1.00) ;;
    g025) EXTRA_ARGS=(--mode global --max_scale 0.25 --lr 0.01 --gate_bias -1.0 --routing_weight 0) ;;
    g050) EXTRA_ARGS=(--mode global --max_scale 0.50 --lr 0.01 --gate_bias -1.0 --routing_weight 0) ;;
    g100) EXTRA_ARGS=(--mode global --max_scale 1.00 --lr 0.01 --gate_bias -1.0 --routing_weight 0) ;;
    v035_i01) EXTRA_ARGS=(--target_entropy_weight 0.01 --target_diversity_weight 0.01) ;;
    v035_i03) EXTRA_ARGS=(--target_entropy_weight 0.03 --target_diversity_weight 0.03) ;;
    v035_i06) EXTRA_ARGS=(--target_entropy_weight 0.06 --target_diversity_weight 0.06) ;;
    v050_i03) EXTRA_ARGS=(--max_scale 0.50 --target_entropy_weight 0.03 --target_diversity_weight 0.03) ;;
    v065_i03) EXTRA_ARGS=(--max_scale 0.65 --target_entropy_weight 0.03 --target_diversity_weight 0.03) ;;
    v100_i03) EXTRA_ARGS=(--max_scale 1.00 --target_entropy_weight 0.03 --target_diversity_weight 0.03) ;;
    v035_p01) EXTRA_ARGS=(--target_pseudo_weight 0.01 --target_pseudo_margin 0.02) ;;
    v035_p03) EXTRA_ARGS=(--target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    v035_p06) EXTRA_ARGS=(--target_pseudo_weight 0.06 --target_pseudo_margin 0.02) ;;
    v050_p03) EXTRA_ARGS=(--max_scale 0.50 --target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    v065_p03) EXTRA_ARGS=(--max_scale 0.65 --target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    v100_p03) EXTRA_ARGS=(--max_scale 1.00 --target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    v035_ip) EXTRA_ARGS=(--target_entropy_weight 0.03 --target_diversity_weight 0.03 --target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    v050_ip) EXTRA_ARGS=(--max_scale 0.50 --target_entropy_weight 0.03 --target_diversity_weight 0.03 --target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    v065_ip) EXTRA_ARGS=(--max_scale 0.65 --target_entropy_weight 0.03 --target_diversity_weight 0.03 --target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    v100_ip) EXTRA_ARGS=(--max_scale 1.00 --target_entropy_weight 0.03 --target_diversity_weight 0.03 --target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    v035_dist0) EXTRA_ARGS=(--distill_weight 0) ;;
    v035_dist4) EXTRA_ARGS=(--distill_weight 0.4) ;;
    v035_route1) EXTRA_ARGS=(--routing_weight 1.0 --routing_margin 0.02) ;;
    v035_bias3) EXTRA_ARGS=(--gate_bias -3.0) ;;
    v035_bias1) EXTRA_ARGS=(--gate_bias -1.0) ;;
    v035_lr1) EXTRA_ARGS=(--lr 0.0001) ;;
    v035_lr10) EXTRA_ARGS=(--lr 0.001) ;;
    v035_h64) EXTRA_ARGS=(--hidden_dim 64 --visual_gate_dim 32) ;;
    vs035_ip) EXTRA_ARGS=(--mode visual_sample --target_entropy_weight 0.03 --target_diversity_weight 0.03 --target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    vs100_ip) EXTRA_ARGS=(--mode visual_sample --max_scale 1.00 --target_entropy_weight 0.03 --target_diversity_weight 0.03 --target_pseudo_weight 0.03 --target_pseudo_margin 0.02) ;;
    *) echo "[pareto-sweep] unknown config=$1" >&2; return 2 ;;
  esac
}

lookup_floor() {
  awk -F, -v baseline="$BASELINE" -v corpus="$1" -v seed="$2" '
    NR > 1 && $1 == baseline && $2 == corpus && $3 == seed {print $5; exit}
  ' "$THRESHOLD_CSV"
}

for corpus in $CORPORA; do
  for seed in $SEEDS; do
    cache_dir="$CACHE_ROOT/$corpus/seed${seed}"
    source_floor=$(lookup_floor "$corpus" "$seed")
    if [[ -z "$source_floor" ]]; then
      echo "[pareto-sweep] missing threshold baseline=$BASELINE corpus=$corpus seed=$seed" >&2
      exit 2
    fi
    for config in $CONFIGS; do
      resolve_config "$config"
      out_dir="$OUT_ROOT/$corpus/${config}_seed${seed}"
      if [[ -s "$out_dir/external_test_best_summary.json" ]]; then
        echo "[pareto-sweep] skip $out_dir"
        continue
      fi
      mkdir -p "$out_dir"
      "$PYTHON_BIN" new/train_safe_transfer_gate.py \
        --train_cache "$cache_dir/source_train.npz" \
        --source_test_cache "$cache_dir/source_test.npz" \
        --external_test_cache "$cache_dir/external_test.npz" \
        --out_dir "$out_dir" --seed "$seed" --epochs "$EPOCHS" \
        --selection_split pareto_external --source_floor "$source_floor" \
        --source_floor_tolerance "$SOURCE_FLOOR_TOLERANCE" --allow_source_fallback \
        "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$OUT_ROOT/logs/${corpus}_${config}_seed${seed}.log"
    done
  done
done

echo "[pareto-sweep] completed baseline=$BASELINE out=$OUT_ROOT"
