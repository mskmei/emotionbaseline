#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/sast_plain_joint_refine/ConxGNN}
CORPUS=${CORPUS:-meld}
BOBSL_K=${BOBSL_K:-5}
SOURCE_PROFILE=${SOURCE_PROFILE:-rank1_b10_kl1}
OUT_PROFILE=${OUT_PROFILE:-rank1_b10_kl1_scaled}
SEEDS=${SEEDS:-"35 36 37 41 42"}
SCALE_FACTORS=${SCALE_FACTORS:-"0,0.1,0.25,0.5,0.75,1.0,1.25"}

for seed in $SEEDS; do
  source_dir="$WORK_ROOT/${CORPUS}/b${BOBSL_K}k/${SOURCE_PROFILE}_seed${seed}"
  out_dir="$WORK_ROOT/${CORPUS}/b${BOBSL_K}k/${OUT_PROFILE}_seed${seed}"
  if [[ -s "$out_dir/summary.json" ]]; then
    echo "[SAST-SCALE] skip $out_dir"
    continue
  fi
  "$PYTHON_BIN" new2/ConxGNN/eval_conxgnn_sast_scale.py \
    --adapter_checkpoint "$source_dir/model_best.pt" \
    --out_dir "$out_dir" \
    --scale_factors "$SCALE_FACTORS"
done
