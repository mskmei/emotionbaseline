#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl}
UNIFIED_ROOT=${UNIFIED_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}
OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/video_bobsl_seed_param_sweep"}

BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-"$WORK_ROOT/bobsl_anjs4_train_val.pkl"}
BOBSL_TEST_PKL=${BOBSL_TEST_PKL:-"$WORK_ROOT/bobsl_anjs4_test.pkl"}
BASE_PRETRAIN_CHECKPOINT=${BASE_PRETRAIN_CHECKPOINT:-"$WORK_ROOT/bobsl_video_pretrain/video/model_best.pt"}
MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$UNIFIED_ROOT/meld_anjs4_unified.pkl"}
EJSL_UNIFIED_PKL=${EJSL_UNIFIED_PKL:-"$UNIFIED_ROOT/ejsl_anjs4_unified.pkl"}

RESUME=${RESUME:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
RUN_PRETRAIN_VARIANTS=${RUN_PRETRAIN_VARIANTS:-1}
FT_GRAPH_TYPE=${FT_GRAPH_TYPE:-MMGCN}
BOBSL_GRAPH_TYPE=${BOBSL_GRAPH_TYPE:-MMGCN}
FT_EPOCHS=${FT_EPOCHS:-15}
BOBSL_EPOCHS=${BOBSL_EPOCHS:-30}
SELECTION_SPLIT=${SELECTION_SPLIT:-source}
SELECTION_METRIC=${SELECTION_METRIC:-weighted_f1}
SAVE_EPOCH_EVERY=${SAVE_EPOCH_EVERY:-0}
TOP_K=${TOP_K:-30}

TRIALS=${TRIALS:-"base_seed42 seed35 seed36 seed37 seed38 seed39 seed40 seed41 seed43 seed44 seed45 lr100 lr150 lr200 lr300 lr500 drop20 drop30 drop50 gamma10 gamma15 gamma25 nll nocw batch4 batch16 layer2 layer6 nores pre_seed43 pre_seed44 pre_lr100 pre_lr500 pre_drop25 pre_drop55 pre_nll"}

mkdir -p "$OUT_ROOT"

for required in "$BOBSL_TRAIN_VAL_PKL" "$BOBSL_TEST_PKL" "$MELD_UNIFIED_PKL" "$EJSL_UNIFIED_PKL"; do
  if [ ! -f "$required" ]; then
    echo "[MMGCN-VIDEO-SWEEP] missing required file: $required" >&2
    exit 1
  fi
done

write_config() {
  local path="$1"
  shift
  python3 - "$path" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = {}
for item in sys.argv[2:]:
    key, sep, value = item.partition("=")
    if not sep:
        continue
    data[key] = value
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

run_logged() {
  local log_path="$1"
  shift
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$@" | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e
  if [ "$status" -ne 0 ]; then
    echo "[MMGCN-VIDEO-SWEEP] command failed with status=$status" | tee "$log_path.failed"
    if [ "$CONTINUE_ON_ERROR" = "1" ]; then
      echo "[MMGCN-VIDEO-SWEEP] CONTINUE_ON_ERROR=1, continuing to next trial"
      return 0
    fi
    exit "$status"
  fi
}

pretrain_checkpoint_for() {
  local key="$1"
  if [ "$key" = "existing" ]; then
    echo "$BASE_PRETRAIN_CHECKPOINT"
  else
    echo "$OUT_ROOT/pretrains/$key/video/model_best.pt"
  fi
}

run_bobsl_pretrain() {
  local key="$1"
  local seed="$2"
  local epochs="$3"
  local batch_size="$4"
  local lr="$5"
  local l2="$6"
  local dropout="$7"
  local loss="$8"
  local focal_gamma="$9"
  local max_grad_norm="${10}"
  shift 10
  local extra_args=("$@")
  local out_dir="$OUT_ROOT/pretrains/$key"
  local checkpoint="$out_dir/video/model_best.pt"

  if [ "$RESUME" = "1" ] && [ -f "$checkpoint" ]; then
    echo "[MMGCN-VIDEO-SWEEP][pretrain:$key] skip existing $checkpoint"
    return
  fi

  mkdir -p "$out_dir"
  write_config "$out_dir/pretrain_config.json" \
    "pretrain_key=$key" \
    "pre_seed=$seed" \
    "pre_epochs=$epochs" \
    "pre_batch_size=$batch_size" \
    "pre_lr=$lr" \
    "pre_l2=$l2" \
    "pre_dropout=$dropout" \
    "pre_loss=$loss" \
    "pre_focal_gamma=$focal_gamma" \
    "pre_max_grad_norm=$max_grad_norm" \
    "pre_extra_args=${extra_args[*]}"

  echo "[MMGCN-VIDEO-SWEEP][pretrain:$key] seed=$seed epochs=$epochs batch=$batch_size lr=$lr l2=$l2 dropout=$dropout loss=$loss gamma=$focal_gamma extra=${extra_args[*]}"
  run_logged "$out_dir/train.log" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$BOBSL_TRAIN_VAL_PKL" \
    --external_test_pkl "$BOBSL_TEST_PKL" \
    --out_dir "$out_dir" \
    --modalities video \
    --graph_type "$BOBSL_GRAPH_TYPE" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --l2 "$l2" \
    --dropout "$dropout" \
    --loss "$loss" \
    --focal_gamma "$focal_gamma" \
    --max_grad_norm "$max_grad_norm" \
    --selection_split source \
    --selection_metric "$SELECTION_METRIC" \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    --seed "$seed" \
    "${extra_args[@]}"
}

ensure_pretrain() {
  local key="$1"
  if [ "$key" = "existing" ]; then
    if [ ! -f "$BASE_PRETRAIN_CHECKPOINT" ]; then
      echo "[MMGCN-VIDEO-SWEEP] missing BASE_PRETRAIN_CHECKPOINT: $BASE_PRETRAIN_CHECKPOINT" >&2
      exit 1
    fi
    return
  fi

  if [ "$RUN_PRETRAIN_VARIANTS" != "1" ]; then
    local checkpoint
    checkpoint="$(pretrain_checkpoint_for "$key")"
    if [ ! -f "$checkpoint" ]; then
      echo "[MMGCN-VIDEO-SWEEP] missing pretrain variant checkpoint: $checkpoint" >&2
      echo "[MMGCN-VIDEO-SWEEP] set RUN_PRETRAIN_VARIANTS=1 or remove trial using pretrain_key=$key." >&2
      exit 1
    fi
    return
  fi

  case "$key" in
    p_seed43)
      run_bobsl_pretrain "$key" 43 "$BOBSL_EPOCHS" 32 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    p_seed44)
      run_bobsl_pretrain "$key" 44 "$BOBSL_EPOCHS" 32 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    p_lr100)
      run_bobsl_pretrain "$key" 42 "$BOBSL_EPOCHS" 32 0.0001 0.00003 0.40 focal 2.0 5.0
      ;;
    p_lr500)
      run_bobsl_pretrain "$key" 42 "$BOBSL_EPOCHS" 32 0.0005 0.00003 0.40 focal 2.0 5.0
      ;;
    p_drop25)
      run_bobsl_pretrain "$key" 42 "$BOBSL_EPOCHS" 32 0.0003 0.00003 0.25 focal 2.0 5.0
      ;;
    p_drop55)
      run_bobsl_pretrain "$key" 42 "$BOBSL_EPOCHS" 32 0.0003 0.00003 0.55 focal 2.0 5.0
      ;;
    p_nll)
      run_bobsl_pretrain "$key" 42 "$BOBSL_EPOCHS" 32 0.0003 0.00003 0.40 nll 2.0 5.0
      ;;
    *)
      echo "[MMGCN-VIDEO-SWEEP] unknown pretrain key: $key" >&2
      exit 1
      ;;
  esac
}

run_meld_finetune() {
  local variant="$1"
  local pair_out="$2"
  local init_checkpoint="$3"
  local seed="$4"
  local epochs="$5"
  local batch_size="$6"
  local lr="$7"
  local l2="$8"
  local dropout="$9"
  local loss="${10}"
  local focal_gamma="${11}"
  local max_grad_norm="${12}"
  shift 12
  local extra_args=("$@")
  local out_dir="$pair_out/$variant"
  local best_summary="$out_dir/video/external_test_best_summary.json"

  if [ "$RESUME" = "1" ] && [ -f "$best_summary" ]; then
    echo "[MMGCN-VIDEO-SWEEP][$pair_out][$variant] skip existing $best_summary"
    return
  fi

  mkdir -p "$out_dir"
  echo "[MMGCN-VIDEO-SWEEP][$(basename "$pair_out")][$variant] seed=$seed epochs=$epochs batch=$batch_size lr=$lr l2=$l2 dropout=$dropout loss=$loss gamma=$focal_gamma extra=${extra_args[*]}"

  local cmd=(
    python MMGCN/train_eval_mmgcn_unified.py
    --train_pkl "$MELD_UNIFIED_PKL"
    --external_test_pkl "$EJSL_UNIFIED_PKL"
    --out_dir "$out_dir"
    --modalities video
    --graph_type "$FT_GRAPH_TYPE"
    --epochs "$epochs"
    --batch_size "$batch_size"
    --lr "$lr"
    --l2 "$l2"
    --dropout "$dropout"
    --loss "$loss"
    --focal_gamma "$focal_gamma"
    --max_grad_norm "$max_grad_norm"
    --selection_split "$SELECTION_SPLIT"
    --selection_metric "$SELECTION_METRIC"
    --save_epoch_every "$SAVE_EPOCH_EVERY"
    --seed "$seed"
  )
  if [ -n "$init_checkpoint" ]; then
    cmd+=(--init_checkpoint "$init_checkpoint")
  fi
  cmd+=("${extra_args[@]}")

  run_logged "$out_dir/train.log" "${cmd[@]}"
}

run_pair_trial() {
  local name="$1"
  local pretrain_key="$2"
  local seed="$3"
  local epochs="$4"
  local batch_size="$5"
  local lr="$6"
  local l2="$7"
  local dropout="$8"
  local loss="$9"
  local focal_gamma="${10}"
  local max_grad_norm="${11}"
  shift 11
  local extra_args=("$@")
  local pair_out="$OUT_ROOT/$name"
  local checkpoint

  ensure_pretrain "$pretrain_key"
  checkpoint="$(pretrain_checkpoint_for "$pretrain_key")"
  mkdir -p "$pair_out"
  write_config "$pair_out/config.json" \
    "pair=$name" \
    "pretrain_key=$pretrain_key" \
    "pretrain_checkpoint=$checkpoint" \
    "ft_seed=$seed" \
    "ft_epochs=$epochs" \
    "ft_batch_size=$batch_size" \
    "ft_lr=$lr" \
    "ft_l2=$l2" \
    "ft_dropout=$dropout" \
    "ft_loss=$loss" \
    "ft_focal_gamma=$focal_gamma" \
    "ft_max_grad_norm=$max_grad_norm" \
    "ft_extra_args=${extra_args[*]}"

  run_meld_finetune scratch "$pair_out" "" "$seed" "$epochs" "$batch_size" "$lr" "$l2" "$dropout" "$loss" "$focal_gamma" "$max_grad_norm" "${extra_args[@]}"
  run_meld_finetune bobsl "$pair_out" "$checkpoint" "$seed" "$epochs" "$batch_size" "$lr" "$l2" "$dropout" "$loss" "$focal_gamma" "$max_grad_norm" "${extra_args[@]}"
}

for trial in $TRIALS; do
  case "$trial" in
    base_seed42)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed35)
      run_pair_trial "$trial" existing 35 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed36)
      run_pair_trial "$trial" existing 36 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed37)
      run_pair_trial "$trial" existing 37 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed38)
      run_pair_trial "$trial" existing 38 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed39)
      run_pair_trial "$trial" existing 39 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed40)
      run_pair_trial "$trial" existing 40 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed41)
      run_pair_trial "$trial" existing 41 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed43)
      run_pair_trial "$trial" existing 43 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed44)
      run_pair_trial "$trial" existing 44 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    seed45)
      run_pair_trial "$trial" existing 45 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    lr100)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0001 0.00003 0.40 focal 2.0 5.0
      ;;
    lr150)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.00015 0.00003 0.40 focal 2.0 5.0
      ;;
    lr200)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0002 0.00003 0.40 focal 2.0 5.0
      ;;
    lr300)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    lr500)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0005 0.00003 0.40 focal 2.0 5.0
      ;;
    drop20)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.20 focal 2.0 5.0
      ;;
    drop30)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.30 focal 2.0 5.0
      ;;
    drop50)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.50 focal 2.0 5.0
      ;;
    gamma10)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 1.0 5.0
      ;;
    gamma15)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 1.5 5.0
      ;;
    gamma25)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.5 5.0
      ;;
    nll)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 nll 2.0 5.0
      ;;
    nocw)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0 --no_class_weight
      ;;
    batch4)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 4 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    batch16)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 16 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    layer2)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0 --deep_gcn_nlayers 2
      ;;
    layer6)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0 --deep_gcn_nlayers 6
      ;;
    nores)
      run_pair_trial "$trial" existing 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0 --no_residue
      ;;
    pre_seed43)
      run_pair_trial "$trial" p_seed43 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    pre_seed44)
      run_pair_trial "$trial" p_seed44 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    pre_lr100)
      run_pair_trial "$trial" p_lr100 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    pre_lr500)
      run_pair_trial "$trial" p_lr500 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    pre_drop25)
      run_pair_trial "$trial" p_drop25 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    pre_drop55)
      run_pair_trial "$trial" p_drop55 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    pre_nll)
      run_pair_trial "$trial" p_nll 42 "$FT_EPOCHS" 8 0.0003 0.00003 0.40 focal 2.0 5.0
      ;;
    *)
      echo "[MMGCN-VIDEO-SWEEP] unknown trial: $trial" >&2
      exit 1
      ;;
  esac
done

python3 MMGCN/summarize_video_pair_sweep.py \
  --root "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/video_pair_sweep_runs.csv" \
  --out_delta_csv "$OUT_ROOT/video_pair_sweep_deltas.csv" \
  --top_k "$TOP_K" | tee "$OUT_ROOT/video_pair_sweep_summary.log"

echo "[MMGCN-VIDEO-SWEEP] done: $OUT_ROOT"
echo "[MMGCN-VIDEO-SWEEP] paired deltas: $OUT_ROOT/video_pair_sweep_deltas.csv"
echo "[MMGCN-VIDEO-SWEEP] run summaries: $OUT_ROOT/video_pair_sweep_runs.csv"
