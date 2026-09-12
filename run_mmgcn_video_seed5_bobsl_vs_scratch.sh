#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl}
UNIFIED_ROOT=${UNIFIED_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl}
OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/video_seed5_bobsl_vs_scratch"}
PRETRAIN_VARIANT_ROOT=${PRETRAIN_VARIANT_ROOT:-"$WORK_ROOT/video_bobsl_seed_param_sweep/pretrains"}

BOBSL_ROOT=${BOBSL_ROOT:-/raid_zoe/home/lr/wangyi/sign/bobsl}
MELD_RAW_ROOT=${MELD_RAW_ROOT:-./dataset/MELD.Raw}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
FRAME_ROOT=${FRAME_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame}
MP4_ROOT=${MP4_ROOT:-/raid_zoe/home/lr/wangyi/sign/eJSL_dial/video}
export TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ejsl_txt_en_openai}

BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-"$WORK_ROOT/bobsl_anjs4_train_val.pkl"}
BOBSL_TEST_PKL=${BOBSL_TEST_PKL:-"$WORK_ROOT/bobsl_anjs4_test.pkl"}
BOBSL_FEATURE_CACHE_DIR=${BOBSL_FEATURE_CACHE_DIR:-"$WORK_ROOT/bobsl_feature_cache"}
MELD_UNIFIED_PKL=${MELD_UNIFIED_PKL:-"$UNIFIED_ROOT/meld_anjs4_unified.pkl"}
EJSL_UNIFIED_PKL=${EJSL_UNIFIED_PKL:-"$UNIFIED_ROOT/ejsl_anjs4_unified.pkl"}
MELD_EJSL_FEATURE_CACHE_DIR=${MELD_EJSL_FEATURE_CACHE_DIR:-"$UNIFIED_ROOT/feature_cache"}
BASE_PRETRAIN_CHECKPOINT=${BASE_PRETRAIN_CHECKPOINT:-"$WORK_ROOT/bobsl_video_pretrain/video/model_best.pt"}

SEEDS=${SEEDS:-"35 36 37 41 42"}
CONFIGS=${CONFIGS:-"target_pre_lr100 target_pre_lr100_drop20 target_pre_lr100_gamma25 positive_pre_drop25 negative_pre_drop55 baseline_existing finetune_lr100 finetune_drop20 loss_gamma25"}
FT_EPOCHS=${FT_EPOCHS:-15}
BOBSL_EPOCHS=${BOBSL_EPOCHS:-30}
FT_GRAPH_TYPE=${FT_GRAPH_TYPE:-MMGCN}
BOBSL_GRAPH_TYPE=${BOBSL_GRAPH_TYPE:-MMGCN}
RESUME=${RESUME:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
REBUILD_BOBSL_FEATURES=${REBUILD_BOBSL_FEATURES:-0}
REBUILD_MELD_EJSL_FEATURES=${REBUILD_MELD_EJSL_FEATURES:-0}
RUN_PRETRAIN_VARIANTS=${RUN_PRETRAIN_VARIANTS:-1}
SAVE_EPOCH_EVERY=${SAVE_EPOCH_EVERY:-0}
TOP_K=${TOP_K:-20}

mkdir -p "$OUT_ROOT" "$PRETRAIN_VARIANT_ROOT"

if [ ! -d "$TRANSLATED_TXT_ROOT" ]; then
  echo "[MMGCN-SEED5] missing translated eJSL txt root: $TRANSLATED_TXT_ROOT" >&2
  exit 1
fi

run_logged() {
  local log_path="$1"
  shift
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$@" | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e
  if [ "$status" -ne 0 ]; then
    echo "[MMGCN-SEED5] command failed with status=$status" | tee "$log_path.failed"
    if [ "$CONTINUE_ON_ERROR" = "1" ]; then
      echo "[MMGCN-SEED5] CONTINUE_ON_ERROR=1, continuing"
      return 0
    fi
    exit "$status"
  fi
}

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
    if sep:
        data[key] = value
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

if [ "$REBUILD_BOBSL_FEATURES" = "1" ] || [ ! -f "$BOBSL_TRAIN_VAL_PKL" ] || [ ! -f "$BOBSL_TEST_PKL" ]; then
  echo "[MMGCN-SEED5] build BOBSL unified video features"
  run_logged "$OUT_ROOT/build_bobsl_features.log" python MMGCN/build_unified_bobsl_pkl.py \
    --bobsl_root "$BOBSL_ROOT" \
    --out_train_val_pkl "$BOBSL_TRAIN_VAL_PKL" \
    --out_test_pkl "$BOBSL_TEST_PKL" \
    --cache_dir "$BOBSL_FEATURE_CACHE_DIR" \
    --fp16
fi

if [ "$REBUILD_MELD_EJSL_FEATURES" = "1" ] || [ ! -f "$MELD_UNIFIED_PKL" ] || [ ! -f "$EJSL_UNIFIED_PKL" ]; then
  echo "[MMGCN-SEED5] build same-origin MELD/eJSL features from translated eJSL text"
  run_logged "$OUT_ROOT/build_meld_ejsl_features.log" python MMGCN/build_unified_meld_ejsl_pkl.py \
    --meld_root "$MELD_RAW_ROOT" \
    --ejsl_txt_root "$TRANSLATED_TXT_ROOT" \
    --ejsl_dial_list "$DIAL_LIST" \
    --ejsl_frame_root "$FRAME_ROOT" \
    --ejsl_mp4_root "$MP4_ROOT" \
    --out_meld_pkl "$MELD_UNIFIED_PKL" \
    --out_ejsl_pkl "$EJSL_UNIFIED_PKL" \
    --cache_dir "$MELD_EJSL_FEATURE_CACHE_DIR" \
    --fp16
fi

for required in "$BOBSL_TRAIN_VAL_PKL" "$BOBSL_TEST_PKL" "$MELD_UNIFIED_PKL" "$EJSL_UNIFIED_PKL"; do
  if [ ! -f "$required" ]; then
    echo "[MMGCN-SEED5] missing required file: $required" >&2
    exit 1
  fi
done

pretrain_checkpoint_for() {
  local key="$1"
  if [ "$key" = "existing" ]; then
    echo "$BASE_PRETRAIN_CHECKPOINT"
  else
    echo "$PRETRAIN_VARIANT_ROOT/$key/video/model_best.pt"
  fi
}

run_bobsl_pretrain() {
  local key="$1"
  local seed="$2"
  local lr="$3"
  local dropout="$4"
  local loss="$5"
  local gamma="$6"
  local out_dir
  out_dir="$PRETRAIN_VARIANT_ROOT/$key"
  local checkpoint="$out_dir/video/model_best.pt"
  if [ "$RESUME" = "1" ] && [ -f "$checkpoint" ]; then
    echo "[MMGCN-SEED5][pretrain:$key] skip existing $checkpoint"
    return
  fi
  mkdir -p "$out_dir"
  echo "[MMGCN-SEED5][pretrain:$key] seed=$seed lr=$lr dropout=$dropout loss=$loss gamma=$gamma"
  run_logged "$out_dir/train.log" python MMGCN/train_eval_mmgcn_unified.py \
    --train_pkl "$BOBSL_TRAIN_VAL_PKL" \
    --external_test_pkl "$BOBSL_TEST_PKL" \
    --out_dir "$out_dir" \
    --modalities video \
    --graph_type "$BOBSL_GRAPH_TYPE" \
    --epochs "$BOBSL_EPOCHS" \
    --batch_size 32 \
    --lr "$lr" \
    --l2 0.00003 \
    --dropout "$dropout" \
    --loss "$loss" \
    --focal_gamma "$gamma" \
    --max_grad_norm 5.0 \
    --selection_split source \
    --selection_metric weighted_f1 \
    --save_epoch_every "$SAVE_EPOCH_EVERY" \
    --seed "$seed"
}

ensure_pretrain() {
  local key="$1"
  if [ "$key" = "existing" ]; then
    if [ ! -f "$BASE_PRETRAIN_CHECKPOINT" ]; then
      if [ "$RUN_PRETRAIN_VARIANTS" != "1" ]; then
        echo "[MMGCN-SEED5] missing BASE_PRETRAIN_CHECKPOINT: $BASE_PRETRAIN_CHECKPOINT" >&2
        exit 1
      fi
      mkdir -p "$(dirname "$(dirname "$BASE_PRETRAIN_CHECKPOINT")")"
      echo "[MMGCN-SEED5] base BOBSL checkpoint missing; training it at $(dirname "$(dirname "$BASE_PRETRAIN_CHECKPOINT")")"
      local old_root="$PRETRAIN_VARIANT_ROOT"
      PRETRAIN_VARIANT_ROOT="$(dirname "$(dirname "$BASE_PRETRAIN_CHECKPOINT")")"
      run_bobsl_pretrain "." 42 0.0003 0.40 focal 2.0
      PRETRAIN_VARIANT_ROOT="$old_root"
    fi
    return
  fi
  local checkpoint
  checkpoint="$(pretrain_checkpoint_for "$key")"
  if [ "$RUN_PRETRAIN_VARIANTS" != "1" ] && [ ! -f "$checkpoint" ]; then
    echo "[MMGCN-SEED5] missing pretrain variant checkpoint: $checkpoint" >&2
    exit 1
  fi
  case "$key" in
    p_lr100)
      run_bobsl_pretrain "$key" 42 0.0001 0.40 focal 2.0
      ;;
    p_drop25)
      run_bobsl_pretrain "$key" 42 0.0003 0.25 focal 2.0
      ;;
    p_drop55)
      run_bobsl_pretrain "$key" 42 0.0003 0.55 focal 2.0
      ;;
    *)
      echo "[MMGCN-SEED5] unknown pretrain key: $key" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local out_dir="$1"
  local init_checkpoint="$2"
  local seed="$3"
  local lr="$4"
  local dropout="$5"
  local loss="$6"
  local gamma="$7"
  shift 7
  local extra_args=("$@")
  local best_summary="$out_dir/video/external_test_best_summary.json"
  if [ "$RESUME" = "1" ] && [ -f "$best_summary" ]; then
    echo "[MMGCN-SEED5] skip existing $best_summary"
    return
  fi
  mkdir -p "$out_dir"
  local cmd=(
    python MMGCN/train_eval_mmgcn_unified.py
    --train_pkl "$MELD_UNIFIED_PKL"
    --external_test_pkl "$EJSL_UNIFIED_PKL"
    --out_dir "$out_dir"
    --modalities video
    --graph_type "$FT_GRAPH_TYPE"
    --epochs "$FT_EPOCHS"
    --batch_size 8
    --lr "$lr"
    --l2 0.00003
    --dropout "$dropout"
    --loss "$loss"
    --focal_gamma "$gamma"
    --max_grad_norm 5.0
    --selection_split source
    --selection_metric weighted_f1
    --save_epoch_every "$SAVE_EPOCH_EVERY"
    --seed "$seed"
  )
  if [ -n "$init_checkpoint" ]; then
    cmd+=(--init_checkpoint "$init_checkpoint")
  fi
  cmd+=("${extra_args[@]}")
  run_logged "$out_dir/train.log" "${cmd[@]}"
}

run_pair() {
  local config="$1"
  local seed="$2"
  local pretrain_key="$3"
  local lr="$4"
  local dropout="$5"
  local loss="$6"
  local gamma="$7"
  shift 7
  local extra_args=("$@")
  local pair="$config""_seed""$seed"
  local pair_out="$OUT_ROOT/$pair"
  local checkpoint
  ensure_pretrain "$pretrain_key"
  checkpoint="$(pretrain_checkpoint_for "$pretrain_key")"
  mkdir -p "$pair_out"
  write_config "$pair_out/config.json" \
    "pair=$pair" \
    "config=$config" \
    "seed=$seed" \
    "pretrain_key=$pretrain_key" \
    "pretrain_checkpoint=$checkpoint" \
    "ft_lr=$lr" \
    "ft_dropout=$dropout" \
    "ft_loss=$loss" \
    "ft_focal_gamma=$gamma" \
    "ft_extra_args=${extra_args[*]}"
  echo "[MMGCN-SEED5][$pair] scratch then BOBSL-init"
  run_one "$pair_out/scratch" "" "$seed" "$lr" "$dropout" "$loss" "$gamma" "${extra_args[@]}"
  run_one "$pair_out/bobsl" "$checkpoint" "$seed" "$lr" "$dropout" "$loss" "$gamma" "${extra_args[@]}"
}

for config in $CONFIGS; do
  for seed in $SEEDS; do
    case "$config" in
      target_pre_lr100)
        run_pair "$config" "$seed" p_lr100 0.0003 0.40 focal 2.0
        ;;
      target_pre_lr100_drop20)
        run_pair "$config" "$seed" p_lr100 0.0003 0.20 focal 2.0
        ;;
      target_pre_lr100_gamma25)
        run_pair "$config" "$seed" p_lr100 0.0003 0.40 focal 2.5
        ;;
      positive_pre_drop25)
        run_pair "$config" "$seed" p_drop25 0.0003 0.40 focal 2.0
        ;;
      negative_pre_drop55)
        run_pair "$config" "$seed" p_drop55 0.0003 0.40 focal 2.0
        ;;
      baseline_existing)
        run_pair "$config" "$seed" existing 0.0003 0.40 focal 2.0
        ;;
      finetune_lr100)
        run_pair "$config" "$seed" existing 0.0001 0.40 focal 2.0
        ;;
      finetune_drop20)
        run_pair "$config" "$seed" existing 0.0003 0.20 focal 2.0
        ;;
      loss_gamma25)
        run_pair "$config" "$seed" existing 0.0003 0.40 focal 2.5
        ;;
      *)
        echo "[MMGCN-SEED5] unknown config: $config" >&2
        exit 1
        ;;
    esac
  done
done

python3 MMGCN/summarize_video_pair_sweep.py \
  --root "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/video_pair_sweep_runs.csv" \
  --out_delta_csv "$OUT_ROOT/video_pair_sweep_deltas.csv" \
  --top_k "$TOP_K" | tee "$OUT_ROOT/video_pair_sweep_summary.log"

python3 MMGCN/analyze_video_pair_meld_performance.py \
  --delta_csv "$OUT_ROOT/video_pair_sweep_deltas.csv" \
  --runs_csv "$OUT_ROOT/video_pair_sweep_runs.csv" \
  --out_txt "$OUT_ROOT/video_pair_meld_analysis.txt" \
  --out_ranked_csv "$OUT_ROOT/video_pair_meld_ranked.csv" \
  --top_k "$TOP_K" | tee "$OUT_ROOT/video_pair_meld_analysis.log"

python3 MMGCN/summarize_video_seed5_pair_averages.py \
  --root "$OUT_ROOT" \
  --top_k "$TOP_K" | tee "$OUT_ROOT/video_seed5_group_averages.log"

echo "[MMGCN-SEED5] done: $OUT_ROOT"
echo "[MMGCN-SEED5] per-seed deltas: $OUT_ROOT/video_pair_sweep_deltas.csv"
echo "[MMGCN-SEED5] group averages: $OUT_ROOT/video_seed5_group_averages.csv"
