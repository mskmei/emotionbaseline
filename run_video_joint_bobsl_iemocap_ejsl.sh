#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE to MMGCN, ECERC, MAGTKD, CMERC, or ConxGNN}
BASELINE_LOWER=$(printf '%s' "$BASELINE" | tr '[:upper:]' '[:lower:]')
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"

IEMOCAP_ROOT=${IEMOCAP_ROOT:-/raid_zoe/home/lr/maokeyu/IEMOCAP_full_release}
IEMOCAP_METADATA_ROOT=${IEMOCAP_METADATA_ROOT:-"$ROOT_DIR/new/MAGTKD/datasets/IEMOCAP"}
IEMOCAP_FRAME_ROOT=${IEMOCAP_FRAME_ROOT:-/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/frames_8_margin1}
IEMOCAP_MANIFEST_ROOT=${IEMOCAP_MANIFEST_ROOT:-/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/manifests}
IEMOCAP_FEATURE_CACHE=${IEMOCAP_FEATURE_CACHE:-/raid_zoe/home/lr/maokeyu/sign/iemocap_bobsl_ejsl/timesformer_feature_cache}
VIDEO_MODEL=${VIDEO_MODEL:-facebook/timesformer-base-finetuned-k400}
VIDEO_PROCESSOR=${VIDEO_PROCESSOR:-MCG-NJU/videomae-base}
NUM_FRAMES=${NUM_FRAMES:-8}
FRAME_MARGIN_SECONDS=${FRAME_MARGIN_SECONDS:-1.0}

SEEDS=${SEEDS:-"35 36 37 41 42"}
BOBSL_SAMPLE_SEED=${BOBSL_SAMPLE_SEED:-123}
INCLUDE_BOBSL_VAL=${INCLUDE_BOBSL_VAL:-0}
PREPARE_FRAMES=${PREPARE_FRAMES:-1}
BUILD_FEATURES=${BUILD_FEATURES:-1}
REBUILD_IEMOCAP_FEATURES=${REBUILD_IEMOCAP_FEATURES:-0}
REBUILD_JOINT=${REBUILD_JOINT:-0}
RESUME_DONE=${RESUME_DONE:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
SAVE_EPOCH_EVERY=${SAVE_EPOCH_EVERY:-0}

case "$BASELINE" in
  MMGCN)
    WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl}
    IEMOCAP_PKL=${IEMOCAP_PKL:-"$WORK_ROOT/features/iemocap_mmgcn_video_anjs4.pkl"}
    BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/bobsl_anjs4_train_val.pkl}
    EJSL_PKL=${EJSL_PKL:-/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl}
    CONFIGS=${CONFIGS:-"iemocap_only joint_b1k joint_b2k joint_b5k joint_b10k"}
    EPOCHS=${EPOCHS:-15}
    ;;
  ECERC)
    WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl}
    IEMOCAP_PKL=${IEMOCAP_PKL:-"$WORK_ROOT/features/iemocap_ecerc_video_anjs4.pkl"}
    BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/bobsl_ecerc_video_anjs4_train_val.pkl}
    EJSL_PKL=${EJSL_PKL:-/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/ejsl_ecerc_video_anjs4_translated.pkl}
    CONFIGS=${CONFIGS:-"iemocap_only joint_b1k joint_b2k joint_b5k joint_b10k"}
    EPOCHS=${EPOCHS:-40}
    ;;
  MAGTKD)
    WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl}
    IEMOCAP_PKL=${IEMOCAP_PKL:-"$WORK_ROOT/features/iemocap_magtkd_video_anjs4.pkl"}
    BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/bobsl_magtkd_video_anjs4_train_val.pkl}
    EJSL_PKL=${EJSL_PKL:-/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl}
    CONFIGS=${CONFIGS:-"iemocap_only joint_b1k joint_b2k joint_b5k joint_b10k"}
    EPOCHS=${EPOCHS:-30}
    ;;
  CMERC)
    WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl}
    IEMOCAP_PKL=${IEMOCAP_PKL:-"$WORK_ROOT/features/iemocap_cmerc_visual_anjs4.pkl"}
    BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/bobsl_cmerc_visual_anjs4_train_val.pkl}
    EJSL_PKL=${EJSL_PKL:-/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl}
    CONFIGS=${CONFIGS:-"iemocap_only joint_b1k joint_b2k joint_b5k joint_b10k"}
    EPOCHS=${EPOCHS:-40}
    ;;
  ConxGNN)
    WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_iemocap_ejsl}
    IEMOCAP_PKL=${IEMOCAP_PKL:-"$WORK_ROOT/features/iemocap_conxgnn_visual_anjs4.pkl"}
    BOBSL_TRAIN_VAL_PKL=${BOBSL_TRAIN_VAL_PKL:-/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/bobsl_conxgnn_visual_anjs4_train_val.pkl}
    EJSL_PKL=${EJSL_PKL:-/raid_zoe/home/lr/maokeyu/sign/conxgnn_bobsl_meld_ejsl/features/ejsl_conxgnn_visual_anjs4_translated.pkl}
    CONFIGS=${CONFIGS:-"iemocap_only joint_b1k joint_b2k joint_b5k joint_b10k"}
    EPOCHS=${EPOCHS:-30}
    ;;
  *)
    echo "Unknown BASELINE=$BASELINE" >&2
    exit 2
    ;;
esac

OUT_ROOT=${OUT_ROOT:-"$WORK_ROOT/video_joint_bobsl_iemocap"}
JOINT_PKL_ROOT=${JOINT_PKL_ROOT:-"$OUT_ROOT/joint_pkls"}
LOG_ROOT=${LOG_ROOT:-"$OUT_ROOT/logs"}
mkdir -p "$WORK_ROOT/features" "$OUT_ROOT" "$JOINT_PKL_ROOT" "$LOG_ROOT"

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"
  echo "[$BASELINE-IEMOCAP] $*" | tee "$log_path"
  set +e
  "$@" 2>&1 | tee -a "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    touch "$log_path.failed"
    if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
      echo "[$BASELINE-IEMOCAP] failed status=$status; continuing" | tee -a "$log_path"
      return 0
    fi
    return "$status"
  fi
  rm -f "$log_path.failed"
}

if [[ "$PREPARE_FRAMES" == "1" ]]; then
  run_logged "$LOG_ROOT/prepare_iemocap_frames.log" \
    python IEMOCAP/prepare_iemocap_anjs4_frames.py \
      --iemocap_root "$IEMOCAP_ROOT" \
      --metadata_root "$IEMOCAP_METADATA_ROOT" \
      --frame_root "$IEMOCAP_FRAME_ROOT" \
      --manifest_root "$IEMOCAP_MANIFEST_ROOT" \
      --num_frames "$NUM_FRAMES" \
      --margin_seconds "$FRAME_MARGIN_SECONDS" \
      --strict
fi

for required in "$BOBSL_TRAIN_VAL_PKL" "$EJSL_PKL"; do
  if [[ ! -s "$required" ]]; then
    echo "[$BASELINE-IEMOCAP] missing existing MELD-experiment feature pkl: $required" >&2
    echo "Run the corresponding BOBSL+MELD script's feature stage first, or override the path." >&2
    exit 1
  fi
done

if [[ "$BUILD_FEATURES" == "1" && ( "$REBUILD_IEMOCAP_FEATURES" == "1" || ! -s "$IEMOCAP_PKL" ) ]]; then
  feature_args=(
    python IEMOCAP/build_iemocap_anjs4_baseline_features.py
    --baselines "$BASELINE_LOWER"
    --iemocap_root "$IEMOCAP_ROOT"
    --metadata_root "$IEMOCAP_METADATA_ROOT"
    --frame_root "$IEMOCAP_FRAME_ROOT"
    --cache_dir "$IEMOCAP_FEATURE_CACHE"
    --video_model "$VIDEO_MODEL"
    --video_processor "$VIDEO_PROCESSOR"
    --num_frames "$NUM_FRAMES"
    --fp16
  )
  case "$BASELINE" in
    MMGCN) feature_args+=(--reference_mmgcn_pkl "$BOBSL_TRAIN_VAL_PKL" --out_mmgcn_pkl "$IEMOCAP_PKL") ;;
    ECERC) feature_args+=(--reference_ecerc_pkl "$BOBSL_TRAIN_VAL_PKL" --out_ecerc_pkl "$IEMOCAP_PKL") ;;
    MAGTKD) feature_args+=(--reference_magtkd_pkl "$BOBSL_TRAIN_VAL_PKL" --out_magtkd_pkl "$IEMOCAP_PKL" --local_model_root ./new/MAGTKD/pretrained_model) ;;
    CMERC) feature_args+=(--reference_cmerc_pkl "$BOBSL_TRAIN_VAL_PKL" --out_cmerc_pkl "$IEMOCAP_PKL") ;;
    ConxGNN) feature_args+=(--reference_conxgnn_pkl "$BOBSL_TRAIN_VAL_PKL" --out_conxgnn_pkl "$IEMOCAP_PKL") ;;
  esac
  run_logged "$LOG_ROOT/build_iemocap_features.log" "${feature_args[@]}"
fi

if [[ ! -s "$IEMOCAP_PKL" ]]; then
  echo "[$BASELINE-IEMOCAP] missing IEMOCAP pkl: $IEMOCAP_PKL" >&2
  exit 1
fi

joint_pkl_for() {
  local max_bobsl="$1"
  printf '%s\n' "$JOINT_PKL_ROOT/iemocap_bobsl${max_bobsl}_trainonly_sample${BOBSL_SAMPLE_SEED}.pkl"
}

build_joint_if_needed() {
  local max_bobsl="$1"
  local out_pkl="$2"
  if [[ "$REBUILD_JOINT" != "1" && -s "$out_pkl" ]]; then
    return
  fi
  local builder
  case "$BASELINE" in
    MMGCN) builder=MMGCN/build_joint_meld_bobsl_video_pkl.py ;;
    ECERC) builder=new/ECERC/MELD/build_ecerc_joint_video_pkl.py ;;
    MAGTKD) builder=new/MAGTKD/MELD/build_magtkd_joint_video_pkl.py ;;
    CMERC) builder=new2/CMERC/build_cmerc_joint_visual_pkl.py ;;
    ConxGNN) builder=new2/ConxGNN/build_conxgnn_joint_visual_pkl.py ;;
  esac
  local cmd=(
    python "$builder"
    --source_pkl "$IEMOCAP_PKL"
    --source_name IEMOCAP
    --bobsl_train_val_pkl "$BOBSL_TRAIN_VAL_PKL"
    --out_pkl "$out_pkl"
    --max_bobsl_train_dialogues "$max_bobsl"
    --bobsl_sample_seed "$BOBSL_SAMPLE_SEED"
  )
  if [[ "$BASELINE" == "MMGCN" ]]; then
    cmd+=(--force_bobsl_context1)
  fi
  if [[ "$INCLUDE_BOBSL_VAL" == "1" ]]; then
    cmd+=(--include_bobsl_val)
  fi
  run_logged "$out_pkl.build.log" "${cmd[@]}" >&2
}

write_config() {
  local out_dir="$1" config="$2" seed="$3" bobsl_max="$4" train_pkl="$5" lr="$6" dropout="$7"
  python - "$out_dir/config.json" "$config" "$seed" "$bobsl_max" "$train_pkl" "$lr" "$dropout" <<'PY'
import json
import os
import sys
from pathlib import Path

path, config, seed, bobsl_max, train_pkl, lr, dropout = sys.argv[1:8]
data = {
    "baseline": os.environ["BASELINE"],
    "config": config,
    "seed": int(seed),
    "bobsl_max": int(bobsl_max),
    "bobsl_sample_seed": int(os.environ["BOBSL_SAMPLE_SEED"]),
    "train_pkl": train_pkl,
    "source_test": "IEMOCAP Session5 test",
    "external_test": "translated eJSL test",
    "modality": "video",
    "epochs": int(os.environ["EPOCHS"]),
    "lr": float(lr),
    "dropout": float(dropout),
    "selection_split": "source",
    "selection_metric": "weighted_f1",
}
Path(path).parent.mkdir(parents=True, exist_ok=True)
Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

export BASELINE BOBSL_SAMPLE_SEED EPOCHS

run_trial() {
  local config="$1" seed="$2" bobsl_max="$3" train_pkl="$4" lr="$5" dropout="$6"
  local out_dir="$OUT_ROOT/${config}_seed${seed}"
  local result_dir="$out_dir"
  [[ "$BASELINE" == "MMGCN" ]] && result_dir="$out_dir/video"
  if [[ "$RESUME_DONE" == "1" && -s "$result_dir/external_test_best_summary.json" ]]; then
    echo "[$BASELINE-IEMOCAP] skip existing $result_dir/external_test_best_summary.json"
    return
  fi
  mkdir -p "$out_dir"
  write_config "$out_dir" "$config" "$seed" "$bobsl_max" "$train_pkl" "$lr" "$dropout"

  local cmd
  case "$BASELINE" in
    MMGCN)
      cmd=(python MMGCN/train_eval_mmgcn_unified.py --train_pkl "$train_pkl" --external_test_pkl "$EJSL_PKL" --out_dir "$out_dir" --modalities video --graph_type MMGCN --epochs "$EPOCHS" --batch_size 8 --lr "$lr" --l2 0.00003 --dropout "$dropout" --loss focal --focal_gamma 2.0 --max_grad_norm 5.0 --selection_split source --selection_metric weighted_f1 --save_epoch_every "$SAVE_EPOCH_EVERY" --seed "$seed")
      ;;
    ECERC)
      cmd=(python new/ECERC/MELD/train_eval_ecerc_video_anjs.py --train_pkl "$train_pkl" --external_test_pkl "$EJSL_PKL" --out_dir "$out_dir" --seed "$seed" --epochs "$EPOCHS" --batch_size 32 --num_workers 0 --lr 0.00001 --l2 0.0002 --loss_gamma 1.0 --base_layer 1 --hidden_size 128 --max_grad_norm 5.0 --selection_split source --selection_metric weighted_f1 --save_epoch_every "$SAVE_EPOCH_EVERY" --class_weight)
      ;;
    MAGTKD)
      cmd=(python new/MAGTKD/MELD/train_eval_magtkd_video_anjs.py --train_pkl "$train_pkl" --external_test_pkl "$EJSL_PKL" --out_dir "$out_dir" --seed "$seed" --epochs "$EPOCHS" --batch_size 16 --num_workers 0 --lr 0.0001 --l2 0.000001 --dropout 0.5 --hidden_dim 768 --n_head 8 --temp 2.0 --output_head video --max_grad_norm 10.0 --selection_split source --selection_metric weighted_f1 --save_epoch_every "$SAVE_EPOCH_EVERY" --class_weight)
      ;;
    CMERC)
      cmd=(python new2/CMERC/train_eval_cmerc_visual_anjs.py --train_pkl "$train_pkl" --external_test_pkl "$EJSL_PKL" --out_dir "$out_dir" --seed "$seed" --epochs "$EPOCHS" --batch_size 16 --num_workers 0 --lr 0.0001 --l2 0.00003 --dropout 0.4 --loss focal --focal_gamma 2.5 --max_grad_norm 5.0 --selection_split source --selection_metric weighted_f1 --save_epoch_every "$SAVE_EPOCH_EVERY")
      ;;
    ConxGNN)
      cmd=(python new2/ConxGNN/train_eval_conxgnn_visual_anjs.py --train_pkl "$train_pkl" --external_test_pkl "$EJSL_PKL" --out_dir "$out_dir" --seed "$seed" --epochs "$EPOCHS" --batch_size 64 --lr 0.0003 --weight_decay 0.00000001 --drop_rate 0.8 --hidden_size 128 --inter_size 256 --loss ce --focal_gamma 2.5 --ce_loss_param 1.0 --cb_loss_param 0.0 --max_grad_norm 5.0 --selection_split source --selection_metric weighted_f1 --save_epoch_every "$SAVE_EPOCH_EVERY" --class_weight)
      ;;
  esac
  run_logged "$LOG_ROOT/${config}_seed${seed}.log" "${cmd[@]}"
}

for config in $CONFIGS; do
  lr=0
  dropout=0
  case "$config" in
    iemocap_only)
      bobsl_max=0
      train_pkl="$IEMOCAP_PKL"
      ;;
    joint_b1k) bobsl_max=1000 ;;
    joint_b2k) bobsl_max=2000 ;;
    joint_b5k) bobsl_max=5000 ;;
    joint_b10k) bobsl_max=10000 ;;
    *) echo "[$BASELINE-IEMOCAP] unknown config: $config" >&2; exit 2 ;;
  esac
  if [[ "$config" != "iemocap_only" ]]; then
    train_pkl=$(joint_pkl_for "$bobsl_max")
    build_joint_if_needed "$bobsl_max" "$train_pkl"
  fi
  case "$BASELINE" in
    MMGCN) lr=0.0003; dropout=0.4 ;;
    ECERC) lr=0.00001; dropout=0.0 ;;
    MAGTKD) lr=0.0001; dropout=0.5 ;;
    CMERC) lr=0.0001; dropout=0.4 ;;
    ConxGNN) lr=0.0003; dropout=0.8 ;;
  esac
  for seed in $SEEDS; do
    run_trial "$config" "$seed" "$bobsl_max" "$train_pkl" "$lr" "$dropout"
  done
done

python IEMOCAP/summarize_iemocap_joint_baselines.py \
  --root "$BASELINE=$OUT_ROOT" \
  --out_dir "$OUT_ROOT/summary"

echo "[$BASELINE-IEMOCAP] done: $OUT_ROOT"
