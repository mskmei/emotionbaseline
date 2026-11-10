#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
BOBSL_K=${BOBSL_K:-5}
PROFILES=${PROFILES:-"conservative balanced open fullspace nosafety lowbobsl rank1_open rank1_safe very_low_lr high_lr source2 source4 bobsl2 scale05 scale4"}
BASE_MODE=${BASE_MODE:-source_only}
SEEDS=${SEEDS:-${SEED:-42}}
EPOCHS=${EPOCHS:-20}
RESUME_DONE=${RESUME_DONE:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/cesa_cmerc/screen}

profile_args() {
  case "$1" in
    conservative)
      PROFILE_ARGS=(--lr 0.0001 --residual_scale 0.5 --bobsl_weight 0.5 --safety_weight 2.0 --residual_weight 0.02)
      ;;
    balanced)
      PROFILE_ARGS=(--lr 0.0003 --residual_scale 1.0 --bobsl_weight 1.0 --safety_weight 1.0 --residual_weight 0.01)
      ;;
    open)
      PROFILE_ARGS=(--lr 0.0005 --residual_scale 2.0 --bobsl_weight 1.0 --local_source_weight 0.1 --safety_weight 0.5 --residual_weight 0.001)
      ;;
    fullspace)
      PROFILE_ARGS=(--lr 0.0001 --projection full --residual_scale 0.25 --bobsl_weight 0.5 --safety_weight 2.0 --residual_weight 0.02)
      ;;
    nosafety)
      PROFILE_ARGS=(--lr 0.0003 --residual_scale 1.0 --bobsl_weight 1.0 --safety_weight 0.0 --residual_weight 0.01)
      ;;
    lowbobsl)
      PROFILE_ARGS=(--lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.25 --local_source_weight 0.5 --safety_weight 2.0 --residual_weight 0.01)
      ;;
    rank1_open) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0005 --residual_scale 2.0 --bobsl_weight 1.0 --local_source_weight 0.1 --safety_weight 0.5 --residual_weight 0.001) ;;
    rank1_safe) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.5 --local_source_weight 0.25 --safety_weight 2.0 --residual_weight 0.01) ;;
    very_low_lr) PROFILE_ARGS=(--lr 0.00005 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 2.0 --residual_weight 0.02) ;;
    high_lr) PROFILE_ARGS=(--lr 0.001 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 2.0 --residual_weight 0.02) ;;
    source2) PROFILE_ARGS=(--lr 0.0003 --source_weight 2.0 --residual_scale 1.0 --bobsl_weight 1.0 --safety_weight 1.0 --residual_weight 0.01) ;;
    source4) PROFILE_ARGS=(--lr 0.0002 --source_weight 4.0 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 4.0 --residual_weight 0.02) ;;
    bobsl2) PROFILE_ARGS=(--lr 0.0003 --residual_scale 1.0 --bobsl_weight 2.0 --safety_weight 1.0 --residual_weight 0.01) ;;
    scale05) PROFILE_ARGS=(--lr 0.0003 --residual_scale 0.5 --bobsl_weight 1.0 --safety_weight 1.0 --residual_weight 0.01) ;;
    scale4) PROFILE_ARGS=(--lr 0.0003 --residual_scale 4.0 --bobsl_weight 1.0 --safety_weight 1.0 --residual_weight 0.01) ;;
    *) echo "Unknown profile: $1" >&2; return 2 ;;
  esac
}

corpus_paths() {
  case "$1" in
    MELD)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld/joint_pkls/joint_b${BOBSL_K}k.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld
      BASE_PREFIX=meld_only
      ;;
    IEMOCAP)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl${BOBSL_K}000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      BASE_PREFIX=iemocap_only
      ;;
    *) echo "Unknown corpus: $1" >&2; return 2 ;;
  esac
}

checkpoint_for_seed() {
  if [[ "$BASE_MODE" == plain_joint ]]; then
    printf '%s/joint_b%sk_seed%s/model_best.pt\n' "$BASE_ROOT" "$BOBSL_K" "$1"
  elif [[ "$BASE_MODE" == source_only ]]; then
    printf '%s/%s_seed%s/model_best.pt\n' "$BASE_ROOT" "$BASE_PREFIX" "$1"
  else
    echo "Unsupported BASE_MODE=$BASE_MODE" >&2
    return 2
  fi
}

mkdir -p "$WORK_ROOT/logs"
for corpus in $CORPORA; do
  corpus_paths "$corpus"
  for profile in $PROFILES; do
    profile_args "$profile"
    for seed in $SEEDS; do
    out_dir="$WORK_ROOT/${corpus,,}/b${BOBSL_K}k/${profile}_seed${seed}"
    log_path="$WORK_ROOT/logs/${corpus,,}_b${BOBSL_K}k_${profile}_seed${seed}.log"
    if [[ "$RESUME_DONE" == 1 && -s "$out_dir/external_test_best_summary.json" ]]; then
      echo "[CESA-SCREEN] skip $out_dir"
      continue
    fi
    mkdir -p "$out_dir"
    base_checkpoint=$(checkpoint_for_seed "$seed")
    echo "[CESA-SCREEN] corpus=$corpus bobsl=${BOBSL_K}k base_mode=$BASE_MODE profile=$profile seed=$seed gpu=$GPU" | tee "$log_path"
    "$PYTHON_BIN" new2/CMERC/train_eval_cmerc_cesa_visual_anjs.py \
      --train_pkl "$TRAIN_PKL" \
      --external_test_pkl "$EXTERNAL_PKL" \
      --base_checkpoint "$base_checkpoint" \
      --out_dir "$out_dir" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --source_batch_size 8 \
      --bobsl_batch_size 64 \
      --eval_batch_size 8 \
      --adapter_dim 128 \
      --local_hidden_dim 128 \
      --adapter_dropout 0.2 \
      --source_weight 1.0 \
      --local_source_weight 0.25 \
      --max_grad_norm 5.0 \
      "${PROFILE_ARGS[@]}" 2>&1 | tee -a "$log_path"
    done
  done
done

"$PYTHON_BIN" analysis/summarize_cesa_screen.py --root "$WORK_ROOT"
