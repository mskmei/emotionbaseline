#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
BOBSL_K=${BOBSL_K:-5}
SEEDS=${SEEDS:-42}
PROFILES=${PROFILES:-"gamma1_lr10 gamma3_lr75 gamma2_lr75 gamma4_lr75 gamma3_lr5 gamma3_lr10 gamma3_lr15 gamma05_lr10 gamma15_lr10 gamma3_l2low gamma3_l2high gamma3_layer2 gamma3_hidden256 gamma3_nocw"}
EPOCHS=${EPOCHS:-40}
RESUME_DONE=${RESUME_DONE:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/sast_large_sweep/ECERC}

set_paths() {
  case "$1" in
    MELD)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_pkls/joint_b${BOBSL_K}k.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/ejsl_ecerc_video_anjs4_translated.pkl
      ;;
    IEMOCAP)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl${BOBSL_K}000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/ecerc_bobsl_meld_ejsl/features/ejsl_ecerc_video_anjs4_translated.pkl
      ;;
    *) echo "Unknown corpus=$1" >&2; return 2 ;;
  esac
}

profile_args() {
  CLASS_WEIGHT=1
  case "$1" in
    gamma1_lr10) PROFILE_ARGS=(--lr 0.00001 --loss_gamma 1.0 --l2 0.0002 --base_layer 1 --hidden_size 128) ;;
    gamma3_lr75) PROFILE_ARGS=(--lr 0.0000075 --loss_gamma 3.0 --l2 0.0002 --base_layer 1 --hidden_size 128) ;;
    gamma2_lr75) PROFILE_ARGS=(--lr 0.0000075 --loss_gamma 2.0 --l2 0.0002 --base_layer 1 --hidden_size 128) ;;
    gamma4_lr75) PROFILE_ARGS=(--lr 0.0000075 --loss_gamma 4.0 --l2 0.0002 --base_layer 1 --hidden_size 128) ;;
    gamma3_lr5) PROFILE_ARGS=(--lr 0.000005 --loss_gamma 3.0 --l2 0.0002 --base_layer 1 --hidden_size 128) ;;
    gamma3_lr10) PROFILE_ARGS=(--lr 0.00001 --loss_gamma 3.0 --l2 0.0002 --base_layer 1 --hidden_size 128) ;;
    gamma3_lr15) PROFILE_ARGS=(--lr 0.000015 --loss_gamma 3.0 --l2 0.0002 --base_layer 1 --hidden_size 128) ;;
    gamma05_lr10) PROFILE_ARGS=(--lr 0.00001 --loss_gamma 0.5 --l2 0.0002 --base_layer 1 --hidden_size 128) ;;
    gamma15_lr10) PROFILE_ARGS=(--lr 0.00001 --loss_gamma 1.5 --l2 0.0002 --base_layer 1 --hidden_size 128) ;;
    gamma3_l2low) PROFILE_ARGS=(--lr 0.0000075 --loss_gamma 3.0 --l2 0.00005 --base_layer 1 --hidden_size 128) ;;
    gamma3_l2high) PROFILE_ARGS=(--lr 0.0000075 --loss_gamma 3.0 --l2 0.0004 --base_layer 1 --hidden_size 128) ;;
    gamma3_layer2) PROFILE_ARGS=(--lr 0.0000075 --loss_gamma 3.0 --l2 0.0002 --base_layer 2 --hidden_size 128) ;;
    gamma3_hidden256) PROFILE_ARGS=(--lr 0.0000075 --loss_gamma 3.0 --l2 0.0002 --base_layer 1 --hidden_size 256) ;;
    gamma3_nocw) PROFILE_ARGS=(--lr 0.0000075 --loss_gamma 3.0 --l2 0.0002 --base_layer 1 --hidden_size 128); CLASS_WEIGHT=0 ;;
    *) echo "Unknown profile=$1" >&2; return 2 ;;
  esac
}

mkdir -p "$WORK_ROOT/logs"
for corpus in $CORPORA; do
  set_paths "$corpus"
  for profile in $PROFILES; do
    profile_args "$profile"
    for seed in $SEEDS; do
      out_dir="$WORK_ROOT/${corpus,,}/b${BOBSL_K}k/${profile}_seed${seed}"
      log_path="$WORK_ROOT/logs/${corpus,,}_b${BOBSL_K}k_${profile}_seed${seed}.log"
      if [[ "$RESUME_DONE" == 1 && -s "$out_dir/external_test_best_summary.json" ]]; then
        echo "[ECERC-LARGE] skip $out_dir"
        continue
      fi
      class_weight_args=()
      if [[ "$CLASS_WEIGHT" == 1 ]]; then class_weight_args+=(--class_weight); fi
      echo "[ECERC-LARGE] corpus=$corpus bobsl=${BOBSL_K}k profile=$profile seed=$seed gpu=$GPU" | tee "$log_path"
      "$PYTHON_BIN" new/ECERC/MELD/train_eval_ecerc_video_anjs.py \
        --train_pkl "$TRAIN_PKL" \
        --external_test_pkl "$EXTERNAL_PKL" \
        --out_dir "$out_dir" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch_size 32 \
        --num_workers 0 \
        --max_grad_norm 5.0 \
        --selection_split source \
        --selection_metric weighted_f1 \
        --save_epoch_every 0 \
        "${PROFILE_ARGS[@]}" \
        "${class_weight_args[@]}" 2>&1 | tee -a "$log_path"
    done
  done
done
