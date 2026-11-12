#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

BASELINE=${BASELINE:?Set BASELINE=MMGCN or MAGTKD}
CORPORA=${CORPORA:-MELD}
BOBSL_K=${BOBSL_K:-5}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
BASE_MODE=${BASE_MODE:-source_only}
JOINT_TEACHER=${JOINT_TEACHER:-0}
PEER_MODE=${PEER_MODE:-opposite}
SEEDS=${SEEDS:-42}
PROFILES=${PROFILES:-"context_r3 context_r1 proto10_r3 proto50_r3 safe_r3 open_r3 source2_r3 full_safe low_lr very_low_lr rank1_low rank1_safe b25_low source4_r3 scale05_r3 scale1_r3 no_weight"}
EPOCHS=${EPOCHS:-12}
RESUME_DONE=${RESUME_DONE:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/sast_negative_transfer}

case "$BASELINE" in
  MMGCN)
    TRAINER=MMGCN/train_eval_mmgcn_sast_video_anjs.py
    ;;
  MAGTKD)
    TRAINER=new/MAGTKD/MELD/train_eval_magtkd_sast_video_anjs.py
    ;;
  *) echo "Unsupported BASELINE=$BASELINE" >&2; exit 2 ;;
esac

set_paths() {
  local corpus="$1"
  case "$BASELINE/$corpus" in
    MMGCN/MELD)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_pkls/meld_bobsl${BOBSL_K}000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld
      BASE_PREFIX=meld_only
      ;;
    MMGCN/IEMOCAP)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl${BOBSL_K}000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      BASE_PREFIX=iemocap_only
      ;;
    MAGTKD/MELD)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld/joint_pkls/joint_b${BOBSL_K}k.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld
      BASE_PREFIX=meld_only
      ;;
    MAGTKD/IEMOCAP)
      TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl${BOBSL_K}000_trainonly_sample123.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl
      BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      BASE_PREFIX=iemocap_only
      ;;
    *) echo "Unsupported BASELINE/CORPUS=$BASELINE/$corpus" >&2; return 2 ;;
  esac
}

checkpoint_for_seed() {
  if [[ "$BASE_MODE" == plain_joint ]]; then
    if [[ "$BASELINE" == MMGCN ]]; then
      printf '%s/joint_b%sk_seed%s/video/model_best.pt\n' "$BASE_ROOT" "$BOBSL_K" "$1"
    else
      printf '%s/joint_b%sk_seed%s/model_best.pt\n' "$BASE_ROOT" "$BOBSL_K" "$1"
    fi
    return
  fi
  if [[ "$BASE_MODE" != source_only ]]; then
    echo "Unsupported BASE_MODE=$BASE_MODE" >&2
    return 2
  fi
  if [[ "$BASELINE" == MMGCN ]]; then
    printf '%s/%s_seed%s/video/model_best.pt\n' "$BASE_ROOT" "$BASE_PREFIX" "$1"
  else
    printf '%s/%s_seed%s/model_best.pt\n' "$BASE_ROOT" "$BASE_PREFIX" "$1"
  fi
}

joint_checkpoint_for_seed() {
  local peer_mode="$PEER_MODE"
  if [[ "$peer_mode" == opposite ]]; then
    if [[ "$BASE_MODE" == source_only ]]; then peer_mode=plain_joint; else peer_mode=source_only; fi
  fi
  if [[ "$peer_mode" == plain_joint ]]; then
    if [[ "$BASELINE" == MMGCN ]]; then
      printf '%s/joint_b%sk_seed%s/video/model_best.pt\n' "$BASE_ROOT" "$BOBSL_K" "$1"
    else
      printf '%s/joint_b%sk_seed%s/model_best.pt\n' "$BASE_ROOT" "$BOBSL_K" "$1"
    fi
  elif [[ "$peer_mode" == source_only ]]; then
    if [[ "$BASELINE" == MMGCN ]]; then
      printf '%s/%s_seed%s/video/model_best.pt\n' "$BASE_ROOT" "$BASE_PREFIX" "$1"
    else
      printf '%s/%s_seed%s/model_best.pt\n' "$BASE_ROOT" "$BASE_PREFIX" "$1"
    fi
  else
    echo "Unsupported PEER_MODE=$PEER_MODE" >&2
    return 2
  fi
}

profile_args() {
  case "$1" in
    context_r3) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.0) ;;
    context_r1) PROFILE_ARGS=(--subspace_rank 1 --prototype_weight 0.0) ;;
    proto10_r3) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.1) ;;
    proto50_r3) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.5) ;;
    safe_r3) PROFILE_ARGS=(--subspace_rank 3 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --prototype_weight 0.1) ;;
    open_r3) PROFILE_ARGS=(--subspace_rank 3 --residual_scale 4.0 --safety_weight 0.25 --stability_weight 0.05 --prototype_weight 0.1 --residual_weight 0.0001) ;;
    source2_r3) PROFILE_ARGS=(--subspace_rank 3 --source_weight 2.0 --prototype_weight 0.1) ;;
    full_safe) PROFILE_ARGS=(--projection full --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --prototype_weight 0.1) ;;
    low_lr) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --prototype_weight 0.1) ;;
    very_low_lr) PROFILE_ARGS=(--subspace_rank 3 --lr 0.00005 --prototype_weight 0.1) ;;
    rank1_low) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0001 --prototype_weight 0.0) ;;
    rank1_safe) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0002 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --prototype_weight 0.0) ;;
    b25_low) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --bobsl_weight 0.25 --safety_weight 1.0 --stability_weight 0.5 --prototype_weight 0.1) ;;
    source4_r3) PROFILE_ARGS=(--subspace_rank 3 --source_weight 4.0 --bobsl_weight 0.5 --safety_weight 2.0 --stability_weight 0.5 --prototype_weight 0.1) ;;
    scale05_r3) PROFILE_ARGS=(--subspace_rank 3 --residual_scale 0.5 --lr 0.0002 --prototype_weight 0.1) ;;
    scale1_r3) PROFILE_ARGS=(--subspace_rank 3 --residual_scale 1.0 --lr 0.0002 --prototype_weight 0.1) ;;
    no_weight) PROFILE_ARGS=(--subspace_rank 3 --prototype_weight 0.1 --no-class_weight) ;;
    dual_all_w01) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 0.1) ;;
    dual_all_w025) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 0.25) ;;
    dual_all_w05) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 0.5) ;;
    dual_all_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 1.0) ;;
    dual_all_w2) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 2.0) ;;
    dual_better_w05) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 0.5 --joint_teacher_better_only) ;;
    dual_better_w01) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 0.1 --joint_teacher_better_only) ;;
    dual_better_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 1.0 --joint_teacher_better_only) ;;
    dual_better_w2) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 2.0 --joint_teacher_better_only) ;;
    dual_better_w4) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 4.0 --joint_teacher_better_only) ;;
    dual_correct_w05) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 0.5 --joint_teacher_correct_only) ;;
    dual_correct_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 1.0 --joint_teacher_correct_only) ;;
    dual_correct_w2) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 2.0 --joint_teacher_correct_only) ;;
    dual_better_correct_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 1.0 --joint_teacher_better_only --joint_teacher_correct_only) ;;
    dual_better_correct_w2) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 2.0 --joint_teacher_better_only --joint_teacher_correct_only) ;;
    dual_rank1_better_w1) PROFILE_ARGS=(--subspace_rank 1 --lr 0.0001 --prototype_weight 0.0 --joint_distill_weight 1.0 --joint_teacher_better_only) ;;
    dual_soft_better_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 1.0 --joint_distill_temperature 4.0 --joint_teacher_better_only) ;;
    dual_safe_better_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --joint_distill_weight 1.0 --joint_teacher_better_only) ;;
    dual_full_safe_w05) PROFILE_ARGS=(--projection full --lr 0.0001 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --joint_distill_weight 0.5 --joint_teacher_better_only) ;;
    dual_full_safe_w1) PROFILE_ARGS=(--projection full --lr 0.0001 --residual_scale 1.0 --bobsl_weight 0.5 --safety_weight 1.0 --stability_weight 0.5 --joint_distill_weight 1.0 --joint_teacher_better_only) ;;
    dual_no_weight_w05) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 0.5 --joint_teacher_better_only --no-class_weight) ;;
    dual_no_weight_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 1.0 --joint_teacher_better_only --no-class_weight) ;;
    dual_no_weight_w2) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 2.0 --joint_teacher_better_only --no-class_weight) ;;
    dual_no_weight_w4) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --joint_distill_weight 4.0 --joint_teacher_better_only --no-class_weight) ;;
    dual_p25_w05) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.25 --joint_distill_weight 0.5 --joint_teacher_better_only) ;;
    dual_p25_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.25 --joint_distill_weight 1.0 --joint_teacher_better_only) ;;
    dual_p25_w2) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.25 --joint_distill_weight 2.0 --joint_teacher_better_only) ;;
    dual_p25_w4) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.25 --joint_distill_weight 4.0 --joint_teacher_better_only) ;;
    dual_p50_w05) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.5 --joint_distill_weight 0.5 --joint_teacher_better_only) ;;
    dual_p50_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.5 --joint_distill_weight 1.0 --joint_teacher_better_only) ;;
    dual_p50_w2) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.5 --joint_distill_weight 2.0 --joint_teacher_better_only) ;;
    dual_p50_w4) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.5 --joint_distill_weight 4.0 --joint_teacher_better_only) ;;
    dual_p75_w05) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.75 --joint_distill_weight 0.5 --joint_teacher_better_only) ;;
    dual_p75_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.75 --joint_distill_weight 1.0 --joint_teacher_better_only) ;;
    dual_p75_w2) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.75 --joint_distill_weight 2.0 --joint_teacher_better_only) ;;
    dual_p75_w4) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.75 --joint_distill_weight 4.0 --joint_teacher_better_only) ;;
    dual_p0_source2_w05) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --source_weight 2.0 --bobsl_weight 0.25 --safety_weight 1.0 --stability_weight 0.25 --joint_distill_weight 0.5 --joint_teacher_better_only --no-class_weight) ;;
    dual_p0_source2_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --source_weight 2.0 --bobsl_weight 0.25 --safety_weight 1.0 --stability_weight 0.25 --joint_distill_weight 1.0 --joint_teacher_better_only --no-class_weight) ;;
    dual_p50_source2_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0001 --class_weight_power 0.5 --source_weight 2.0 --bobsl_weight 0.25 --safety_weight 1.0 --stability_weight 0.25 --joint_distill_weight 1.0 --joint_teacher_better_only) ;;
    dual_open_better_w1) PROFILE_ARGS=(--subspace_rank 3 --lr 0.0002 --residual_scale 4.0 --safety_weight 0.25 --stability_weight 0.05 --prototype_weight 0.1 --residual_weight 0.0001 --joint_distill_weight 1.0 --joint_teacher_better_only) ;;
    *) echo "Unknown profile=$1" >&2; return 2 ;;
  esac
}

mkdir -p "$WORK_ROOT/$BASELINE/logs"
for corpus in $CORPORA; do
  set_paths "$corpus"
  for profile in $PROFILES; do
    profile_args "$profile"
    for seed in $SEEDS; do
    out_dir="$WORK_ROOT/$BASELINE/${corpus,,}/b${BOBSL_K}k/${profile}_seed${seed}"
    log_path="$WORK_ROOT/$BASELINE/logs/${corpus,,}_b${BOBSL_K}k_${profile}_seed${seed}.log"
    if [[ "$RESUME_DONE" == 1 && -s "$out_dir/summary.json" ]]; then
      echo "[SAST-SCREEN] skip $out_dir"
      continue
    fi
    checkpoint=$(checkpoint_for_seed "$seed")
    JOINT_ARGS=()
    if [[ "$JOINT_TEACHER" == 1 ]]; then
      joint_checkpoint=$(joint_checkpoint_for_seed "$seed")
      JOINT_ARGS=(--joint_teacher_checkpoint "$joint_checkpoint")
    fi
    echo "[SAST-SCREEN] baseline=$BASELINE corpus=$corpus bobsl=${BOBSL_K}k base_mode=$BASE_MODE profile=$profile seed=$seed gpu=$GPU" | tee "$log_path"
    "$PYTHON_BIN" "$TRAINER" \
      --train_pkl "$TRAIN_PKL" \
      --external_test_pkl "$EXTERNAL_PKL" \
      --base_checkpoint "$checkpoint" \
      --anchor_name "$BASE_MODE" \
      "${JOINT_ARGS[@]}" \
      --out_dir "$out_dir" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --adapter_dim 128 \
      --adapter_dropout 0.2 \
      --lr 0.0005 \
      --residual_scale 2.0 \
      --source_weight 1.0 \
      --bobsl_weight 1.0 \
      --safety_weight 0.5 \
      --stability_weight 0.1 \
      --prototype_weight 0.1 \
      --residual_weight 0.001 \
      --max_grad_norm 5.0 \
      "${PROFILE_ARGS[@]}" 2>&1 | tee -a "$log_path"
    done
  done
done

"$PYTHON_BIN" analysis/summarize_sast_negative_transfer.py \
  --root "$WORK_ROOT" \
  --out_csv "$WORK_ROOT/screen_summary.csv"
