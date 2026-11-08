#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

CORPUS=${CORPUS:?Set CORPUS=MELD or IEMOCAP}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-/home/lr/maokeyu/.conda/envs/telme39/bin/python}
SEEDS=${SEEDS:-"35 36 37 41 42"}
PROFILES=${PROFILES:-"fullspace no_safety context_joint rank1"}
RESUME_DONE=${RESUME_DONE:-1}
WORK_ROOT=${WORK_ROOT:-/raid_zoe/home/lr/maokeyu/sign/cesa_cmerc/ablations}

case "$CORPUS" in
  MELD)
    EPOCHS=${EPOCHS:-7}
    TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld/joint_pkls/joint_b5k.pkl
    EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
    BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld
    BASE_PREFIX=meld_only
    ;;
  IEMOCAP)
    EPOCHS=${EPOCHS:-25}
    TRAIN_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl
    EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
    BASE_ROOT=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
    BASE_PREFIX=iemocap_only
    ;;
  *) echo "Unknown CORPUS=$CORPUS" >&2; exit 2 ;;
esac

profile_args() {
  case "$1" in
    fullspace) PROFILE_ARGS=(--projection full) ;;
    no_safety) PROFILE_ARGS=(--safety_weight 0.0) ;;
    context_joint) PROFILE_ARGS=(--bobsl_weight 0.0 --bobsl_context_weight 1.0) ;;
    rank1) PROFILE_ARGS=(--subspace_rank 1) ;;
    *) echo "Unknown profile=$1" >&2; return 2 ;;
  esac
}

mkdir -p "$WORK_ROOT/logs"
for profile in $PROFILES; do
  profile_args "$profile"
  for seed in $SEEDS; do
    out_dir="$WORK_ROOT/${CORPUS,,}/${profile}_fixed${EPOCHS}_seed${seed}"
    log_path="$WORK_ROOT/logs/${CORPUS,,}_${profile}_fixed${EPOCHS}_seed${seed}.log"
    if [[ "$RESUME_DONE" == 1 && -s "$out_dir/external_test_final_summary.json" ]]; then
      echo "[CESA-ABLATE] skip $out_dir"
      continue
    fi
    base_checkpoint="$BASE_ROOT/${BASE_PREFIX}_seed${seed}/model_best.pt"
    echo "[CESA-ABLATE] corpus=$CORPUS profile=$profile seed=$seed epochs=$EPOCHS gpu=$GPU" | tee "$log_path"
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
      --lr 0.0005 \
      --residual_scale 2.0 \
      --source_weight 1.0 \
      --bobsl_weight 1.0 \
      --local_source_weight 0.1 \
      --safety_weight 0.5 \
      --residual_weight 0.001 \
      --max_grad_norm 5.0 \
      "${PROFILE_ARGS[@]}" 2>&1 | tee -a "$log_path"
  done
done

"$PYTHON_BIN" analysis/analyze_cesa_study.py \
  --main_root /raid_zoe/home/lr/maokeyu/sign/cesa_cmerc/fiveseed \
  --ablation_root "$WORK_ROOT" \
  --out_dir /raid_zoe/home/lr/maokeyu/sign/cesa_cmerc/study_summary
