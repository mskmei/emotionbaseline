#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
PYTHON_BIN=${PYTHON_BIN:-python}
WORKER_ID=${WORKER_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-1}
BASELINES=${BASELINES:-"MMGCN MAGTKD CMERC"}
CORPORA=${CORPORA:-"MELD IEMOCAP"}
CONFIGS=${CONFIGS:-"source_only joint_b5k"}
SEEDS=${SEEDS:-"35 36 37"}
MODES=${MODES:-"normal isolate"}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
OUT_ROOT=${OUT_ROOT:-/raid_zoe/home/lr/maokeyu/sign/bobsl_b5k_mechanism_analysis/context_ablation}
LOG_ROOT=${LOG_ROOT:-$OUT_ROOT/logs}
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

resolve_paths() {
  local baseline="$1"
  local corpus="$2"
  local config="$3"
  local seed="$4"
  local result_root

  case "$baseline/$corpus" in
    MMGCN/MELD)
      result_root=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_meld_ejsl/video_joint_bobsl_meld
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/meld_anjs4_unified.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl
      if [[ "$config" == source_only ]]; then
        TRAIN_PKL="$SOURCE_PKL"
        trial=meld_only
      else
        TRAIN_PKL="$result_root/joint_pkls/meld_bobsl5000_trainonly_sample123.pkl"
        trial=joint_b5k
      fi
      CHECKPOINT="$result_root/${trial}_seed${seed}/video/model_best.pt"
      ;;
    MMGCN/IEMOCAP)
      result_root=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_bobsl_iemocap_ejsl/features/iemocap_mmgcn_video_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/mmgcn_unified_meld_ejsl/ejsl_anjs4_unified.pkl
      if [[ "$config" == source_only ]]; then
        TRAIN_PKL="$SOURCE_PKL"
        trial=iemocap_only
      else
        TRAIN_PKL="$result_root/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl"
        trial=joint_b5k
      fi
      CHECKPOINT="$result_root/${trial}_seed${seed}/video/model_best.pt"
      ;;
    MAGTKD/MELD)
      result_root=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/video_joint_bobsl_meld
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/meld_magtkd_video_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl
      if [[ "$config" == source_only ]]; then
        TRAIN_PKL="$SOURCE_PKL"
        trial=meld_only
      else
        TRAIN_PKL="$result_root/joint_pkls/joint_b5k.pkl"
        trial=joint_b5k
      fi
      CHECKPOINT="$result_root/${trial}_seed${seed}/model_best.pt"
      ;;
    MAGTKD/IEMOCAP)
      result_root=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_iemocap_ejsl/features/iemocap_magtkd_video_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/magtkd_bobsl_meld_ejsl/features/ejsl_magtkd_video_anjs4_translated.pkl
      if [[ "$config" == source_only ]]; then
        TRAIN_PKL="$SOURCE_PKL"
        trial=iemocap_only
      else
        TRAIN_PKL="$result_root/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl"
        trial=joint_b5k
      fi
      CHECKPOINT="$result_root/${trial}_seed${seed}/model_best.pt"
      ;;
    CMERC/MELD)
      result_root=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/visual_joint_bobsl_meld
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/meld_cmerc_visual_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      if [[ "$config" == source_only ]]; then
        TRAIN_PKL="$SOURCE_PKL"
        trial=meld_only
      else
        TRAIN_PKL="$result_root/joint_pkls/joint_b5k.pkl"
        trial=joint_b5k
      fi
      CHECKPOINT="$result_root/${trial}_seed${seed}/model_best.pt"
      ;;
    CMERC/IEMOCAP)
      result_root=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/video_joint_bobsl_iemocap
      SOURCE_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_iemocap_ejsl/features/iemocap_cmerc_visual_anjs4.pkl
      EXTERNAL_PKL=/raid_zoe/home/lr/maokeyu/sign/cmerc_bobsl_meld_ejsl/features/ejsl_cmerc_visual_anjs4_translated.pkl
      if [[ "$config" == source_only ]]; then
        TRAIN_PKL="$SOURCE_PKL"
        trial=iemocap_only
      else
        TRAIN_PKL="$result_root/joint_pkls/iemocap_bobsl5000_trainonly_sample123.pkl"
        trial=joint_b5k
      fi
      CHECKPOINT="$result_root/${trial}_seed${seed}/model_best.pt"
      ;;
    *)
      echo "[context-ablation] unsupported baseline/corpus: $baseline/$corpus" >&2
      return 2
      ;;
  esac
}

task_index=0
failures=0
for baseline in $BASELINES; do
  for corpus in $CORPORA; do
    for config in $CONFIGS; do
      for seed in $SEEDS; do
        for mode in $MODES; do
          current=$task_index
          task_index=$((task_index + 1))
          if (( current % NUM_WORKERS != WORKER_ID )); then
            continue
          fi
          resolve_paths "$baseline" "$corpus" "$config" "$seed"
          out_dir="$OUT_ROOT/$baseline/$corpus/$config/seed${seed}/$mode"
          done_path="$out_dir/summary.json"
          log_path="$LOG_ROOT/${baseline}_${corpus}_${config}_seed${seed}_${mode}.log"
          if [[ -s "$done_path" ]]; then
            echo "[context-ablation] skip existing $done_path"
            continue
          fi
          for required in "$CHECKPOINT" "$TRAIN_PKL" "$SOURCE_PKL" "$EXTERNAL_PKL"; do
            if [[ ! -s "$required" ]]; then
              echo "[context-ablation] missing required file: $required" >&2
              exit 1
            fi
          done
          mkdir -p "$out_dir"
          echo "[context-ablation] worker=$WORKER_ID task=$current $baseline/$corpus/$config/seed$seed/$mode" | tee "$log_path"
          set +e
          "$PYTHON_BIN" analysis/evaluate_context_ablation.py \
            --baseline "$baseline" \
            --corpus "$corpus" \
            --config "$config" \
            --seed "$seed" \
            --mode "$mode" \
            --checkpoint "$CHECKPOINT" \
            --train_pkl "$TRAIN_PKL" \
            --source_pkl "$SOURCE_PKL" \
            --external_pkl "$EXTERNAL_PKL" \
            --out_dir "$out_dir" 2>&1 | tee -a "$log_path"
          status=${PIPESTATUS[0]}
          set -e
          if [[ "$status" -ne 0 ]]; then
            failures=$((failures + 1))
            echo "[context-ablation] failed status=$status log=$log_path" | tee -a "$log_path"
            if [[ "$CONTINUE_ON_ERROR" != 1 ]]; then
              exit "$status"
            fi
          fi
        done
      done
    done
  done
done

echo "[context-ablation] worker=$WORKER_ID done tasks=$task_index failures=$failures"
if [[ "$NUM_WORKERS" == 1 ]]; then
  "$PYTHON_BIN" analysis/summarize_context_ablation.py --root "$OUT_ROOT"
fi
