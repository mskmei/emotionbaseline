#!/bin/sh
set -eu

# Use this script from any cwd; resolve script directory in POSIX sh.
SCRIPT_DIR=`cd "`dirname "$0"`" && pwd`
cd "${SCRIPT_DIR}"

# Configurable paths (can be overridden by env vars).
: "${BASE_DATASET:=meld}"
: "${NEW_DATASET:=meld_dial}"
: "${DIAL_TEST_CSV:=/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame/_list.csv}"
: "${TXT_ROOT:=/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}"
: "${TRANSLATE_TO_EN:=1}"
: "${TRANSLATION_MODEL:=Helsinki-NLP/opus-mt-ja-en}"

echo "[Run] emotrans dir: ${SCRIPT_DIR}"
echo "[Run] preparing dataset: ${NEW_DATASET}"

if [ "${TRANSLATE_TO_EN}" = "1" ]; then
  python prepare_emotrans_dial_test.py \
    --base_dataset "${BASE_DATASET}" \
    --new_dataset "${NEW_DATASET}" \
    --dial_test_csv "${DIAL_TEST_CSV}" \
    --txt_root "${TXT_ROOT}" \
    --translate_to_en \
    --translation_model "${TRANSLATION_MODEL}"
else
  python prepare_emotrans_dial_test.py \
    --base_dataset "${BASE_DATASET}" \
    --new_dataset "${NEW_DATASET}" \
    --dial_test_csv "${DIAL_TEST_CSV}" \
    --txt_root "${TXT_ROOT}"
fi

echo "[Run] training + per-epoch dial test"
python run_emotrans_dial_test.py

REPORT_DIR="${SCRIPT_DIR}/saved/meld_dial_metrics"
echo "[Run] checking reports in: ${REPORT_DIR}"

if [ ! -d "${REPORT_DIR}" ]; then
  echo "[Error] report dir not found: ${REPORT_DIR}" >&2
  exit 1
fi

report_count=$(find "${REPORT_DIR}" -maxdepth 1 -type f -name 'test_epoch*_classification_report.txt' | wc -l)
cm_count=$(find "${REPORT_DIR}" -maxdepth 1 -type f -name 'test_epoch*_confusion_matrix.txt' | wc -l)

if [ "${report_count}" -eq 0 ] || [ "${cm_count}" -eq 0 ]; then
  echo "[Error] dial test outputs missing. report_count=${report_count}, cm_count=${cm_count}" >&2
  echo "[Error] inspect process logs for '[test] skip report:' reasons" >&2
  exit 1
fi

echo "[OK] saved reports: ${report_count}, confusion matrices: ${cm_count}"
ls -1 "${REPORT_DIR}" | sed -n '1,30p'