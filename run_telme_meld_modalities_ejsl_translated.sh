#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "$ROOT_DIR"

SOURCE_TXT_ROOT=${SOURCE_TXT_ROOT:-/raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue}
DIAL_LIST=${DIAL_LIST:-/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv}
TRANSLATED_TXT_ROOT=${TRANSLATED_TXT_ROOT:-./IEMOCAP/ejsl_txt_en_openai}
TRANSLATION_CACHE=${TRANSLATION_CACHE:-"$TRANSLATED_TXT_ROOT/translation_cache.jsonl"}
TRANSLATION_BACKEND=${TRANSLATION_BACKEND:-openai}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}

TRANSLATE_LIMIT_FILES=${TRANSLATE_LIMIT_FILES:-0}
TRANSLATE_SLEEP=${TRANSLATE_SLEEP:-0}
TRANSLATE_RETRIES=${TRANSLATE_RETRIES:-3}

if [ "$TRANSLATED_TXT_ROOT" = "$SOURCE_TXT_ROOT" ]; then
  echo "[TELME-TRANSLATE] refusing to overwrite SOURCE_TXT_ROOT: $SOURCE_TXT_ROOT" >&2
  exit 1
fi

echo "[TELME-TRANSLATE] source txt: $SOURCE_TXT_ROOT"
echo "[TELME-TRANSLATE] english txt: $TRANSLATED_TXT_ROOT"
python JSL/translate_ejsl_txt_root.py \
  --input_txt_root "$SOURCE_TXT_ROOT" \
  --output_txt_root "$TRANSLATED_TXT_ROOT" \
  --dial_list "$DIAL_LIST" \
  --cache_jsonl "$TRANSLATION_CACHE" \
  --backend "$TRANSLATION_BACKEND" \
  --openai_model "$OPENAI_MODEL" \
  --sleep "$TRANSLATE_SLEEP" \
  --retries "$TRANSLATE_RETRIES" \
  --limit_files "$TRANSLATE_LIMIT_FILES"

export TXT_ROOT="$TRANSLATED_TXT_ROOT"
export OUT_ROOT=${OUT_ROOT:-./IEMOCAP/outputs_ejsl_telme_meld_modalities_translated}
export REUSE_SHARED=${REUSE_SHARED:-1}
export REUSE_FUSION=${REUSE_FUSION:-1}

echo "[TELME-TRANSLATE] run TELME with translated TXT_ROOT=$TXT_ROOT"
bash run_telme_meld_modalities_ejsl.sh
