# JSL Non-Oracle Text Pipeline

This folder builds non-oracle Japanese text for eJSL by fine-tuning a
keypoint-to-text sign translation model.

Verified J-Shuwa structure:

- Hugging Face dataset: `mouwjone/J-Shuwa`
- Split: currently a single `train` split
- Fields: `vid`, `yid`, `start`, `end`, `source`
- The HF release is metadata only. It does not include videos, subtitle text,
  transcripts, OCR text, or translations.

Because of that, training requires a prepared local manifest that contains real
local video paths and Japanese target text. The scripts below fail fast if those
columns are missing.

## Expected Flow

The default server paths are:

- Work/cache root: `/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle`
- J-Shuwa metadata: `/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/manifests/jshuwa_metadata_train.csv`
- Downloaded J-Shuwa videos: `/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/jshuwa_youtube_videos`
- Downloaded J-Shuwa subtitles: `/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/jshuwa_youtube_subtitles`
- Extracted J-Shuwa keypoints: `/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/keypoints/jshuwa_cc`
- Saved JSL translator: `/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/models/qwen3_jsl_lora_cc`
- Generated eJSL non-oracle txt root: `/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/ejsl_nonoracle_txt`

One-line CC-subset pipeline:

```bash
cd /raid_zoe/home/lr/maokeyu/sign/emotionbaseline && read -rsp "HF_TOKEN: " HF_TOKEN && echo && export HF_TOKEN && python -m pip install -r JSL/requirements-jsl.txt && bash JSL/run_all_cc_nonoracle_pipeline.sh
```

This downloads the gated J-Shuwa metadata, downloads Japanese YouTube subtitles
and videos for the `cc` subset, extracts MediaPipe keypoints, trains the model,
and generates the 1920 eJSL non-oracle txt files.

1. Prepare a J-Shuwa training manifest after setting these variables to real
   server paths.

```bash
python JSL/prepare_jshuwa_manifest.py \
  --video_dir "$JSHUWA_VIDEO_DIR" \
  --subtitle_text "$JSHUWA_SUBTITLE_TEXT" \
  --out_csv "$JSHUWA_MANIFEST"
```

2. Extract MediaPipe Holistic keypoints:

```bash
python JSL/extract_mediapipe_keypoints.py \
  --manifest_csv "$JSHUWA_MANIFEST" \
  --output_dir "$JSHUWA_KEYPOINT_DIR" \
  --out_manifest_csv "$JSHUWA_KEYPOINT_MANIFEST"
```

3. Train MLP + Qwen3 LoRA:

```bash
python JSL/train_jsl_translation.py \
  --manifest_csv "$JSHUWA_KEYPOINT_MANIFEST" \
  --output_dir "$JSL_MODEL_DIR"
```

4. Generate non-oracle eJSL txt files:

```bash
python JSL/generate_ejsl_non_oracle_txt.py \
  --model_dir "$JSL_MODEL_DIR" \
  --dial_list /home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv \
  --video_root /raid_zoe/home/lr/wangyi/sign/eJSL_dial/video \
  --structure_txt_root /raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue \
  --output_txt_root "$OUTPUT_TXT_ROOT"
```

`structure_txt_root` is read only for speaker/dialogue line structure. The
oracle utterance text column is ignored.

## Notes

- Model route: MediaPipe Holistic keypoints -> two-layer MLP visual token
  projector -> Qwen3 causal LM with LoRA.
- Default base model is `Qwen/Qwen3-1.7B`, which is the current HF Qwen3 1.7B
  chat/instruction-style release. Override `--base_model` if your server has a
  different exact Qwen3-1.7B-Instruct checkpoint name.
- The generated eJSL txt tree mirrors the oracle txt layout expected by the
  existing TELME/MMGCN eJSL code: `SDxx/txt/SDxx-Dialogue-YY.txt`.
