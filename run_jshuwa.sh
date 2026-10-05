CUDA_VISIBLE_DEVICES=0 python JSL/train_jsl_translation.py \
  --manifest_csv /raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/manifests/jshuwa_cc_train_keypoints.csv \
  --output_dir /raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/models/qwen3_jsl_lora_cc \
  --base_model Qwen/Qwen3-1.7B \
  --epochs 5 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lr 1e-4 \
  --num_visual_tokens 64 \
  --max_target_tokens 96 \
  --bf16

CUDA_VISIBLE_DEVICES=0 python JSL/generate_ejsl_non_oracle_txt.py \
  --model_dir /raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/models/qwen3_jsl_lora_cc \
  --dial_list /home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv \
  --video_root /raid_zoe/home/lr/wangyi/sign/eJSL_dial/video \
  --frame_root /raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame \
  --structure_txt_root /raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue \
  --output_txt_root /raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/ejsl_nonoracle_txt \
  --keypoint_cache_dir /raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/keypoints/ejsl_1920 \
  --predictions_jsonl /raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle/ejsl_nonoracle_predictions.jsonl \
  --batch_size 2 \
  --num_visual_tokens 64 \
  --sample_fps 10 \
  --max_frames 0 \
  --model_complexity 1 \
  --max_new_tokens 64 \
  --temperature 0.0 \
  --resume