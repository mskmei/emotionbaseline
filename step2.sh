python IEMOCAP/inference_ejsl_frame.py \
  --frame_root /raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame \
  --txt_root /raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue \
  --save_model_root /raid_zoe/home/lr/maokeyu/sign/open/IEMOCAP/save_model \
  --batch_size 4 \
  --num_workers 4 \
  --save_dir ./IEMOCAP/outputs_ejsl \
  --report_prefix telme_ejsl_test \
  --save_predictions