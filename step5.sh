python analysis/build_case_analysis.py \
  --frame_root /raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame \
  --txt_root /raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue \
  --telme_pred_csv ./IEMOCAP/outputs_ejsl/telme_ejsl_test_predictions.csv \
  --emotrans_pred_csv ./emotrans/saved/meld_dial_metrics/test_epoch001_predictions.csv \
  --eanwh_pred_csv ./eanwh_res.csv \
  --out_dir ./analysis/outputs_cases \
  --expected_samples 1920