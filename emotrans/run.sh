cd /raid_zoe/home/lr/maokeyu/sign/emotionbaseline/emotrans
python prepare_emotrans_dial_test.py \
  --base_dataset meld \
  --new_dataset meld_dial \
  --dial_test_csv /home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv \
  --txt_root /raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue

  rm -f ./Cache/meld_dial/emotrans.large

  CUDA_VISIBLE_DEVICES=1 python run_emotrans_dial_test.py