pip install sentencepiece sacremoses

cd /raid_zoe/home/lr/maokeyu/sign/emotionbaseline/emotrans

# 1) 生成 dial test（启用日->英）
python prepare_emotrans_dial_test.py \
  --base_dataset meld \
  --new_dataset meld_dial \
  --dial_test_csv /raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame/_list.csv \
  --txt_root /raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue \
  --translate_to_en \
  --translation_model Helsinki-NLP/opus-mt-ja-en

# 2) 跑训练+每epoch dial测试
python run_emotrans_dial_test.py