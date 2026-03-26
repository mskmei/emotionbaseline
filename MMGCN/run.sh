cd /raid_zoe/home/lr/maokeyu/sign/emotionbaseline/MMGCN

python build_dial_test_pkl.py \
  --dial_list /home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/ejsldial_filenames.csv \
  --txt_root /raid_elmo/home/lr/wangyi/PTR/STUDIES-Japanese/Short_dialogue \
  --frame_root /raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame \
  --dataset IEMOCAP \
  --out_pkl /raid_zoe/home/lr/wangyi/sign/eJSL_dial/mmgcn/dial_test_iemocap.pkl


CUDA_VISIBLE_DEVICES=1 python train.py \
  --base-model LSTM \
  --graph-model \
  --nodal-attention \
  --dropout 0.4 \
  --lr 0.0003 \
  --batch-size 16 \
  --l2 0.00003 \
  --graph_type MMGCN \
  --epochs 20 \
  --graph_construct direct \
  --multi_modal \
  --mm_fusion_mthd concat_subsequently \
  --modals avl \
  --Dataset IEMOCAP \
  --Deep_GCN_nlayers 4 \
  --class-weight \
  --use_speaker \
  --dial_test_path /raid_zoe/home/lr/wangyi/sign/eJSL_dial/mmgcn/dial_test_iemocap.pkl \
  --dial_eval_every 1 \
  --dial_save_dir ./saved/dial_eval