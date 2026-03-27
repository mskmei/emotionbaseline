cd /raid_zoe/home/lr/maokeyu/sign/emotionbaseline/emotrans
sed -i 's/\r$//' run.sh
EMOTRANS_EPOCHS=1 sh run.sh