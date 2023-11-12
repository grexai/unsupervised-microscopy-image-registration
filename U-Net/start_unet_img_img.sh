
#source /CUDA/cuda-9.0/activate.sh
# source /storage01/grexai/dev/envs/pix2pix/bin/activate

#cd /storage01/grexai/dev/imreg/contrastive-unpaired-translation/
# jupyter-notebook --no-browser --port=8087

# python3 train.py config_HeLa.json 'HeLa' --epoch=1000
# python3 train.py config_Hella.json 'Hella' --epoch=1000
# python3 train.py config_IHC.json 'IHC' --epoch=1000
# python3 train.py config_dstrom.json 'dstorm' --epoch=1000


CUDA_VISIBLE_DEVICES=0 python3 train.py config_HeLa.json 'HeLa_max_image' --epoch=1000 --max_images=25

CUDA_VISIBLE_DEVICES=0 python3 train.py config_HeLa.json 'HeLa_max_image' --epoch=1000 --max_images=50

CUDA_VISIBLE_DEVICES=0 python3 train.py config_HeLa.json 'HeLa_max_image' --epoch=1000 --max_images=75