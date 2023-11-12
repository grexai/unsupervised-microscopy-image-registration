
#source /CUDA/cuda-9.0/activate.sh
# source /storage01/grexai/dev/envs/pix2pix/bin/activate

#cd /storage01/grexai/dev/imreg/contrastive-unpaired-translation/
# jupyter-notebook --no-browser --port=8087

python3 train.py config_HeLa_ba.json 'HeLa_ba' --epoch=1000

python3 train.py config_Hella_BA.json 'Hella_ba' --epoch=1000

python3 train.py config_IHC_BA.json 'IHC_ba' --epoch=1000

python3 train.py config_dstrom_ba.json 'dstorm_ba' --epoch=1000
