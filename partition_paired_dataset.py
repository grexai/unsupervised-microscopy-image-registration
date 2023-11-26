import glob
import json
from tqdm import tqdm
import random
import shutil
import os

# select you dataset
# path = './datasets/Image_registration/Renal_cancer_tissue/'
# path = './datasets/Image_registration/Skin_tissue/'
path = './datasets/Image_registration/HeLa/'


fiximag_path = f'{path}lmd63x/'
moving_path = f'{path}registration/'
annot_path = f'{path}annotations/'

this_path = 'splitted/'
training_path = f"{path}{this_path}/train/lmd63x/"
training_path2 = f"{path}{this_path}/train/registration/"
training_path3 = f"{path}{this_path}/train/annotation/"

test_path = f"{path}{this_path}/test/lmd63x/"
test_path2 = f"{path}{this_path}/test/registration/"
test_path3 = f"{path}{this_path}/test/annotation/"


if not os.path.exists(training_path):
    os.makedirs(training_path)

if not os.path.exists(training_path2):
    os.makedirs(training_path2)

if not os.path.exists(training_path3):
    os.makedirs(training_path3)

if not os.path.exists(test_path):
    os.makedirs(test_path)

if not os.path.exists(test_path2):
    os.makedirs(test_path2)

if not os.path.exists(test_path3):
    os.makedirs(test_path3)


files = glob.glob(f"{annot_path}*.mat")
print(files)
random.seed(0)
random.shuffle(files)
n_files = len(files)
n_training = 0.8 * n_files
n_test = 0.2 * n_files

for idx in tqdm(range(n_files)):
    fname, ext = os.path.splitext(files[idx])
    path, fname = os.path.split(fname)
    if idx < n_training:
        for focus in range(1, 4):
            shutil.copy(f"{fiximag_path}{fname.replace('z1','z0')}_{focus}.BMP", f"{training_path}{fname.replace('z1','z0')}_{focus}.BMP")
        shutil.copy(f"{moving_path}{fname.replace('c0','c1').replace('z1','z0')}.png", f"{training_path2}{fname.replace('c0','c1').replace('z1','z0')}.png")
        shutil.copy(f"{annot_path}{fname}.mat", f"{training_path3}{fname}.mat")
    else:
        for focus in range(1, 4):
            shutil.copy(f"{fiximag_path}{fname.replace('z1','z0')}_{focus}.BMP", f"{test_path}{fname.replace('z1','z0')}_{focus}.BMP")
        shutil.copy(f"{moving_path}{fname.replace('c0','c1').replace('z1','z0')}.png", f"{test_path2}{fname.replace('c0','c1').replace('z1','z0')}.png")
        shutil.copy(f"{annot_path}{fname}.mat", f"{test_path3}{fname}.mat")





