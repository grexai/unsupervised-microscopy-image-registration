import cv2
import numpy as np
from numpy.core.fromnumeric import resize, shape
from numpy.core.records import array
from numpy.lib.npyio import save
from os import listdir, mkdir, path
from os.path import isfile, join
import matplotlib.pyplot as plt
from numpy.lib.ufunclike import _fix_out_named_y
import scipy.io
import re
from tqdm import tqdm


'''   
Aligns image pairs using annotated registration points.
Estimate transformation matrix using the points
Warps the moving image using the estmation
Finds the greatest overlapping area between the fix and the warped image
Calculates the greatest overlapping rectangle to remove the rotation effects

Fix images: BIAS images 
Moving images: Leica LMD 40x

LEICA LMD data:
magnification 40x
Pixelwidth:
x: 0.194531
y: 0.194531
BIAS
magnification 40x
Pixelwidth:
x: 0.228
y: 0.228

0.853
'''


def crop_image(img1, img2, crop_size):
    height, width = img1.shape[0], img1.shape[1]
    dy, dx = crop_size
    nx = round(img1.shape[0]/dx)
    ny = round(img1.shape[1]/dy)
    arr1 = []
    arr2 = []
    for i in range(nx-1):
        for j in range(ny-1):
            arr1.append(img1[i*dx:(i+1)*dx,j*dy:(j+1)*dy])
            arr2.append(img2[i*dx:(i+1)*dx,j*dy:(j+1)*dy])
    return arr1, arr2


# file extensions
fix_extension = '.BMP'
moving_extension = ".png"

plotgen = False

small_crops_gen = True
# Paths: where annotation folder, fix image folder, moving image folder located
pathreg = "d:/datasets/Image_registration/211109-HK-60x/splitted/train/"


# foldername of the fix images
pathoffix = pathreg + 'lmd63x/'
# foldername of the moving images
pathofmoving = pathreg + 'registration/'

#check for annotated images and create
pathofannotation = pathreg+'annotation/'


files = [f for f in listdir(pathofannotation) if isfile(join(pathofannotation, f))]
# Save directories

alignpathF = pathreg + 'trainA/'
alignpathM = pathreg + 'trainB/'

# Create save directories
if not path.exists(alignpathM):
    mkdir(alignpathM)
if not path.exists(alignpathF):
    mkdir(alignpathF)



print("number of files to align:", len(files))
nerror = 0
count = 0
#

#focal_plane=3
for focal_plane in range(1, 4):
    for idx in tqdm(range(len(files))):
        fname, extension = path.splitext(files[idx])
        numbers = re.compile(r'\d+')
        res = numbers.findall(fname)
        moving_str = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{1}_z0_l1_o0'
        fix_str = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c1_z0_l1_o0_{focal_plane}'
        # read images
        count = count + 1
        # LEICA
        fix = cv2.imread(join(pathoffix, f'{fix_str}{fix_extension}'))
        # BIAS
        moving = cv2.imread(join(pathofmoving, f'{moving_str}{moving_extension}'))
        # create squred image
        fixcut = fix[:1440, :1440, :]
        fixcut = cv2.resize(fixcut, (1024, 1024))
        # create squred image
        movingcut = moving[:850, :850, :]
        #
        # moving_rs = cv2.resize(movingcut, (1638, 1638))

        moving_rs = cv2.resize(movingcut, (1024, 1024))

        cv2.imwrite(alignpathF + fix_str + '.png', fixcut)
        cv2.imwrite(alignpathM + fix_str + '.png', moving_rs)


print(f"n imges {count}")
print("failed alignment(s): ", nerror)