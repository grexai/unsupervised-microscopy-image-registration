import math

import cv2
import numpy as np
import skimage.transform
from numpy import matrix
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
import csv



'''   
Aligns image pairs using annotated registration points.
Estimate transformation matrix using the points
Warps the moving image using the estmation
Finds the greatest overlapping area between the fix and the warped image
Calculates the greatest overlapping rectangle to remove the rotation effects

Fix images: Screening images
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

# gives the result image pair size
output_image_size = 1024
# file extensions
fix_extension = '.BMP'
moving_extension = "png"
# Generate visualization of cropsJ:\p1_wA
plotgen = False
small_crops_gen = True
use_skiimage = False
load_points = False
load_tm = True
load_warped = True
write_transform_data = True
crops1024_gen = True
focal_plane = 3

# Paths: where annotation folder, fix image folder, moving image folder located
pathreg = "d:/datasets/Image_registration/211109-HK-60x/splitted/train/"
if load_points:
    pathofannotation = pathreg+'annotation/'
if load_tm:
   # pathofannotation = pathreg + 'annot_T/'
    #for scaled
    pathofannotation = pathreg + 'annot_scaled/'

annot = [f for f in listdir(pathofannotation) if isfile(join(pathofannotation, f))]
# foldername of the fix images
pathoffix = pathreg+'lmd63x/'
# foldername of the moving images
if load_warped:
    pathofmoving = pathofannotation + 'warped_bias/'
else:
    pathofmoving = pathreg+'registration/'
# Save directories
savepath = pathreg + 'res_boti/'
alignpathF = pathreg + 'aligned63x/'
alignpathM = pathreg + 'alignedbias/'


# Create save directories
if not path.exists(alignpathM):
    mkdir(alignpathM)
if not path.exists(alignpathF):
    mkdir(alignpathF)
if not path.exists(savepath):
    mkdir(savepath)
if small_crops_gen:
    if not path.exists(alignpathM+'small/'):
        mkdir( alignpathM+'small/')
    if not path.exists(alignpathF+'small/'):
        mkdir(alignpathF+'small/')
if crops1024_gen:
    if not path.exists(alignpathM+'1024crops/'):
        mkdir(alignpathM+'1024crops/')
    if not path.exists(alignpathF+'1024crops/'):
        mkdir(alignpathF+'1024crops/')


def crop_image(p_img1, p_img2, p_crop_size):
    height, width = p_img1.shape[0], p_img1.shape[1]
    dy, dx = p_crop_size
    nx = round(p_img1.shape[0]/dx)
    ny = round(p_img1.shape[1]/dy)
    arr1 = []
    arr2 = []
    for i in range(nx-1):
        for j in range(ny-1):
            arr1.append(p_img1[i*dx:(i+1)*dx, j*dy:(j+1)*dy])
            arr2.append(p_img2[i*dx:(i+1)*dx, j*dy:(j+1)*dy])
    return arr1, arr2


def get_secondminmax(p_list):
    p_list.sort()
    p_list = np.squeeze(p_list)
    sec_min = p_list[1]
    sec_max = p_list[-2]
    return sec_min, sec_max


def calculate_translation_angle_scale(transformation_matrix: np.ndarray):
    xtransform = transformation_matrix[0, 2]
    ytransform = transformation_matrix[1, 2]
    angle = math.asin(transformation_matrix[0, 1])
    scale_x = transformation_matrix[0, 0] / math.cos(angle)
    scale_y = transformation_matrix[1, 1] / math.cos(angle)
    angled = math.degrees(angle)
    return xtransform, ytransform, angled, scale_x, scale_y


nerror = 0
ndrops = 0
fname_arr = []
xt_array = []
yt_array = []
angle_array = []
sx_array = []

print("number of files to align:", len(annot))
for idx in tqdm(range(len(annot))):
    # Extract annotation coordinates from mat files.
    mat = scipy.io.loadmat(join(pathofannotation, annot[idx]))
    if load_points:
        mP = mat['movingPoints']
        fP = mat['fixedPoints']
    if load_tm:
        Tm = mat['Tm']
        Tm[0, 2] = Tm[2, 0]
        Tm[1, 2] = Tm[2, 1]
        Tm = np.delete(Tm, 2, 0)

    # Find if there are more annotations for one image.
    fname, extension = path.splitext(annot[idx])
    numbers = re.compile(r'\d+')
    res = numbers.findall(annot[idx])
    xt, yt, angle, sx, _ = calculate_translation_angle_scale(Tm)

    fname_arr.append(fname)
    xt_array.append(xt)
    yt_array.append(yt)
    angle_array.append(angle)
    sx_array.append(sx)
    fix_str = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c1_z0_l1_o0_{focal_plane}{fix_extension}'
    # read images

    fix = cv2.imread(join(pathoffix, fix_str))

    if load_warped:
        moving_str = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{1}_z0_l1_o0_ws.{moving_extension}'
        moving = cv2.imread(join(pathofmoving, moving_str))

        if fix is None or moving is None:
            nerror = nerror + 1
            continue

        wp2 = moving
    else:
        moving_str = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{1}_z0_l1_o0.{moving_extension}'
        moving = cv2.imread(join(pathofmoving, moving_str))
        if fix is None or moving is None:
            nerror = nerror + 1
            continue
        if load_points:
            if use_skiimage:
                tform2 = skimage.transform.estimate_transform('similarity', mP, fP)
                Tm = tform2.params
            else:
                Tm, inliers = cv2.estimateAffinePartial2D(mP, fP, cv2.RANSAC)
            if Tm is None:
                print("failed to align: " + fname + ",\nat least 3 - 3 corrdinate required")
                # print("found only",len(fP)+1,"-",len(mP)+1)
                nerror = nerror + 1
                continue
            xt, yt, angle, sx, _ = calculate_translation_angle_scale(Tm)
            if use_skiimage:
                warped_moving = skimage.transform.warp(moving, tform2, output_shape=fix.shape)
                # transform = skimage.transform.AffineTransform(translation=[xt, yt], rotation=-angle, scale=scalex)
                skiomoving = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY)
                skiomoving = np.asarray(skiomoving, dtype=np.float32)
                # warped_moving = skimage.transform.warp(skiomoving, transform)
            else:
                wp2 = cv2.warpAffine(moving, Tm, (fix.shape[1], fix.shape[0]))


    '''
    T_inv = np.append(Tm, np.array([[0, 0, 1]]), axis=0)
    T_inv = np.linalg.inv(T_inv)

    T_inv = np.delete(T_inv, 2, 0)
    wp1 = cv2.warpAffine(fix, T_inv, (fix.shape[0], fix.shape[1]))
    '''
    # overlay = cv2.addWeighted(fix, 0.5, wp2, 0.5, 0.7)

    # overlay2 = cv2.addWeighted(fix, 0.5, warped_moving.astype(np.uint8), 0.5, 0.7)
    from sklearn.metrics import mutual_info_score

    from pyitlib import discrete_random_variable as drv

    # Remove black regions of the image
    # Create an image area sized mask

    gray = cv2.cvtColor(wp2, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    ret, thresh = cv2.threshold(gray, 1, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    gray2 = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

    mi = drv.information_mutual(gray.astype(np.int32), gray2[:1440, :1440].astype(np.int32))
    mi2 = drv.information_mutual(gray2[:1440, :1440].astype(np.int32), gray.astype(np.int32))
    # Calculate the greatest area maybe unnecessary
    max_area = -1
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    # approx the greatest rect inside the intersect of 2 images
    approx = cv2.approxPolyDP(best_cnt, 0.01 * cv2.arcLength(best_cnt, True), True)
    far = approx[np.product(approx, 2).argmax()][0]

    # second min second max y,x
    list1 = np.transpose(approx[:, :, 1])
    ymin, ymax = get_secondminmax(list1) 
    list2 = np.transpose(approx[:, :, 0])
    xmin, xmax = get_secondminmax(list2)
    # Crop the images
    img2 = wp2[ymin:ymax, xmin:xmax].copy()
    img1 = fix[ymin:ymax, xmin:xmax].copy()
    newshape = [0, 0]
    if img2.shape != img1.shape:
        ndrops = ndrops + 1
        if img1.shape[0] < img2.shape[0]:
            newshape[0] = img1.shape[0]
        else:
            newshape[0] = img2.shape[0]
        if img1.shape[1] < img2.shape[1]:
            newshape[1] = img1.shape[1]
        else:
            newshape[1] = img2.shape[1]
        img2 = img2[ymin:newshape[0], xmin:newshape[1]].copy()
        img1 = img1[ymin:newshape[0], xmin:newshape[1]].copy()

    # small crops

    if small_crops_gen:
        crop_size = 256
        if (img2.shape[0] or img1.shape[1]) < crop_size:
            continue
        img1_list, img2_list = crop_image(img1, img2, (crop_size, crop_size))

        # crop to smaller size for pix2pix

        for idx_s in range(len(img1_list)):
            cv2.imwrite(alignpathF+'small/' + fname+'_'+str(idx_s) + '.png', img1_list[idx_s])
            cv2.imwrite(alignpathM+'small/' + fname+'_'+str(idx_s) + '.png', img2_list[idx_s])



    '''
    # resize image
    
    #moving_sq = moving[:1440,:1440,:]
    moving_sq = moving[:width,:height,:]
    moving_resized = cv2.resize(moving_sq, dim, interpolation = cv2.INTER_AREA)

    #squared_images
    #fix 
    #moving_sq = moving[:width,:width]
    dimsmall = (width, width)
    fix_resized = cv2.resize(fix, dimsmall, interpolation = cv2.INTER_AREA)
    #fix_resized = fix[:width,:height,:]
    #moving_resized = cv2.resize(moving_resized, dimsmall, interpolation = cv2.INTER_AREA)
    '''

    # Crop visualization compared to original images

    if plotgen:
        plt.figure()
        plt.subplot(2, 3, 1), plt.imshow(moving, 'gray')
        plt.subplot(2, 3, 2), plt.imshow(fix, 'gray')
        plt.subplot(2, 3, 3), plt.imshow(thresh, 'gray')
        plt.subplot(2, 3, 4), plt.imshow(overlay, "gray")
        plt.subplot(2, 3, 5), plt.imshow(wp2, 'hot')
        plt.subplot(2, 3, 6), plt.imshow(img1, 'gray')
        plt.savefig(savepath + fname + '.png')
    # Save aligned images // BF, SHG --> to fit CoMIR's image handling.
    if crops1024_gen:
        f_i_1024 = cv2.resize(img1[:1440, :1440, :], (output_image_size, output_image_size))
        m_i_1024 = cv2.resize(img2[:1440, :1440, :], (output_image_size, output_image_size))
        cv2.imwrite(alignpathF + '1024crops/'+fname + f'_{focal_plane}.png', f_i_1024)
        cv2.imwrite(alignpathM + '1024crops/' + fname + f'_{focal_plane}.png', m_i_1024)

    #cv2.imwrite(savepath+fname+'overlay.png', overlay)
   # cv2.imwrite(savepath + fname + 'skio.png', warped_moving)

if write_transform_data:
    with open('hela63x_transformation_scaled.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fname_arr)
        write.writerow(xt_array)
        write.writerow(yt_array)
        write.writerow(angle_array)
        write.writerow(sx_array)

print(f'droped images because of different shape: {ndrops}')
print("failed alignment(s): ", nerror)