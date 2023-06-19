import json
import math
import pandas as pb
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
import timeit
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate
from datetime import timedelta
from skimage.transform import AffineTransform, warp
from scipy.fft import fft2, fftshift
from skimage.filters import window, difference_of_gaussians
import glob
import csv
import scipy.io
from scipy import signal

from statistics import mean
write_transform_data = True
write_error = False
write_annotated = True
from skimage.metrics import structural_similarity as ssim
create_file_for_gt_corners = True
'''
transform the 4 cornerpoints of the image with the result transformation matrix,
transform the 4 cornerpoints of the image with the annotated transformation matrix,
compare the results with eucledian distance and avarage them

params: path of regs
size of the result transformation method

size of image which influences the results image
METHDS for style: pix2pix cyclegan, CUT
methods for reg: sift and superpoint

calculates mean squared error and structural similiraty beetween originl and transformed image

'''
size = 256
regmethod = 'SP'
style = 'pix2pix'
experiment_name = 'hela6063x_fl_spv6'
experiment_name = 'HeLa_BA'
path_results = f'd:/datasets/Image_registration/211109-HK-60x/results/registration/{experiment_name}/'

# path_results = f'e:/deeplearning/hela/RES_RES/CUT/CUT_hela63_BF2FL_all_small/SIFT_AND_SP{size}/'
# path_results = f'e:/deeplearning/hela/RES_RES/Cyclegan/cycle_hela63_BF2FL_all_small/SIFT_AND_SP{size}/'
path_annot = "d:/datasets/Image_registration/211109-HK-60x/splitted/test/annot_scaled/"

path_bias = 'd:/datasets/Image_registration/211109-HK-60x/splitted/test/notaligned_samescale_bias/'
path_bias_mask = 'd:/datasets/Image_registration/211109-HK-60x/splitted/test/annot_scaled/masks/'
# test with original masks prepare them here
path_bias_mask = 'd:/datasets/Image_registration/211109-HK-60x/registration/mask/'

path_leica = 'd:/datasets/Image_registration/211109-HK-60x/splitted/test/notaligned_samescale_LMD/'
reg_methods = ['corr', 'sp', 'sift']
#style_methods = ['Comirr2r1', 'cyclegan_aligned', 'pix2pix_aligned', 'pix2pix_unaligned']
# style_methods = ['comir', 'cyclegan_aligned', 'pix2pix_aligned', 'pix2pix_unaligned', 'cut_hela63_aligned_reize', 'cut_hela63_unaligned_reize', 'no_style', 'unetl1_256']

style_methods = ['nostyle', 'comir', 'unet_aligned', 'cyclegan_aligned', 'cut_aligned', 'pix2pix_aligned',
                 'cyclegan_unaligned', \
                 'cut_unaligned', 'pix2pix_unaligned' ]
style_methods = ['pix2pix_aligned', 'cyclegan_aligned', 'cut_aligned']
csvname = f'hela63x_scaled_{style}_and_{regmethod}_{size}_avg_error.csv'

csvraw_data = f'hela63x_scaled_{style}_and_{regmethod}_{size}_rawdata.csv'
sizecorr = 1024 / size


# np settings
np.set_printoptions(suppress=True)


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    imageA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    cor = signal.correlate2d(imageA, imageB)
    return m, s, cor


def format_annotation_matrix(transformation_matrix: np.array):
    transformation_matrix = transformation_matrix.T
    # transformation_matrix = np.delete(transformation_matrix, 2, 0)
    return transformation_matrix


def get_transformed_coordinates_coorelation(p_transformation):
    """
    inputs expected transformation = [shtifty, shiftx,angle,scale]
    returns a transformed 4 element corner point set
    """
    cp2 = np.array([[-512, -512, 1], [-512, 512, 1], [512, 512, 1], [512, -512, 1]]).T
    radangle = math.radians(p_transformation[2])
    rot_matrix = np.array(
        [[p_transformation[3] * math.cos(radangle), p_transformation[3] * (-math.sin(radangle)), 0], [p_transformation[3] * math.sin(radangle), p_transformation[3] * math.cos(radangle), 0], [0, 0, 1]])
    rotated = rot_matrix @ cp2
    translation_matrix = np.array([[1, 0, p_transformation[1] + 512], [0, 1, p_transformation[0] + 512], [0, 0, 1]])
    final = translation_matrix @ rotated
    return final


def get_avarage_corner_distance(x_s, y_s, transformation_matrix, annot_matrix, corr=False):
    corner_points = [[0, 0, 1], [0, y_s, 1], [x_s, y_s, 1], [x_s, 0, 1]]
    corner_points = np.array(corner_points).T
    result_annot = annot_matrix @ corner_points

    if corr:
        result_coor = get_transformed_coordinates_coorelation(transformation_matrix)
    else:
        result_coor = transformation_matrix @ corner_points
    fig = plt.figure()
    # print(result_annot)
    # print(result_coor)
    ax1 = fig.add_subplot(111)
    ax1.scatter(result_coor[0, :], result_coor[1, :], s=20, c='g', marker="s", label='results')
    ax1.scatter(result_annot[0, :], result_annot[1, :], s=20, c='r', marker="s", label='annotation')
    ax1.legend(['results', 'annotation'])
    ax1.axis('square')
    return fig


def warp_correlation(img1, img2, transformation):
    rad = math.radians(transformation[2])
    t2 = [-transformation[1], -transformation[0]]

    aff = AffineTransform(translation=t2)
    moving_t = warp(img2, aff, order=0)
    moving_t = rotate(moving_t, transformation[2],order=0)
    return moving_t


avg_dist_error_array_array = []
fname_arr = []
avg_dist_error_array = []
angle_error_array = []
angle_array = []
scale_array = []
mean_squared_err_arr = []
ssim_arr = []
corr_arr = []
mindex = 0
d2 = []

df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()

for stylemethod in style_methods:

    for regmethod in reg_methods:
        folder_str = f"{path_results}/visulaization/{stylemethod}_{regmethod}/"
        mask_folder_str = f"{folder_str}/masks/"
        corner_folder_str = f"{folder_str}/corner_points/"
        if not os.path.exists(folder_str):
            os.makedirs(folder_str)
        if not os.path.exists(mask_folder_str):
            os.makedirs(mask_folder_str)
        if not os.path.exists(corner_folder_str):
            os.makedirs(corner_folder_str)

        files = glob.glob(f'{path_results}{stylemethod}/{regmethod}/json/*.json')
        # print(f"{stylemethod} {regmethod}")
        for idx in tqdm(range(len(files))):
            fname, ext = os.path.splitext(files[idx])
            path, fname = os.path.split(fname)
            numbers = re.compile(r'\d+')
            res = numbers.findall(fname)
            result_string = f'{path}/p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{res[4]}_z0_l1_o0_1{ext}'
            rs2 = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{res[4]}_z0_l1_o0_{res[8]}'
            maskstr = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{0}_z0_l1_o0'
            bias_image = cv2.imread(f'{path_bias}{rs2}.png')
           # print(f'{path_bias_mask}{maskstr}mask.png')
            # mask_image = cv2.imread(f'{path_bias_mask}{maskstr}mask.png',cv2.IMREAD_UNCHANGED)
            # testing with original masks
            mask_image = cv2.imread(f'{path_bias_mask}{maskstr}.png', cv2.IMREAD_UNCHANGED)

            #mask_image = mask_image * 65535

            mask_image = mask_image[:850, :850]

            mask_image = cv2.resize(mask_image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            # print(np.max(mask_image))
            #mask_image = mask_image / mask_image.max()  # normalizes data in range 0 - 255
            #mask_image = 255 * mask_image
            #mask_image = mask_image.astype(np.uint8)

            LMD_image = cv2.imread(f'{path_leica}{rs2}.png')
            # load annotations TM to compare
            annot_str = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{1}_z0_l1_o0.mat'
            if not os.path.isfile(os.path.join(path_annot, annot_str)):
                annot_str = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{1}_z{1}_l1_o0.mat'
                if not os.path.isfile(os.path.join(path_annot, annot_str)):
                    continue
            mat = scipy.io.loadmat(os.path.join(path_annot, annot_str))
            annot_tm = mat['Tm']

            annot_tm = format_annotation_matrix(annot_tm)
            f = open(files[idx])
            res_tm = json.load(f)
           # print(fname)
           # print(annot_tm)
            #print(res_tm)
            # print(regmethod)
            if regmethod == 'corr':
                res_tm[1] = int(res_tm[1]) * sizecorr
                res_tm[0] = int(res_tm[0]) * sizecorr
               # print(res_tm)
                warped = warp_correlation(LMD_image, bias_image, res_tm)
                warped_mask = warp_correlation(LMD_image, mask_image, res_tm)
                #warped_mask = mask_image
                corenr_points = get_avarage_corner_distance(1024, 1024, res_tm, annot_tm, corr=True)
            else:
                res_tm = np.array(res_tm)
                res_tm[:2, 2] = res_tm[:2, 2] * sizecorr
                #print(bias_image.shape)
                warped = cv2.warpAffine(bias_image, res_tm, (bias_image.shape[0], bias_image.shape[1]))
                warped_mask = cv2.warpAffine(mask_image, res_tm, (bias_image.shape[0], bias_image.shape[1]), cv2.INTER_NEAREST)
                # warped_mask = mask_image
                np.vstack([res_tm, np.array([0, 0, 1])])
                corenr_points = get_avarage_corner_distance(1024, 1024, res_tm, annot_tm)

            warped = cv2.normalize(warped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
            ol_SP_real = cv2.addWeighted(LMD_image, 0.5, warped.astype(np.uint8), 0.5, 0.7)

            # warped_mask = cv2.normalize(warped_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # m_err, ssim_val, cor_val = compare_images(LMD_image[128:128+768, 128:128+768, :], warped[128:128 + 768, 128:128 + 768, :])

            m_err2, ssim_val2 = compare_images(LMD_image, bias_image)

            #mean_squared_err_arr.append(m_err)
            ssim_arr.append(ssim_val)
            # corr_arr.apped(cor_val)
            #print(f'{fname} mse {m_err} ssim = {ssim_val} diff: {m_err2-m_err}, {ssim_val2-ssim_val}')
            # warped_mask = warped_mask.astype(np.uint8)
            warped_mask = (warped_mask * 65535).astype(np.uint8)
            contours_only_4channel = np.zeros_like(mask_image)

            # for j in range(1, int(np.max(mask_image))):
            #     jth_object = mask_image == j
            #     jth_object = jth_object.astype(np.uint8)
            #     jth_object = cv2.erode(jth_object, np.ones((2,2)))
            #     jth_object = cv2.dilate(jth_object, np.ones((2,2)))
            #     jth_object = jth_object
            #     contours, _ = cv2.findContours(jth_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     #contours, _ = cv2.findContours(jth_object.copy(), 1, 1)
            #     cv2.drawContours(contours_only_4channel, contours, -1, (255, 255, 255, 255), 5)
            for j in range(1, int(np.max(warped_mask))):
                jth_object = warped_mask == j
                jth_object = jth_object.astype(np.uint8)
                jth_object = cv2.erode(jth_object, np.ones((2,2)))
                jth_object = cv2.dilate(jth_object, np.ones((2,2)))
                jth_object = jth_object
                contours, _ = cv2.findContours(jth_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #contours, _ = cv2.findContours(jth_object.copy(), 1, 1)
                cv2.drawContours(LMD_image, contours, -1, (0, 255, 0), 5)


            # imshow("", LMD_image)
            # cv2.waitKey(0)

            warped_mask.astype(np.uint8)
            k = np.ones((2, 2), np.uint8)

            ret, thresh = cv2.threshold(warped_mask, 1, 255, 0)
            warped_mask = cv2.erode(warped_mask, k)
            # print(thresh.dtype)
            # contours, hierarchy = cv2.findContours(thresh[:,:,1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # ol_real_masked = cv2.addWeighted(LMD_image, 0.5, warped_mask.astype(np.uint8), 0.5, 0.7)

           # cv2.imshow("asd", ol_SP_real)
            #cv2.waitKey(0)
            cv2.imwrite(f'{folder_str}{rs2}.png', ol_SP_real.astype(np.uint8))
            # cv2.imwrite(f'{mask_folder_str}{rs2}.png', ol_real_masked.astype(np.uint8))
            cv2.imwrite(f'{mask_folder_str}{rs2}.png', LMD_image.astype(np.uint8))
            #  cv2.imwrite(f'{mask_folder_str}{rs2}_mask_only.png', contours_only_4channel.astype(np.uint8))
            corenr_points.savefig(f'{corner_folder_str}{rs2}.png')
            plt.close(corenr_points)
            #avg_dist_error_array.append(avg_dist_error)
            #angle_error_array.append(a_err)
           # print(f"{fname}: avg error:{avg_dist_error}")
        if mindex == 0:
            df['files'] = fname_arr
            df2['files'] = fname_arr
            df3['files'] = fname_arr
        mindex = mindex + 1
        df[f'{stylemethod} {regmethod}'] = ssim_arr
        df2[f'{stylemethod} {regmethod}'] = mean_squared_err_arr
        df3[f'{stylemethod} {regmethod}'] = corr_arr
        ssim_arr = []
        mean_squared_err_arr = []
        corr_arr = []

df.to_csv('ssim.csv')
df2.to_csv('meansquared.csv')
df3.to_csv('corr.csv')
'''
if write_error:
    with open(csvname, 'w') as f:
        write = csv.writer(f)
        write.writerow(fname_arr)
        write.writerow(avg_dist_error_array)
if write_annotated:
    with open(csvraw_data, 'w') as f:
        write = csv.writer(f)
        write.writerow(fname_arr)
        write.writerow(angle_array)
'''