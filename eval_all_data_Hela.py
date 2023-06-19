import json
import math

import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
import glob

import scipy.io
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

write_transform_data = True
write_error = False
write_annotated = True
from statistics import *

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

'''

size = 256
inverted = False
save_plot = False
save_barplot = False
save_highlight_ecdf = False
save_highlight_bar = False
ecdf_name = 'ecdf_hela'

experiment_name = 'hela6063x_fl_spv6'
experiment_name = 'hela_v2_ssv_test2'
experiment_name = 'hela_ssv_test2'

path_results = f'd:/datasets/Image_registration/211109-HK-60x/results/registration/{experiment_name}/'
# path_results = f'e:/deeplearning/hela/RES_RES/CUT/CUT_hela63_BF2FL_all_small/SIFT_AND_SP{size}/'
# path_results = f'e:/deeplearning/hela/RES_RES/Cyclegan/cycle_hela63_BF2FL_all_small/SIFT_AND_SP{size}/'
path_annot = "d:/datasets/Image_registration/211109-HK-60x/splitted/test/annot_scaled/"
reg_methods = ['corr', 'sp', 'sift']
# style_methods = ['cut', 'pix2pix', 'cyclegan']
style_methods = ['nostyle', 'comir', 'unet_aligned', 'pix2pix_aligned', 'cyclegan_aligned', \
                 'cut_aligned', \
                 'cyclegan_unaligned', \
                 'cut_unaligned']

# style_methods = ['pix2pix_aligned', 'cyclegan_aligned', 'cut_aligned']

csvname = f'{experiment_name}_avg_error.csv'

path_bias = 'f:/datasets/211109-HK-60x_120annot/splitted/test/notaligned_samescale_bias/'
path_leica = 'f:/datasets/211109-HK-60x_120annot/splitted/test/notaligned_samescale_LMD/'

sizecorr = 1024 / size

np.set_printoptions(suppress=True)

avg_dist_error_array_array = []


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
        [[p_transformation[3] * math.cos(radangle), p_transformation[3] * (-math.sin(radangle)), 0],
         [p_transformation[3] * math.sin(radangle), p_transformation[3] * math.cos(radangle), 0], [0, 0, 1]])
    rotated = rot_matrix @ cp2
    translation_matrix = np.array([[1, 0, p_transformation[1] + 512], [0, 1, p_transformation[0] + 512], [0, 0, 1]])
    final = translation_matrix @ rotated
    return final


def get_avarage_corner_distance(x_s, y_s, transformation_matrix, annot_matrix, corr=False):
    corner_points = [[0, 0, 1], [0, y_s, 1], [x_s, y_s, 1], [x_s, 0, 1]]
    corner_points = np.array(corner_points).T
    result_annot = annot_matrix @ corner_points

    scale_annot = math.sqrt(annot_matrix[0, 0] * annot_matrix[0, 0] + annot_matrix[1, 0] * annot_matrix[1, 0])
    angle_annot = math.atan2(annot_matrix[1, 0], annot_matrix[0, 0]) * 180 / math.pi
    if corr:
        result_coor = get_transformed_coordinates_coorelation(transformation_matrix)
        angle_res = transformation_matrix[2]
        scale_res = transformation_matrix[3]
    else:
        result_coor = transformation_matrix @ corner_points
        angle_res = math.atan2(transformation_matrix[0, 1], transformation_matrix[0, 0]) * 180 / math.pi
        scale_res = math.sqrt(
            transformation_matrix[0, 0] * transformation_matrix[0, 0] + transformation_matrix[0, 1] *
            transformation_matrix[
                0, 1])
    # print(f'scale =  {scale_annot},resscel = {scale_res} angleannot  = {angle_annot}, angle = {angle_res} ')

    # print(result_coor)
    # print(result_annot)
    angle_err = angle_annot - angle_res
    angle_array.append(angle_res)
    scale_array.append(scale_res)

    dist = list(range(corner_points.shape[1]))
    for i in range(corner_points.shape[1]):
        dist[i] = np.linalg.norm(result_coor[:2, i] - result_annot[:2, i])
    dist = np.array(dist)

    avgarge_distance = dist.mean()
    return avgarge_distance, angle_err, result_annot


avg_dist_error_array_array = []
fname_arr = []
avg_dist_error_array = []
angle_error_array = []
angle_array = []
scale_array = []
mindex = 0
d2 = []
csvname = 'asdallavaragedistanceerror2.csv'
# f = open(csvname, 'w', newline='')
# write = csv.writer(f)

df = pd.DataFrame()
for stylemethod in style_methods:

    for regmethod in reg_methods:

        files = glob.glob(f'{path_results}{stylemethod}/{regmethod}/json/*_1*.json')

        for idx in range(len(files)):
            fname, ext = os.path.splitext(files[idx])
            path, fname = os.path.split(fname)
            numbers = re.compile(r'\d+')
            res = numbers.findall(fname)

            result_string = f'{path}/p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{res[4]}_z0_l1_o0_1{ext}'
            rs2 = f'p{res[0]}_wA{res[1]}_t{res[2]}_m{res[3]}_c{res[4]}_z0_l1_o0_{res[8]}'

            # bias_image = cv2.imread(f'{path_bias}{rs2}.png')
            # LMD_image = cv2.imread(f'{path_leica}{rs2}.png')

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
            # print(res_tm)

            if regmethod == 'corr':
                res_tm[1] = int(res_tm[1]) * sizecorr
                res_tm[0] = int(res_tm[0]) * sizecorr
                avg_dist_error, a_err, res_ann = get_avarage_corner_distance(1024, 1024, res_tm, annot_tm, corr=True)
            else:
                res_tm.append([0, 0, 1])
                res_tm = np.array(res_tm)
                res_tm[:2, 2] = res_tm[:2, 2] * sizecorr
                # print(res_tm)
                if inverted:
                    res_tm = np.linalg.pinv(res_tm)
                avg_dist_error, a_err, res_ann = get_avarage_corner_distance(1024, 1024, res_tm, annot_tm)

            avg_dist_error_array.append((avg_dist_error / 1024) * 100)
            # angle_error_array.append(a_err)
            fname_arr.append(fname)
            # print(f"{fname}: avg error:{avg_dist_error}")

        if mindex == 0:
            df['files'] = fname_arr
        mindex = mindex + 1

        df[f'{stylemethod} {regmethod}'] = avg_dist_error_array
        print(
            f'cindenx {mindex} styletransfer: {stylemethod} registration: {regmethod} error: {mean(avg_dist_error_array):.2f},+/-{stdev(avg_dist_error_array):.2f} ' \
            f'median {median(avg_dist_error_array):.2f}, max {max(avg_dist_error_array):.2f}, min {min(avg_dist_error_array):.2f}')

        avg_dist_error_array_array.append(avg_dist_error_array)
        avg_dist_error_array = []

        # write.writerow(fname_arr)
        # write.writerow(avg_dist_error_array)

print(len(fname_arr))
print(df)

df.to_csv(csvname)
np_arr_arr = []
mindex = 1
ecdf_arr = []
lgnd_arr = []
ecdf_integral = []
prob_arr = []

print("***integral**************************")
for stylemethod in style_methods:
    for regmethod in reg_methods:
        np_arr_arr.append(df.iloc[:, mindex].to_numpy())
        ecdf = ECDF(np_arr_arr[mindex - 1])
        ecdf_arr.append(ecdf)
        # print(ecdf.x)
        # TODO fix this here

        for idx in range(0, 100):
            x = float(idx)
            prob_arr.append(ecdf(x))

        integral = np.trapz(prob_arr)
        shifted_x = np.roll(ecdf.x[1:], -1)
        diff_x = np.abs(ecdf.x[1:] - shifted_x)
        shifted_y = np.roll(ecdf.y[1:], -1)
        avg_y = np.abs((ecdf.y[1:] + shifted_y) / 2)
        area = avg_y * diff_x

        ecdf_integral.append(integral)
        # print(integral,  {0:.2f})
        print("%.2f" % integral)
        # print(f'cindenx {mindex} styletransfer: {stylemethod} registration: {regmethod} integral:{integral}')
        lgnd_arr.append([f'{stylemethod} and {regmethod}'])
        mindex = mindex + 1
        prob_arr = []

print(ecdf_integral)
print(len(ecdf_integral))
print("***end of integral**************************")

print(f'{np.argmax(ecdf_integral)} ,{np.argmin(ecdf_integral)}')

new_arr = [s[0] for s in lgnd_arr]
colors = ['forestgreen', 'lime', 'deepskyblue', 'violet', 'sienna', 'navy', 'tomato', 'gray', 'brown']
colors = ['forestgreen', 'deepskyblue', 'navy', 'violet', 'brown', 'lime', 'tomato', 'sienna']
colors = ['forestgreen', 'deepskyblue',  'navy','violet',   'brown',  'tomato', 'sienna', 'lime']
color_idx = 0
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=10)  # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
fig = plt.figure()


def addlabels(y):
    for i in range(len(y)):
        plt.text(int(i), int(y[i] // 2), int(y[i]), ha='center', )


# barplot

for plotidx in range(0, len(ecdf_integral), 3):
    print(*lgnd_arr[plotidx + 1], f'pix2pix_aligned and sp')
    # if ', '.join(lgnd_arr[plotidx + 1]) == 'pix2pix_aligned and sp' or ', '.join(
    #         lgnd_arr[plotidx + 1]) == 'nostyle and sp' or ','.join(lgnd_arr[plotidx + 1]) == 'cut_unaligned and sp':
    plt.bar(new_arr[plotidx], ecdf_integral[plotidx], color=colors[color_idx])
    plt.bar(new_arr[plotidx + 1], ecdf_integral[plotidx + 1], color=colors[color_idx])
    plt.bar(new_arr[plotidx + 2], ecdf_integral[plotidx + 2], color=colors[color_idx])

    color_idx = color_idx + 1

# addlabels(ecdf_integral)
# plt.legend(loc='upper center', bbox_to_anchor=(0.8, 1.5), fancybox=True, ncol=3)

plt.xticks(rotation=90)
plt.tight_layout()
if save_barplot:
    plt.savefig('barplot_hela.pdf')

plt.show()
"""
highlight barplot
"""
plt.clf()
colors = ['forestgreen', 'deepskyblue', 'navy', 'violet', 'brown', 'lime', 'tomato', 'sienna']
colors = ['forestgreen', 'deepskyblue',  'navy','violet',   'brown',  'tomato', 'sienna', 'lime']
color_idx = 0
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=10)  # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
fig = plt.figure()

for plotidx in range(0, len(ecdf_integral), 3):
    if ', '.join(lgnd_arr[plotidx]) == 'comir and corr' or ', '.join(
            lgnd_arr[plotidx]) == 'nostyle and corr':
        plt.bar(new_arr[plotidx], ecdf_integral[plotidx], color=colors[color_idx])
        print(f"{ecdf_integral[plotidx]:.2f}")
    if ', '.join(lgnd_arr[plotidx + 1]) == 'pix2pix_aligned and sp' or ', '.join(
            lgnd_arr[plotidx + 1]) == 'unet_aligned and sp' or ','.join(
        lgnd_arr[plotidx + 1]) == 'cut_aligned and sp' or ', '.join(
        lgnd_arr[plotidx + 1]) == 'nostyle and sp' or ','.join(lgnd_arr[plotidx + 1]) == 'cut_unaligned and sp':
        plt.bar(new_arr[plotidx + 1], ecdf_integral[plotidx + 1], color=colors[color_idx])
        print(f"{ecdf_integral[plotidx + 1]:.2f}")
    if ', '.join(lgnd_arr[plotidx + 2]) == 'comir and sift' or ', '.join(
            lgnd_arr[plotidx + 2]) == 'nostyle and sift' or ','.join(lgnd_arr[plotidx + 2]) == 'cut_unaligned and sift':
        plt.bar(new_arr[plotidx + 2], ecdf_integral[plotidx + 2], color=colors[color_idx])
        print(f"{ecdf_integral[plotidx + 2]:.2f}")

    color_idx = color_idx + 1

# addlabels(ecdf_integral)
# plt.legend(loc='upper center', bbox_to_anchor=(0.8, 1.5), fancybox=True, ncol=3)
plt.xticks([])
fig.set_size_inches(3, 3)
plt.xticks(rotation=90)
plt.tight_layout()
if save_highlight_bar:
    plt.savefig('barplot_hela_high.pdf')

plt.show()


# generate the ultimate plot

import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# vivi_colors = ['crimson', 'deepskyblue', 'magenta', 'navy', 'orange', 'limegreen', 'gray', 'chocolate', 'sienna']
#  colors = vivi_colors
plt.clf()
fig = plt.figure()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
colors = ['forestgreen', 'deepskyblue', 'lime', 'sienna', 'violet', 'gray', 'navy', 'tomato', 'brown']
colors = ['deepskyblue', 'forestgreen', 'violet', 'navy', 'lime', 'gray', 'tomato', 'sienna', 'brown']
colors = ['forestgreen', 'deepskyblue', 'navy', 'violet', 'brown', 'lime', 'tomato', 'sienna']
colors = ['forestgreen', 'deepskyblue',  'navy','violet',   'brown',  'tomato', 'sienna', 'lime']

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=10)  # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title
color_idx = 0
for plotidx in range(0, len(ecdf_arr), 3):
    plt.plot(ecdf_arr[plotidx].x, ecdf_arr[plotidx].y * 100, c=colors[color_idx], linestyle='dotted',
             label=''.join(lgnd_arr[plotidx]), )
    plt.plot(ecdf_arr[plotidx + 1].x, ecdf_arr[plotidx + 1].y * 100, c=colors[color_idx], linestyle='dashed',
             label=''.join(lgnd_arr[plotidx + 1]))
    plt.plot(ecdf_arr[plotidx + 2].x, ecdf_arr[plotidx + 2].y * 100, c=colors[color_idx],
             label=''.join(lgnd_arr[plotidx + 2]))
    color_idx = color_idx + 1

plt.suptitle('HeLa Kyoto', fontsize=16)
fig.set_size_inches(5, 5)
plt.legend(bbox_to_anchor=(1.03, 0.8), loc='upper left', ncol=4)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 0.5), fancybox=True, ncol=3)


# plt.xticks([0.7, 1, 2, 3, 4])

plt.xscale('log')
plt.ylabel('cummultative succes')
plt.xlabel('relative error %')

plt.xlim(right=4)
plt.xlim(left=0.6)
ax = plt.axes()
ax.set_xticks([0.8, 1, 2, 3, 4])
# plt.xticks([0.6, 0.8, 1, 2, 3, 4])
ax.grid(which='minor', axis='both', linestyle='-')
ax.grid(which='major', axis='both', linestyle='-')
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_formatter(FormatStrFormatter("%0.1f"))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%5.f'))
# plt.gca().xaxis.grid(True)

plt.tight_layout()
if save_plot:
    plt.savefig(f"{ecdf_name}.pdf")
plt.show()

# generate the ultimate  highlight plot

import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# vivi_colors = ['crimson', 'deepskyblue', 'magenta', 'navy', 'orange', 'limegreen', 'gray', 'chocolate', 'sienna']
#  colors = vivi_colors
plt.clf()
fig = plt.figure()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
colors = ['forestgreen', 'deepskyblue', 'lime', 'sienna', 'violet', 'gray', 'navy', 'tomato', 'brown']
colors = ['deepskyblue', 'forestgreen', 'violet', 'navy', 'lime', 'gray', 'tomato', 'sienna', 'brown']
colors = ['forestgreen', 'deepskyblue', 'navy', 'violet', 'brown', 'lime', 'tomato', 'sienna']
colors = ['forestgreen', 'deepskyblue',  'navy','violet',   'brown',  'tomato', 'sienna', 'lime']

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=10)  # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title
color_idx = 0
for plotidx in range(0, len(ecdf_arr), 3):
    if ', '.join(lgnd_arr[plotidx]) == 'comir and corr' or ', '.join(
            lgnd_arr[plotidx]) == 'nostyle and corr':
        plt.plot(ecdf_arr[plotidx].x, ecdf_arr[plotidx].y * 100, c=colors[color_idx], linestyle='dotted',
                 label=''.join(lgnd_arr[plotidx]))
    if ', '.join(lgnd_arr[plotidx + 1]) == 'pix2pix_aligned and sp' or ', '.join(
            lgnd_arr[plotidx + 1]) == 'unet_aligned and sp' or ','.join(
            lgnd_arr[plotidx + 1]) == 'cut_aligned and sp' or ', '.join(
            lgnd_arr[plotidx + 1]) == 'nostyle and sp' or ','.join(lgnd_arr[plotidx + 1]) == 'cut_unaligned and sp':
        plt.plot(ecdf_arr[plotidx + 1].x, ecdf_arr[plotidx + 1].y * 100, c=colors[color_idx], linestyle='dashed',
                 label=''.join(lgnd_arr[plotidx + 1]), )
    if ', '.join(lgnd_arr[plotidx + 2]) == 'comir and sift' or ', '.join(
            lgnd_arr[plotidx + 2]) == 'nostyle and sift' or ','.join(lgnd_arr[plotidx + 2]) == 'cut_unaligned and sift':
        plt.plot(ecdf_arr[plotidx + 2].x, ecdf_arr[plotidx + 2].y * 100, c=colors[color_idx],
                 label=''.join(lgnd_arr[plotidx + 2]))
    color_idx = color_idx + 1

plt.suptitle('HeLa Kyoto', fontsize=16)
fig.set_size_inches(5, 5)
# plt.legend(bbox_to_anchor=(1.03, 0.8), loc='upper left', ncol=4)
# plt.legend(loc='upper left', bbox_to_anchor=(1.05, 0.5), fancybox=True, ncol=3)


# plt.xticks([0.7, 1, 2, 3, 4])

plt.xscale('log')
plt.ylabel('cummultative succes')
plt.xlabel('relative error %')

plt.xlim(right=5)
plt.xlim(left=0.5)
ax = plt.axes()
ax.set_xticks([0.6, 0.8, 1, 2, 3, 4, 5])
plt.xticks([0.6, 0.8, 1, 2, 3, 4, 5])
ax.grid(which='minor', axis='both', linestyle='-')
ax.grid(which='major', axis='both', linestyle='-')
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_formatter(FormatStrFormatter("%0.1f"))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%5.f'))
# plt.gca().xaxis.grid(True)

plt.tight_layout()
if save_highlight_ecdf:
    plt.savefig(f"{ecdf_name}_high.pdf")
plt.show()

# significane
np_arr_arr = []
mindex = 1
ecdf_arr = []
# lgnd_arr = []
ecdf_integral = []
yes = 0
for stylemethod in style_methods:
    for regmethod in reg_methods:
        np_arr_arr.append(df.iloc[:, mindex].to_numpy())
        mindex = mindex + 1
mindex = 1
res_w = np.zeros([len(np_arr_arr), len(np_arr_arr)])
print(res_w)
for idx in range(len(np_arr_arr)):
    for jdx in range(len(np_arr_arr)):
        if (np_arr_arr[idx] - np_arr_arr[jdx]).all() == 0:
            res_w[idx, jdx] = 0
        else:
            val = stats.wilcoxon(np_arr_arr[idx], np_arr_arr[jdx])[1]

            if yes:
                if val < 0.05:
                    res_w[idx, jdx] = 1
                if val < 0.01:
                    res_w[idx, jdx] = 2
                if val < 0.001:
                    res_w[idx, jdx] = 3
                else:
                    res_w[idx, jdx] = 0
            else:
                res_w[idx, jdx] = val

ax = plt.axes()

data = res_w
plt.imshow(data)
plt.title("2-D Heat Map")
lgnd_arr = np.array(lgnd_arr)
plt.xticks(range(0, len(lgnd_arr)), lgnd_arr, rotation='vertical')
plt.yticks(range(0, len(lgnd_arr)), lgnd_arr)
plt.tight_layout()
plt.show()
significance_name = 'Hela_sig.csv'
# res_w = np.insert(res_w, 0, lgnd_arr, axis=1)
nplegend_array = np.array([lgnd_arr[0] for lgnd_arr in lgnd_arr])
res_w2 = np.vstack((nplegend_array, res_w))
nplegend_array1 = np.insert(nplegend_array, 0, 0)
res_w2 = np.insert(res_w2, 0, nplegend_array1, axis=1)
sig_df = pd.DataFrame(res_w2)
sig_df.to_csv(significance_name, float_format='%.5f')

"""
NICE PLOT colors = style methods
linestyles = registration methods
"""

colors = ['forestgreen', 'lime', 'deepskyblue', 'violet', 'sienna', 'navy', 'tomato', 'gray', 'brown']
colors = ['forestgreen', 'deepskyblue',  'navy','violet',   'brown',  'tomato', 'sienna', 'lime']
color_idx = 0
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=10)  # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title
fig = plt.figure()

for plotidx in range(0, len(ecdf_arr), 3):
    print(*lgnd_arr[plotidx + 1], f'pix2pix_aligned and sp')
    # if ', '.join(lgnd_arr[plotidx+1]) == 'pix2pix_aligned and sp' or ', '.join(lgnd_arr[plotidx+1]) == 'nostyle and sp' or ','.join(lgnd_arr[plotidx+1]) == 'cut_unaligned and sp':
    plt.plot(ecdf_arr[plotidx].x, ecdf_arr[plotidx].y * 100, c=colors[color_idx], linestyle='dotted',
             label=''.join(lgnd_arr[plotidx]), )
    plt.plot(ecdf_arr[plotidx + 1].x, ecdf_arr[plotidx + 1].y * 100, c=colors[color_idx], linestyle='dashed',
             label=''.join(lgnd_arr[plotidx + 1]))
    plt.plot(ecdf_arr[plotidx + 2].x, ecdf_arr[plotidx + 2].y * 100, c=colors[color_idx],
             label=''.join(lgnd_arr[plotidx + 2]))
    color_idx = color_idx + 1

plt.suptitle('Hella', fontsize=16)
# plt.legend(bbox_to_anchor=(1.03, 0.8), loc='upper left', ncol=2)
plt.legend(loc='upper center', bbox_to_anchor=(0.8, 1.5), fancybox=True, ncol=3)
#

plt.xlim(right=5)
plt.xlim(left=0.7)
plt.xticks([1, 3, 5])
plt.xscale('log')
plt.ylabel('cummultative succes')
plt.xlabel('relative error %')
plt.grid(linestyle='-', linewidth=1)
fig.set_size_inches(5, 5)
ax = plt.axes()
import matplotlib.ticker as ticker
# ax.set_xticks(ax.get_xticks()[::2])
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_formatter(FormatStrFormatter("%0.1f"))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%5.f'))
plt.tight_layout()
plt.show()

np_array1 = df.iloc[:, 1].to_numpy()
np_array2 = df.iloc[:, 2].to_numpy()
np_array3 = df.iloc[:, 3].to_numpy()
np_array4 = df.iloc[:, 4].to_numpy()
np_array5 = df.iloc[:, 5].to_numpy()
np_array6 = df.iloc[:, 6].to_numpy()
np_array7 = df.iloc[:, 7].to_numpy()
np_array8 = df.iloc[:, 8].to_numpy()
np_array9 = df.iloc[:, 9].to_numpy()

# np_no_style_sift = df.iloc[:, 21].to_numpy()
# np_no_style_sp = df.iloc[:, 20].to_numpy()
# np_no_style_cor = df.iloc[:, 19].to_numpy()
# np_cut_sp_a = df.iloc[:, 14].to_numpy()
# np_cut_sp_ua = df.iloc[:, 17].to_numpy()
# np_arrayunet = df.iloc[:, -2].to_numpy()

print(len(np_array1))
ecdf1 = ECDF(np_array1)
ecdf8 = ECDF(np_array8)
ecdf2 = ECDF(np_array2)
ecdf3 = ECDF(np_array3)
ecdf4 = ECDF(np_array4)
ecdf5 = ECDF(np_array5)
ecdf6 = ECDF(np_array6)
ecdf7 = ECDF(np_array7)
ecdf9 = ECDF(np_array9)
# ecdfcuta = ECDF(np_cut_sp_a)
# ecdfcutua = ECDF(np_cut_sp_ua)
# ecdfunet = ECDF(np_arrayunet)
# ecdf_no_style_cor = ECDF(np_no_style_cor)

print(np_array8.shape)

# x = np.linspace(min(np_array), max(np_array))

from matplotlib.ticker import (FormatStrFormatter)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# plt.plot(ecdf1.x, ecdf1.y, 'b', ecdf2.x, ecdf2.y,'b', ecdf3.x, ecdf3.y, 'b', ecdf4.x, ecdf4.y, 'r', ecdf5.x,
# ecdf5.y,'r', ecdf6.x, ecdf6.y,'r', ecdf7.x,ecdf7.y,'g', ecdf8.x, ecdf8.y, 'g', ecdf9.x,ecdf9.y,'g') plt.plot(
# ecdf1.x, ecdf1.y, 'r', ecdf2.x, ecdf2.y,'g', ecdf3.x, ecdf3.y, 'b', ecdf4.x, ecdf4.y, 'r', ecdf5.x, ecdf5.y,'g',
# ecdf6.x, ecdf6.y,'b', ecdf7.x,ecdf7.y,'r', ecdf8.x, ecdf8.y, 'g', ecdf9.x,ecdf9.y,'b') plt.plot(ecdfcuta.x,
# ecdfcuta.y, 'g', ecdfcutua.x, ecdfcutua.y, 'r', ecdfunet.x,ecdfunet.y,'y', ecdf8.x,ecdf8.y,'b', ecdf1.x, ecdf1.y,
# ecdf_no_style_cor.x, ecdf_no_style_cor.y, 'c') plt.legend(['cut aligned superpoint', 'cut unaligned superpoint',
# 'Unet sift', 'p2p superpoint', 'Comir correlation', 'No style correlation']) ax.legend(['cut unaligned',
# 'cut unaligned'])
plt.xlim(right=100)
plt.xlim(left=5)
# ax.xaxis.set_major_locator(MultipleLocator(10))
plt.grid(linestyle='-', linewidth=1)
plt.legend([])
plt.show()

#
# for index in range(0,len(ecdf8.x),2):
#     print(f"{ecdf8.y[index]} % p2p sp {ecdf8.x[index]}, Comir {ecdf1.x[index]} unet {ecdfunet.x[index]}")

#  todo wilcoxon every row with every row
res_w = stats.wilcoxon(np_array1, np_array9)
# res_w = stats.wilcoxon(np_cut_sp_a, np_cut_sp_ua)
print(f"wilcoxon {res_w}")

# plt.plot(ecdf1.x, )
plt.xscale('log')
plt.ylabel('cummultative succes')
plt.xlabel('error in pixels')
ax = plt.axes()
ax.xaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))
import matplotlib.ticker as ticker

# ax.set_xticks(ax.get_xticks()[::2])
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

plt.xlim(right=100)
plt.xlim(left=5)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

ax.xaxis.set_major_locator(MultipleLocator(10))
plt.grid(linestyle='-', linewidth=1)

plt.show()

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
