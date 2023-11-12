import argparse
from itertools import cycle
from pathlib import Path
#from this import d

import cv2
import numpy as np
import tensorflow as tf  # noqa: E402
import os
from settings import EXPER_PATH  # noqa: E402
import glob
import json
from tqdm import tqdm
# logging.set_verbosity(logging.INFO)
import math

# correlation related
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate,  rescale
from datetime import timedelta
from skimage.transform import AffineTransform, warp
from scipy.fft import fft2, fftshift
from skimage.filters import window, difference_of_gaussians



'''
p2p images must exists for testing
'''


'''
CUDA_VISIBLE_DEVICES=0 python3 calc_sp_sift_corr.py superpoint_tissue_finetuned 
'/storage01/grexai/datasets/imgreg/Image_registration/Aligned/test/testA/*.png' 
/storage01/grexai/datasets/imgreg/Image_registration/Aligned/test/testB/
'''


'''Run for comir 
CUDA_VISIBLE_DEVICES=1, python3 Comir_test.py model_L1_20222005_115850 
/storage01/grexai/datasets/imgreg/Image_registration/Aligned/Comir/test/notaligned_samescale_LMD/
'''


def estimate_rotation_and_scaling(fix: np.array, moving: np.array,dowindowing=False):
    fix = difference_of_gaussians(fix, 5, 20)
    moving = difference_of_gaussians(moving, 5, 20)
    if dowindowing:
        w_fix = fix * window('hann', fix.shape)
        w_moving = moving * window('hann', moving.shape)
    else:
        w_fix = fix
        w_moving = moving
    fix_fs = np.abs(fftshift(fft2(w_fix)))
    moving_fs = np.abs(fftshift(fft2(w_moving)))

    shape = fix_fs.shape
    radius = shape[0] // 8 # checking on lower freq

    warped_fix = warp_polar(fix_fs, radius=radius, output_shape=shape, scaling='log', order=0)
    warped_moving = warp_polar(moving_fs, radius=radius, output_shape=shape, scaling='log', order=0)
    warped_fix = warped_fix[:shape[0] // 2, :]
    warped_moving = warped_moving[:shape[0] // 2, :]
    shift, error, phasediff = phase_cross_correlation(warped_fix, warped_moving, upsample_factor=20)
    shiftr, shiftc = shift[:2]
    angle = (360 / shape[0]) * shiftr
    klog = radius / np.log(radius)
    shift_scale = 1 / (np.exp(shiftc / klog))
    #rescaled = rescale(rotated, scale, channel_axis=-1)
    '''
    if visualize:
        plt.figure()
        plt.subplot(2, 3, 1), plt.imshow(w_fix, 'hot')
        plt.subplot(2, 3, 2), plt.imshow(w_moving, 'hot')
        plt.subplot(2, 3, 3), plt.imshow(warped_fix, 'gray')
        plt.subplot(2, 3, 4), plt.imshow(warped_moving, 'gray')
        plt.savefig('rotation.png')
    '''
    return angle,shift_scale


def calculate_correlation_and_warp_image(fix, moving,do_rotation=True):
    #fix2 = cv2.resize(fix,(256, 256))
    #moving2 = cv2.resize(moving, (256, 256))
    fix =  cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)
    moving = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    shift, error, diffphase = phase_cross_correlation(fix, moving, overlap_ratio=0.8)
    shift2 = [shift[1], shift[0]]
    shift2 = np.array(shift2)
    shift2 = shift2*-1
    transform = AffineTransform(translation=shift2)
    moving_t = warp(moving, transform)
    
    rotation_angle = 0
    scaling = 0
    if do_rotation:
        rotation_angle, scaling = estimate_rotation_and_scaling(fix, moving_t)
        rad_angle = math.radians(rotation_angle)
       # print(rotation_angle)
        t_r = AffineTransform(rotation=rad_angle)

        temp = transform.params
        TM = t_r.params
        TM[0, 1] = temp[0, 1]
        TM[0, 2] = temp[0, 2]
        #moving_t_r = warp(moving_t, t_r)
        moving_t_r = rotate(moving_t, rotation_angle)
       # moving_t_r = rescale(moving_t_r, scaling)
    else:
        TM = transform.params
        moving_t_r = moving_t
    return moving_t_r, shift, rotation_angle, scaling


def extract_SIFT_keypoints_and_descriptors(p_img):
    #print(p_img.shape)
    # p_img = p_img.astype('uint8')
    gray_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)
    # gray_img= p_img
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.xfeatures2d.SURF_create()
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)
    return kp, desc


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]
    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc


def compute_rigid(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
    # print(len(matched_pts1), len(matched_pts2))
    
    if len(matched_pts1) == 0 or len(matched_pts1)==0:
        H = np.array([[1, 0, 0],[0, 1, 0]], dtype=np.float32)
        inliers = np.array(0)
        # print(H)
        return H, inliers
        
    H, inliers = cv2.estimateAffinePartial2D(matched_pts1[:, [0, 1]],
                                             matched_pts2[:, [0, 1]],
                                             cv2.RANSAC)
                                 
    # print(H)
    inliers = inliers.flatten()
    #print("transformation matrix:", H)
    return H, inliers


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
    if len(matched_pts1) == 0 or len(matched_pts1)==0:
        H = np.array([[1, 0, 0],[0, 1, 0]], dtype=np.float32)
        inliers = np.array(0)
        # print(H)
        return H, inliers                
    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [0, 1]],
                                    matched_pts2[:, [0, 1]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers



def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file,  cv2.IMREAD_COLOR)
    #print(img.shape)
    img = cv2.resize(img, img_size)
    img_orig = img.copy()
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)   
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.
    return img_preprocessed, img_orig


def compute_sift(i1, i2 ):
    # img1_orig, img1_orig
    sift_kp1, sift_desc1 = extract_SIFT_keypoints_and_descriptors(i1)
    sift_kp2, sift_desc2 = extract_SIFT_keypoints_and_descriptors(i2)
    if len(sift_kp2) == 0 or len(sift_kp2) == 0:
        sift_Hr = np.array([[1, 0, 0],[0, 1, 0]], dtype=np.float32)
        sift_inliersr = np.array(0)
        sift_matched_img = np.zeros((256,512))
        # print(sift_Hr.shape)
        return sift_Hr, sift_inliersr, sift_matched_img
    sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
    sift_kp1, sift_desc1, sift_kp2, sift_desc2)
    if use_homography:
        sift_Hr, sift_inliersr = compute_homography(sift_m_kp1, sift_m_kp2)
    else:
        sift_Hr, sift_inliersr = compute_rigid(sift_m_kp1, sift_m_kp2)
    sift_matches = np.array(sift_matches)[sift_inliersr.astype(bool)].tolist()
    sift_matched_img = cv2.drawMatches(img1_orig, sift_kp1, img2_orig,
                                        sift_kp2, sift_matches, None,
                                        matchColor=(0, 255, 0),
                                        singlePointColor=(0, 0, 255))
    if sift_Hr is None:
        sift_Hr = np.array([[1, 0, 0],[0, 1, 0]], dtype=np.float32)
        sift_inliersr = np.array(0)
        sift_matched_img = np.zeros((256,512))
        # print(sift_Hr.shape)
        return sift_Hr, sift_inliersr, sift_matched_img
    return  sift_Hr, sift_inliersr, sift_matched_img


def load_tensorflow_model(p_weights_dir):
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(graph= graph,config=config)
    tf.saved_model.loader.load(session,
                            [tf.saved_model.tag_constants.SERVING],
                            str(p_weights_dir))
    input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
    output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
    output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')
    return session ,input_img_tensor, output_prob_nms_tensor, output_desc_tensors


def compute_superpoint(i1, i2, p_sess, pinput_img_tensor, poutput_prob_nms_tensor, poutput_desc_tensors):
    out1 = p_sess.run([poutput_prob_nms_tensor, poutput_desc_tensors],
                        feed_dict={pinput_img_tensor: np.expand_dims(i1, 0)})
    keypoint_map1 = np.squeeze(out1[0])
    descriptor_map1 = np.squeeze(out1[1])
    kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
                            keypoint_map1, descriptor_map1, keep_k_best)
    out2 = p_sess.run([poutput_prob_nms_tensor, poutput_desc_tensors],
                       feed_dict={pinput_img_tensor: np.expand_dims(i2, 0)})
    keypoint_map2 = np.squeeze(out2[0])
    descriptor_map2 = np.squeeze(out2[1])
    kp2, desc2 = extract_superpoint_keypoints_and_descriptors(
                keypoint_map2, descriptor_map2, keep_k_best)

    # Match and get rid of outliers

    m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
    if use_homography:
        Hr, inliersr = compute_homography(m_kp1, m_kp2)
    else:
        Hr, inliersr = compute_rigid(m_kp1, m_kp2)
    matches = np.array(matches)[inliersr.astype(bool)].tolist()
    matched_img = cv2.drawMatches(img1_orig, kp1, img2_orig, kp2, matches,
                                 None, matchColor=(0, 255, 0),
                                 singlePointColor=(0, 0, 255))
           
    return Hr, inliersr, matched_img
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('weights_name', type=str)
    # parser.add_argument('img1_path', type=str, )
    # parser.add_argument('img2_path', type=str)
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--H', type=int, default=256,
                        help='The height in pixels to resize the images to. \
                                (default: 384)')
    parser.add_argument('--W', type=int, default=256,
                        help='The width in pixels to resize the images to. \
                                (default: 384)')
    parser.add_argument('--k_best', type=int, default=1000,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    args = parser.parse_args()
    weights_name = args.weights_name
    use_homography = False
    # img1_file = args.img1_path
    # img2_file = args.img2_path
    
    img_size = (args.W, args.H)
    keep_k_best = args.k_best
    exp_name = args.exp_name

    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)


    methods = ['nostyle','comir','unet_aligned','cyclegan_aligned','cut_aligned','pix2pix_aligned','cyclegan_unaligned','cut_unaligned']
    # methods = ['nostyle','pix2pix_aligned','cut_aligned','unet_aligned']

    sess, input_img_tensor, output_prob_nms_tensor, output_desc_tensors = load_tensorflow_model(weights_dir)
    for method in methods:
        path = os.path.abspath('./')
        save_sift_path = f'results/{exp_name}/{method}/sift/'
        save_sp_path = f'results/{exp_name}/{method}/sp/'
        save_corr_path = f'results/{exp_name}/{method}/corr/'
        save_sift_json = f"{save_sift_path}json/"
        save_sp_json = f"{save_sp_path}json/"
        save_corr_json = f"{save_corr_path}json/"
        print(method)
        for name in [save_sift_path, save_sp_path, save_corr_path,save_sift_json,save_sp_json, save_corr_json]:
            if not os.path.exists(path+'/'+name):
                os.makedirs(path+'/'  + name)


        if method == 'pix2pix_aligned':
            p2p_fake_images = './datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_IHC_aligned_1f/test_latest/images/*_fake_B.png' 
            p2p_path = './datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_IHC_aligned_1f/test_latest/images/'
            images = glob.glob(f"{p2p_path}*_fake_B.png")
            # fix_p = f'{p2p_path}/{fname}{ext}'.replace("fake_B","real_B")
        elif method == 'cyclegan_aligned':
            p2p_fake_images = './datasets//pytorch-CycleGAN-and-pix2pix/results/cycle_IHC_mode_aligned/test_latest/images/*_fake.png'
            p2p_path = './datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_IHC_aligned_1f/test_latest/images/'
            images = glob.glob(p2p_fake_images)
            # fix_p = f'{p2p_path}/{fname}{ext}'.replace("fake","real_B")
        elif method == 'comir':
            p2p_fake_images = './datasets//CoMIR/results/export_IHC/*R2.tif'
            p2p_path = './datasets//CoMIR/results/export_IHC'
            images = glob.glob(p2p_fake_images)
            # fix_p = f'{p2p_path}/{fname}{ext}'.replace("fake","real_B")    
        elif method =='cut_aligned':
            p2p_fake_images = './datasets//contrastive-unpaired-translation/results/cut_IHC_aligned_mode_aligned/test_latest/images/fake_B/*.png'
            p2p_path = './datasets//contrastive-unpaired-translation/results/cut_IHC_aligned_mode_aligned/test_latest/images/real_B/'
            images = glob.glob(p2p_fake_images)
        elif method == 'unet_aligned':
            p2p_fake_images = '/storage01/grexai/dev/Unet-torch/test_config_IHC/*.png'
            p2p_path = '/storage01/grexai/datasets/imgreg/Image_registration/IHC/test/trainB/'
            images = glob.glob(p2p_fake_images)
            # fix_p = f'{img2_file}/{fname}{ext}'
            # fix_p = f'{p2p_path}/{fname}{ext}'.replace("fake","real_B")
        elif method == 'pix2pix_unaligned':
            p2p_fake_images = './datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_IHC_unaligned_1f/test_latest/images/*_fake_B.png' 
            p2p_path = './datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_IHC_unaligned_1f/test_latest/images/'
            images = glob.glob(f"{p2p_path}*_fake_B.png")
            # fix_p = f'{p2p_path}/{fname}{ext}'.replace("fake_B","real_B")
        elif method == 'cyclegan_unaligned':
            p2p_fake_images = './datasets//pytorch-CycleGAN-and-pix2pix/results/cycle_IHC_unaligned/test_latest/images/*_fake.png'
            p2p_path = './datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_IHC_unaligned_1f/test_latest/images/'
            images = glob.glob(p2p_fake_images)
            # fix_p = f'{p2p_path}/{fname}{ext}'.replace("fake","real_B")
        elif method =='cut_unaligned':
            p2p_fake_images = './datasets//contrastive-unpaired-translation/results/cut_IHC_unaligned/train_latest/images/fake_B/*.png'
            p2p_path = './datasets//contrastive-unpaired-translation/results/cut_IHC_unaligned/train_latest/images/real_B/'
            images = glob.glob(p2p_fake_images)    
        elif method == 'nostyle':
            p2p_fake_images  = '/storage01/grexai/datasets/imgreg/Image_registration/IHC/test2/trainA/*.png' 
            p2p_path ='/storage01/grexai/datasets/imgreg/Image_registration/IHC/test/trainB/'
            images = glob.glob(p2p_fake_images)
        else:
            continue
        for image in tqdm(images):
            fname, ext = os.path.splitext(image)
            path, fname = os.path.split(fname)
        
            if method == 'pix2pix_aligned' or method ==  'pix2pix_unaligned':
                fix_p = f'{p2p_path}/{fname}{ext}'.replace("fake_B","real_B")
            elif method == 'cyclegan_aligned' or method ==  'cyclegan_unaligned':                
                fix_p = f'{p2p_path}/{fname}{ext}'.replace("fake","real_B")
            elif method == 'comir':
                fix_p = f'{p2p_path}/{fname}{ext}'.replace("R2","R1")    
            elif method =='cut_aligned' or method ==  'cut_unaligned':
                fix_p = f'{p2p_path}/{fname}{ext}'
            elif method =='unet_aligned' or method ==  'unet_unaligned':
                fix_p = f'{p2p_path}/{fname}{ext}'    
            elif method == 'nostyle':
                fix_p = f'{p2p_path}/{fname}{ext}'
            else:
                continue

            

            """
            normals
            CUDA_VISIBLE_DEVICES=0 python3 calc_sp_sift_corr.py superpoint_tissue_finetuned '/storage01/grexai/datasets/imgreg/Image_registration/Aligned/test/trainA/*.png' /storage01/grexai/datasets/imgreg/Image_registration/Aligned/test/trainB/
            UNET:
            CUDA_VISIBLE_DEVICES=0 python3 calc_sp_sift_corr.py superpoint_tissue_finetuned '/storage01/grexai/dev/Unet-torch/saved_images/*.png' /storage01/grexai/datasets/imgreg/Image_registration/Aligned/test/notaligned_samescale_bias/
            """
            #fix_p = f'{img2_file}/{fname}{ext}'

            '''
            Comir
            python3 calc_sp_sift_corr.py superpoint_tissue_finetuned './datasets//CoMIR/results/export/*R1.tif' ./datasets//CoMIR/results/export

            '''
            #fix_p = f'{img2_file}/{fname}{ext}'.replace("R2","R1")
            
            '''
            #pix2pix
            CUDA_VISIBLE_DEVICES=1 python3 calc_sp_sift_corr.py superpoint_tissue_finetuned './datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_hela63x_/test_latest/images/*_fake_B.png' ./datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_hela63x_/test_latest/images/
            '''
            
            #fix_p = f'{img2_file}/{fname}{ext}'.replace("fake_B","real_B")
            
            '''
            #cyclegan
            HeLa
            CUDA_VISIBLE_DEVICES=1 python3 calc_sp_sift_corr.py superpoint_tissue_finetuned './datasets//pytorch-CycleGAN-and-pix2pix/results/cycle_hela63_unaligned_scale_crop/test_latest/images/*_fake.png' ./datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_hela63x_/test_latest/images/
            HELLA
            CUDA_VISIBLE_DEVICES=1 python3 calc_sp_sift_corr.py superpoint_tissue_finetuned './datasets//pytorch-CycleGAN-and-pix2pix/results/cycle_b18z40-63x_resize_aligned/test_latest/images/*_fake.png' ./datasets//pytorch-CycleGAN-and-pix2pix/results/p2p_b18zHella40-63x/test_latest/images/
            '''
            #fix_p = f'{img2_file}/{fname}{ext}'.replace("fake","real_B")
            

            """
            CUT
            example run
            CUDA_VISIBLE_DEVICES=1 python3 calc_sp_sift_corr.py superpoint_tissue_finetuned './datasets//contrastive-unpaired-translation/results/cut_hela63_unaligned_resize/train_latest/images/fake_B/*.png' ./datasets//contrastive-unpaired-translation/results/cut_hela63_unaligned_resize/train_latest/images/real_B/
            """

            # fix_p = f'{img2_file}/{fname}{ext}'
           
            # print(image)
            # print(fix_p)
            

            # print(os.path.exists(image), os.path.exists(fix_p))
            # original registration
            # img1, img1_orig = preprocess_image(image, img_size)
            # img2, img2_orig = preprocess_image(fix_p, img_size)
            # change fix and moving
            img1, img1_orig = preprocess_image(image, img_size)
            img2, img2_orig = preprocess_image(fix_p, img_size)

            Hr, inliersr, matched_img  = compute_superpoint(img2, img1, sess, input_img_tensor, output_prob_nms_tensor, output_desc_tensors)
            
            
            sift_Hr, sift_inliersr, sift_matched_img = compute_sift(img2_orig, img1_orig)

            corr_warped, c_shift, c_angle, c_scale = calculate_correlation_and_warp_image(img1_orig,img2_orig, do_rotation=True)
            # print(f"{c_angle}, {c_shift}, {c_scale}")
            
            base = os.path.basename(image)
            base2 = os.path.basename(fix_p)
            base = os.path.splitext(base)[0]
            base2 = os.path.splitext(base2)[0]
            fname = base + "_" + base2 + ".png"
            # matches
            cv2.imwrite(save_sp_path + "matches_" + fname, matched_img)
            cv2.imwrite(save_sift_path + "matches_"+fname, sift_matched_img)

            # WARP
            #rigid
            
            # homography
            if use_homography:
                wp1 = cv2.warpPerspective(img2_orig,Hr,(img1_orig.shape[0], img1_orig.shape[1]),flags=cv2.INTER_LINEAR)
            else:
                wp1 = cv2.warpAffine(img2_orig, Hr, (img1_orig.shape[0], img1_orig.shape[1]))
            # print(img2_orig.shape, sift_Hr.shape, (img1_orig.shape[0],img1_orig.shape[1]))
            #
            if use_homography:
                wp3 = cv2.warpPerspective(img2_orig,sift_Hr,(img1_orig.shape[0], img1_orig.shape[1]),flags=cv2.INTER_LINEAR)
            else:
                wp3 = cv2.warpAffine(img2_orig, sift_Hr, (img1_orig.shape[0],img1_orig.shape[1]))
            # cv2.imwrite(save_sp_path +"warped_" + fname, wp1)
            # cv2.imwrite(save_sift_path + "warped" + fname, wp3)
            # cv2.imwrite(save_corr_path + "warped" + fname, corr_warped)

            # OVERLAY
            overlay_SUPER = cv2.addWeighted(img1_orig, 0.5, wp1, 0.5, 0.7)
            overlay_SIFT = cv2.addWeighted(img1_orig,0.5,wp3,0.5,0.7)

            img1_gray = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY)
            corr_warped =  cv2.normalize(corr_warped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            corr_warped =  corr_warped.astype(np.uint8)

            # print(f"{img1_gray.dtype} , {corr_warped.dtype}")
            
            overlay_corr = cv2.addWeighted(img1_gray, 0.5, corr_warped, 0.5, 0.7)
            # ol_SP_real = cv2.addWeighted(img3_orig, 0.5, wp1, 0.5, 0.7)
            cv2.imwrite(save_sp_path+"SP_rigid" + fname, overlay_SUPER)
            cv2.imwrite(save_sift_path+"sift_rigid" + fname, overlay_SIFT)
            cv2.imwrite(save_corr_path+"corr_rigid" + fname, overlay_corr)
            # cv2.imwrite("SP_real_OL" + fname, ol_SP_real)

            corner_points = [[0, 0, 1], [0, 1024, 1], [1024, 1024, 1], [1024, 0, 1]]
            corner_points = np.array(corner_points).T
            # print(corner_points)
            cp2 = np.array([[-512, -512, 1], [-512, 512, 1], [512, 512, 1], [512, -512, 1]]).T
            radangle = math.radians(c_angle)
            rot_matrix = np.array([[math.cos(radangle), -math.sin(radangle),0],[math.sin(radangle),math.cos(radangle),0],[0,0,1]])
            rotated = rot_matrix@cp2
            translation_matrix = np.array([[1,0,c_shift[0]+512],[0,1,c_shift[1]+512],[0,0,1]])
            final = translation_matrix@rotated
            # print(final)
            # transform = AffineTransform(translation=c_shift)
            # corr_t = warp(corner_points, transform.inverse)
            #corr_t = rotate(corr_t, c_angle)
    
            result_coor = sift_Hr @ corner_points
            #print(result_coor)

            listH = Hr.tolist()
            with open(save_sp_json + base + '.json', 'w', encoding='utf-8') as f:
                json.dump(listH, f, indent=2)
            listSH = sift_Hr.tolist()
            with open(save_sift_json+base + '.json', 'w', encoding='utf-8') as f:
                json.dump(listSH, f, indent=2)
            listc = [c_shift[0], c_shift[1], c_angle, c_scale]
            with open(save_corr_json + base + '.json', 'w', encoding='utf-8') as f:
                json.dump(listc, f, indent=2)

