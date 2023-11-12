from pathlib import Path
#from this import d

import cv2
import numpy as np
import tensorflow as tf  # noqa: E402
import os
# from settings import EXPER_PATH  # noqa: E402
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
# EXPER_PATH = "./exper"
EXPER_PATH = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
    H, inliers = cv2.estimateAffinePartial2D(matched_pts1[:, [0, 1]],
                                             matched_pts2[:, [0, 1]],
                                             cv2.RANSAC)
    inliers = inliers.flatten()
    #print("transformation matrix:", H)
    return H, inliers


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

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
    """
    img preprocessing
    returns the original resized image, and a prepared image for sp

    """
    # img = cv2.imread(img_file,  cv2.IMREAD_COLOR)
    img = img_file
    img = cv2.resize(img, img_size)
    img_orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.
    return img_preprocessed, img_orig


def load_tensorflow_model(p_weights_dir):
    """
    loads tensorflow model: input graph path
    outputs: tf session,  img input tensor, output nms tensor, output prob descriptor tensor,
    """

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(graph=graph, config=config)
    print(f"p_weights_dir = {p_weights_dir}")
    weights_root_dir = Path(EXPER_PATH, '')
    weights_path = Path(weights_root_dir, p_weights_dir)
    tf.saved_model.loader.load(session,
                               [tf.saved_model.tag_constants.SERVING],
                               str(weights_path))
    input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
    output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
    output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')
    global __model
    __model = {
        "session": session,
        "input_img_tensor": input_img_tensor,
        "output_prob_nms_tensor": output_prob_nms_tensor,
        "output_desc_tensors": output_desc_tensors
    }


def compute_superpoint(i1, i2, keep_k_best=1000):
    """
    calculates superpoint feautres for 2 images and estimates a rigid transformation matrix
    inputs: image1 , image 2, points to keep default =1000
    from load model function:
    tensorflow session, img input tensor, output nms tensor, output prob tesnor,
    returns a rigid Hr matrix 2x3
    """

    global __model
    session = __model["session"]
    input_img_tensor = __model["input_img_tensor"]
    output_prob_nms_tensor = __model["output_prob_nms_tensor"]
    output_desc_tensors = __model["output_desc_tensors"]

    out1 = session.run([output_prob_nms_tensor, output_desc_tensors],
                      feed_dict={input_img_tensor: np.expand_dims(i1, 0)})
    keypoint_map1 = np.squeeze(out1[0])
    descriptor_map1 = np.squeeze(out1[1])
    kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
        keypoint_map1, descriptor_map1, keep_k_best)
    out2 = session.run([output_prob_nms_tensor, output_desc_tensors],
                      feed_dict={input_img_tensor: np.expand_dims(i2, 0)})
    keypoint_map2 = np.squeeze(out2[0])
    descriptor_map2 = np.squeeze(out2[1])
    kp2, desc2 = extract_superpoint_keypoints_and_descriptors(
        keypoint_map2, descriptor_map2, keep_k_best)

    # Match and get rid of outliers
    m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)

    # H, inliers = compute_homography(m_kp1, m_kp2)
    Hr, inliersr = compute_rigid(m_kp1, m_kp2)

    matches = np.array(matches)[inliersr.astype(bool)].tolist()
    matched_img = cv2.drawMatches(i1, kp1, i2, kp2, matches,
                                  None, matchColor=(0, 255, 0),
                                  singlePointColor=(0, 0, 255))
    cv2.imwrite("SuperPoint matches.png", matched_img)

    return Hr




