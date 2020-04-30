import sys
import numpy as np
import cv2
import skimage.io
import matplotlib.pyplot as plt

import utils, ransac


###############################################################################
#   CONFIG
###############################################################################
corner_detector = 'harris'

ransac_params = {'s':3, 'N':10, 'd':0.5, 'T':10}

###############################################################################
image_set_dir = "./Images/Pair-1/"
img1_rgb = skimage.io.imread(image_set_dir+'left.png')
img2_rgb = skimage.io.imread(image_set_dir+'right.png')
img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)

if corner_detector == 'harris':
    # Get Harris corners: np array of (row,col) pairs each representing a point
    img1_kpts, img2_kpts = utils.get_Harris_pts(img1_gray, img2_gray)
    # Convert to list of cv2 KeyPoint objects
    img1_kpts = utils.cvt_to_cv2KeyPoints(img1_kpts)
    img2_kpts = utils.cvt_to_cv2KeyPoints(img2_kpts)

if corner_detector == 'sift':
    # Get SIFT corners: list of cv2.KeyPoint() objects
    img1_kpts = sift.detect(img1_gray,None)
    img2_kpts = sift.detect(img2_gray,None)

# Extract descriptors using the images and their keypoints
img1_kpts, img1_descriptors = sift.compute(img1_gray,img1_kpts)
img2_kpts, img2_descriptors = sift.compute(img2_gray,img2_kpts)

# Normalize the descriptors
img1_descriptors = utils.normalize(img1_descriptors)
img2_descriptors = utils.normalize(img2_descriptors)


# Get similarity matrices
#   - High similarity = Low euc distance, High correlation
#   - Shape: [n_img1_kpts, n_img2_kpts]
euc_distance_matrix = utils.compute_euclidean_distances(img1_descriptors, img2_descriptors)
correlation_matrix = utils.compute_correlation(img1_descriptors, img2_descriptors)

# Matching keypoint pair indices.
matching_kpt_pair_indices = utils.get_matchings(euc_distance_matrix,
                                                similarity_type='euc_distance',
                                                threshold=0.3)

# Visualize keypoints and matchings
# utils.display_image_and_keypts(img1_gray, img2_gray,
#                                img1_kpts, img2_kpts,
#                                matching_kpt_pair_indices)

# Perform RANSAC
affine_matrix = ransac.apply_RANSAC(img1_kpts, img2_kpts,
                                    matching_kpt_pair_indices,
                                    ransac_params)