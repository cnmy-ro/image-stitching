import sys
import numpy as np
import cv2
import skimage.io
import matplotlib.pyplot as plt

import utils, ransac, visualizer

np.random.seed(100)

###############################################################################
#   CONFIG
###############################################################################
corner_detector = 'harris'

ransac_imp = 'custom'   # 'custom' / 'opencv'
ransac_params = {'s':3, 'N':100, 'd':0.5, 'T':5}

###############################################################################
image_set_dir = "./Images/Pair-2/"
img1_rgb = skimage.io.imread(image_set_dir+'left.jpeg')
img1_rgb = img1_rgb[:,:,:3]
img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)

img2_rgb = skimage.io.imread(image_set_dir+'right.jpeg')
img2_rgb = img2_rgb[:,:,:3]
img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

vis = visualizer.Visualizer(img1_rgb, img2_rgb, save_figs=True)

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

vis.set_keypoints(img1_kpts, img2_kpts)

# Normalize the descriptors
img1_descriptors = utils.normalize(img1_descriptors)
img2_descriptors = utils.normalize(img2_descriptors)


# Get similarity matrices
#   - High similarity = Low euc distance, High correlation
#   - Shape: [n_img1_kpts, n_img2_kpts]
euc_distance_matrix = utils.compute_euclidean_distances(img1_descriptors, img2_descriptors)
correlation_matrix = utils.compute_correlation(img1_descriptors, img2_descriptors)

# Matching keypoint pair indices.
matching_kpt_pair_indices = utils.get_matchings(correlation_matrix,
                                                similarity_type='correlation',
                                                threshold=0.97)

# Visualize keypoints and matchings
vis.set_matches(matching_kpt_pair_indices)
vis.show_keypoints(best_matches=True)


# Perform RANSAC to obtain the affine matrix
if ransac_imp == 'custom':
    affine_matrix = ransac.apply_RANSAC(img1_kpts, img2_kpts,
                                        matching_kpt_pair_indices,
                                        ransac_params)
if ransac_imp == 'opencv':
    affine_matrix = ransac.apply_RANSAC_opencv(img1_kpts, img2_kpts, matching_kpt_pair_indices)


# Apply Affine transform
img2_warped = cv2.warpPerspective(img2_rgb, affine_matrix, (img1_gray.shape[1]+img2_gray.shape[1],img2_gray.shape[0]))

vis.stitch_and_display(img2_warped)