import sys
import numpy as np
import cv2
import skimage.io
import matplotlib.pyplot as plt

import utils, descriptors, ransac, visualizer

np.random.seed(100)

###############################################################################
#   CONFIG
###############################################################################
corner_detector = 'harris' # 'harris' / 'sift'
descriptor = 'custom_rgb_intensities'  # 'custom_gray_intensities' / 'custom_rgb_intensities'/ 'opencv_sift'
patch_size = 7

matching_threshold = 0.995

ransac_implementation = 'custom'   # 'custom' / 'opencv's
sample_size = 5
n_iterations = 100
tolerance = 10
T_frac = 0.4  # Fraction of the no. of matched kpts


###############################################################################
# Define the metrics for RANSAC

metrics = {'inlier-outlier-ratio': None,
           'avg-residual': None,
           'avg-euc-dist-all-matches': None}      # Avg of Euc dist between transformed img2 kpts and the correspnding matching img1 kpts

###############################################################################
image_set_dir = "./Images/Pair-1/"
img1_rgb = skimage.io.imread(image_set_dir+'left.png')
img1_rgb = img1_rgb[:,:,:3]
img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)

img2_rgb = skimage.io.imread(image_set_dir+'right.png')
img2_rgb = img2_rgb[:,:,:3]
img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

vis = visualizer.Visualizer(img1_rgb, img2_rgb, save_figs=False)

sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)

###############################################################################
# Corner detection
if corner_detector == 'harris':
    # Get Harris corners: np array of (row,col) pairs each representing a point
    img1_kpts, img2_kpts = utils.get_Harris_corners(img1_gray, img2_gray)
    # Convert to list of cv2 KeyPoint objects
    img1_kpts = utils.cvt_to_cv2KeyPoints(img1_kpts)
    img2_kpts = utils.cvt_to_cv2KeyPoints(img2_kpts)

if corner_detector == 'sift':
    # Get SIFT corners: list of cv2.KeyPoint() objects
    img1_kpts = sift.detect(img1_gray,None)
    img2_kpts = sift.detect(img2_gray,None)

vis.set_keypoints(img1_kpts, img2_kpts)
vis.draw_keypoints()

###############################################################################
# Extract descriptors using the images and their keypoints
if descriptor == 'opencv_sift':
    img1_kpts, img1_descriptors = sift.compute(img1_gray,img1_kpts)
    img2_kpts, img2_descriptors = sift.compute(img2_gray,img2_kpts)

elif descriptor == 'custom_gray_intensities':
    img1_descriptors = descriptors.gray_intensities(img1_gray, img1_kpts, patch_size)
    img2_descriptors = descriptors.gray_intensities(img2_gray, img2_kpts, patch_size)

elif descriptor == 'custom_rgb_intensities':
    img1_descriptors = descriptors.rgb_intensities(img1_rgb, img1_kpts, patch_size)
    img2_descriptors = descriptors.rgb_intensities(img2_rgb, img2_kpts, patch_size)

# Normalize the descriptors
img1_descriptors = utils.normalize(img1_descriptors)
img2_descriptors = utils.normalize(img2_descriptors)

###############################################################################
# Get similarity matrices
#   - High similarity = Low euc distance, High correlation
#   - Shape: [n_img1_kpts, n_img2_kpts]
euc_distance_matrix = utils.compute_euclidean_distances(img1_descriptors, img2_descriptors)
correlation_matrix = utils.compute_correlation(img1_descriptors, img2_descriptors)

# Matching keypoint pair indices.
matching_kpt_pair_indices = utils.get_matchings(correlation_matrix,
                                                similarity_type='correlation',
                                                threshold=matching_threshold)


# Visualize matchings
vis.set_matches(matching_kpt_pair_indices)
vis.draw_matches(title="All matches | Patch size: {} | Correlation threshold: {}".format(patch_size, matching_threshold))

###############################################################################
# Perform RANSAC to obtain the affine matrix
if ransac_implementation == 'custom':
    inlier_threshold = round(T_frac * matching_kpt_pair_indices.shape[0])
    ransac_estimator = ransac.RANSAC_Estimator(sample_size, n_iterations, tolerance, inlier_threshold)

    affine_matrix, avg_residual, inlier_indices = ransac_estimator.estimate_affine_matrix(img1_kpts, img2_kpts,
                                                                                          matching_kpt_pair_indices)

    metrics['avg-residual'] = avg_residual
    metrics['inlier-outlier-ratio'] = inlier_indices.shape[0]/(matching_kpt_pair_indices.shape[0]-inlier_indices.shape[0])
    metrics['avg-euc-dist-all-matches'] = ransac.evaluate_model(affine_matrix, img1_kpts, img2_kpts, matching_kpt_pair_indices)

    print("Avg residual for the inliers: {:.3f}".format(metrics['avg-residual']))
    print("Inlier to Outlier ratio: {:.3f}".format(metrics['inlier-outlier-ratio']))
    print("Average Euclidean distance: {:.3f}".format(metrics['avg-euc-dist-all-matches']) )

    vis.set_inliers(inlier_indices)


if ransac_implementation == 'opencv':
    affine_matrix = ransac.apply_RANSAC_opencv(img1_kpts, img2_kpts, matching_kpt_pair_indices)

###############################################################################
# Apply Affine transform
img2_warped = cv2.warpPerspective(img2_rgb, affine_matrix, (img1_gray.shape[1]+img2_gray.shape[1],img2_gray.shape[0]))
vis.draw_matches(title="Inliers and Outliers among all matches")
vis.stitch_and_display(img2_warped, display_all=False)