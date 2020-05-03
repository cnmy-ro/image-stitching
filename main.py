import os, sys, argparse
import numpy as np
import cv2

import utils, descriptors, ransac, visualizer

np.random.seed(100)

###############################################################################

def run_app(img1_path, img2_path,
            descriptor, patch_size,
            matching_threshold,
            ransac_sample_size, ransac_n_iterations, ransac_tolerance, ransac_inlier_fraction_threshold,
            experiment_id=0,
            save_results=False):


    # Prepare the results directory
    results_dir = None
    if experiment_id:
        results_dir = "./Results/Exp-" + experiment_id + "/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with open(results_dir+"config_info.txt", 'w') as info_file:
            config = """CONFIG --\n
                        Descriptor: {}\n
                        Patch size: {}\n
                        RANSAC sample size: {}\n
                        RANSAC N iterations: {}\n
                        RANSAC tolerance: {}\n
                        RANSAC inlier fraction threshold: {}""".format(descriptor,patch_size,ransac_sample_size, ransac_n_iterations, ransac_tolerance, ransac_inlier_fraction_threshold)
            info_file.write(config)


    # Read and prepare the images
    img1 = cv2.imread(img1_path)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)

    img2 = cv2.imread(img2_path)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

    # Create the visualizer object
    vis = visualizer.Visualizer(img1_rgb, img2_rgb, save_results, results_dir)

    # Create sift object (used only if enabled)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)

    # -------------------------------------------------------------------------
    # Corner detection

    img1_kpts, img2_kpts = utils.get_Harris_corners(img1_gray, img2_gray)

    vis.set_keypoints(img1_kpts, img2_kpts)
    vis.draw_keypoints()

    # -------------------------------------------------------------------------
    # Extract descriptors

    if descriptor == 'opencv_sift':
        _, img1_descriptors = sift.compute(img1_gray, utils.cvt_to_cv2KeyPoints(img1_kpts))
        _, img2_descriptors = sift.compute(img2_gray, utils.cvt_to_cv2KeyPoints(img2_kpts))

    elif descriptor == 'custom_gray_intensities':
        img1_descriptors = descriptors.gray_intensities(img1_gray, img1_kpts, patch_size)
        img2_descriptors = descriptors.gray_intensities(img2_gray, img2_kpts, patch_size)

    elif descriptor == 'custom_rgb_intensities':
        img1_descriptors = descriptors.rgb_intensities(img1_rgb, img1_kpts, patch_size)
        img2_descriptors = descriptors.rgb_intensities(img2_rgb, img2_kpts, patch_size)

    # Normalize the descriptors
    img1_descriptors = utils.normalize(img1_descriptors)
    img2_descriptors = utils.normalize(img2_descriptors)

    # -------------------------------------------------------------------------
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
    vis.draw_matches(title="Matching pairs of keypoints")

    # -------------------------------------------------------------------------
    # Perform RANSAC to obtain the affine matrix

    ransac_estimator = ransac.RANSAC_Estimator(ransac_sample_size,
                                               ransac_n_iterations,
                                               ransac_tolerance,
                                               ransac_inlier_fraction_threshold)

    affine_matrix, avg_residual, inlier_indices = ransac_estimator.estimate_affine_matrix(img1_kpts, img2_kpts,
                                                                                          matching_kpt_pair_indices)

    metrics = {'inlier-fraction': None,
               'avg-inlier-residual': None,
               'avg-inlier-euc-dist': None}

    metrics['avg-inlier-residual'] = avg_residual
    metrics['inlier-fraction'] = inlier_indices.shape[0]/matching_kpt_pair_indices.shape[0]
    metrics['avg-inlier-euc-dist'] = ransac.evaluate_model(affine_matrix, img1_kpts, img2_kpts, inlier_indices)

    print("Avg residual for the inliers (before refitting): {:.3f}".format(metrics['avg-inlier-residual']))
    print("Inlier fraction: {:.3f}".format(metrics['inlier-fraction']))
    print("Average Euclidean distance: {:.3f}".format(metrics['avg-inlier-euc-dist']) )

    vis.set_inliers(inlier_indices)

    # -------------------------------------------------------------------------
    # Apply Affine transform

    img2_warped = cv2.warpPerspective(img2_rgb, affine_matrix, (img1_gray.shape[1]+img2_gray.shape[1],img2_gray.shape[0]))
    vis.draw_matches(title="Inliers (blue) and Outliers (red)")
    vis.stitch_and_display(img2_warped, display_all=True)


###############################################################################
#  Run the panorama application
###############################################################################

if __name__ == '__main__':

    image_set_dir = "./Images/Pair-4/"
    image_paths = [ image_set_dir + filename for filename in sorted(os.listdir(image_set_dir)) ]
    img1_path, img2_path = image_paths[0:2]

    descriptor = 'custom_gray_intensities'  # 'custom_gray_intensities' / 'custom_rgb_intensities'/ 'opencv_sift'
    patch_size = 7

    matching_threshold = 0.98

    ransac_sample_size = 5
    ransac_n_iterations = 100
    ransac_tolerance =20 # Tolerance value for a pair to be considered an inlier
    ransac_inlier_fraction_threshold = 0.2 # Fraction of the total matching pairs that are inliers.

    run_app(img1_path, img2_path,
            descriptor, patch_size,
            matching_threshold,
            ransac_sample_size, ransac_n_iterations, ransac_tolerance, ransac_inlier_fraction_threshold,
            experiment_id='1',
            save_results=True)