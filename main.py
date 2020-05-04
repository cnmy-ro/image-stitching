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
            visualize=True,
            experiment_id=None,
            case_id=None):


    # Prepare the results directory
    save_results = False
    results_dir = None
    if experiment_id:
        save_results = True
        results_dir = "./Results/Exp-" + experiment_id + "/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with open(results_dir+"result.txt", 'a') as result_file:
            result_file.write(case_id+'\n')
            config = "Config --\nDescriptor: {}\nPatch size: {}\nRANSAC sample size: {}\nRANSAC iterations: {}\nRANSAC tolerance: {}\nRANSAC inlier fraction threshold: {}".format(descriptor,patch_size,ransac_sample_size, ransac_n_iterations, ransac_tolerance, ransac_inlier_fraction_threshold)
            result_file.write(config)


    # Read and prepare the images
    img1 = cv2.imread(img1_path)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)

    img2 = cv2.imread(img2_path)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

    # Create the visualizer object
    vis = visualizer.Visualizer(img1_rgb, img2_rgb, visualize, save_results, results_dir, case_id)

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

    if experiment_id:
        with open(results_dir+"result.txt", 'a') as result_file:
            result = "\n\nResult --\nAverage inlier residual (before refitting): {:.3f}\nFraction of inliers: {:.3f}\nAverage euclidean distance: {:.3f}".format( metrics['avg-inlier-residual'], metrics['inlier-fraction'], metrics['avg-inlier-euc-dist'])
            result_file.write(result)
            result_file.write("\n\n----------------------------------------------\n\n")

    print("Avg residual for the inliers (before refitting): {:.3f}".format(metrics['avg-inlier-residual']))
    print("Inlier fraction: {:.3f}".format(metrics['inlier-fraction']))
    print("Average Euclidean distance: {:.3f}".format(metrics['avg-inlier-euc-dist']) )

    vis.set_inliers(inlier_indices)

    # -------------------------------------------------------------------------
    # Apply Affine transform

    img2_warped = cv2.warpPerspective(img2_rgb, affine_matrix, (img1_gray.shape[1]+img2_gray.shape[1],img2_gray.shape[0]))
    vis.draw_matches(title="Inliers (blue) and Outliers (red)")
    vis.stitch_and_display(img2_warped, display_all=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img1', type=str, help="Path to image 1",
                        default="./Images/Pair-6/1.jpeg")
    parser.add_argument('--img2', type=str, help="Path to image 2",
                        default="./Images/Pair-6/2.jpeg")
    parser.add_argument('--descriptor', type=str, help="Options: 'custom_gray_intensities', custom_rgb_intensities, 'opencv_sift'",
                        default='custom_rgb_intensities')
    parser.add_argument('--patch_size', type=int, help="5,7,9,...",
                        default=5)
    parser.add_argument('--matching_threshold', type=float, help="Minimum correlation value for a kpt descriptor pair to match",
                        default=0.990)
    parser.add_argument('--ransac_sample_size', type=int, help="3,4,5,...",
                        default=3)
    parser.add_argument('--ransac_n_iterations', type=int,
                        default=100)
    parser.add_argument('--ransac_tolerance', type=int, help="Tolerance value for a kpt pair to be considered an inlier",
                        default=40)
    parser.add_argument('--ransac_inlier_fraction_threshold', type=float, help="Fraction of the total matching pairs that need to be inliers",
                        default=0.1)
    parser.add_argument('--visualize', type=int,
                        default=1)
    parser.add_argument('--experiment_id', type=str,
                        default=None)
    parser.add_argument('--case_id', type=str,
                        default=None)

    args = parser.parse_args()
    return args


###############################################################################
#  Run the panorama application
###############################################################################

if __name__ == '__main__':

    args = parse_args()

    run_app(args.img1, args.img2,
            args.descriptor, args.patch_size,
            args.matching_threshold,
            args.ransac_sample_size, args.ransac_n_iterations, args.ransac_tolerance, args.ransac_inlier_fraction_threshold,
            args.visualize==1,
            args.experiment_id, args.case_id)