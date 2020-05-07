import numpy as np
import scipy.spatial
import cv2


def get_Harris_corners(img1_gray, img2_gray, harris_thr=0.05):
    '''
    Custom wrapper over the OpenCV Harris detector

    PARAMETERS: img1_gray, img2_gray - Gray-scaled input images
                harris_thr - To remove weak harris detections

    RETURNS:    img1_kpts, img2_kpts - np array of keypoints in (x,y) coordinate format
    '''
    img1_harris = cv2.cornerHarris(img1_gray, 2, 3, 0.04)
    img1_harris = cv2.dilate(img1_harris,None)
    ret, img1_harris = cv2.threshold(img1_harris, harris_thr*img1_harris.max(), 255, 0)
    img1_harris = np.uint8(img1_harris)
    _, _, _, centroids = cv2.connectedComponentsWithStats(img1_harris)
    img1_kpts = centroids.copy().astype(np.uint16)

    img2_harris = cv2.cornerHarris(img2_gray, 2, 3, 0.04)
    img2_harris = cv2.dilate(img2_harris,None)
    ret, img2_harris = cv2.threshold(img2_harris, harris_thr*img2_harris.max(), 255, 0)
    img2_harris = np.uint8(img2_harris)
    _, _, _, centroids = cv2.connectedComponentsWithStats(img2_harris)
    img2_kpts = centroids.copy().astype(np.uint16)

    return img1_kpts, img2_kpts



def cvt_to_cv2KeyPoints(xy_pairs):
    '''
    Converts keypoints from numpy arrays to list of OpenCV KeyPoint objects
    '''
    xy_pairs = list(xy_pairs)
    cv2_kpts = []
    for i in range(len(xy_pairs)):
        cv2_kpts.append( cv2.KeyPoint(xy_pairs[i][0], xy_pairs[i][1], 1) )
    return cv2_kpts



def normalize(img_descriptors):
    '''
    Normalizes the each descriptor vector to unit length
    '''
    for i in range(img_descriptors.shape[0]):
        img_descriptors[i,:] = img_descriptors[i,:]/np.linalg.norm(img_descriptors[i,:],ord=2)
    return img_descriptors



def compute_euclidean_distances(img1_descriptors, img2_descriptors):
    '''
    Computes Euclidean distance between every img1 and img2 (normalized) descriptors

    PARAMETERS: An np array of descriptors from each image

    RETURNS:   Distance matrix of shape (img1_descriptors.shape[0], img2_descriptors[0])
    '''
    distance_matrix = scipy.spatial.distance_matrix(img1_descriptors, img2_descriptors)
    return distance_matrix



def compute_correlation(img1_descriptors, img2_descriptors):
    '''
    Computes correlation between every img1 and img2 (normalized) descriptors

    PARAMETERS: An np array of descriptors from each image

    RETURNS:   Correlation matrix of shape (img1_descriptors.shape[0], img2_descriptors[0])
    '''
    correlation_matrix = np.dot(img1_descriptors, img2_descriptors.T)
    return correlation_matrix



def get_matchings(similarity_matrix, similarity_type, threshold):
    '''
    Two-way NN matching without replacement

    PARAMETERS: similarity_matrix - Either distance matrix or correlation matrix
                similarity_type - 'euc_distance' or 'correlation'
                threshold - scalar value

    RETURNS:    np array listing the indices of matching keypoint pairs
    '''
    similarity_matrix_copy = similarity_matrix.copy()
    matching_kpt_pair_indices = []

    if similarity_type == 'euc_distance':
        for i in range(similarity_matrix.shape[0]):
            match_for_i = np.argmin(similarity_matrix_copy[i,:]) # Find best match for i
            # Check if the vice versa is also true, and apply threshold
            if i == np.argmin(similarity_matrix_copy[:,match_for_i]) and similarity_matrix[i,match_for_i] <= threshold :
                matching_kpt_pair_indices.append(tuple([i,match_for_i]))

    if similarity_type == 'correlation':
        for i in range(similarity_matrix.shape[0]):
            match_for_i = np.argmax(similarity_matrix_copy[i,:]) # Find best match for i
            # Check if the vice versa is also true, and apply threshold
            if i == np.argmax(similarity_matrix_copy[:,match_for_i]) and similarity_matrix[i,match_for_i] >= threshold :
                matching_kpt_pair_indices.append(tuple([i,match_for_i]))

    return np.array(matching_kpt_pair_indices)

