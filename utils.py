import numpy as np
import scipy.spatial
import cv2
import skimage.feature

def get_Harris_corners(img1_gray, img2_gray):
    img1_harris = cv2.cornerHarris(img1_gray, 2, 3, 0.04)
    img1_kpts = skimage.feature.corner_peaks(img1_harris, min_distance=1) # Format: (row,col)
    img1_kpts[:,[0,1]] = img1_kpts[:,[1,0]] # Transorm into (x,y)

    img2_harris = cv2.cornerHarris(img2_gray, 2, 3, 0.04)
    img2_kpts = skimage.feature.corner_peaks(img2_harris, min_distance=1)
    img2_kpts[:,[0,1]] = img2_kpts[:,[1,0]]
    return img1_kpts, img2_kpts


def get_Harris_corners_2(img1_gray, img2_gray):
    img1_harris = cv2.cornerHarris(img1_gray, 2, 3, 0.04)
    img1_harris = cv2.dilate(img1_harris,None)
    ret, img1_harris = cv2.threshold(img1_harris, 0.01*img1_harris.max(), 255, 0)
    img1_harris = np.uint8(img1_harris)
    _, _, _, centroids = cv2.connectedComponentsWithStats(img1_harris)
    img1_kpts = centroids.copy().astype(np.uint16)

    img2_harris = cv2.cornerHarris(img2_gray, 2, 3, 0.04)
    img2_harris = cv2.dilate(img2_harris,None)
    ret, img2_harris = cv2.threshold(img2_harris, 0.01*img2_harris.max(), 255, 0)
    img2_harris = np.uint8(img2_harris)
    _, _, _, centroids = cv2.connectedComponentsWithStats(img2_harris)
    img2_kpts = centroids.copy().astype(np.uint16)

    return img1_kpts, img2_kpts


def cvt_to_cv2KeyPoints(xy_pairs):
    xy_pairs = list(xy_pairs)
    cv2_kpts = []
    for i in range(len(xy_pairs)):
        cv2_kpts.append( cv2.KeyPoint(xy_pairs[i][0], xy_pairs[i][1], 1) )
    return cv2_kpts


def normalize(img_descriptors):
    for i in range(img_descriptors.shape[0]):
        img_descriptors[i,:] = img_descriptors[i,:]/np.linalg.norm(img_descriptors[i,:],ord=2)
    return img_descriptors

def compute_euclidean_distances(img1_descriptors, img2_descriptors):
    distance_matrix = scipy.spatial.distance_matrix(img1_descriptors, img2_descriptors)
    return distance_matrix

def compute_correlation(img1_descriptors, img2_descriptors):
    correlation_matrix = np.dot(img1_descriptors, img2_descriptors.T)
    return correlation_matrix


def get_matchings(similarity_matrix, similarity_type, threshold):
    # 1-way NN matching with replacement (no vice versa condition; one target element can match with multiple pts)
    if similarity_type == 'euc_distance':
        # Threshold defines the max allowable distance
        matching_kpt_pair_indices = np.argwhere(similarity_matrix <= threshold)
    if similarity_type == 'correlation':
        # Threshold defines the min allowable correlation
        matching_kpt_pair_indices = np.argwhere(similarity_matrix >= threshold)
    return matching_kpt_pair_indices

def get_matchings_2(similarity_matrix, similarity_type, threshold):
    # 2-way NN matching without replacement
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

