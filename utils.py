import numpy as np
import cv2
import skimage.feature
import matplotlib.pyplot as plt

###############################################################################
# Key point extraction and matching stuff
###############################################################################

def get_Harris_pts(img1_gray, img2_gray):
    img1_harris = cv2.cornerHarris(img1_gray, 3, 3, 0.04)
    img1_kpts = skimage.feature.corner_peaks(img1_harris, min_distance=1)
    img1_kpts[:,[0,1]] = img1_kpts[:,[1,0]]

    img2_harris = cv2.cornerHarris(img2_gray, 3, 3, 0.04)
    img2_kpts = skimage.feature.corner_peaks(img2_harris, min_distance=1)
    img2_kpts[:,[0,1]] = img2_kpts[:,[1,0]]
    return img1_kpts, img2_kpts


def cvt_to_cv2KeyPoints(xy_pairs):
    xy_pairs = list(xy_pairs)
    cv2_kpts = []
    for i in range(len(xy_pairs)):
        cv2_kpts.append( cv2.KeyPoint(xy_pairs[i][0], xy_pairs[i][1], 1) )
    return cv2_kpts

def normalize(descriptors):
    for i in range(descriptors.shape[0]):
        descriptors[i,:] = descriptors[i,:]/np.linalg.norm(descriptors[i,:],ord=2)
    return descriptors

def compute_euclidean_distances(img1_descriptors, img2_descriptors):
    distance_matrix = np.empty((img1_descriptors.shape[0], img2_descriptors.shape[0]))
    for i, d1 in enumerate(img1_descriptors):
        for j, d2 in enumerate(img2_descriptors):
            distance_matrix[i,j] = np.linalg.norm(d2-d1,ord=2)
    return distance_matrix

def compute_correlation(img1_descriptors, img2_descriptors):
    correlation_matrix = np.empty((img1_descriptors.shape[0], img2_descriptors.shape[0]))
    for i, d1 in enumerate(img1_descriptors):
        for j, d2 in enumerate(img2_descriptors):
            correlation_matrix[i,j] = np.dot(d1,d2)
    return correlation_matrix

def get_matchings(similarity_matrix, similarity_type, threshold):
    if similarity_type == 'euc_distance':
        # Threshold defines the max allowable distance
        matching_kpt_pair_indices = np.argwhere(similarity_matrix <= threshold)
    return matching_kpt_pair_indices
