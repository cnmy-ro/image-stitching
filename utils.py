import numpy as np
import scipy.spatial
import cv2
import skimage.feature
import matplotlib.pyplot as plt

###############################################################################
# Key point extraction and matching stuff
###############################################################################

def get_Harris_corners(img1_gray, img2_gray):
    img1_harris = cv2.cornerHarris(img1_gray, 3, 3, 0.04)
    #img1_harris = skimage.feature.corner_harris(img1_gray, k=0.05)
    img1_kpts = skimage.feature.corner_peaks(img1_harris, min_distance=1) # Format: (row,col)
    img1_kpts[:,[0,1]] = img1_kpts[:,[1,0]] # Transorm into (x,y)

    img2_harris = cv2.cornerHarris(img2_gray, 3, 3, 0.04)
    #img2_harris = skimage.feature.corner_harris(img2_gray, k=0.05)
    img2_kpts = skimage.feature.corner_peaks(img2_harris, min_distance=1)
    img2_kpts[:,[0,1]] = img2_kpts[:,[1,0]]
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
    if similarity_type == 'euc_distance':
        # Threshold defines the max allowable distance
        matching_kpt_pair_indices = np.argwhere(similarity_matrix <= threshold)
    if similarity_type == 'correlation':
        matching_kpt_pair_indices = np.argwhere(similarity_matrix >= threshold)
    return matching_kpt_pair_indices


