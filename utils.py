import numpy as np
import cv2
import skimage.feature
import matplotlib.pyplot as plt

def display_images(img1, img2):
    fig, [ax1, ax2] = plt.subplots(1,2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()

def display_image_and_keypts(img1, img2, img1_kpts, img2_kpts, matching_kpt_pair_indices):
    img1_kpts = np.array([kp.pt for kp in img1_kpts])
    img2_kpts = np.array([kp.pt for kp in img2_kpts])

    fig, [ax1, ax2] = plt.subplots(1,2)
    ax1.imshow(img1)
    ax1.plot(img1_kpts[:, 0], img1_kpts[:, 1], color='cyan', marker='o', linestyle='None', markersize=2)
    ax2.imshow(img2)
    ax2.plot(img2_kpts[:, 0], img2_kpts[:, 1], color='cyan', marker='o', linestyle='None', markersize=2)

    # Plot matchings
    for m_idxs in matching_kpt_pair_indices:
        color = tuple(np.random.random((3,)))
        ax1.plot(img1_kpts[m_idxs[0], 0], img1_kpts[m_idxs[0], 1], color=color, marker='D', linestyle='None', markersize=5)
        ax2.plot(img2_kpts[m_idxs[1], 0], img2_kpts[m_idxs[1], 1], color=color, marker='D', linestyle='None', markersize=5)

    plt.show()


def get_Harris_pts(img1_gray, img2_gray):
    img1_harris = cv2.cornerHarris(img1_gray, 3, 3, 0.04)
    img2_harris = cv2.cornerHarris(img2_gray, 3, 3, 0.04)

    img1_kpts = skimage.feature.corner_peaks(img1_harris, min_distance=5)
    img2_kpts = skimage.feature.corner_peaks(img2_harris, min_distance=5)

    return img1_kpts, img2_kpts

def cvt_to_cv2KeyPoints(xy_pairs):
    xy_pairs = list(xy_pairs)
    cv2_kpts = []
    for i in range(len(xy_pairs)):
        cv2_kpts.append( cv2.KeyPoint(xy_pairs[i][1], xy_pairs[i][0], 1) )
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