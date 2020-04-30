import numpy as np
import cv2
import skimage.feature
import matplotlib.pyplot as plt

def display_images(img1, img2):
    fig, [ax1, ax2] = plt.subplots(1,2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()

def display_image_and_keypts(img1, img2, img1_kpts, img2_kp):
    fig, [ax1, ax2] = plt.subplots(1,2)
    ax1.imshow(img1)
    ax1.plot(img1_kpts[:, 1], img1_kpts[:, 0], color='cyan', marker='o', linestyle='None',markersize=3)
    ax2.imshow(img2)
    ax2.plot(img2_kp[:, 1], img2_kp[:, 0], color='cyan', marker='o', linestyle='None', markersize=3)
    plt.show()

def get_Harris_pts(img1_gray, img2_gray):
    img1_harris = cv2.cornerHarris(img1_gray, 3, 3, 0.04)
    img2_harris = cv2.cornerHarris(img2_gray, 3, 3, 0.04)

    img1_kpts = skimage.feature.corner_peaks(img1_harris, min_distance=5)
    img2_kp = skimage.feature.corner_peaks(img2_harris, min_distance=5)

    return img1_kpts, img2_kp

def cvt_to_cv2KeyPoints(img1_kpts, img2_kp):
    img1_kpts = list(img1_kpts)
    for i in range(len(img1_kpts)):
        img1_kpts[i] = cv2.KeyPoint(img1_kpts[i][1], img1_kpts[i][0], 1)
    img2_kp = list(img2_kp)
    for i in range(len(img2_kp)):
        img2_kp[i] = cv2.KeyPoint(img2_kp[i][1], img2_kp[i][0], 1)
    return img1_kpts, img2_kp

def normalize(descriptors):
    for i in range(descriptors.shape[0]):
        descriptors[i,:] = descriptors[i,:]/np.linalg.norm(descriptors[i,:],ord=2)
    return descriptors

def compute_distances(img1_descriptors, img2_descriptors):
    distance_matrix = np.empty((img1_descriptors.shape[0], img2_descriptors.shape[0]))
    for i, d1 in enumerate(img1_descriptors):
        for j, d2 in enumerate(img2_descriptors):
            distance_matrix[i,j] = np.linalg.norm(d2-d1,ord=2)
    return distance_matrix
