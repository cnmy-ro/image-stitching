import numpy as np
import scipy.spatial
import cv2
import skimage.feature

def gray_intensities(img_gray, img_kpts, patch_size=5):
    if patch_size%2 == 0:
        raise Exception("Patch size should be an odd number")

    img_descriptors = np.zeros((img_kpts.shape[0], patch_size*patch_size))
    for i,kp in enumerate(img_kpts):
        # key-points are xy. But indexing needs (row,col) -> swap them
        patch = img_gray[max(0, kp[1]-patch_size//2)  :  min(img_gray.shape[0], kp[1]+patch_size//2+1),
                         max(0, kp[0]-patch_size//2)  :  min(img_gray.shape[1], kp[0]+patch_size//2+1)]

        if kp[0] - patch_size//2 < 0:
            n = - (kp[0] - patch_size//2)
            pad = np.zeros((patch_size, n))
            patch = np.hstack((pad, patch))

        if kp[1] - patch_size//2 < 0:
            n = - (kp[1] - patch_size//2)
            pad = np.zeros((n, patch_size))
            patch = np.vstack((pad, patch))

        if kp[0] + patch_size//2 >= img_gray.shape[1]:
            n = kp[0] + patch_size//2 + 1 - img_gray.shape[1]
            pad = np.zeros((patch_size, n))
            patch = np.hstack((patch, pad))

        if kp[1] + patch_size//2 >= img_gray.shape[0]:
            n = kp[1] + patch_size//2 + 1 - img_gray.shape[0]
            pad = np.zeros((n, patch_size))
            patch = np.vstack((patch, pad))

        img_descriptors[i,:] = patch.flatten()
    return img_descriptors



def rgb_intensities(img_rgb, img_kpts, patch_size=5):
    if patch_size%2 == 0:
        raise Exception("Patch size should be an odd number")

    img_descriptors = np.zeros((img_kpts.shape[0], patch_size*patch_size*3))
    for i,kp in enumerate(img_kpts):
        # key-points are xy. But indexing needs (row,col) -> swap them
        patch = img_rgb[max(0, kp[1]-patch_size//2)  :  min(img_rgb.shape[0], kp[1]+patch_size//2+1),
                        max(0, kp[0]-patch_size//2)  :  min(img_rgb.shape[1], kp[0]+patch_size//2+1),
                        :]

        if kp[0] - patch_size//2 < 0:
            n = - (kp[0] - patch_size//2)
            pad = np.zeros((patch_size, n, 3))
            patch = np.hstack((pad, patch))

        if kp[1] - patch_size//2 < 0:
            n = - (kp[1] - patch_size//2)
            pad = np.zeros((n, patch_size, 3))
            patch = np.vstack((pad, patch))

        if kp[0] + patch_size//2 >= img_rgb.shape[1]:
            n = kp[0] + patch_size//2 + 1 - img_rgb.shape[1]
            pad = np.zeros((patch_size, n, 3))
            patch = np.hstack((patch, pad))

        if kp[1] + patch_size//2 >= img_rgb.shape[0]:
            n = kp[1] + patch_size//2 + 1 - img_rgb.shape[0]
            pad = np.zeros((n, patch_size, 3))
            patch = np.vstack((patch, pad))

        img_descriptors[i,:] = patch.flatten()
    return img_descriptors