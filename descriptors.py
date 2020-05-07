import numpy as np

'''
Two feature descriptors are implemented:

    1. RGB intensity patch
    2. Gray-scale intensity patch (not used in experiments)

'''

def rgb_intensities(img_rgb, img_kpts, patch_size=5):
    '''
    Computes RGB intensities of a fixed size local neighbourhood around each keypoint.

    PARAMETERS: img_rgb - RGB image
                img_kpts - np array of keypoints in (x,y) coordinate format
                patch_size - size of the local neighbourhood

    RETURNS:    img_descriptors - np array of descriptors, each of length (patch_size x patch_size x 3)

    '''
    if patch_size%2 == 0:
        raise Exception("Patch size should be an odd number")

    # Initialize an empty list of descriptors
    img_descriptors = np.zeros((img_kpts.shape[0], patch_size*patch_size*3))
    for i,kp in enumerate(img_kpts):
        # Keypoints are in xy coords, but indexing needs (row,col) -> swap x and y
        patch = img_rgb[max(0, kp[1]-patch_size//2)  :  min(img_rgb.shape[0], kp[1]+patch_size//2+1),
                        max(0, kp[0]-patch_size//2)  :  min(img_rgb.shape[1], kp[0]+patch_size//2+1),
                        :]

        # Handle edge cases
        if kp[0] - patch_size//2 < 0:
            n = - (kp[0] - patch_size//2)
            pad = np.zeros((patch.shape[0], n, 3))
            patch = np.hstack((pad, patch))

        if kp[1] - patch_size//2 < 0:
            n = - (kp[1] - patch_size//2)
            pad = np.zeros((n, patch.shape[1], 3))
            patch = np.vstack((pad, patch))

        if kp[0] + patch_size//2 >= img_rgb.shape[1]:
            n = kp[0] + patch_size//2 + 1 - img_rgb.shape[1]
            pad = np.zeros((patch.shape[0], n, 3))
            patch = np.hstack((patch, pad))

        if kp[1] + patch_size//2 >= img_rgb.shape[0]:
            n = kp[1] + patch_size//2 + 1 - img_rgb.shape[0]
            pad = np.zeros((n, patch.shape[1], 3))
            patch = np.vstack((patch, pad))

        # Flatten the RGB patch into a vector and store it
        img_descriptors[i,:] = patch.flatten()
    return img_descriptors



def gray_intensities(img_gray, img_kpts, patch_size=5):
    '''
    Gray level intensities of a fixed size local neighbourhood

    PARAMETERS: img_gray - Gray-scale image
                img_kpts - np array of keypoints in (x,y) coordinate format
                patch_size - size of the local neighbourhood

    RETURNS:    img_descriptors - np array of descriptors, each of length (patch_size x patch_size)

    '''

    if patch_size%2 == 0:
        raise Exception("Patch size should be an odd number")

    img_descriptors = np.zeros((img_kpts.shape[0], patch_size*patch_size))
    for i,kp in enumerate(img_kpts):
        # Keypoints are in xy coords, but indexing needs (row,col) -> swap x and y
        patch = img_gray[max(0, kp[1]-patch_size//2)  :  min(img_gray.shape[0], kp[1]+patch_size//2+1),
                         max(0, kp[0]-patch_size//2)  :  min(img_gray.shape[1], kp[0]+patch_size//2+1)]

        # Handle edge cases
        if kp[0] - patch_size//2 < 0:
            n = - (kp[0] - patch_size//2)
            pad = np.zeros((patch.shape[0], n))
            patch = np.hstack((pad, patch))

        if kp[1] - patch_size//2 < 0:
            n = - (kp[1] - patch_size//2)
            pad = np.zeros((n, patch.shape[1]))
            patch = np.vstack((pad, patch))

        if kp[0] + patch_size//2 >= img_gray.shape[1]:
            n = kp[0] + patch_size//2 + 1 - img_gray.shape[1]
            pad = np.zeros((patch.shape[0], n))
            patch = np.hstack((patch, pad))

        if kp[1] + patch_size//2 >= img_gray.shape[0]:
            n = kp[1] + patch_size//2 + 1 - img_gray.shape[0]
            pad = np.zeros((n, patch.shape[1]))
            patch = np.vstack((patch, pad))

        # Flatten the gray patch into a vector and store it
        img_descriptors[i,:] = patch.flatten()
    return img_descriptors



