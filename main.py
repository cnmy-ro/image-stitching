import sys
import numpy as np
import cv2
import skimage.io
import matplotlib.pyplot as plt

import utils

###############################################################################
#   CONFIG
###############################################################################
corners = 'harris'

###############################################################################
image_set_dir = "./Images/Pair-1/"
img1_rgb = skimage.io.imread(image_set_dir+'left.png')
img2_rgb = skimage.io.imread(image_set_dir+'right.png')
img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)

if corners == 'harris':
    # Get Harris corners: np array of (row,col) pairs each representing a point
    img1_kpts, img2_kpts = utils.get_Harris_pts(img1_gray, img2_gray)
    # Convert to list of cv2 KeyPoint objects
    img1_kpts, img2_kpts = utils.cvt_to_cv2KeyPoints(img1_kpts, img2_kpts)

if corners == 'sift':
    # Get SIFT corners: list of cv2.KeyPoint() objects
    img1_kpts = sift.detect(img1_gray,None)
    img2_kpts = sift.detect(img2_gray,None)

# Extract descriptors using the images and their keypoints
img1_kpts, img1_descriptors = sift.compute(img1_gray,img1_kpts)
img2_kpts, img2_descriptors = sift.compute(img2_gray,img2_kpts)

# Normalize the descriptors
img1_descriptors = utils.normalize(img1_descriptors)
img2_descriptors = utils.normalize(img2_descriptors)


# Get distance matrix
distance_matrix = utils.compute_distances(img1_descriptors, img2_descriptors)



# visualize keypoints
img1_rgb=cv2.drawKeypoints(img1_gray, img1_kpts, img1_rgb, color=(0,255,0))
img2_rgb=cv2.drawKeypoints(img2_gray, img2_kpts, img2_rgb, color=(0,255,0))
utils.display_images(img1_rgb, img2_rgb)
