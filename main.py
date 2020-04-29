import sys
import numpy as np
import cv2
from skimage import io
from skimage.feature import corner_peaks
import matplotlib.pyplot as plt

import utils

def run_app(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    img1_harris = cv2.cornerHarris(img1_gray, 5, 3, 0.04)
    img2_harris = cv2.cornerHarris(img2_gray, 5, 3, 0.04)

    key_pts_img1 = corner_peaks(img1_harris, min_distance=5)
    key_pts_img2 = corner_peaks(img2_harris, min_distance=5)
    #utils.display_image_and_keypts(img1, img2, key_pts_img1, key_pts_img2)

    descriptor_patch_size = 5

###############################################################################

if __name__ == '__main__':

    image_set_dir = "./Images/Pair-1/"
    img1 = io.imread(image_set_dir+'left.png')
    img2 = io.imread(image_set_dir+'right.png')

    run_app(img1, img2)