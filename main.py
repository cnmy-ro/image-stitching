import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


def run_app(img1, img2):
    pass

###############################################################################
if __name__ == '__main__':

    image_set_dir = "./Images/Pair-1/"
    img1 = cv2.imread(image_set_dir+'right.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(image_set_dir+'left.png')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    run_app(img1, img2)