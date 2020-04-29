import numpy as np
import matplotlib.pyplot as plt


def display_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.show()


def display_image_and_keypts(img1, img2, key_pts1, key_pts2):
    fig, [ax1, ax2] = plt.subplots(1,2)
    ax1.imshow(img1)
    ax1.plot(key_pts1[:, 1], key_pts1[:, 0],
             color='cyan',
             marker='o',
             linestyle='None',
             markersize=3)
    ax2.imshow(img2)
    ax2.plot(key_pts2[:, 1], key_pts2[:, 0],
             color='cyan',
             marker='o',
             linestyle='None',
             markersize=3)
    plt.show()