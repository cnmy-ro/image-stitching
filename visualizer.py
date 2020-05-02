import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils

class Visualizer:
    def __init__(self, img1, img2, save_figs):
        self.img1_rgb = img1
        self.img1_gray = cv2.cvtColor(self.img1_rgb, cv2.COLOR_RGB2GRAY)
        self.img2_rgb = img2
        self.img2_gray = cv2.cvtColor(self.img2_rgb, cv2.COLOR_RGB2GRAY)

        self.img1_kpts = None
        self.img2_kpts = None
        self.matching_kpt_pair_indices = None
        self.inlier_indices = None

        self.save_figs = save_figs

    def set_keypoints(self, img1_kpts, img2_kpts):
        self.img1_kpts = np.array([kp.pt for kp in img1_kpts])
        self.img2_kpts = np.array([kp.pt for kp in img2_kpts])

    def set_matches(self, matching_kpt_pair_indices):
        self.matching_kpt_pair_indices = matching_kpt_pair_indices

    def set_inliers(self, inlier_indices):
        self.inlier_indices = inlier_indices


    def draw_keypoints(self):
        fig, [ax1, ax2] = plt.subplots(1,2)
        fig.set_size_inches(0.25*18.5, 0.20*10.5)
        fig.set_dpi(200)
        fig.subplots_adjust(wspace=0.2, hspace=0.1, top=0.95, bottom=0.05, left=0.05, right=0.95)

        ax1.imshow(self.img1_gray, cmap='gray')
        ax1.plot(self.img1_kpts[:, 0], self.img1_kpts[:, 1], color='red', marker='o', linestyle='None', markersize=1)
        ax2.imshow(self.img2_gray, cmap='gray')
        ax2.plot(self.img2_kpts[:, 0], self.img2_kpts[:, 1], color='red', marker='o', linestyle='None', markersize=1)

        # # Plot matchings
        # for m_idxs in self.matching_kpt_pair_indices:
        #     color = tuple(np.random.random((3,)))
        #     ax1.plot(self.img1_kpts[m_idxs[0], 0], self.img1_kpts[m_idxs[0], 1], color=color, marker='*', linestyle='None', markersize=5)
        #     ax2.plot(self.img2_kpts[m_idxs[1], 0], self.img2_kpts[m_idxs[1], 1], color=color, marker='*', linestyle='None', markersize=5)

        ax1.axis('off')
        ax1.set_title("Left")
        ax2.axis('off')
        ax2.set_title("Right")
        #fig.suptitle("Detected keypoints and best matches", fontsize='x-large')
        if self.save_figs:
            fig.savefig("./Results/Keypoints_and_best_matches.png")
        plt.show()

    def draw_matches(self, title):
        fig, ax = plt.subplots()
        #fig.set_size_inches(0.25*18.5, 0.20*10.5)
        fig.set_dpi(200)
        #fig.subplots_adjust(wspace=0.2, hspace=0.1, top=0.95, bottom=0.05, left=0.05, right=0.95)
        combined_img = np.hstack((self.img1_rgb, self.img2_rgb))
        ax.imshow(combined_img)

        if self.inlier_indices is not None:
            inlier_color = 'blue'
            outlier_color = 'red'

            ax.plot( [ self.img1_kpts[self.inlier_indices[:,0], 0], self.img1_rgb.shape[1] + self.img2_kpts[self.inlier_indices[:,1], 0]  ],
                     [ self.img1_kpts[self.inlier_indices[:,0], 1], self.img2_kpts[self.inlier_indices[:,1], 1] ],
                      color=inlier_color, marker='*', linestyle='-', linewidth=1, markersize=5)

            outlier_indices = []
            for indices in self.matching_kpt_pair_indices:
                if indices not in self.inlier_indices:
                    outlier_indices.append(indices)
            outlier_indices = np.array(outlier_indices)
            ax.plot( [ self.img1_kpts[outlier_indices[:,0], 0], self.img1_rgb.shape[1] + self.img2_kpts[outlier_indices[:,1], 0]  ],
                     [ self.img1_kpts[outlier_indices[:,0], 1], self.img2_kpts[outlier_indices[:,1], 1] ],
                      color=outlier_color, marker='*', linestyle='-', linewidth=1, markersize=5)

        else:
            color = 'purple'
            ax.plot( [ self.img1_kpts[self.matching_kpt_pair_indices[:,0], 0], self.img1_rgb.shape[1] + self.img2_kpts[self.matching_kpt_pair_indices[:,1], 0]  ],
                     [ self.img1_kpts[self.matching_kpt_pair_indices[:,0], 1], self.img2_kpts[self.matching_kpt_pair_indices[:,1], 1] ],
                     color=color, marker='*', linestyle='-', linewidth=1, markersize=5)

        ax.set_title(title)
        ax.axis('off')
        plt.show()


    def stitch_and_display(self, img2_warped, display_all=False):
        img1_expanded = np.zeros( (self.img1_rgb.shape[0], self.img1_rgb.shape[1]+self.img2_rgb.shape[1], 3) ).astype(np.uint8)
        img1_expanded[:, :self.img1_rgb.shape[1], :] = self.img1_rgb[:,:,:]

        stitched_image = img1_expanded.copy()

        #stitched_image = np.maximum(stitched_image, img2_warped)

        img2_warped_gray = cv2.cvtColor(img2_warped, cv2.COLOR_RGB2GRAY)
        non_zeros = np.argwhere(img2_warped_gray > 0)
        stitched_image[non_zeros[:,0], non_zeros[:,1], :] = img2_warped[non_zeros[:,0],non_zeros[:,1],:]

        if display_all:
            fig, [ax1,ax2,ax3] = plt.subplots(3,1)
            fig.set_size_inches(0.15*18.5, 0.4*10.5)
            fig.set_dpi(200)
            #fig.subplots_adjust(hspace=0.3, top=0.85, bottom=0.15)

            ax1.imshow(img1_expanded)
            ax2.imshow(img2_warped)
            ax3.imshow(stitched_image)

            ax1.axis('off')
            ax1.set_title("Left image")
            ax2.axis('off')
            ax2.set_title("Right image warped")
            ax3.axis('off')
            ax3.set_title("Stitched image")
            if self.save_figs:
                fig.savefig("./Results/Stitching_result.png")
            plt.show()

        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(0.5*18.5, 0.4*10.5)
            fig.set_dpi(200)
            ax.imshow(stitched_image)
            ax.set_title("Stitched Image")
            ax.axis('off')
            plt.show()
