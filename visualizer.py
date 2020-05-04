import numpy as np
import matplotlib.pyplot as plt
import cv2

class Visualizer:
    def __init__(self, img1, img2, visualize=True, save_results=False, results_dir=None, case_id=None):
        self.visualize = visualize

        self.img1_rgb = img1
        self.img1_gray = cv2.cvtColor(self.img1_rgb, cv2.COLOR_RGB2GRAY)
        self.img2_rgb = img2
        self.img2_gray = cv2.cvtColor(self.img2_rgb, cv2.COLOR_RGB2GRAY)

        self.img1_kpts = None
        self.img2_kpts = None
        self.matching_kpt_pair_indices = None
        self.inlier_indices = None

        self.save_results = save_results
        self.results_dir = results_dir
        self.case_id = case_id

    def set_keypoints(self, img1_kpts, img2_kpts):
        self.img1_kpts = img1_kpts
        self.img2_kpts = img2_kpts

    def set_matches(self, matching_kpt_pair_indices):
        self.matching_kpt_pair_indices = matching_kpt_pair_indices

    def set_inliers(self, inlier_indices):
        self.inlier_indices = inlier_indices


    def draw_keypoints(self):
        fig, [ax1, ax2] = plt.subplots(1,2)
        fig.set_size_inches(0.5*18.5, 0.4*10.5)
        fig.set_dpi(200)
        fig.subplots_adjust(wspace=0.2, hspace=0.1, top=0.95, bottom=0.05, left=0.05, right=0.95)

        ax1.imshow(self.img1_gray, cmap='gray')
        ax1.plot(self.img1_kpts[:, 0], self.img1_kpts[:, 1], color='red', marker='o', linestyle='None', markersize=2)
        ax2.imshow(self.img2_gray, cmap='gray')
        ax2.plot(self.img2_kpts[:, 0], self.img2_kpts[:, 1], color='red', marker='o', linestyle='None', markersize=2)

        ax1.axis('off')
        ax1.set_title("Image 1")
        ax2.axis('off')
        ax2.set_title("Image 2")
        #fig.suptitle("Detected keypoints and best matches", fontsize='x-large')
        if self.save_results:
            fig.savefig(self.results_dir + self.case_id.replace('.','') + "_keypoints.png")
        if self.visualize:
            plt.show()


    def draw_matches(self, title):
        fig, ax = plt.subplots()
        fig.set_size_inches(0.5*18.5, 0.4*10.5)
        fig.set_dpi(200)
        #fig.subplots_adjust(wspace=0.2, hspace=0.1, top=0.95, bottom=0.05, left=0.05, right=0.95)
        white_strip = (np.ones((self.img1_rgb.shape[0], 100, 3))*255).astype(np.uint8)
        combined_img = np.hstack((self.img1_rgb, white_strip, self.img2_rgb))
        ax.imshow(combined_img)

        if self.inlier_indices is not None:
            inlier_color = 'blue'
            outlier_color = 'red'

            # Plot outliers
            outlier_indices = []
            for indices in self.matching_kpt_pair_indices:
                if indices not in self.inlier_indices:
                    outlier_indices.append(indices)
            outlier_indices = np.array(outlier_indices)
            ax.plot( [ self.img1_kpts[outlier_indices[:,0], 0], self.img1_rgb.shape[1] + white_strip.shape[1] + self.img2_kpts[outlier_indices[:,1], 0]  ],
                     [ self.img1_kpts[outlier_indices[:,0], 1], self.img2_kpts[outlier_indices[:,1], 1] ],
                     color=outlier_color, marker='*', linestyle='-', linewidth=1, markersize=5)

            # Plot inliers
            ax.plot( [ self.img1_kpts[self.inlier_indices[:,0], 0], self.img1_rgb.shape[1] + white_strip.shape[1] + self.img2_kpts[self.inlier_indices[:,1], 0]  ],
                     [ self.img1_kpts[self.inlier_indices[:,0], 1], self.img2_kpts[self.inlier_indices[:,1], 1] ],
                     color=inlier_color, marker='*', linestyle='-', linewidth=1, markersize=5)

        else:
            color = 'green'
            ax.plot( [ self.img1_kpts[self.matching_kpt_pair_indices[:,0], 0], self.img1_rgb.shape[1] + white_strip.shape[1] + self.img2_kpts[self.matching_kpt_pair_indices[:,1], 0]  ],
                     [ self.img1_kpts[self.matching_kpt_pair_indices[:,0], 1], self.img2_kpts[self.matching_kpt_pair_indices[:,1], 1] ],
                     color=color, marker='*', linestyle='-', linewidth=1, markersize=5)

        ax.set_title(title)
        ax.axis('off')
        if self.save_results:
            fig.savefig(self.results_dir + self.case_id.replace('.','') + "_matches.png")
        if self.visualize:
            plt.show()


    def stitch_and_display(self, img2_warped, display_all=False):
        img1_expanded = np.zeros( (self.img1_rgb.shape[0], self.img1_rgb.shape[1]+self.img2_rgb.shape[1], 3) ).astype(np.uint8)
        img1_expanded[:, :self.img1_rgb.shape[1], :] = self.img1_rgb[:,:,:]

        stitched_image = img1_expanded.copy()

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
            ax1.set_title("Image 1")
            ax2.axis('off')
            ax2.set_title("Image 2 warped")
            ax3.axis('off')
            ax3.set_title("Stitched image")

        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(0.5*18.5, 0.4*10.5)
            fig.set_dpi(200)
            ax.imshow(stitched_image)
            ax.set_title("Stitched Image")
            ax.axis('off')

        if self.save_results:
            fig.savefig(self.results_dir + self.case_id.replace('.','') + "_stitching_result.png")
        if self.visualize:
            plt.show()