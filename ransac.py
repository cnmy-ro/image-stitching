import numpy as np
import cv2


def evaluate_model(affine_matrix, img1_kpts, img2_kpts, inlier_indices):
    # img1_kpts = np.array([kp.pt for kp in img1_kpts])
    inlier_img1_kpts = img1_kpts[inlier_indices[:,0]]
    inlier_img1_kpts = np.hstack( (inlier_img1_kpts, np.ones((inlier_img1_kpts.shape[0],1))) )

    # img2_kpts = np.array([kp.pt for kp in img2_kpts])
    inlier_img2_kpts = img2_kpts[inlier_indices[:,1]]
    inlier_img2_kpts  = np.hstack((inlier_img2_kpts, np.ones((inlier_img2_kpts.shape[0],1))) )

    inlier_img2_kpts_warped = np.dot(inlier_img2_kpts , affine_matrix.T)

    euc_distance = np.mean(np.sqrt(np.sum(np.square(inlier_img1_kpts-inlier_img2_kpts_warped), axis=1)), axis=0)
    return euc_distance


###############################################################################
# RANSAC implementation
###############################################################################

class RANSAC_Estimator:

    def __init__(self, sample_size, n_iterations, tolerance, inlier_fraction_threshold):
        self.sample_size = sample_size
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.inlier_fraction_threshold = inlier_fraction_threshold


    def estimate_affine_matrix(self, img1_kpts, img2_kpts, matching_kpt_pair_indices):
        # img1_kpts = np.array([kp.pt for kp in img1_kpts])
        # img2_kpts = np.array([kp.pt for kp in img2_kpts])

        inlier_threshold = round(self.inlier_fraction_threshold * matching_kpt_pair_indices.shape[0])

        candidate_model_list = []
        for i in range(self.n_iterations):
            # Sample random matching pairs
            indices_of_indices = np.random.choice(matching_kpt_pair_indices.shape[0], size=self.sample_size, replace=False)
            sampled_kpt_pair_indices = matching_kpt_pair_indices[indices_of_indices]
            sampled_img1_kpts, sampled_img2_kpts  = img1_kpts[sampled_kpt_pair_indices[:,0]], img2_kpts[sampled_kpt_pair_indices[:,1]]

            # Fit a linear model
            sampled_img1_kpts = np.append(sampled_img1_kpts, np.ones((self.sample_size, 1)), axis=1)
            sampled_img2_kpts = np.append(sampled_img2_kpts, np.ones((self.sample_size, 1)), axis=1)
            '''
            Solve for X in --  A.X = B

              > A: sampled_img2_kpts; Format: [[x1,y1,1],
                                               [x2,y2,1],
                                               [x3,y3,1],
                                                  ...  ]

              > B: sampled_img1_kpts; Format: [[u1,v1,1],
                                               [u2,v2,1],
                                               [u3,v3,1],
                                                  ...  ]

              > X: model; Format: [[a,d,1],
                                   [b,e,1],
                                   [c,f,1]]

            '''
            model = np.linalg.lstsq(sampled_img2_kpts, sampled_img1_kpts, rcond=None)[0]

            # Check how many other points lie within the tolerance zone
            inlier_count = 0
            inlier_indices = []
            for pair_indices in matching_kpt_pair_indices:
                if pair_indices not in sampled_kpt_pair_indices:
                    pt2 = img2_kpts[pair_indices[1]]
                    pt1_true = np.append(img1_kpts[pair_indices[0]], [1])
                    pt1_hypothesis = np.dot(np.append(pt2, [1]), model)

                    # Calculate error(euc distance b/w prediction and truth). In image coordinates.
                    dist_from_model = np.linalg.norm(pt1_true-pt1_hypothesis,ord=2)

                    if dist_from_model <= self.tolerance:
                        #print(dist_from_model)
                        inlier_count += 1
                        inlier_indices.append(pair_indices)

            # Calculate avg. inlier residual for this model
            if inlier_count == 0:
                continue
            inlier_indices = np.array(inlier_indices)
            inlier_img1_kpts, inlier_img2_kpts = img1_kpts[inlier_indices[:,0]], img2_kpts[inlier_indices[:,1]]
            inlier_img1_kpts = np.append(inlier_img1_kpts, np.ones((inlier_img1_kpts.shape[0],1)), axis=1)
            inlier_img2_kpts = np.append(inlier_img2_kpts, np.ones((inlier_img2_kpts.shape[0],1)), axis=1)

            hypothesis = np.dot(inlier_img2_kpts, model)
            inlier_residuals = np.sum(np.square(inlier_img1_kpts-hypothesis), axis=1)
            avg_inlier_residual = np.mean(inlier_residuals, axis=0)

            # Apply threshold and Refit
            if inlier_count >= inlier_threshold:
                # The model is good -- Fit the model on all the inliers
                candidate_model, _, _, _ = np.linalg.lstsq(inlier_img2_kpts, inlier_img1_kpts, rcond=None)
                candidate_model_list.append(tuple([candidate_model, avg_inlier_residual, inlier_indices]))

        # Choose the best model (one with least residual sum value)
        if len(candidate_model_list) == 0:
            raise Exception("Couldn't find a good model for the given constraints")

        candidate_model_list = sorted(candidate_model_list, key=lambda c: c[1])
        affine_matrix, avg_residual, inlier_indices = candidate_model_list[0]
        return affine_matrix.T, avg_residual, inlier_indices

