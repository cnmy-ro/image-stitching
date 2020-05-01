import numpy as np

def apply_RANSAC(img1_kpts, img2_kpts, matching_kpt_pair_indices, ransac_params):
    img1_kpts = np.array([kp.pt for kp in img1_kpts])
    img2_kpts = np.array([kp.pt for kp in img2_kpts])

    candidate_model_list = []
    for i in range(ransac_params['N']):
        # Sample random matching pairs
        indices_of_indices = np.random.choice(matching_kpt_pair_indices.shape[0], size=ransac_params['s'], replace=False)
        sampled_kpt_pair_indices = matching_kpt_pair_indices[indices_of_indices]
        sampled_img1_kpts, sampled_img2_kpts  = img1_kpts[sampled_kpt_pair_indices[:,0]], img2_kpts[sampled_kpt_pair_indices[:,1]]

        # Fit a line
        sampled_img1_kpts = np.append(sampled_img1_kpts, np.ones((ransac_params['s'],1)), axis=1)
        sampled_img2_kpts = np.append(sampled_img2_kpts, np.ones((ransac_params['s'],1)), axis=1)
        model = np.linalg.lstsq(sampled_img2_kpts, sampled_img1_kpts, rcond=None)[0]

        # Check how many other points lie within the tolerance zone
        inlier_count = 0
        inlier_indices = []
        for indices in matching_kpt_pair_indices:
            if indices not in sampled_kpt_pair_indices:
                pt2 = img2_kpts[indices[1]]
                pt1_true = np.append(img1_kpts[indices[0]], [1])
                pt1_hypotesis = np.dot(model.T, np.append(pt2, [1]))
                if np.linalg.norm(pt1_true-pt1_hypotesis,ord=2) <= ransac_params['d']:
                    inlier_count += 1
                    inlier_indices.append(indices)
        inlier_indices = np.array(inlier_indices)


        # Apply threshold
        if inlier_count >= ransac_params['T']:
            # The model is good -- Fit the model on all the inliers
            inlier_img1_kpts, inlier_img2_kpts = img1_kpts[inlier_indices[:,0]], img2_kpts[inlier_indices[:,1]]

            inlier_img1_kpts = np.append(inlier_img1_kpts, np.ones((inlier_img1_kpts.shape[0],1)), axis=1)
            inlier_img2_kpts = np.append(inlier_img2_kpts, np.ones((inlier_img2_kpts.shape[0],1)), axis=1)

            candidate_model, residuals, _, _ = np.linalg.lstsq(inlier_img2_kpts, inlier_img1_kpts, rcond=None)
            candidate_model_list.append(tuple([candidate_model, np.sum(residuals)]))

    # if inlier_indices.shape[0] == 0:
    #     raise Exception("No satisfactory inliers for given constraints.")

    # Choose the best model (one with least residual sum value)
    candidate_model_list = sorted(candidate_model_list, key=lambda c: c[1])
    affine_matrix = candidate_model_list[0][0]

    return affine_matrix.T