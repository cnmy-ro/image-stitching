import matplotlib.pyplot as plt
import numpy as np

exp_results_dir = "./Results/Exp-1/"


patch_size = np.array([11, 13 ,15, 17, 19, 21, 23, 25 ,27 ,29])
percent_change_in_patch_size = (patch_size[1:] - patch_size[:-1])/patch_size[:-1]


results_file_path = exp_results_dir + "result.txt"
data = np.loadtxt(results_file_path, delimiter=',')

n_inliers_list = data[:,0]
n_outliers_list = data[:,1]
avg_inlier_residual_list = data[:,2]
avg_euc_dist_list = data[:,3]

inlier_outlier_ratios = n_inliers_list/n_outliers_list
percent_change_in_ratio = (inlier_outlier_ratios[1:] - inlier_outlier_ratios[:-1]) / inlier_outlier_ratios[:-1]
avg_ratio_sensitivity =  np.mean(percent_change_in_ratio / percent_change_in_patch_size)

percent_change_in_res = (avg_inlier_residual_list[1:] - avg_inlier_residual_list[:-1])/avg_inlier_residual_list[:-1]
avg_res_sensitivity = np.mean(percent_change_in_res / percent_change_in_patch_size)

percent_change_in_dist = (avg_euc_dist_list[1:] - avg_euc_dist_list[:-1]) / avg_euc_dist_list[:-1]
avg_dist_sensitivity = np.mean(percent_change_in_dist/percent_change_in_patch_size)

# plt.plot(patch_size, n_inliers_list+n_outliers_list, 'go-', label="Total matches")
# plt.plot(patch_size, n_inliers_list, 'bo-', label="Inliers")
# plt.plot(patch_size, n_outliers_list, 'ro-', label="Outliers")
# plt.xticks(np.arange(patch_size[0], patch_size[-1]+1, 2))
# plt.legend()
# plt.xlabel("Patch size")
# plt.ylabel("No. of pairs")
# plt.grid()
# plt.savefig(exp_results_dir + "inlier_curve.png")
# plt.show()

# plt.plot(patch_size, avg_inlier_residual_list, marker='o', color='magenta', label="Avg. inlier residual before refitting")
# plt.xticks(np.arange(patch_size[0], patch_size[-1]+1, 2))
# plt.legend()
# plt.xlabel("Patch size")
# plt.ylabel("Residual")
# plt.grid()
# plt.savefig(exp_results_dir + "residual_curve.png")
# plt.show()

# plt.plot(patch_size, avg_euc_dist_list, marker='o', color='darkcyan', label="Avg. inlier euclidean distance after refitting")
# plt.xticks(np.arange(patch_size[0], patch_size[-1]+1, 2))
# plt.legend()
# plt.xlabel("Patch size")
# plt.ylabel("Euclidean distance")
# plt.grid()
# plt.savefig(exp_results_dir + "eucdist_curve.png")
# plt.show()