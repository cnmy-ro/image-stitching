#!/bin/bash

img1_path="./Images/Pair-7/1.jpeg"
img2_path="./Images/Pair-7/2.jpeg"

visualize=0

# Effect of Correlation threshold
# using custom descriptor -----------------------------------------------------
experiment_id="2a"

# Base case -- (comment out the experiment variable)
descriptor="custom_rgb_intensities"
patch_size=11

#matching_threshold=0.980

sample_size=3
n_iterations=1000
tolerance=40
inlier_threshold=15

# Start execution -------------------------------------------------------------

for threshold in 0.980 0.981 0.982 0.983 0.984 0.985 0.986 0.987 0.988 0.989
do
    python main.py --img1 $img1_path --img2 $img2_path --descriptor $descriptor --patch_size $patch_size \
                   --matching_threshold $threshold \
                   --ransac_sample_size $sample_size --ransac_n_iterations $n_iterations \
                   --ransac_tolerance $tolerance --ransac_inlier_threshold $inlier_threshold \
                   --visualize $visualize \
                   --experiment_id $experiment_id \
                   --case_id "thresh$threshold"
done


# using OpenCV SIFT descriptor ------------------------------------------------
experiment_id="2b"

# Base case -- (comment out the experiment variable)
descriptor="opencv_sift"

#matching_threshold=0.950

sample_size=3
n_iterations=1000
tolerance=40
inlier_threshold=15

# Start execution ---

for threshold in 0.950 0.960 0.970 0.980 0.990
do
    python main.py --img1 $img1_path --img2 $img2_path --descriptor $descriptor --patch_size $patch_size \
                   --matching_threshold $threshold \
                   --ransac_sample_size $sample_size --ransac_n_iterations $n_iterations \
                   --ransac_tolerance $tolerance --ransac_inlier_threshold $inlier_threshold \
                   --visualize $visualize \
                   --experiment_id $experiment_id \
                   --case_id "thresh$threshold"
done
