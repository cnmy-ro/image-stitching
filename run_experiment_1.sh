#!/bin/bash

img1_path="./Images/Pair-7/1.jpeg"
img2_path="./Images/Pair-7/2.jpeg"

visualize=0

# Effect of descriptor patch_size
experiment_id="1"

# Base case -- (comment out the experiment variable)
descriptor="custom_rgb_intensities"
#patch_size=11

matching_threshold=0.980

sample_size=3
n_iterations=1000
tolerance=40
inlier_threshold=15


# Start execution -------------------------------------------------------------

for patch_size in 11 13 15 17 19
do
    python main.py --img1 $img1_path --img2 $img2_path --descriptor $descriptor --patch_size $patch_size \
                   --matching_threshold $matching_threshold \
                   --ransac_sample_size $sample_size --ransac_n_iterations $n_iterations \
                   --ransac_tolerance $tolerance --ransac_inlier_threshold $inlier_threshold \
                   --visualize $visualize \
                   --experiment_id $experiment_id \
                   --case_id "patch$patch_size"
done
