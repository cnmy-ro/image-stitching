#!/bin/bash

# Effect of descriptor patch_size
experiment_id="1"

img1_path="./Images/Pair-1/1.png"
img2_path="./Images/Pair-1/2.png"

descriptor="custom_gray_intensities"

matching_threshold=0.995

sample_size=3
n_iterations=100
tolerance=20
inlier_fraction_threshold=0.2

# Start execution

python main.py --img1 $img1_path --img2 $img2_path --descriptor $descriptor --patch_size 5 \
               --matching_threshold $matching_threshold \
               --ransac_sample_size $sample_size --ransac_n_iterations $n_iterations \
               --ransac_tolerance $tolerance --ransac_inlier_fraction_threshold $inlier_fraction_threshold \
               --experiment_id $experiment_id

