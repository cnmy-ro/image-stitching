#!/bin/bash

# Effect of descriptor patch_size
experiment_id="1"

img1_path="./Images/Pair-6/1.jpeg"
img2_path="./Images/Pair-6/2.jpeg"

descriptor="custom_rgb_intensities"

matching_threshold=0.990

sample_size=3
n_iterations=500
tolerance=40
inlier_fraction_threshold=0.1

visualize=0

# Start execution ---

# patch size = 5
python main.py --img1 $img1_path --img2 $img2_path --descriptor $descriptor --patch_size 5 \
               --matching_threshold $matching_threshold \
               --ransac_sample_size $sample_size --ransac_n_iterations $n_iterations \
               --ransac_tolerance $tolerance --ransac_inlier_fraction_threshold $inlier_fraction_threshold \
               --visualize $visualize \
               --experiment_id $experiment_id \
               --case_id "patch5"

# patch size = 7
python main.py --img1 $img1_path --img2 $img2_path --descriptor $descriptor --patch_size 7 \
               --matching_threshold $matching_threshold \
               --ransac_sample_size $sample_size --ransac_n_iterations $n_iterations \
               --ransac_tolerance $tolerance --ransac_inlier_fraction_threshold $inlier_fraction_threshold \
               --visualize $visualize \
               --experiment_id $experiment_id \
               --case_id "patch7"

# patch size = 9
python main.py --img1 $img1_path --img2 $img2_path --descriptor $descriptor --patch_size 9 \
               --matching_threshold $matching_threshold \
               --ransac_sample_size $sample_size --ransac_n_iterations $n_iterations \
               --ransac_tolerance $tolerance --ransac_inlier_fraction_threshold $inlier_fraction_threshold \
               --visualize $visualize \
               --experiment_id $experiment_id \
               --case_id "patch9"

# patch size = 11
python main.py --img1 $img1_path --img2 $img2_path --descriptor $descriptor --patch_size 11 \
               --matching_threshold $matching_threshold \
               --ransac_sample_size $sample_size --ransac_n_iterations $n_iterations \
               --ransac_tolerance $tolerance --ransac_inlier_fraction_threshold $inlier_fraction_threshold \
               --visualize $visualize \
               --experiment_id $experiment_id \
               --case_id "patch11"

# patch size = 13
python main.py --img1 $img1_path --img2 $img2_path --descriptor $descriptor --patch_size 13 \
               --matching_threshold $matching_threshold \
               --ransac_sample_size $sample_size --ransac_n_iterations $n_iterations \
               --ransac_tolerance $tolerance --ransac_inlier_fraction_threshold $inlier_fraction_threshold \
               --visualize $visualize \
               --experiment_id $experiment_id \
               --case_id "patch13"