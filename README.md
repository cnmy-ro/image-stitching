# Panoramic Image Stitching Program

The pipeline is composed of 4 parts:
1. Feature detection - Harris corner detector
2. Feature description - Simple custom descriptors or SIFT
3. Extimation of the affine transformation matrix - RANSAC algorithm
4. Transforming and overlaying the second image over the first

----------------------------------------------

## Qualitative Results

![alt text](https://github.com/cnmy-ro/image-stitching/blob/master/Results/Qualitative%20tests/simple2-stitched.png)
![alt text](https://github.com/cnmy-ro/image-stitching/blob/master/Results/Qualitative%20tests/simple3-stitched.png)
![alt text](https://github.com/cnmy-ro/image-stitching/blob/master/Results/Qualitative%20tests/simple4-stitched.png)
![alt text](https://github.com/cnmy-ro/image-stitching/blob/master/Results/Qualitative%20tests/hard-stitched.png)

----------------------------------------------

## Structure and Instructions

Directories:
- Test cases: Contains pairs of input images for testing
- Results: Contains plots and logs from all experiments

Code:
- main.py: The main file for the application
- utils.py: Collection of utility functions
- descriptors.py: Implementation of feature descriptors
- ransac.py: RANSAC implementation
- visualizer.py: Visualization code

Instructions:

1. To use one of the supplied test examples in base case configuration, execute:

`
    $ python3 main.py --img1 ./test_images/Original/1.jpeg 
                      --img2 ./test_images/Original/2.jpeg
`

2. To use the hard test example, execute:

`
    $ python3 main.py --img1 ./test_images/Hard/1.jpeg 
                      --img2 ./test_images/Hard/2.jpeg 
                      --harris_thr 0.01 
                      --descriptor "custom_rgb_intensities" 
                      --patch_size 21 
                      --matching_threshold 0.940 
                      --ransac_sample_size 3 
                      --ransac_n_iterations 1000 
                      --ransac_tolerance 50 
                      --ransac_inlier_threshold 5
`
