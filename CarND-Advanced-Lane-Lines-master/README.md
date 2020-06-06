## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Project Instructions / Project Scope

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The requirements are defined in [Rubric](https://review.udacity.com/#!/rubrics/1966/view)

### Project change log
* Ver 0.0 : Initial submission
* Ver 0.1 : Add LAB channel A color filter to enhance left lane detection, HLS channel L color filter to remove shadow (low brightness), sanity_check() function for lane pixel correction, more unit tests, video_test.mp4 to test robustness in shadow, update documentation

### Project folder structure
* Implementation code is in file [Advanced_Lane_Finding.ipynb](./Advanced_Lane_Finding.ipynb)
* The images for camera calibration are in folder `camera_cal`, camera calibration result is stored in file [calibration_result.p](./calibration_result.p)
* Helper function is in file [helperfunction.py](./helperfunction.py)
* Output images are stored in folder `output_images`
* Unit test folder is stored in folder `output_images/test_images`
* Video output are [output_video.mp4](./output_video.mp4), [output_test_video.mp4](./output_test_video.mp4), [challenge_output_video.mp4](./challenge_output_video.mp4), [harder_challenge_output_video.mp4](./harder_challenge_output_video.mp4)

## Project document
The code for each step is correspondingly in each section of `Advanced_Lane_Finding.ipynb`

## Camera Calibration and correct distortion calculation for 9x6 chessboard image

I begin the camera calibration with "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

Then I scan through every images in the list of calibration_images to find and draw corners in an image of a chessboard pattern.

![](output_images/camera_calibration_result.png)

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:

![](output_images/distorted_and_undistorted_image.png)

## Pipeline for single image

### 1. Distortion-corrected for example image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images:

* Load the calibration result from file `calibration_result.p`.
* Read the example image and then call `cv2.undistort()` to correct distortion
* Plot the original image and after distorted-correction image

#### Example of undistored image from test_image2.jpg

![](output_images/example_distort_and_undistorted.png)

### 2. Apply color transforms, gradients to create a thresholded binary image

I used a combination of gradient and color thresholds to generate a binary image (thresholding steps with sobelx operation threshold `xgrad_threshold = (40,100)` and HLS color space S threshold `s_threshold = (150,255)` and call `apply_threshold_v2()` in `helperfunction.py`). Here's an example of my output for this step.

#### Binary image after HLS S threshold color transform and Sobelx operation

![](output_images/color_transform_gradient_color_binary_image.png)

Filter LAB color space channel a to enhance the detection of left lane (yellow color) to reduce the effect of shadow on the road with `a_thresh = (80,120)`. The result after the LAB color space A threshold:

#### Binary image after LAB A threshold color filter

![](output_images/LAB_A_Channel_Filter.png)

Filter HLS color space channel L as mask image to remove the low brightness component on the road with `l_thresh = (100,255)`. The result after the HLS color space L threshold:

#### Binary image after HLS L threshold brightness

![](output_images/HLS_L_Channel_Filter.png)

Combine all the filters together to create the combined binary image of lane detection:

#### Combined binary image after filters
![](output_images/combined_filter_binary.png)

### 3. Perform perspective transform to create "bird-eye-view" image

I select only a hard-coded region of interest using a binary image mask. I only keep the region of the image defined by the polygon defined by `vertices` vector. The rest of the image is set to black.

#### Masked binary image region of interest  

![](output_images/binary_image_mask.png)

The code for my perspective transform includes a function called warper(). The warper() function takes as inputs an image (img), as well as source (src) and destination (dst) points. I chose the hardcode the source and destination points for my test image (`img_size = (720,1280)`) are:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 740      | 200, 720      | 
| 550, 470      | 200, 0        |
| 720, 470      | 1080, 0       |
| 1160, 720     | 1080, 720     |

#### Bird-eye-view binary image

![](output_images/bird_eye_view.png)

I verified that my perspective transform was working as expected by drawing the src and dst points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

#### Verification between original image and warped image counter part

![](output_images/original_image_and_warped_image.png)

### 4.Identify lane-line pixels and fit their positions with a polynomial

#### Identify lane line pixels process

To identify the lane line, we perform:
* Divide the image into n horizontal strips (steps) of equal height.
* For each step, take a count of all the pixels at each x-value within the step window using a histogram generated from np.sum.
* Smoothen the histogram using scipy.signal.medfilt.
* Find the peaks in the left and right halves (one half for each lane line) histogram using signal.find_peaks_swt.
* Get (add to our collection of lane line pixels) the pixels in that horizontal strip that have x coordinates close to the two peak x coordinates.

These steps are implemented by `histogram_pixels()` in `helperfunction.py`.

Implement the sanity check whether the set of pixels are correct, I implement the `sanity_check` function to do the following:

* Choose the pixel set with shorter length. If `shorter length pixel < 0.35 (33%) length of longer pixel set`, `sanity_check` flag set to false which requires correction afterwards.
* Scan through all the pixels in shorter set, check whether the distance between the corresponding right and left pixel smaller than the define `distance_thresh = (800,1000)` pixels.
* If `sanity_check` flag is `False`, pass the collection of pixels to the next step.
* Else `sanity_check` flag is `True`, correct the right lane pixel collection by adding `distance_correction = 835` pixels to the left collection pixel (the reason behind is the LAB color filter enhance the left lane detection. Thus, the left lane detection is always better than the right lane.

These steps are implemented by `sanity_check()` in `Advanced_Lane_Finding.ipynb`.

#### Fit position of lane-line pixels with a polynomial

Fit a second order polynomial to each lane line using np.fitpoly by `fit_second_order_poly()` in `helperfunction.py`.

##### Fit second order polynomial to left and right lane lines

![](output_images/identify_lane_pixel.png)

Draw the polynomial binary image with the coeffients found by `draw_poly()` in helper function `helperfunction.py`.

##### Draw polynomial binary image

![](output_images/draw_polynomial_binary_image.png)

Highlight the area between 2 lanes by `hightlight_lane_line_area()` in `helperfunction.py`

##### Highlight the area between 2 lanes

![](output_images/highlight_area_between_2_lanes.png)

### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

radius of curvature: 
* left radius: `left_curve_radius = np.absolute(((1 + (2 * left_coefficients[0] * y_eval + left_coefficients[1])**2) ** 1.5)/(2 * left_coefficients[0]))`
* right radius: `right_curve_radius = np.absolute(((1 + (2 * right_coefficients[0] * y_eval + right_coefficients[1]) ** 2) ** 1.5)/(2 * right_coeffs[0]))`

### 6. Example image plotted back down onto the road such that the lane area is identified clearly
The process is as followed:
* Warp lane lines back onto original image using `cv2.warpPerspective`
* Plot the warped image
* Output visual information display onto the image

#### Warped image onto the original image

![](output_images/plot_back_warped_image_on_original_image.png)

#### Output visual information onto the original image

![](output_images/add_information_on_image.png)

## Unit test (image)

### Case Straight_lines_1: Vehicle on left lane
![](output_images/test_images/test_straight_lines1.jpg)

### Case Straight_lines_2: Vehicle on right lane
![](output_images/test_images/test_straight_lines2.jpg)

### Case 1: Vehicle is approaching shadow on left lane
![](output_images/test_images/test1.jpg)

### Case 2: Vehicle is on open road with no shadow
![](output_images/test_images/test2.jpg)

### Case 3: Vehicle is on open road
![](output_images/test_images/test3.jpg)

### Case 4: Vehicle under tree shadow
![](output_images/test_images/test4.jpg)

### Case 5: Vehicle is approaching shadow area
![](output_images/test_images/test5.jpg)

### Case 6: Another test case for approaching shadow area (test remove low brightness)
![](output_images/test_images/test6.jpg)

### Case 7: Another test case for approaching shadow area (test remove low brightness)
![](output_images/test_images/test7.jpg)

### Case 8: Vehicle approaching damage road (test remove damage on the road)
![](output_images/test_images/test8.jpg)

## Image pipeline

Combine all the steps above to create the `image_pipeline()` in file `Advanced_Lane_Finding.ipynb` for video pipeline.

### Test image pipeline with different test image
![](output_images/test_with_different_image.png)

## Video pipeline

### Test Video
Test the image pipeline in shadow by testing 4s in original video from 38s to 42s. Here's the result [test_output_video](./test_output_video.mp4).

### Project Output Video
Use the image pipeline to generate video project output with lane finding information. Here's the [output_video](./output_video.mp4).

### Challenge Output Video
Use the same image pipeline for challenge output video. Here's the [challenge_output_video](./challenge_output_video.mp4).

### Challenge Output Video
Use the same image pipeline for harder challenge output video. Here's the [harder_challenge_output_video](./harder_challenge_output_video.mp4).

### Discussion
Remarks:
* To find the correct threshold for each filter. Use the [color picker website](http://colorizer.org/)
* Apply the correction conversion for each color space from [OpenCv color conversion](https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html)
* To filter the damage on the road, increase the threshold for sobelx, sobely or use combine dir_threshold operation
* Enhance the left lane detection so I could use left lane as the reference lane
* Remove the low brightness component of the image for better shadow/ dark scenario
* With sanity check, I could choose one lane detection to correct the other lane ( if the polynomial fit result is wrong) 

Feedback from review:
| Suggestion                                | Section   | Implementation   | Status   | 
|:-----------------------------------------:|:---------:|:----------------:|:---------| 
| Try out different color space LAB or LUV  | 2  	| LAB A channel threshold `(80,120)`| pushed to git         |
| Try out sanity check for better detection | 3    	| Distance threshold = `(800,1000)`, Distance correction = `835`| pushed to git      |

 
