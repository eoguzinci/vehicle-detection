## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./figures/hog_sample.png
[image2]: ./figures/color_hist.png
[image3]: ./figures/car_noncar.png
[image4]: ./figures/camera_img.png
[image5]: ./figures/cropped_img.png
[image6]: ./figures/cropped_YCrCb.png
[image7]: ./figures/scales.png
[image8]: ./figures/nothresh_heat.png
[image9]: ./figures/heat.png
[image10]: ./figures/box_list.png
[video1]: ./project_video.mp4

## Introduction

In this project, we will detect the vehicles on the road through the images   For this project, I have two main files:
1. `project.py` : which illustrates how to read the images with car and noncar labels, how to extract HOG features from those images, train a support vector machines classifier by using the features and their associated labels and how to capture the vehicles on the road by using the fitted model on specific partition of frames in each image. The result is the image on which the vehicles are marked with a blue box. 
2. `pipeline.py` : which demonstrates how we can use these algorithms with an already trained model on sequential video frames to build a video which highlights the detected vehicles on your journey.

The main files use libraries:

* `settings.py` : which contains all the parameters that can be tweaked to extract features, to fit classifiet and to determine the locations and the size of the windows to find the vehicles.  
* `feature_extraction.py` : holds the functions to extract the features from the images. The features are can contain the color histograms and/or histogram of oriented gradients. 
* `train.py` : reads the two datasets which are small and large and trains the labeled data with a linear support vector classifier. The function also gives the accuracy on test dataset.  
* `detect_cars.py`: which returns a `box_list` which contains the pixels representing a car. In this file, a sliding window technique is used to divide the active area of search for cars and to check if the image on the window represents an image of a car. 
* `image_process.py`: holds the functions to create an heatmap from the box_list obtained by the `find_cars()` function inside `detect_cars.py` and returns only the hot boxes which detects the cars only by eliminating the false positives.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### The Data

There are two datasets used with different sizes. The small dataset provided by Udacity to compare the accuracy of the classifier in different parameters sets in `settings.py`. The large dataset contains Udacity data for this project, in which [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images of size 64x64 pixels are provided. The vehicle and non-vehicle images were extracted from the [GTI](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI](http://www.cvlibs.net/datasets/kitti/) datasets.

Here is an example of vehicle and non_vehicle images from the extracted data:

![alt text][image3]

After the data is extracted, the number of cars and noncars are balanced to some 10,000 samples. 

### Histogram of Oriented Gradients (HOG) and Color Histogram

#### 1. Explain how (and identify where in your code) you extracted features from the training images.

As I have mentioned in the introduction section, two feature extraction methods are available in the `feature_extration.py`: 

1. The color histogram
2. The histogram of the oriented gradients

##### The Color Histogram

The color histogram divides the image into color bins RGB color space and extract the features that which RGB values are common or scarce. This feature could be of interest if the cars had specific color. However, as you can see below the cars might have different colors and therefore different dominat color axis. As any color cannot define if the object/pattern is a car or not, I decided not to feed this feature to my classifier.

![alt text][image2]

##### The Histogram of Oriented Gradients (HOG)

I will not go deep to the theory, but you can find extensive information in [this presentation](https://www.youtube.com/watch?v=7S5qXET179I) and in [this paper](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf). To briefly explain HOG divides the images into bins(cells) in 2D and look at the dominant gradient in each cell instead of looking each pixels. This gives really good estimate of the shape of an image and it is immune to the small variations in shape and size.

The hog feature extraction is done by the function `get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)`. 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

Heuristically, I ended up with the parameters of:

* `color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb`
* `hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"`
* `orient = 9  # HOG orientations`
* `pix_per_cell = 8 # HOG pixels per cell`
* `cell_per_block = 2 # HOG cells per block`

The features are normalized anyways in order to prevent a feature type dominates over other types, even if we don't use any other features training the dataset.

### Training the Model

#### The Classifier

I used a linear Support Vector Classifier kernel in [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) to fit HOG features from each image to the corresponding label.

With the parameters in the `settings.py` file, the classifier performs an accuracy of 97.32% in 121 seconds.

### Car Detection

#### Sliding Window Search
#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After the classifier is trained and tested on the dataset, we need to predict objects at the image correctly whether it is a car or not. However, the image obtained by single camera contains much more than only cars, but the road, trees, traffic signs and even the blue sky. 

![alt text][image4]

First thing to do is to restrict the area of search for the vehicle to an area where the cars might be on the image. Therefore, masked the region expect the road.

![alt text][image5]

Then we convert the image from 'RGB' to 'YCrCb' scale as we have trained our car-noncar images in the dataset in this color scale and got better accuracy than RGB scale.

![alt text][image6]

Now, we are ready to search for the area of interest. Now we need to divide the areas into small portions, inour case windows, to check whether a car is present inside this window or not. If yes, we return this window as a positive detection. You can find the fundamental idea behind the sliding window search at [this video](https://www.youtube.com/watch?v=HMtd9EQooCk&feature=youtu.be&t=28).  However, the trick is how frequent the windows should slide for the next search to detect the car and how big the window size should be. For this I have divided the region of interest into 4 overlapping screens to capture big/small (close or far) cars accurately. Window size is set to be 64 px however each screen, blue, turquoise, green and red, has its own scale, 3.0, 2.0, 1.5 and 1.0, respectively. 

![alt text][image7]

The sliding rate of window is determined not by pixels but the cells of the HOG window. I have selected the **slide rate as 1 cell per iteration** make it more sensitive to false positives (because I can eliminate by setting a higher threshold at heatmap which I will explain later.) Then each window is classified to be car or noncar pattern by the already trained model. The positive windows are stored and returned to be used for the visualization.

#### HeatMap and Thresholding the False Positives
#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Now, we have the windows which classified as true by our linearSVC() model. However, the classifier is not perfect and it can make errors while detecting car images inside a window. 

![alt text][image10]

Therefore, we need to eliminate the false positives. The method eliminating them is using a heatmap of the multiple detected pixels by using `add_heat` function in `image_process.py`

```python
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
```

![alt text][image8]

After we got the heatmap, it is obvious that the pixels representing a car image are hotter. So, we can threshold the false positives to avoid classifying them as a vehicle at the output image. I have selected a threshold value of 4 considering the settings above such as sliding rate of the window.
After we threshold the heatmap, we obtain only the vehicles highlight for the output image, in which different vehicles are colored differently and label separately by `scipy.ndimage.measurements.label()` avoiding multiple detections.

![alt text][image9]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

##### Problems
* The biggest problem I faced at this project was about detection accuracy at test frames, thus the accuracy of the classifiers. I tweak all the parameters via brute-force to find the best accuracy in detection of the vehicles. Although I got 97% of accuracy in test dataset, you can see that there are still many false positives to be thresholded. 

* Another problem I got was when two vehicles get too close to each other, they cannot be labeled separately always and treat as one vehicle. I suppose, that we need to use color and shape information to separate one from another.

##### Further Improvements
* One improvement could be to add more screens with smaller scales to identify the cars far from the ego vehicle. However, this can be computationally expensive considering its little benefit in performing the driving task. 

* Another improvement might be widening the region of interest to dtect the vehicles on inclined road. As you can observe from the screens that we use sliding windows to search for vehicles, the upper bound is the horizon as we estimated the road is flat. However, there are many roads that are bumpy which can disorient the camera and disrupt the vehicle detection system.

* We can enhance the data by including several unique cars, because if the car does not resemble the ones in the data set classifier will have the problem to label it as a car. 

* I used so many screens with different scales and small sliding window rate (only 1 cell(8px*scale) per iteration). I did it to be on the safe side and detect the vehicle correctly by adjusting an appropriate threshold for eliminating all the false positives. However, if you want to use this method in real-time, I suggest you to decrease the number windows screened and overlap between the windows, thus decreasing the threshold. My recommendation is to use a better classifier with less windows searched for computational efficiency.

