# Vehicle Detection Project
Project on vehicle detection in real road images. A machine learning model is trained with vehicle images and is used to detect the parts of the camera image containing cars using sliding window technique. 

__Key lessons learnt__:
- Feature extraction
- HOG features
- Sliding window technique
- SVM and classification
- Video frames handling and tracking


__Files in the submission__:
- Preprocessing.html - All the preprocessing steps for images
- Feature_Extraction(*)./html - Attempts at extracting features 
- Sliding_window.html - Trying different things with sliding window size adjustments
- Submission_notebook.html - Final ipython notebook


### Feature Extraction and Machine Learning Models
Following steps are performed for feature extractions
#### Preprocessing
Images are read from the source files and visualized. One important thing to noticed here is that - most images belong to rear view of the car (_this is has some effect on detection, as discussed in last section_) Once images are read, I tried changing colorspaces and plot the histogram of colors to see how vehicle and non-vehicle classes deferred. I tried RGB, YCrCb, HSV and HSL colorspaces with 16, 32 and 64 bin sizes. HSV and HLS colorspaces with 32 bins were choosen as they looked promising.

#### Histogram of Oriented Gradients (HOG)
This was done as a part of [preprocessing][ppr] and [feature selection](fsel1) steps. I tried following options:
- pix_per_cell: [4,8]
- cell_per_block: [2,4]
- orient: [7,8,9]
- Image size: (32x32), (64x64)
- Color spaces: Gray, RGB (individual channels), HSV (individual channels)

Based on different tests, I could make following observations:
- Using (32x32) image size helped reduce the feature vector size. Using (64x64) images gave better accuracy.
- 9 orientations gave better results than 7 or 8 orientations
- cell_per_block=2,pix_per_cell=4,orient=8, img_resize=(32,32) for HOG feature on grayscale images with 16 bin histogram of RGB channels gave a 99.07%/98.98% train/test accuracy. But was very poor on actual road images. This was a clear call to use different feature vectors
- cell_per_block=2,pix_per_cell=8,orient=9, img_resize=(64,64) for HOG features and 32 bin histogram for HSV colorspaces gave a fair performance of 98.93% accuracy on test set, but could detect cars on actual road images fairly well.

#### Machine Learning Models
This process is done togather with previous step. I tried Random Forests (RF) and Support Vector Machines (SVM) on the HOG+Color features and conducted a grid search to evaluate their performance.


__SVM Classifier__
I tried following parameters for SVM clasifier:
- kernel: 'linear', 'rbf'
- C: [1, 10, 100, 1000]
- gamma: [0.1, 0.001,0.0001]

Top performing results:

| Rank | kernel | gamma | C | Validation Accuracy | 
| --- | --- | --- | --- | --- |
| 1 | rbf | 0.001 | 10 | 0.993 |
| 2 | rbf | 0.001 | 100 | 0.993 |
| 3 | rbf | 0.001 | 1000 | 0.993 |

Clearly 'rbf' kernel with 0.001 gamma performs well. I double checked the accuracy numbers to be sliglty different in 4th or 5ht precision nummbers. 

__Random Forest Classifier__
I tried RF classifier cross validation with following parameters:
- max_depth": [3, None]
- max_features: [1, 3, 10]
- min_samples_split: [1, 3, 10]
- min_samples_leaf: [1, 3, 10]
- bootstrap: [True, False]
- criterion": ["gini", "entropy"]

Best performing RF classifiers gave a good accuracy of ~99% on validation set, but were not effective on real road images sliding window search. 


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
#### <a name="back_view"></a> Back view problem
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

[//]: # (References)
[fsel1]: ./Feature_Selection.html
[fsel2]: ./Feature_Selection_trial2.html
[fsel3]: ./Feature_Selection_trial3.html
[ppr]: ./Preproceessing.html
[swin]: ./Sliding_Window.html
[subm]: ./Submission_Notebook.html
