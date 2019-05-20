# Behavioral Cloning

### Project #4 for Udacity Self-Driving Car Nanodegree

#### By Adam Gyarmati

![](examples/track2.gif | width=200)

---

### Overview: Behavioral Cloning

The goals of this project were the following:
* Use the simulator build by Udacity to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/center_2018_12_04_10_00_49_239.jpg "Example image from dataset"
[image2]: ./examples/center_2018_12_04_10_00_49_239_cropped.jpg "Example cropped image from dataset"
[image3]: ./examples/cnn-architecture-624x890.png "NVIDIA convolutional neural network model"


#### 1. Files

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* P4_writeup.md summarizing the results
* video_tr1.mp4 showing a video recording of the vehicle driving autonomously with model.h5 on track 1
* video_tr2.mp4 showing a video recording of the vehicle driving autonomously with model.h5 on track 2


#### 2. Code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Since I wanted to make my model work on both tracks, I collected data from both tracks. Because I wanted the vehicle to keep its lane on the second track, I decided to teach it to always keep to the right, even on the first track. Therefore I also recorded recoveries from the left or center back to the right side. The final dataset consisted of:
* 1 + 1 lap on track 1 in both directions keeping to the right
* 2 laps on track 2 in the same direction staying in the right lane
* recordings of recovery from the side of the lanes for both tracks (8-10 recoveries/track)
* multiple recordings of "tricky parts", e.g. sharp turns


#### 2. Model architecture

I got to my final model architecture after a lot of experiments using the NVIDIA model suggested by Udacity and modifying it to get better results.

The model accepts preprocessed images of the road made by the center camera. An example of a raw camera image from the recording is provided below:

![alt text][image1]

Preprocessing consists of:
* cropping the images to delete irrelevant sections like trees and the front of the vehicle (see example images below)
* normalizing the pixel values from the range (0, 255) to (-0.5, 0.5)

Example of a cropped image (this may be a lot of cropping, but I found it useful especially for the second track):

![alt text][image2]

The convolutional neutal network model is based on [NVIDIA's end-to-end deep learning model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), which uses the following architecture (image sizes are different in my network due to the different input size):

![alt text][image3]

I made the following modifications to make the model work for me:
* added Batch Normalization after all convolutional layers and the first three dense layers
* used *ELU* (Exponential Linear Unit) activation functions after Batch Normalization
* added Dropout layers (with 50% dropout rate) after the first two dense layers


#### 3. Training the model

After the collection process, I had about 57000 number of data points.

I randomly shuffled the data set and put 5% of the data into a validation set. 

I used the training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as this provided the best results in the simulator (although the validation loss was not always the lowest). I used an adam optimizer so that manually training the learning rate wasn't necessary.


#### 4. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 91 and 96). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 5. Results and further improvement potential

My model is able to output the correct angle to steer the car autonomously on both tracks, and keeps to the right on both tracks, which I think is more realistic than driving in the middle even in the case of the first track. However there is still room for improvement:
* driving could be smoother, and in some cases the vehicle keeps to much to the right, risking hitting a curb or going off track
* since I trained the model using images from the simulator in the fastest (= lower quality graphics) mode, there are features in the high quality mode (e.g. shadows) that it never saw during training
* this is not an issue for the first track, but the model fails on the second track in high quality mode
* therefore, data should be collected from high quality mode to train the model with the additional features (most importantly the shadows)
