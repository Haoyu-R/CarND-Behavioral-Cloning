# Behavioral Cloning Project


Overview
---
This repository contains files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.


### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## Rubric Points

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

Yes, my project inclueds train.py, model.h5 and readme.md these three files.

### 2. Submission includes functional code

Yes, the code works pretty fine

### 3. Submission code is usable and readable

Yes, the code is well commentted

### 4. An appropriate model architecture has been employed

My model consists of several conv2d layer, maxpooling2d, and dense layer. At the beginning of network, a lambda and a cropping2d layer are included for image normalization and cropping. Also, the activation function "relu" is chosen for all layers. 


### 5. Attempts to reduce overfitting in the model

Yes, a dropout layer is applied right after conv2d layer.


### 6. Model parameter tuning

Since the network pretty well at the first time, not much hyperparameter tunning is needed. And the model used an adam optimizer, so the learning rate was not tuned manually.

### 7. Appropriate training data

Only the provided data is used for training and validation.

### 8. Solution Design Approach + Final Model Architecture

1. Lambda layer: image normalization
2. Cropping2D layer: crop off unrelevant area
3. Conv2D+MaxPool2D layer: extract features in the images
4. Dropout layer: prevent overfitting
5. Dense layer: fully connected

The following is the end model structure:

```
model.add(Convolution2D(32, 5, activation='relu'))
model.add(MaxPool2D(3, 3))
model.add(Convolution2D(64, 5, activation='relu'))
model.add(MaxPool2D(3, 3))
model.add(Convolution2D(128, 5, activation='relu'))
model.add(MaxPool2D(3, 3))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

### 10. Creation of the Training Set & Training Process

At each timestamp, total 3 images will be collected using three camera (towards front left, towards front center, towards front right). By using data augumentation, we can transfer the side view images into center view and add them to dataset. Also, by flipping the image, we can also augumente the dataset by twice (not forget also flipp the measurement value). The code block is as following:

```
for line in batch_samples:
    current_steerings = [float(line[3]), float(line[3]) + correction, float(line[3]) - correction]
    for i in range(3):
        name = '/opt/carnd_p3/data/IMG/' + line[i].split('/')[-1]
        img = ndimage.imread(name)
        images.append(img)
        measurements.append(current_steerings[i])
        image_flipped = np.fliplr(img)
        images.append(image_flipped)
        measurements.append(-current_steerings[i])
 ```          
 
By using sklearn pacakge, the input images are successfully seperated into 80/20 training and validation data in the generator. The code block is as following:

```
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
```


