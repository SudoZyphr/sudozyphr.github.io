
# Introduction

Facial Recognition is used in almost every application that is uses some kind of picture. It is primary used in social media to run facial recognition alogithms to find people in other people pictures. Now they even use facial recognition to unlock your cell phone.

My approach will use a machine learning algorithm to take my own dataset as inputs. It will then learn from this data set to attempt to identify me in a crowd in a frame.

OpenCV is the main package that will be used in this program. It uses a lot of algorithms that use machine learning to find specific things in a frame. For my example I am using OpenCV to get a frame of a face.

Cascade are XML files that contain OpenCV data used to detect objects. For my examples I use the front face xml which helps me identify the faces in a video that I am using. This is just a simple detection file that will help me narrow down my model to a specific area in a frame.


```python
for x in {1..50}; do imagesnap -w 2 dataset/User.1.$x.jpg; done

```

This is where I am adding in my dataset. This is a picture of myself to be able to help the model identify who I am. I will take 50 pictures and label them as User.1.

By doing this I will create a dataset folder that I can use to train my model and be able to get a high percentage of probability that I am in a video.



```python
import cv2
import numpy as np
from PIL import Image
import os

path = 'datasetnew'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


```

# The Model

There are a couple of libraries that I had to use in this case. CV2, Numpy, PIL, and OS.

All of these were required to be able to get the information that I needed. The data will be in the format of 300,300,1 were I can feed that into a CNN. With this CNN I will be able to add layers to the neural network that will be able to take the image through layers and make it so that I can package the CNN to better recongize my face.


```python
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

```

# The results

The model will be able to be used when I am capturing video through a camera. The video will use the Cascade Classifier to be able to do frontal face detection. This means it will find my face or other faces in the video. Then it will track my face in the video and track faces by outlining it with a box.

Then based on how well my model was training with the data pictures the model will predict who I am and how confident it was of that decision. The purpose of this project is to be able to find the person of interest out of a group of people. So when the video does play I am looking for the output to predict who that person of interest is.
