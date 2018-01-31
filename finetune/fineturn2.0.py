# -*- coding: utf-8 -*-
"""
This script reuses pieces of code from the post:
"Building powerful image classification models using very little data"
from blog.keras.io
and from:
https://www.kaggle.com/tnhabc/state-farm-distracted-driver-detection/keras-sample
The training data can be downloaded at:
https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

https://blog.keras.io/

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

https://github.com/fchollet/keras-resources

https://github.com/keras-team/keras/tree/master/examples

https://github.com/keunwoochoi/music-auto_tagging-keras

http://blog.csdn.net/sinat_26917383/article/details/72861152

https://github.com/fchollet/deep-learning-models/releases

"""

import os
import h5py
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from EigenvalueDecay import EigenvalueRegularizer
#from keras import EigenvalueRegularizer
#from regularizers import EigenvalueRegularizer
from numpy.random import permutation
from keras.optimizers import SGD
import pandas as pd
import datetime
import glob
import cv2
import math
import pickle
from collections import OrderedDict
from keras import backend as K
print(keras.__version__)
import sys  

#WEIGHTS_PATH = '/home/ubuntu/keras/animal5/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
#WEIGHTS_PATH_NO_TOP = '/home/ubuntu/keras/animal5/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


model = VGG16(include_top=False, weights='imagenet')
#VGG16(include_top=False, weights='imagenet',
                               # input_tensor=None, input_shape=None,
                                #pooling=None,
                                #classes=1000)




