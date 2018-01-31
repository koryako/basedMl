# importing required libraries
from keras.models import Sequential
from scipy.misc import imread
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import kerasfrom keras.layers import Dense
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
train=pd.read_csv("R/Data/Train/train.csv")
test=pd.read_csv("R/Data/test.csv")
train_path="R/Data/Train/Images/train/"
test_path="R/Data/Train/Images/test/"
#http://yann.lecun.com/exdb/mnist/
from scipy.misc import imresize# preparing the train dataset
train_img=[]
for i in range(len(train)):
    temp_img=image.load_img(train_path+train['filename'][i],target_size=(224,224))
    temp_img=image.img_to_array(temp_img)
    train_img.append(temp_img)
    #converting train images to array and applying mean subtraction processing
    train_img=np.array(train_img) 
    train_img=preprocess_input(train_img)
# applying the same procedure with the test dataset
test_img=[]
for i in range(len(test)):
    temp_img=image.load_img(test_path+test['filename'][i],target_size=(224,224))
    temp_img=image.img_to_array(temp_img)
    test_img.append(temp_img)
    test_img=np.array(test_img) 
    test_img=preprocess_input(test_img)# loading VGG16 model weights
    
model = VGG16(weights='imagenet', include_top=False)

# Extracting features from the train dataset using the VGG16 pre-trained model

features_train=model.predict(train_img)

# Extracting features from the train dataset using the VGG16 pre-trained model

features_test=model.predict(test_img)
# flattening the layers to conform to MLP input

train_x=features_train.reshape(49000,25088)
# converting target variable to array
train_y=np.asarray(train['label'])
# performing one-hot encoding for the target variable
train_y=pd.get_dummies(train_y)
train_y=np.array(train_y)# creating training and validation set

from sklearn.model_selection import train_test_split



X_train, X_valid, Y_train, Y_valid=train_test_split(train_x,train_y,test_size=0.3, random_state=42)# creating a mlp model

from keras.layers import Dense, Activation
model=Sequential()

model.add(Dense(1000, input_dim=25088, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

model.add(Dense(500,input_dim=1000,activation='sigmoid'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

model.add(Dense(150,input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])# fitting the model model.fit(X_train, Y_train, epochs=20, batch_size=128,validation_data=(X_valid,Y_valid))