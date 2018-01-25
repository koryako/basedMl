import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data
import sys
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


from util.minst_input_data import read_data_sets

#DATA_DIR = '../datasets/minist/'

#data = read_data_sets(DATA_DIR, one_hot=True)

def create_model():
    return model

model = KerasClassifier(build_fn=create_model, epochs=20)


def create_model():
    return model

model = KerasClassifier(build_fn=create_model, epochs=20)


def create_model(dropout_rate=0.0):
    return model

model = KerasClassifier(build_fn=create_model, dropout_rate=0.2)


param_grid = dict(nb_epochs=[10,30,50]) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)


batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)


optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)


learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)


init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)


def build_model():
   model=Sequential()
   model.add(Convolution2D(32,3,3,input_shape=(img_width,img_height,3)))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   
   model.add(Convolution2D(32,3,3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2,2)))

   model.add(Convolution2D(64,3,3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2,2)))
   
   model.add(Flatten())
   model.add(Dense(64))
   model.add(Activation('relu'))
   
   model.add(Dropout(0.5))
   model.add(Dense(1))
   modle.add(Activation('sigmoid'))
