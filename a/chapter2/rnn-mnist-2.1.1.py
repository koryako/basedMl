'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependencies: keras 
Usage: python3 <this file>
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, SimpleRNN
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_labels = np.amax(y_train)+1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1,image_size*image_size])
x_train = x_train.astype('float32')/255
x_test = np.reshape(x_test, [-1,image_size*image_size])
x_test = x_test.astype('float32')/255

input_size = image_size*image_size
batch_size = 128
hidden_units = 256
dropout = 0.4

model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
# model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
# model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=batch_size)

score = np.asarray(model.evaluate(x_test, y_test, batch_size=batch_size))*100.0
print("\nTest accuracy: %.1f%%" % score[1])
