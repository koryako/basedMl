'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependencies: keras 
Usage: python3 <this file>
'''

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import RMSprop

n_digits = 10
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', \
        strides=2, input_shape=(28, 28, 1), padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu', \
        strides=2))
model.add(Flatten())
model.add(Dense(n_digits, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), \
        metrics=['accuracy'])
model.summary()
